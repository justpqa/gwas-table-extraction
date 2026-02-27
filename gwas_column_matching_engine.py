import re
from typing import List, Union, Tuple, Dict
from collections.abc import Iterable
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, BitsAndBytesConfig, LogitsProcessor, LogitsProcessorList
    
class GWASColumnMatchingEngine:
    def __init__(self, referencing_col_df: pd.DataFrame, embeddings_model_name: str = "NeuML/pubmedbert-base-embeddings"):
        # df of referencing col
        self.referencing_col_lst = referencing_col_df["column"].to_list()
        self.referencing_col_context_lst = referencing_col_df.apply(lambda x: x["column"] if pd.isna(x["description"]) else x["column"] + ": " + x["description"], axis = 1).to_list()

        # Try to first convert any abbreviation, then we match column with right semantic
        self.gwas_abbreviation_dict = {
            "CHR": "Chromosome number",
            "BP": "Base-pair position",
            "POS": "Position",
            "SNP": "Single nucleotide polymorphism identifier",
            "RS": "Reference Single nucleotide polymorphism",
            "VAR": "Variant",
            "ID": "identifier",
            "A1": "Effect allele / tested allele",
            "A2": "Other allele / non-effect allele",
            "REF": "Reference allele (genome reference)",
            "ALT": "Alternate allele",
            "EA": "Effect Allele",
            "NEA": "Non-Effect Allele",
            "RA": "Risk Allele",
            "OA": "Other Allele",
            "AF": "Allele Frequency (general term)",
            "RAF": "Risk Allele Frequency",
            "EAF": "Effect Allele Frequency",
            "MAF": "Minor Allele Frequency",
            "BETA": "Effect size (regression coefficient)",
            "OR": "Odds Ratio",
            "SE": "Standard Error of effect estimate",
            "Z": "Z-score statistic",
            "T": "T-statistic",
            "CI": "Confidence Interval",
            "P": "P-value",
            "PVAL": "P-value",
            "LOGP": "Negative log10 P-value",
            "Q": "Heterogeneity statistic (meta-analysis)",
            "I2": "I-squared heterogeneity metric",
            "HWE": "Hardy-Weinberg Equilibrium test statistic",
            "INFO": "Imputation quality score",
            "R2": "Imputation accuracy metric",
            "CALLRATE": "Genotype call rate",
            "MISSING": "Missing genotype rate",
            "N": "Total sample size",
            "N_CASES": "Number of cases (for binary traits)",
            "N_CONTROLS": "Number of controls (for binary traits)",
            "EUR": "European ancestry",
            "AFR": "African ancestry",
            "ASN": "Asian ancestry",
            "AMR": "Admixed American ancestry",
            "SAS": "South Asian ancestry",
            "EAS": "East Asian ancestry",
            "LD": "Linkage Disequilibrium",
            "DPRIME": "LD D’ value",
            "CADD": "CADD score (functional impact)",
            "EQTL": "Expression quantitative trait locus",
            "PQTL": "Protein QTL",
            "GWGAS": "Gene-wide association study",
            "PRS": "Polygenic Risk Score",
            "PGS": "Polygenic Score",
            "QC": "Quality Control",
            "MA": "Meta-analysis",
            "HLA": "Human Leukocyte Antigen region",
            "HR": "Hazard ratio",
            "HET": "Heterogeneity test",
            "APOE4": "APOE ε4",
            "APOE*4": "APOE ε4",
            "#": "Number of",
            "frq": "Frequency",
            'β': "Effect",
            "nsnps": "Number of Variants"
        }

        # embeddings model
        self.embeddings_model_tokenizer = AutoTokenizer.from_pretrained(embeddings_model_name)
        self.embeddings_model = AutoModel.from_pretrained(embeddings_model_name)
        self.embeddings_model.eval()

        # also make col embeddings
        self.referencing_col_embeddings = self.create_col_embeddings_from_model(self.referencing_col_context_lst)

        # # llm model
        # self.use_llm = use_llm
        # if use_llm:
        #     self.llm_model_tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
        #     self.llm_model = AutoModelForCausalLM.from_pretrained(
        #         llm_model_name,
        #         torch_dtype=torch.bfloat16
        #     ).to(device)
        #     self.llm_model.eval()

        # # device
        # self.device = device

        # column list
        # self.col_with_multiple_copies = ["P-value", "Effect Size", "AF"]

    def clean_col(self, col: str) -> str:
        """
        Clean a column by replacing any possible abbreviation with their actual meaning for better semantic matching
        """
        new_col = col
        for abb in self.gwas_abbreviation_dict:
            if re.search(fr"[^a-zA-Z]{abb.lower()}[^a-zA-Z]", new_col.lower()):
                new_col = re.sub(fr"([^a-zA-Z]){abb.lower()}([^a-zA-Z])", fr"\1{self.gwas_abbreviation_dict[abb]}\2", new_col.lower())
            elif re.search(fr"^{abb.lower()}[^a-zA-Z]", new_col.lower()):
                new_col = re.sub(fr"^{abb.lower()}([^a-zA-Z])", fr"{self.gwas_abbreviation_dict[abb]}\1", new_col.lower())
            elif re.search(fr"[^a-zA-Z]{abb.lower()}$", new_col.lower()):
                new_col = re.sub(fr"([^a-zA-Z]){abb.lower()}$", fr"\1{self.gwas_abbreviation_dict[abb]}", new_col.lower())
            elif re.search(fr"^{abb.lower()}$", new_col.lower()):
                new_col = re.sub(fr"{abb.lower()}", self.gwas_abbreviation_dict[abb], new_col.lower())
        # new_col = new_col.replace(".", " ")
        return new_col
    
    def make_col_prompt(self, col: str, example_values: Iterable, num_example_values: int = 5) -> str:
        """
        Based on the column title and some possible values of that columns, try to make a prompt
        """
        col_prompt = f"{col}: "
        for i in range(min(num_example_values, len(example_values))):
            col_prompt += f"{example_values[i]}, "
        return col_prompt

    def create_col_embeddings_from_model(self, col: str | List[str]) -> np.ndarray | torch.Tensor:
        """
        Create embeddings from string represent col name or a prompt of that col
        """
        input = self.embeddings_model_tokenizer(col, padding=True, truncation=True, return_tensors='pt')

        # get token embeddings
        with torch.no_grad():
            output = self.embeddings_model(**input)
        token_embeddings = output[0]

        # extract mask and mean pooling for sentence embeddings
        input_mask_expanded = input['attention_mask'].unsqueeze(-1).expand(token_embeddings.size()).float()
        col_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

        # final normalization
        col_embeddings = F.normalize(col_embeddings, p=2, dim=1)

        return col_embeddings
        
    def match_single_col_to_ref_col(self, col: str) -> Tuple[str, float]:
        """
        match a single column to reference col given column, the list of referencing col and their embeddings
        """
        # calculate column embedding
        # col_embeddings = embeddings_model.encode(col, normalize_embeddings=True)
        # multi_index_pattern = r".+\..+"
        # if re.search(multi_index_pattern, col):
        #     col = col.replace(".", " ")
        #     # sub_col_lst = col.split(".")
        #     # sub_col_embeddings = create_embeddings_from_model(sub_col_lst, embeddings_model, embeddings_model_tokenizer)
        #     # col_embeddings = torch.mean(sub_col_embeddings, dim = 0)
        #     col = col.replace(".", " ")
        #     col_embeddings = create_embeddings_from_model(col, embeddings_model, embeddings_model_tokenizer)
        # else:
        #     col_embeddings = create_embeddings_from_model(col, embeddings_model, embeddings_model_tokenizer)
        col_embeddings = self.create_col_embeddings_from_model(col)
        col_embeddings = col_embeddings.reshape(-1, 1)

        # calculate similarity score
        scores = torch.matmul(self.referencing_col_embeddings, col_embeddings).reshape(-1)

        # sort similairty score
        top_k_indices = torch.argsort(scores, descending=True)

        # verify if we even got good enough similarity
        best_inx = top_k_indices[0].item()
        best_score = scores[best_inx].item()
        # second_best_inx = top_k_indices[1].item()
        # second_best_score = scores[second_best_inx].item()
        # need a threshold for score or else, just return col
        # if best_score < 0.4: 
        #     return (col, 1)
        
        # # now do rerank
        # candidates = [ref_col_lst[inx] for inx in top_k_indices if scores[inx] >= 0.4]
        # candidates_scores = [scores[inx] for inx in top_k_indices if scores[inx] >= 0.4]
        # best_ref_col, best_ref_col_scores = reranking_from_model(col, candidates, candidates_scores)
        # return (best_ref_col, best_ref_col_scores)

        if best_score >= 0.4:
            return (self.referencing_col_lst[best_inx], best_score)
        return (col, 1)

    def match_many_col_to_ref_col(self, df: pd.DataFrame) -> Dict:
        """
        Given a list of column and a dataframe (could be dict later on if that fits better),
        try to match each column to the best fitted reference col, 
        return a dict of ref col : list of (col, cleaned col prompt, score)
        """

        # conduct matching
        multi_index_pattern_1 = r"^.+\..+$" # need multi index pattern for handling multi index
        multi_index_pattern_2 = r"^.+\|.+$"
        ref_col_to_col_lst = {}
        for col in df.columns:

            # extract values needed for prompt
            cleaned_col = self.clean_col(col)
            example_values = df[col].unique().tolist()
            # prompt: {col}: example, need to delete all : first

            if re.search(multi_index_pattern_1, cleaned_col.strip()):
                # try to assess each part and see if which one have highest score
                best_ref_col, best_score = None, 0
                best_cleaned_sub_col_prompt = None
                for sub_col in cleaned_col.split("."):

                    # make column prompt for each sub col
                    cleaned_sub_col_prompt = self.make_col_prompt(sub_col, example_values)
                    
                    # matching and compare
                    ref_col, score = self.match_single_col_to_ref_col(cleaned_sub_col_prompt)
                    if score > best_score and ref_col != cleaned_sub_col_prompt:
                        best_ref_col = ref_col
                        best_score = score
                        best_cleaned_sub_col_prompt = cleaned_sub_col_prompt
                
                # if we have a best one vs not
                if best_ref_col is not None:
                    if best_ref_col not in ref_col_to_col_lst:
                        ref_col_to_col_lst[best_ref_col] = []
                    ref_col_to_col_lst[best_ref_col].append((col, best_cleaned_sub_col_prompt, score))
                else:
                    if col not in ref_col_to_col_lst:
                        ref_col_to_col_lst[col] = []
                    ref_col_to_col_lst[col].append((col, best_cleaned_sub_col_prompt, 1))
            elif re.search(multi_index_pattern_2, cleaned_col.strip()):
                # try to assess each part and see if which one have highest score
                best_ref_col, best_score = None, 0
                best_cleaned_sub_col_prompt = None
                for sub_col in cleaned_col.split("|"):

                    # make column prompt for each sub col
                    cleaned_sub_col_prompt = self.make_col_prompt(sub_col, example_values)
                    
                    # matching and compare
                    ref_col, score = self.match_single_col_to_ref_col(cleaned_sub_col_prompt)
                    if score > best_score and ref_col != cleaned_sub_col_prompt:
                        best_ref_col = ref_col
                        best_score = score
                        best_cleaned_sub_col_prompt = cleaned_sub_col_prompt
                
                # if we have a best one vs not
                if best_ref_col is not None:
                    if best_ref_col not in ref_col_to_col_lst:
                        ref_col_to_col_lst[best_ref_col] = []
                    ref_col_to_col_lst[best_ref_col].append((col, best_cleaned_sub_col_prompt, score))
                else:
                    if col not in ref_col_to_col_lst:
                        ref_col_to_col_lst[col] = []
                    ref_col_to_col_lst[col].append((col, best_cleaned_sub_col_prompt, 1))
            else:
                # match a single col with best fit referencing col and return a dict
                # make the col prompt
                # extra steps to remove .
                cleaned_col = cleaned_col.replace(".", "")
                cleaned_col_prompt = self.make_col_prompt(cleaned_col, example_values)

                # matching for best col
                ref_col, score = self.match_single_col_to_ref_col(cleaned_col_prompt)
                # if we still get same col
                if ref_col == cleaned_col_prompt: 
                    ref_col = col
                if ref_col not in ref_col_to_col_lst:
                    ref_col_to_col_lst[ref_col] = []
                ref_col_to_col_lst[ref_col].append((col, cleaned_col_prompt, score))
            # ref_col, score = match_single_col_to_ref_col(cleaned_col_prompt, ref_col_lst, ref_col_embeddings)
            # if ref_col == cleaned_col_prompt: 
            #     ref_col = col
            # if ref_col not in ref_col_to_col_lst:
            #     ref_col_to_col_lst[ref_col] = []
            # ref_col_to_col_lst[ref_col].append((col, cleaned_col_prompt, score))

        return ref_col_to_col_lst