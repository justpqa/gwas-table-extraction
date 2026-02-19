import os
from copy import deepcopy
import io
from typing import List, Union, Tuple, Optional, Any
from collections.abc import Iterable
from tqdm import tqdm
import numpy as np
import pandas as pd
import requests
import re
from pypdf import PdfReader, PdfWriter
from docling.datamodel import vlm_model_specs
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode, VlmPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.document import DocumentStream
from docling.pipeline.vlm_pipeline import VlmPipeline
from gwas_column_matching_engine import GWASColumnMatchingEngine

def extract_tables_num_col_lst_from_pmc(pmcid: str) -> List[int]:
    """
    Extract the number of col of a table to make sure we try the right orientation with docling
    """

    url = f"https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_json/{pmcid}/unicode"

    response = requests.get(url)
    tables_num_col_lst = []
    if response.status_code == 200:
        data = response.json()
        for d in data:
            doc = d["documents"]
            for p in doc:
                passage = p["passages"]
                for item in passage:
                    if item.get("infons", "").get("type", "").lower() == "table" and "text" in item:
                        table_str = item["text"]
                        num_col = 0
                        for row in table_str.split("\t \t"):
                            row_lst = row.split("\t")
                            num_col = max(num_col, len(row_lst))
                        tables_num_col_lst.append(num_col)
        print(f"Successfully retrieve number of columns")
    else:
        print(f"Failed to retrieve number of columns: {response.status_code}")

    return tables_num_col_lst

def clean_cell(val):
    """
    Clean any cell string with the tag for extra note (like "3.14 a")
    """
    tag_pattern = r'\s[a-z]$'
    if isinstance(val, str):
        return re.sub(tag_pattern, '', val)
    return val

def clean_headers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Similar to clean_cell, but for col index of a dataframe
    """
    tag_pattern = r'\s[a-z]$'
    new_cols = []
    seen = {}
    for col in df.columns:
        if pd.isna(col):
            new_cols.append("")
        else:
            # 1. Apply the regex to the name string
            clean_name = re.sub(tag_pattern, '', str(col))
            # 2. Handle duplicates (e.g., if 'Price a' and 'Price b' both become 'Price')
            if clean_name in seen:
                seen[clean_name] += 1
                clean_name = f"{clean_name}_{seen[clean_name]}"
            else:
                seen[clean_name] = 0
            new_cols.append(clean_name)
    df.columns = new_cols
    return df

def extract_tables_lst_from_pdf_and_num_col(file_name: str, tables_num_col_lst: List[int]) -> List[pd.DataFrame]:
    """
    Extract tables from a paper given a file_name string and a list of number of col for each tables 
    (for cross-check to see if we need to do rotation)
    """
    reader = PdfReader(file_name)
    options = PdfPipelineOptions()
    options.table_structure_options.mode = TableFormerMode.ACCURATE
    ocr_converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=options)
        }
    )
    df_lst = []
    row_pattern = r"\b(?:rs|s)\d+\S*"

    page_num = 1
    while len(df_lst) < len(tables_num_col_lst) and page_num <= len(reader.pages):
        filled = False # flag if we found table
        for angle in [0, 90]: # Try normal, then try rotated
            
            writer = PdfWriter()
            page = reader.pages[page_num - 1]
            
            if angle != 0:
                page.rotate(angle)
            
            writer.add_page(page)
            
            # Convert just this one page
            pdf_buffer = io.BytesIO()
            writer.write(pdf_buffer)
            pdf_buffer.seek(0)
            
            doc_stream = DocumentStream(name=f"page_{page_num}.pdf", stream=pdf_buffer)
            result = ocr_converter.convert(doc_stream)

            # Check if this rotation produced valid table rows
            temp_dfs = []
            for table in result.document.tables:
                df = table.export_to_dataframe()
                # check if table is empty
                if (not df.empty):
                    # Case 1: continue from previous table
                    if len(df_lst) > 0 and df.shape[1] == df_lst[-1].shape[1]:
                        # extra filters for tables that are snp related, we need to remove rows that do not have snp id
                        # often are separation between sections
                        for col in df.columns:
                            # first modify "" -> nan
                            df[col] = df[col].replace(r'^\s*$', np.nan, regex=True).ffill()
                        df["valid_row"] = df.apply(lambda x: any(re.search(row_pattern, str(c)) for c in x), axis=1)
                        df = df[df["valid_row"]].drop("valid_row", axis=1).reset_index().drop("index", axis = 1)
                        # some cell have the tag the end (often include a space and a small letter)
                        df = df.map(clean_cell) 
                        df = clean_headers(df) 
                        if df_lst[-1].columns.equals(df.columns):
                            df_lst[-1] = pd.concat([df_lst[-1], df], ignore_index = True)
                            filled = True
                        elif df.shape[1] == tables_num_col_lst[len(df_lst) + len(temp_dfs)]:
                            # fail that test => add to temp since this is a new table
                            temp_dfs.append(df)
                            filled = True
                    # Case 2: new table
                    elif (len(df_lst) + len(temp_dfs)) < len(tables_num_col_lst) and df.shape[1] == tables_num_col_lst[len(df_lst) + len(temp_dfs)]: 
                        # extra filters for tables that are snp related, we need to remove rows that do not have snp id
                        # often are separation between sections
                        for col in df.columns:
                            # first modify "" -> nan
                            df[col] = df[col].replace(r'^\s*$', np.nan, regex=True).ffill()
                        df["valid_row"] = df.apply(lambda x: any(re.search(row_pattern, str(c)) for c in x), axis=1)
                        df = df[df["valid_row"]].drop("valid_row", axis=1).reset_index().drop("index", axis = 1)
                        # some cell have the tag the end (often include a space and a small letter)
                        df = df.map(clean_cell) 
                        df = clean_headers(df)             
                        temp_dfs.append(df)
                        filled = True
            if temp_dfs:
                df_lst.extend(temp_dfs)
            
            if filled:
                break
            
        page_num += 1

    return df_lst

# def extract_tables_lst_from_pdf_and_num_col(file_name: str, tables_num_col_lst: List[int]) -> List[pd.DataFrame]:
#     """
#     Extract tables from a paper given a file_name string and a list of number of col for each tables 
#     (for cross-check to see if we need to do rotation)
#     """
#     print(tables_num_col_lst)
#     reader = PdfReader(file_name)
#     options = PdfPipelineOptions()
#     options.table_structure_options.mode = TableFormerMode.ACCURATE
#     ocr_converter = DocumentConverter(
#         format_options={
#             InputFormat.PDF: PdfFormatOption(pipeline_options=options)
#         }
#     )
#     pipeline_options = VlmPipelineOptions(
#         vlm_options=vlm_model_specs.QWEN25_VL_3B_MLX,
#     )
#     vlm_converter = DocumentConverter(
#         format_options={
#             InputFormat.PDF: PdfFormatOption(
#                 pipeline_cls=VlmPipeline,
#                 pipeline_options=pipeline_options,
#             ),
#         }
#     )
#     df_lst = []
#     row_pattern = r"\b(?:rs|s)\d+\S*"

#     page_num = 1
#     while len(df_lst) < len(tables_num_col_lst) and page_num <= len(reader.pages):
#         filled = False # flag if we found table
#         has_table = False
#         for pipeline in ["OCR", "VLM"]:
#             if pipeline == "OCR" or (has_table and pipeline == "VLM"):
#                 print(f"Start trying {pipeline} pipeline")
#                 for angle in [0, 90]: # Try normal, then try rotated
#                     print(f"Start trying {angle}")
                    
#                     writer = PdfWriter()
#                     page = reader.pages[page_num - 1]
                    
#                     if angle != 0:
#                         page.rotate(angle)
                    
#                     writer.add_page(page)
                    
#                     # Convert just this one page
#                     pdf_buffer = io.BytesIO()
#                     writer.write(pdf_buffer)
#                     pdf_buffer.seek(0)
                    
#                     doc_stream = DocumentStream(name=f"page_{page_num}.pdf", stream=pdf_buffer)
#                     if pipeline == "OCR":
#                         result = ocr_converter.convert(doc_stream)
#                     else:
#                         result = vlm_converter.convert(doc_stream)
                    
#                     if pipeline == "OCR" and len(result.document.tables) > 0:
#                         has_table = True

#                     # Check if this rotation produced valid table rows
#                     temp_dfs = []
#                     for table in result.document.tables:
#                         df = table.export_to_dataframe()
#                         print(df.shape)
#                         # check if table is empty
#                         if (not df.empty):
#                             # Case 1: continue from previous table
#                             if len(df_lst) > 0 and df.shape[1] == df_lst[-1].shape[1]:
#                                 # extra filters for tables that are snp related, we need to remove rows that do not have snp id
#                                 # often are separation between sections
#                                 for col in df.columns:
#                                     # first modify "" -> nan
#                                     df[col] = df[col].replace(r'^\s*$', np.nan, regex=True).ffill()
#                                 df["valid_row"] = df.apply(lambda x: any(re.search(row_pattern, str(c)) for c in x), axis=1)
#                                 df = df[df["valid_row"]].drop("valid_row", axis=1).reset_index().drop("index", axis = 1)
#                                 # some cell have the tag the end (often include a space and a small letter)
#                                 df = df.map(clean_cell) 
#                                 df = clean_headers(df) 
#                                 if df_lst[-1].columns.equals(df.columns):
#                                     df_lst[-1] = pd.concat([df_lst[-1], df], ignore_index = True)
#                                     filled = True
#                                 elif df.shape[1] == tables_num_col_lst[len(df_lst) + len(temp_dfs)]:
#                                     # fail that test => add to temp since this is a new table
#                                     temp_dfs.append(df)
#                                     filled = True
#                             # Case 2: new table
#                             elif (len(df_lst) + len(temp_dfs)) < len(tables_num_col_lst) and df.shape[1] == tables_num_col_lst[len(df_lst) + len(temp_dfs)]: 
#                                 # extra filters for tables that are snp related, we need to remove rows that do not have snp id
#                                 # often are separation between sections
#                                 for col in df.columns:
#                                     # first modify "" -> nan
#                                     df[col] = df[col].replace(r'^\s*$', np.nan, regex=True).ffill()
#                                 df["valid_row"] = df.apply(lambda x: any(re.search(row_pattern, str(c)) for c in x), axis=1)
#                                 df = df[df["valid_row"]].drop("valid_row", axis=1).reset_index().drop("index", axis = 1)
#                                 # some cell have the tag the end (often include a space and a small letter)
#                                 df = df.map(clean_cell) 
#                                 df = clean_headers(df)             
#                                 temp_dfs.append(df)
#                                 filled = True
#                     print(filled)
#                     if temp_dfs:
#                         df_lst.extend(temp_dfs)
                    
#                     if filled:
#                         break
            
#             if filled:
#                 break
            
#         page_num += 1

#     return df_lst

def extract_tables_lst_from_paper(pmcid: str, file_name: str, table_inx_to_extract: List[int] = []) -> list[pd.DataFrame]:
    """
    Given a paper with pmcid and file name of the pdf file, extract all tables
    """
    tables_num_col_lst = extract_tables_num_col_lst_from_pmc(pmcid)
    if len(table_inx_to_extract) > 0:
        tables_num_col_lst = [tables_num_col_lst[i] for i in table_inx_to_extract]
    df_lst = extract_tables_lst_from_pdf_and_num_col(file_name, tables_num_col_lst)
    return df_lst

# def create_embeddings_from_model(sentences: str | List[str], model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase):
#     """
#     Generate embeddings from any model with mean pooling for sentence embeddings
#     """
#     input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

#     # get token embeddings
#     with torch.no_grad():
#         output = model(**input)
#     token_embeddings = output[0]

#     # extract mask and mean pooling for sentence embeddings
#     input_mask_expanded = input['attention_mask'].unsqueeze(-1).expand(token_embeddings.size()).float()
#     sentence_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

#     # final normalization
#     sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

#     return sentence_embeddings

# def clean_col(col: str) -> str:
#     """
#     Clean a column by replacing any possible abbreviation with their actual meaning for better semantic matching
#     """
#     gwas_abbreviation_dict = {
#         "CHR": "Chromosome number",
#         "BP": "Base-pair position",
#         "POS": "Position",
#         "SNP": "Single nucleotide polymorphism identifier",
#         "RS": "Reference Single nucleotide polymorphism",
#         "VAR": "Variant",
#         "ID": "identifier",
#         "A1": "Effect allele / tested allele",
#         "A2": "Other allele / non-effect allele",
#         "REF": "Reference allele (genome reference)",
#         "ALT": "Alternate allele",
#         "EA": "Effect Allele",
#         "NEA": "Non-Effect Allele",
#         "RA": "Risk Allele",
#         "OA": "Other Allele",
#         "AF": "Allele Frequency (general term)",
#         "RAF": "Risk Allele Frequency",
#         "EAF": "Effect Allele Frequency",
#         "MAF": "Minor Allele Frequency",
#         "BETA": "Effect size (regression coefficient)",
#         "OR": "Odds Ratio",
#         "SE": "Standard Error of effect estimate",
#         "Z": "Z-score statistic",
#         "T": "T-statistic",
#         "CI": "Confidence Interval",
#         "P": "P-value",
#         "PVAL": "P-value",
#         "LOGP": "Negative log10 P-value",
#         "Q": "Heterogeneity statistic (meta-analysis)",
#         "I2": "I-squared heterogeneity metric",
#         "HWE": "Hardy-Weinberg Equilibrium test statistic",
#         "INFO": "Imputation quality score",
#         "R2": "Imputation accuracy metric",
#         "CALLRATE": "Genotype call rate",
#         "MISSING": "Missing genotype rate",
#         "N": "Total sample size",
#         "N_CASES": "Number of cases (for binary traits)",
#         "N_CONTROLS": "Number of controls (for binary traits)",
#         "EUR": "European ancestry",
#         "AFR": "African ancestry",
#         "ASN": "Asian ancestry",
#         "AMR": "Admixed American ancestry",
#         "SAS": "South Asian ancestry",
#         "EAS": "East Asian ancestry",
#         "LD": "Linkage Disequilibrium",
#         "DPRIME": "LD D’ value",
#         "CADD": "CADD score (functional impact)",
#         "EQTL": "Expression quantitative trait locus",
#         "PQTL": "Protein QTL",
#         "GWGAS": "Gene-wide association study",
#         "PRS": "Polygenic Risk Score",
#         "PGS": "Polygenic Score",
#         "QC": "Quality Control",
#         "MA": "Meta-analysis",
#         "HLA": "Human Leukocyte Antigen region",
#         "HR": "Hazard ratio",
#         "HET": "Heterogeneity test",
#         "APOE4": "APOE ε4",
#         "APOE*4": "APOE ε4",
#         "#": "Number of",
#         "frq": "Frequency",
#         'β': "Effect",
#         "nsnps": "Number of Variants"
#     }
    
#     new_col = col
#     for abb in gwas_abbreviation_dict:
#         if re.search(fr"[^a-zA-Z]{abb.lower()}[^a-zA-Z]", new_col.lower()):
#             new_col = re.sub(fr"([^a-zA-Z]){abb.lower()}([^a-zA-Z])", fr"\1{gwas_abbreviation_dict[abb]}\2", new_col.lower())
#         elif re.search(fr"^{abb.lower()}[^a-zA-Z]", new_col.lower()):
#             new_col = re.sub(fr"^{abb.lower()}([^a-zA-Z])", fr"{gwas_abbreviation_dict[abb]}\1", new_col.lower())
#         elif re.search(fr"[^a-zA-Z]{abb.lower()}$", new_col.lower()):
#             new_col = re.sub(fr"([^a-zA-Z]){abb.lower()}$", fr"\1{gwas_abbreviation_dict[abb]}", new_col.lower())
#         elif re.search(fr"^{abb.lower()}$", new_col.lower()):
#             new_col = re.sub(fr"{abb.lower()}", gwas_abbreviation_dict[abb], new_col.lower())
#     # new_col = new_col.replace(".", " ")
#     return new_col

# def make_col_prompt(col: str, example_values: Iterable, num_example_values: int = 5):
#     """
#     Based on the column title and some possible values of that columns, try to make a prompt
#     """
#     col_prompt = f"{col}: "
#     for i in range(min(num_example_values, len(example_values))):
#         col_prompt += f"{example_values[i]}, "
#     return col_prompt

# def match_single_col_to_ref_col(col: str, ref_col_lst: List[str], ref_col_embeddings: np.ndarray | torch.Tensor,
#                                 embeddings_model: PreTrainedModel, embeddings_model_tokenizer: PreTrainedTokenizerBase):
#     """
#     match a single column to reference col given column, the list of referencing col and their embeddings
#     """
#     # calculate column embedding
#     # col_embeddings = embeddings_model.encode(col, normalize_embeddings=True)
#     # multi_index_pattern = r".+\..+"
#     # if re.search(multi_index_pattern, col):
#     #     col = col.replace(".", " ")
#     #     # sub_col_lst = col.split(".")
#     #     # sub_col_embeddings = create_embeddings_from_model(sub_col_lst, embeddings_model, embeddings_model_tokenizer)
#     #     # col_embeddings = torch.mean(sub_col_embeddings, dim = 0)
#     #     col = col.replace(".", " ")
#     #     col_embeddings = create_embeddings_from_model(col, embeddings_model, embeddings_model_tokenizer)
#     # else:
#     #     col_embeddings = create_embeddings_from_model(col, embeddings_model, embeddings_model_tokenizer)
#     col_embeddings = create_embeddings_from_model(col, embeddings_model, embeddings_model_tokenizer)
#     col_embeddings = col_embeddings.reshape(-1, 1)

#     # calculate similarity score
#     scores = torch.matmul(ref_col_embeddings, col_embeddings).reshape(-1)

#     # sort similairty score
#     top_k_indices = torch.argsort(scores, descending=True)

#     # verify if we even got good enough similarity
#     best_inx = top_k_indices[0].item()
#     best_score = scores[best_inx].item()
#     # second_best_inx = top_k_indices[1].item()
#     # second_best_score = scores[second_best_inx].item()
#     # need a threshold for score or else, just return col
#     # if best_score < 0.4: 
#     #     return (col, 1)
    
#     # # now do rerank
#     # candidates = [ref_col_lst[inx] for inx in top_k_indices if scores[inx] >= 0.4]
#     # candidates_scores = [scores[inx] for inx in top_k_indices if scores[inx] >= 0.4]
#     # best_ref_col, best_ref_col_scores = reranking_from_model(col, candidates, candidates_scores)
#     # return (best_ref_col, best_ref_col_scores)

#     if best_score >= 0.4:
#         return (ref_col_lst[best_inx], best_score)
#     return (col, 1)

# def match_many_col_to_ref_col(df: pd.DataFrame, ref_col_df: pd.DataFrame, 
#                               embeddings_model: PreTrainedModel, embeddings_model_tokenizer: PreTrainedTokenizerBase):
#     """
#     Given a list of column and a dataframe (could be dict later on if that fits better),
#     try to match each column to the best fitted reference col, 
#     return a dict of ref col : list of (col, cleaned col prompt, score)
#     """
#     # prepare the embeddings for reference col since we reuse them
#     ref_col_lst = ref_col_df["column"].to_list()
#     ref_col_context_lst = ref_col_df["column_with_context"].to_list()
#     ref_col_embeddings = create_embeddings_from_model(ref_col_context_lst, embeddings_model, embeddings_model_tokenizer)

#     # conduct matching
#     multi_index_pattern = r"^.+\..+$" # need multi index pattern for handling multi index
#     ref_col_to_col_lst = {}
#     for col in df.columns:

#         # extract values needed for prompt
#         cleaned_col = clean_col(col)
#         example_values = df[col].unique().tolist()
#         # prompt: {col}: example, need to delete all : first

#         if re.search(multi_index_pattern, cleaned_col.strip()):
#             # try to assess each part and see if which one have highest score
#             best_ref_col, best_score = None, 0
#             best_cleaned_sub_col_prompt = None
#             for sub_col in cleaned_col.split("."):

#                 # make column prompt for each sub col
#                 cleaned_sub_col_prompt = make_col_prompt(sub_col, example_values)
                
#                 # matching and compare
#                 ref_col, score = match_single_col_to_ref_col(cleaned_sub_col_prompt, ref_col_lst, ref_col_embeddings, embeddings_model, embeddings_model_tokenizer)
#                 if score > best_score and ref_col != cleaned_sub_col_prompt:
#                     best_ref_col = ref_col
#                     best_score = score
#                     best_cleaned_sub_col_prompt = cleaned_sub_col_prompt
            
#             # if we have a best one vs not
#             if best_ref_col is not None:
#                 if best_ref_col not in ref_col_to_col_lst:
#                     ref_col_to_col_lst[best_ref_col] = []
#                 ref_col_to_col_lst[best_ref_col].append((col, best_cleaned_sub_col_prompt, score))
#             else:
#                 if col not in ref_col_to_col_lst:
#                     ref_col_to_col_lst[col] = []
#                 ref_col_to_col_lst[col].append((col, best_cleaned_sub_col_prompt, 1))
#         else:
#             # match a single col with best fit referencing col and return a dict
#             # make the col prompt
#             # extra steps to remove .
#             cleaned_col = cleaned_col.replace(".", "")
#             cleaned_col_prompt = make_col_prompt(cleaned_col, example_values)

#             # matching for best col
#             ref_col, score = match_single_col_to_ref_col(cleaned_col_prompt, ref_col_lst, ref_col_embeddings, embeddings_model, embeddings_model_tokenizer)
#             # if we still get same col
#             if ref_col == cleaned_col_prompt: 
#                 ref_col = col
#             if ref_col not in ref_col_to_col_lst:
#                 ref_col_to_col_lst[ref_col] = []
#             ref_col_to_col_lst[ref_col].append((col, cleaned_col_prompt, score))
#         # ref_col, score = match_single_col_to_ref_col(cleaned_col_prompt, ref_col_lst, ref_col_embeddings)
#         # if ref_col == cleaned_col_prompt: 
#         #     ref_col = col
#         # if ref_col not in ref_col_to_col_lst:
#         #     ref_col_to_col_lst[ref_col] = []
#         # ref_col_to_col_lst[ref_col].append((col, cleaned_col_prompt, score))

#     return ref_col_to_col_lst

def format_original_table(df: pd.DataFrame, gwas_column_matching_engine: GWASColumnMatchingEngine, remove_unique_col: bool = False):
    """
    Format final table by melt/make extra copies of reference columns, or simply rename column to reference column name
    """
    possible_ref_col_to_melt = ["P-value", "Effect Size", "AF"]
    # map the columns
    new_col_to_old_col_lst = gwas_column_matching_engine.match_many_col_to_ref_col(df)
    df_with_ref_col = None
    new_col_to_not_melt = [] # list of columns that are stable and not need to be melt
    new_col_to_old_col_lst_to_melt = {}
    for new_col in new_col_to_old_col_lst:
        if len(new_col_to_old_col_lst[new_col]) == 1:
            if (not remove_unique_col) or (remove_unique_col and new_col_to_old_col_lst[new_col][0][2] != 1):
                if df_with_ref_col is None:
                    df_with_ref_col = df[[new_col_to_old_col_lst[new_col][0][0]]]
                    df_with_ref_col = df_with_ref_col.rename({new_col_to_old_col_lst[new_col][0][0]: new_col}, axis = 1)
                else:
                    df_with_ref_col[new_col] = df[new_col_to_old_col_lst[new_col][0][0]]
                # add these single col to the list of not melt
                new_col_to_not_melt.append(new_col)
        else:
            if new_col in possible_ref_col_to_melt:
                old_col_lst = []
                for col, _, _ in new_col_to_old_col_lst[new_col]:
                    old_col_lst.append(col)
                    if df_with_ref_col is None:
                        df_with_ref_col = df[[col]]
                    else:
                        df_with_ref_col[col] = df[col]
                new_col_to_old_col_lst_to_melt[new_col] = old_col_lst.copy()
            else:
                # make multiple copies with notes
                for inx, (col, _, _) in enumerate(new_col_to_old_col_lst[new_col]):
                    if df_with_ref_col is None:
                        df_with_ref_col = df[[col]]
                        df_with_ref_col = df_with_ref_col.rename({col: f"{new_col}_{inx + 1}"}, axis = 1)
                    else:
                        df_with_ref_col[f"{new_col}_{inx + 1}"] = df[col]
                    df_with_ref_col[f"{new_col}_{inx + 1} notes"] = col
                    # add these cols in group but not need to melt
                    new_col_to_not_melt.append(f"{new_col}_{inx + 1}")
                    new_col_to_not_melt.append(f"{new_col}_{inx + 1} notes")
    # Melting stage
    if len(new_col_to_old_col_lst_to_melt) > 0:
        # now melting column in same groups
        # Instead of keep melting, for each group, we make a new dataset of 
        # [stable col] + [to be melt col] => melt them as a new df
        # do this for each gorup and then join together based on stable col
        df_with_melt_col = None 
        for new_col in new_col_to_old_col_lst_to_melt:
            temp_df = deepcopy(df_with_ref_col[new_col_to_not_melt + new_col_to_old_col_lst_to_melt[new_col]])
            # create a temp row id for stable join
            temp_df["_row_id"] = np.arange(temp_df.shape[0])
            temp_df = temp_df.melt(
                id_vars = new_col_to_not_melt + ["_row_id"],    
                value_vars = new_col_to_old_col_lst_to_melt[new_col], 
                var_name = f"{new_col} notes",      
                value_name = f"{new_col}"
            )
            if df_with_melt_col is None:
                df_with_melt_col = deepcopy(temp_df)
            else:
                df_with_melt_col = df_with_melt_col.merge(temp_df, how = "inner", on = ["_row_id"] + new_col_to_not_melt)
        df_with_melt_col = df_with_melt_col.drop("_row_id", axis = 1)
        return df_with_melt_col
    else:
        return df_with_ref_col
    
if __name__ == "__main__":
    # Extract on some example papers
    pmcid = "PMC10497850"
    file_name = "papers/ACEL-22-e13938.pdf"
    df_lst = extract_tables_lst_from_paper(pmcid, file_name)
    for i in [3, 4]:
        df_lst[i-1].to_csv(f"tables/{file_name.split('/')[-1].replace('.pdf', '')}_table_{i}.csv", index=False)

    pmcid = "PMC10115645"
    file_name = "papers/41591_2023_Article_2268.pdf"  # Can be a local path or a URL
    df_lst = extract_tables_lst_from_paper(pmcid, file_name)
    for i in [1, 2]:
        df_lst[i-1].to_csv(f"tables/{file_name.split('/')[-1].replace('.pdf', '')}_table_{i}.csv", index=False)

    pmcid = "PMC9622429"
    file_name = "papers/nihms-1797266.pdf"  # Can be a local path or a URL
    df_lst = extract_tables_lst_from_paper(pmcid, file_name)  
    for i in [3, 5, 6, 7, 8]:
        df_lst[i-1].to_csv(f"tables/{file_name.split('/')[-1].replace('.pdf', '')}_table_{i}.csv", index=False)

    pmcid = "PMC10615750"
    file_name = "papers/Recent paper on AD GWAS (1).pdf"  # Can be a local path or a URL
    df_lst = extract_tables_lst_from_paper(pmcid, file_name)  
    for i in [1, 2]:
        df_lst[i-1].to_csv(f"tables/{file_name.split('/')[-1].replace('.pdf', '')}_table_{i}.csv", index=False)

    pmcid = "PMC6677735"
    file_name = "papers/s42003-019-0537-9.pdf"  # Can be a local path or a URL
    df_lst = extract_tables_lst_from_paper(pmcid, file_name)  
    for i in [1]:
        df_lst[i-1].to_csv(f"tables/{file_name.split('/')[-1].replace('.pdf', '')}_table_{i}.csv", index=False)

    pmcid = "PMC10286470"
    file_name = "papers/s13024-023-00633-4.pdf"  # Can be a local path or a URL
    df_lst = extract_tables_lst_from_paper(pmcid, file_name)  
    for i in [2, 3]:
        df_lst[i-1].to_csv(f"tables/{file_name.split('/')[-1].replace('.pdf', '')}_table_{i}.csv", index=False)

    # get the df for referencing column
    # Now try to map the columns with the actual col in advp
    referencing_col_df = pd.read_csv("Rules for harmonizing ADVP papers - Main cols.csv")
    referencing_col_df["column_with_context"] = referencing_col_df.apply(lambda x: x["column"] if pd.isna(x["description"]) else x["column"] + ": " + x["description"], axis = 1)

    gwas_column_matching_engine = GWASColumnMatchingEngine(referencing_col_df)

    # format table as needed
    modified_df_all = None
    for file in os.listdir("./tables"):
        if ".csv" in file and "table" in file and "harmonized" not in file:
            df = pd.read_csv(f"./tables/{file}")
            df.columns = ['' if 'Unnamed:' in col else col for col in df.columns]
            modified_df = format_original_table(df, gwas_column_matching_engine, remove_unique_col = False)
            modified_df["file_name"] = file
            if modified_df_all is None:
                modified_df_all = modified_df.copy()
            else:
                modified_df_all = pd.concat([modified_df_all, modified_df], ignore_index = True)
            modified_df.to_csv(f"./harmonized_tables/{file.replace('.csv', '')}_harmonized.csv", index = False)
            print(f"Success in harmonizing tables from {file}")
    modified_df_all.to_csv("./harmonized_tables/harmonized_table.csv", index = False)