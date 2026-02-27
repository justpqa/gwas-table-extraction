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

def contains_valid_snp(row):
    row_snp_pattern = r"\b(?:rs|s)\d+\S*"
    return any(re.search(row_snp_pattern, str(value)) for value in row)

def contains_valid_pvalue(row):
    row_pvalue_pattern = r"\d+\.\d+"
    return any(re.search(row_pvalue_pattern, str(value)) for value in row)

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
                        df["valid_row"] = df.apply(lambda x: contains_valid_snp(x) and contains_valid_pvalue(x), axis=1)
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
                        df["valid_row"] = df.apply(lambda x: contains_valid_snp(x) and contains_valid_pvalue(x), axis=1)
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

    # final filter, since we only consider df with snp
    df_lst = [df for df in df_lst if df.shape[0] > 0]
    return df_lst

def extract_tables_lst_from_paper(pmcid: str, file_name: str, table_inx_to_extract: List[int] = []) -> list[pd.DataFrame]:
    """
    Given a paper with pmcid and file name of the pdf file, extract all tables
    """
    tables_num_col_lst = extract_tables_num_col_lst_from_pmc(pmcid)
    if len(table_inx_to_extract) > 0:
        tables_num_col_lst = [tables_num_col_lst[i] for i in table_inx_to_extract]
    df_lst = extract_tables_lst_from_pdf_and_num_col(file_name, tables_num_col_lst)
    return df_lst

def format_original_table_from_col_matching(df: pd.DataFrame, new_col_to_old_col_lst: dict, remove_unique_col: bool = False):
    """
    Format final table by melt/make extra copies of reference columns, or simply rename column to reference column name
    """
    possible_ref_col_to_melt = ["P-value", "Effect", "AF"]
    # map the columns
    # new_col_to_old_col_lst = gwas_column_matching_engine.match_many_col_to_ref_col(df)
    df_with_ref_col = None
    new_col_to_not_melt = [] # list of columns that are stable and not need to be melt
    new_col_to_old_col_lst_to_melt = {}
    for new_col in new_col_to_old_col_lst:
        if len(new_col_to_old_col_lst[new_col]) == 1:
            # check if we remove unique col or not and if yes, is this a case of a col map to itself (score = 1)
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
                    # NOTE: special case, if col is the same as new_col, then melt will get error
                    if col == new_col:
                        old_col_lst.append(col + " ")
                    else:
                        old_col_lst.append(col)
                    if df_with_ref_col is None:
                        if col == new_col:
                            df_with_ref_col = df[[col]].rename({col: col + " "}, axis = 1)
                        else:
                            df_with_ref_col = df[[col]]
                    else:
                        if col == new_col:
                            df_with_ref_col[col + " "] = df[col]
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

def format_original_table(df: pd.DataFrame, gwas_column_matching_engine: GWASColumnMatchingEngine, remove_unique_col: bool = False):
    """
    Format final table by running pipeline of col matching -> final format
    """
    # first do matching
    ref_col_to_col_lst = gwas_column_matching_engine.match_many_col_to_ref_col(df)

    # finally, filter col that cannot have multiple copies
    # col_with_multiple_copies = ["P-value", "Effect", "AF"]
    # for ref_col in ref_col_to_col_lst:
    #     if ref_col not in col_with_multiple_copies and len(ref_col_to_col_lst[ref_col]) > 1:
    #         best_col, best_cleaned_col_prompt, best_score = None, None, float("-inf")
    #         for col, cleaned_col_prompt, score in ref_col_to_col_lst[ref_col]:
    #             if score > best_score:
    #                 best_col = col
    #                 best_cleaned_col_prompt = cleaned_col_prompt
    #                 best_score = score
    #         ref_col_to_col_lst[ref_col] = [(best_col, best_cleaned_col_prompt, best_score)]

    # then do final melt
    final_df = format_original_table_from_col_matching(df, ref_col_to_col_lst, remove_unique_col)
    return final_df