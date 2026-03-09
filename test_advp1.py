import os
import re
import pandas as pd
import json
from collections import Counter
from typing import Iterable

# Script for testing, require a directory of resulting table, where each test case name is {pmid}_{pmcid}.csv
# run by pytest test_advp1.py --dir-test={insert your dir for all test tables}
# Test log is in test_logs with detail of error for each table

def import_table_and_test_table(dir_path: str, file_name: str):
    if ".csv" in file_name:
        curr_df = pd.read_csv(f"{dir_path}/{file_name}")
        test_df = pd.read_csv(f"test_tables/{file_name[:-4]}.csv")
    elif ".xlsx" in file_name:
        curr_df = pd.read_excel(f"{dir_path}/{file_name}")
        test_df = pd.read_csv(f"test_tables/{file_name[:-5]}.csv")
    return curr_df, test_df

def test_table_dir_exists(dir_path: str):
    assert dir_path is not None, "Please provide --dir_path"

def test_table_name(dir_path: str):
    # We require table name to be in the format of pmid_pmcid.csv
    table_name_pattern = r"^\d+_PMC\d+$"
    for file_name in os.listdir(dir_path):
        # extract filename without tag and check the right pattern
        if ".csv" in file_name:
            assert re.search(table_name_pattern, file_name[:-4]), f"Table {file_name} does not have right name (pmid_pmcid)"
        elif ".xlsx" in file_name:
            assert re.search(table_name_pattern, file_name[:-5]), f"Table {file_name} does not have right name (pmid_pmcid)"
        else:
            raise Exception("Error: table must be .csv or .xlsx")
        assert f"{file_name}" in os.listdir("test_tables"), f"Table {file_name} is in a paper not in test set"

# def test_table_format(dir_path):
#     # Test if table is in right format
#     col_lst = ["SNP", "RA", "P-value", "Effect", "Chr", "Pos", "Cohort", "Population"]
#     for file_name in os.listdir(dir_path):
#         if ".csv" in file_name:
#             curr_df = pd.read_csv(f"{dir_path}/{file_name}")
#         elif ".xlsx" in file_name:
#             curr_df = pd.read_excel(f"{dir_path}/{file_name}")
#         for col in col_lst:
#             assert col in curr_df.columns, f"Table {file_name} does not have column {col}"

def test_unique_snp(dir_path: str):
    # Test if we have the right set of snp
    failed_table = [] # store (table, error)
    for file_name in os.listdir(dir_path):
        curr_df, test_df = import_table_and_test_table(dir_path, file_name)
        if "SNP" not in curr_df.columns:
            failed_table.append((file_name, f"Table {file_name} does not have SNP column"))
        else:
            curr_unique_snp = set(curr_df[["SNP"]].dropna()["SNP"].unique())
            test_unique_snp = set(test_df[["SNP"]].dropna()["SNP"].unique())
            # if curr_unique_snp != test_unique_snp:
            if test_unique_snp.intersection(curr_unique_snp) != test_unique_snp:
                failed_table.append((file_name, f"Table {file_name} do not contain all snp, missing: {test_unique_snp - curr_unique_snp}"))
    try:
        assert len(failed_table) == 0
    except AssertionError:
        print(f"Failed test_unique_snp on {len(failed_table)}")
        with open("test_logs/test_unique_snp.json", "w") as f:
            json.dump(failed_table, f, indent=2)
        raise

def test_num_record_snp(dir_path: str):
    # test if we have the right number of row for each snp
    failed_table = []
    for file_name in os.listdir(dir_path):
        curr_df, test_df = import_table_and_test_table(dir_path, file_name)
        if "SNP" not in curr_df.columns:
            failed_table.append((file_name, f"Table {file_name} does not have SNP column"))
        else:
            test_unique_snp = test_df[["SNP"]].dropna()["SNP"].unique()
            missed_snp = [] # we record exactly how much error do we make on a table
            for snp in test_unique_snp:
                curr_snp_df = curr_df[curr_df["SNP"] == snp]
                test_snp_df = test_df[test_df["SNP"] == snp]
                # NOTE: alternately, we can try to check if we have at least number of row as test
                # to prevent the case of rows that do not pass QC
                # if curr_snp_df.shape[0] != test_snp_df.shape[0]:
                if curr_snp_df.shape[0] < test_snp_df.shape[0]:
                    missed_snp.append(snp)
            if len(missed_snp) > 0:
                failed_table.append((file_name, f"Table {file_name} ({round(100 * (1 - len(missed_snp) / len(test_unique_snp)), 2)}) does not have the enough number of row for SNP {missed_snp}"))
    try:
        assert len(failed_table) == 0
    except AssertionError:
        with open("test_logs/test_num_record_snp.json", "w") as f:
            json.dump(failed_table, f, indent=2)
        raise AssertionError(f"Failed test_num_record_snp on {len(failed_table)} tables")

def check_lst1_contains_lst2(lst1: Iterable, lst2: Iterable):
    counter1 = Counter(lst1)
    counter2 = Counter(lst2)
    return counter2 <= counter1

def check_lst1_equals_lst2(lst1: Iterable, lst2: Iterable):
    counter1 = Counter(lst1)
    counter2 = Counter(lst2)
    return counter2 == counter1

# NOTE: since we do not consider NAs value, 
# there will be cases where extracted table do not contain an SNP
# and target table has that SNP but have all NAs => 2 counter are the same
def get_failed_table_for_test(dir_path: str, col: str, is_numeric: bool = False):
    failed_table = []
    for file_name in os.listdir(dir_path):
        curr_df, test_df = import_table_and_test_table(dir_path, file_name)
        if "SNP" not in curr_df.columns:
            failed_table.append((file_name, f"Table {file_name} does not have SNP column"))
        else:
            if col not in curr_df.columns:
                failed_table.append((file_name, f"Table {file_name} does not have {col} column"))
            else:
                test_unique_snp = test_df[["SNP"]].dropna()["SNP"].unique()
                missed_snp = []
                for snp in test_unique_snp:
                    # NOTE: since Counter in python treat each NaN as a different value since NaN != NaN in pandas, we need to remove them first
                    curr_snp_df = curr_df[curr_df["SNP"] == snp][["SNP", col]].sort_values(col).reset_index().drop("index", axis = 1)
                    curr_snp_col = curr_snp_df[[col]].dropna().reset_index().drop("index", axis = 1)[col]
                    # NOTE: if it is numeric, we round it to 15 digits to prevent unexpected error
                    if is_numeric:
                        curr_snp_col = curr_snp_col.apply(lambda x: round(x, 15))
                    test_snp_df = test_df[test_df["SNP"] == snp][["SNP", col]].sort_values(col).reset_index().drop("index", axis = 1)
                    test_snp_col = test_snp_df[[col]].dropna().reset_index().drop("index", axis = 1)[col]
                    if is_numeric:
                        test_snp_col = test_snp_col.apply(lambda x: round(x, 15))
                    # try:
                    #     if not check_lst1_contains_lst2(curr_snp_col, test_snp_col):
                    #         failed_table.append((file_name, f"Table {file_name} does not contain right set of {col} for SNP {snp}"))
                    #         break
                    # except:
                    #     failed_table.append((file_name, f"Table {file_name} does not contain right set of {col} for SNP {snp}"))
                    #     break
                    # NOTE: alternatively, we can check if our extracted value is a superset of test value
                    # to prevent the case of rows that fails QC
                    # if not (curr_snp_col == test_snp_col).all():
                    # if is_constant:
                    #     if curr_snp_col.nunique() != 1:    
                    #         missed_snp.append(snp)
                    #         failed_table.append((file_name, f"Table {file_name} does not have the right single unique value of {col} for SNP {snp}: {curr_snp_col_value} vs {test_snp_col_value}"))
                    #     else:
                    #         curr_snp_col_value = curr_snp_df[col].unique()[0]
                    #         test_snp_col_value = test_snp_df[col].unique()[0]
                    #         if curr_snp_col_value != test_snp_col_value:
                    #             failed_table.append((file_name, f"Table {file_name} does not have the right single unique value of {col} for SNP {snp}: {curr_snp_col_value} vs {test_snp_col_value}"))
                    #             break
                    # else:
                    if not check_lst1_contains_lst2(curr_snp_col, test_snp_col):
                        missed_snp.append(snp)
                if len(missed_snp) > 0:
                    failed_table.append((file_name, f"Table {file_name} ({round(100 * (1 - len(missed_snp) / len(test_unique_snp)), 2)}) does not contain right set of {col} for SNP {missed_snp}"))
    return failed_table

def test_snp_ra1(dir_path: str):
    failed_table = get_failed_table_for_test(dir_path, "RA1")
    try:
        assert len(failed_table) == 0
    except AssertionError:
        with open("test_logs/test_snp_ra1.json", "w") as f:
            json.dump(failed_table, f, indent=2)
        raise AssertionError(f"Failed test_snp_ra on {len(failed_table)} tables")
    
def test_snp_ra2(dir_path: str):
    failed_table = get_failed_table_for_test(dir_path, "RA2")
    try:
        assert len(failed_table) == 0
    except AssertionError:
        with open("test_logs/test_snp_ra2.json", "w") as f:
            json.dump(failed_table, f, indent=2)
        raise AssertionError(f"Failed test_snp_ra2 on {len(failed_table)} tables")

def test_snp_af(dir_path: str):
    failed_table = get_failed_table_for_test(dir_path, "AF")
    try:
        assert len(failed_table) == 0
    except AssertionError:
        with open("test_logs/test_snp_af.json", "w") as f:
            json.dump(failed_table, f, indent=2)
        raise AssertionError(f"Failed test_snp_af on {len(failed_table)} tables")

def test_snp_chr(dir_path: str):
    # test for each table and for each snp we have right set of Chr
    failed_table = get_failed_table_for_test(dir_path, "Chr")
    try:
        assert len(failed_table) == 0
    except AssertionError:
        with open("test_logs/test_snp_chr.json", "w") as f:
            json.dump(failed_table, f, indent=2)
        raise AssertionError(f"Failed test_snp_chr on {len(failed_table)} tables")

# def test_snp_locus(dir_path: str):
#     # test for each table and for each snp we have right set of Chr
#     failed_table = get_failed_table_for_test(dir_path, "Locus")
#     try:
#         assert len(failed_table) == 0
#     except AssertionError:
#         with open("test_logs/test_snp_locus.json", "w") as f:
#             json.dump(failed_table, f, indent=2)
#         raise AssertionError(f"Failed test_snp_locus on {len(failed_table)} tables")

# def test_snp_pos(dir_path: str):
#     # test for each table and for each snp we have right set of Position
#     failed_table = get_failed_table_for_test(dir_path, "Position")
#     try:
#         assert len(failed_table) == 0
#     except AssertionError:
#         with open("test_logs/test_snp_pos.json", "w") as f:
#             json.dump(failed_table, f, indent=2)
#         raise AssertionError(f"Failed test_snp_pos on {len(failed_table)} tables")

def test_snp_effect(dir_path: str):
    # test for each table and for each snp we have right set of effect
    failed_table = get_failed_table_for_test(dir_path, "Effect", is_numeric = True)
    try:
        assert len(failed_table) == 0
    except AssertionError:
        with open("test_logs/test_snp_effect.json", "w") as f:
            json.dump(failed_table, f, indent=2)
        raise AssertionError(f"Failed test_snp_effect on {len(failed_table)} tables")

def test_snp_pvalue(dir_path: str):
    # test for each table and for each snp we have right set of p-value (numerically)
    failed_table = get_failed_table_for_test(dir_path, "P-value", is_numeric = True)
    try:
        assert len(failed_table) == 0
    except AssertionError:
        with open("test_logs/test_snp_pvalue.json", "w") as f:
            json.dump(failed_table, f, indent=2)
        raise AssertionError(f"Failed test_snp_pvalue on {len(failed_table)} tables")

# def test_snp_cohort(dir_path: str):
#     # test for each table and for each snp we have right set of cohort
#     failed_table = get_failed_table_for_test(dir_path, "Cohort")
#     try:
#         assert len(failed_table) == 0
#     except AssertionError:
#         with open("test_logs/test_snp_cohort.json", "w") as f:
#             json.dump(failed_table, f, indent=2) 
#         raise AssertionError(f"Failed test_snp_cohort on {len(failed_table)} tables")

# def test_snp_population(dir_path: str):
#     # test for each table and for each snp we have right set of population
#     failed_table = get_failed_table_for_test(dir_path, "Population")
#     try:
#         assert len(failed_table) == 0
#     except AssertionError:
#         with open("test_logs/test_snp_population.json", "w") as f:
#             json.dump(failed_table, f, indent=2)
#         raise AssertionError(f"Failed test_snp_population on {len(failed_table)} tables")