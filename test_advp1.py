import os
import re
import pandas as pd

# Script for testing, require a directory of resulting table, where each test case name is {pmid}_{pmcid}.csv
# run by pytest test_advp1.py

# Paper tested
test_papers_info = [
    (30448613, "PMC6331247"), (30979435, "PMC6783343"), (31055733, "PMC6544706"),  (30617256, "PMC6836675"),
    (30820047, "PMC6463297"), (29458411, "PMC5819208"), (29777097, "PMC5959890"), (30651383, "PMC6369905"),
    (31497858, "PMC6736148"), (30930738, "PMC6425305"), (31426376, "PMC6723529"), (29967939, "PMC6280657"),
    (29107063, "PMC5920782"), (29274321, "PMC5938137"), (30413934, "PMC6358498"), (30805717, "PMC7193309"),
    (30636644, "PMC6330399"), (29752348, "PMC5976227"), (28560309, "PMC5440281"), (27899424, "PMC5237405"),
]

def create_test_tables_from_advp():
    advp1 = pd.read_csv("test_tables/advp.variant.records.hg38.tsv", sep = "\t")

    # modify column name of those used to test
    advp1 = advp1.rename({
        "Top SNP": "SNP",
        "RA 1(Reported Allele 1)": "RA",
        "OR_nonref": "Effect",
        "#dbSNP_hg38_chr": "Chr",
        "dbSNP_hg38_position": "Pos",
        "Cohort_simple3": "Cohort",
        "Population_map": "Population"
    }, axis = 1)[[
        "Pubmed PMID", "SNP", "RA", "P-value", "Effect", "Chr", "Pos", "Cohort", "Population"
    ]]

    # Update chr to right format
    advp1["Chr"] = advp1["Chr"].apply(lambda x: (x[3:]))

    for pmid, pmcid in test_papers_info:
        advp1_with_pmid = advp1[advp1["Pubmed PMID"] == pmid]
        # sort by snp
        advp1_with_pmid = advp1_with_pmid.sort_values("SNP").reset_index().drop("index", axis = 1)
        advp1_with_pmid.to_csv(f"test_tables/{pmid}_{pmcid}.csv", index = False)

def import_table_and_test_table(dir_path, file_name):
    if ".csv" in file_name:
        curr_df = pd.read_csv(f"{dir_path}/{file_name}")
        test_df = pd.read_csv(f"test_tables/{file_name[:-4]}")
    elif ".xlsx" in file_name:
        curr_df = pd.read_excel(f"{dir_path}/{file_name}")
        test_df = pd.read_csv(f"test_tables/{file_name[:-5]}")
    return curr_df, test_df

def test_table_dir_exists(dir_path):
    assert dir_path is not None, "Please provide --dir_path"

def test_table_name(dir_path):
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
        assert f"test_tables/{file_name}" in os.listdir(dir_path), f"Table {file_name} is in a paper not in test set"

def test_table_format(dir_path):
    # Test if table is in right format
    col_lst = ["SNP", "RA", "P-value", "Effect", "Chr", "Pos", "Cohort", "Population"]
    for file_name in os.listdir(dir_path):
        if ".csv" in file_name:
            curr_df = pd.read_csv(f"{dir_path}/{file_name}")
        elif ".xlsx" in file_name:
            curr_df = pd.read_excel(f"{dir_path}/{file_name}")
        for col in col_lst:
            assert col in curr_df.columns, f"Table {file_name} does not have column {col}"

def test_unique_snp(dir_path):
    # Test if we have the right set of snp
    for file_name in os.listdir(dir_path):
        curr_df, test_df = import_table_and_test_table(dir_path, file_name)
        curr_unique_snp = set(curr_df["SNP"].unique())
        test_unique_snp = set(test_df["SNP"].unique())
        assert curr_unique_snp == test_unique_snp

def test_num_record_snp(dir_path):
    # test if we have the right number of row for each snp
    for file_name in os.listdir(dir_path):
        curr_df, test_df = import_table_and_test_table(dir_path, file_name)
        test_unique_snp = test_df["SNP"].unique()
        for snp in test_unique_snp:
            curr_snp_df = curr_df[curr_df["SNP"] == snp]
            test_snp_df = test_df[test_df["SNP"] == snp]
            assert curr_snp_df.shape[0] == test_snp_df.shape[0], f"Table {file_name} does not have the right number of row for SNP {snp}"

def test_snp_ra(dir_path):
    # test for each table and for each snp we have right set of RA
    for file_name in os.listdir(dir_path):
        curr_df, test_df = import_table_and_test_table(dir_path, file_name)
        test_unique_snp = test_df["SNP"].unique()
        for snp in test_unique_snp:
            curr_snp_df = curr_df[curr_df["SNP"] == snp][["SNP", "RA"]].sort_values("RA")
            curr_snp_ra = curr_snp_df["RA"]
            test_snp_df = test_df[test_df["SNP"] == snp][["SNP", "RA"]].sort_values("RA")
            test_snp_ra = test_snp_df["RA"]
            assert (curr_snp_ra == test_snp_ra).all(), f"Table {file_name} does not contain right set of RA for SNP {snp}"

def test_snp_chr(dir_path):
    # test for each table and for each snp we have right set of Chr
    for file_name in os.listdir(dir_path):
        curr_df, test_df = import_table_and_test_table(dir_path, file_name)
        test_unique_snp = test_df["SNP"].unique()
        for snp in test_unique_snp:
            curr_snp_df = curr_df[curr_df["SNP"] == snp][["SNP", "Chr"]].sort_values("Chr")
            curr_snp_chr = curr_snp_df["Chr"]
            test_snp_df = test_df[test_df["SNP"] == snp][["SNP", "Chr"]].sort_values("Chr")
            test_snp_chr = test_snp_df["Chr"]
            assert (curr_snp_chr == test_snp_chr).all(), f"Table {file_name} does not contain right set of RA for SNP {snp}"     

def test_snp_pos(dir_path):
    # test for each table and for each snp we have right set of Pos
    for file_name in os.listdir(dir_path):
        curr_df, test_df = import_table_and_test_table(dir_path, file_name)
        test_unique_snp = test_df["SNP"].unique()
        for snp in test_unique_snp:
            curr_snp_df = curr_df[curr_df["SNP"] == snp][["SNP", "Pos"]].sort_values("Pos")
            curr_snp_pos = curr_snp_df["Pos"]
            test_snp_df = test_df[test_df["SNP"] == snp][["SNP", "Pos"]].sort_values("Pos")
            test_snp_pos = test_snp_df["Pos"]
            assert (curr_snp_pos == test_snp_pos).all(), f"Table {file_name} does not contain right set of position for SNP {snp}"

def test_snp_effect(dir_path):
    # test for each table and for each snp we have right set of effect
    for file_name in os.listdir(dir_path):
        curr_df, test_df = import_table_and_test_table(dir_path, file_name)
        test_unique_snp = test_df["SNP"].unique()
        for snp in test_unique_snp:
            curr_snp_df = curr_df[curr_df["SNP"] == snp][["SNP", "Effect"]].sort_values("Effect")
            curr_snp_effect = curr_snp_df["Effect"]
            test_snp_df = test_df[test_df["SNP"] == snp][["SNP", "Effect"]].sort_values("Effect")
            test_snp_effect = test_snp_df["Effect"]
            assert (curr_snp_effect == test_snp_effect).all(), f"Table {file_name} does not contain right set of effect for SNP {snp}"

# def test_snp_effect_str(dir_path):
#     # test for each table and for each snp we have right set of effect (given in str form)
#     for file_name in os.listdir(dir_path):
#         curr_df, test_df = import_table_and_test_table(dir_path, file_name)
#         test_unique_snp = test_df["SNP"].unique()
#         for snp in test_unique_snp:
#             curr_snp_df = curr_df[curr_df["SNP"] == snp][["SNP", "Effect"]].sort_values("Effect")
#             curr_snp_effect = curr_snp_df["Effect"]
#             test_snp_df = test_df[test_df["SNP"] == snp][["SNP", "Effect"]].sort_values("Effect")
#             test_snp_effect = test_snp_df["Effect"].apply(lambda x: str(x))
#             assert (curr_snp_effect == test_snp_effect).all(), f"Table {file_name} does not contain right set of effect for SNP {snp}"

def test_snp_pvalue(dir_path):
    # test for each table and for each snp we have right set of p-value (numerically)
    for file_name in os.listdir(dir_path):
        curr_df, test_df = import_table_and_test_table(dir_path, file_name)
        test_unique_snp = test_df["SNP"].unique()
        for snp in test_unique_snp:
            curr_snp_df = curr_df[curr_df["SNP"] == snp][["SNP", "P-value"]].sort_values("P-value")
            curr_snp_pvalue = curr_snp_df["P-value"]
            test_snp_df = test_df[test_df["SNP"] == snp][["SNP", "P-value"]].sort_values("P-value")
            test_snp_pvalue = test_snp_df["P-value"]
            assert (curr_snp_pvalue == test_snp_pvalue).all(), f"Table {file_name} does not contain right set of p-value for SNP {snp}"

# def test_snp_pvalue_str(dir_path):
#     # test for each table and for each snp we have right set of p-value (str)
#     for file_name in os.listdir(dir_path):
#         curr_df, test_df = import_table_and_test_table(dir_path, file_name)
#         test_unique_snp = test_df["SNP"].unique()
#         for snp in test_unique_snp:
#             curr_snp_df = curr_df[curr_df["SNP"] == snp][["SNP", "P-value"]].sort_values("P-value")
#             curr_snp_pvalue = curr_snp_df["P-value"]
#             test_snp_df = test_df[test_df["SNP"] == snp][["SNP", "P-value"]].sort_values("P-value")
#             test_snp_pvalue = test_snp_df["P-value"].apply(lambda x: str(x))
#             assert (curr_snp_pvalue == test_snp_pvalue).all(), f"Table {file_name} does not contain right set of p-value for SNP {snp}"

def test_snp_cohort(dir_path):
    # test for each table and for each snp we have right set of cohort
    for file_name in os.listdir(dir_path):
        curr_df, test_df = import_table_and_test_table(dir_path, file_name)
        test_unique_snp = test_df["SNP"].unique()
        for snp in test_unique_snp:
            curr_snp_df = curr_df[curr_df["SNP"] == snp][["SNP", "Cohort"]].sort_values("Cohort")
            curr_snp_cohort = curr_snp_df["Cohort"]
            test_snp_df = test_df[test_df["SNP"] == snp][["SNP", "Cohort"]].sort_values("Cohort")
            test_snp_cohort = test_snp_df["Cohort"]
            assert (curr_snp_cohort == test_snp_cohort).all(), f"Table {file_name} does not contain right set of cohort for SNP {snp}"

def test_snp_population(dir_path):
    # test for each table and for each snp we have right set of population
    for file_name in os.listdir(dir_path):
        curr_df, test_df = import_table_and_test_table(dir_path, file_name)
        test_unique_snp = test_df["SNP"].unique()
        for snp in test_unique_snp:
            curr_snp_df = curr_df[curr_df["SNP"] == snp][["SNP", "Population"]].sort_values("Population")
            curr_snp_population = curr_snp_df["Population"]
            test_snp_df = test_df[test_df["SNP"] == snp][["SNP", "Population"]].sort_values("Population")
            test_snp_population = test_snp_df["Population"]
            assert (curr_snp_population == test_snp_population).all(), f"Table {file_name} does not contain right set of population for SNP {snp}"

# if __name__ == "__main__":
    # create_test_tables_from_advp()