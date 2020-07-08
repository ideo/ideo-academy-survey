import ipdb
import pathlib
import pandas as pd 

from settings import (CRUNCHBASE_2015_PATH, CRUNCHBASE_2020_PATH)
from collections import Counter

# maybe belongs in settings.py
CSV_NAMES_2015 = ["acquisitions",
    "additions",
    "companies",
    "investments",
    "rounds"
]

TSV_LABELS_2020 = ["Coaching",
    "Collaboration",
    "E-learning",
    "Edtech",
    "Education",
    "Enterprise Applications",
    "Enterprise Software",
    "Human Resources",
    "Online Portal",
    "Operating Systems",
    "Personal Development",
    "Productivity Tools",
    "Professional Services",
    "Project Management",
    "Real Time",
    "Skill Assessment",
    "Test and Measurement",
    "Training",
    "Crunchbase Top 1000"
]

#This may change once we've worked our way through all of the categories
CATEGORY_DDUP_COLS = ["Organization Name",
    "Organization Name URL",
    "CB Rank (Company)"
]

CATEGORY_DDUP_SORT = [True, True, True]


KEEP_COLUMNS = ["Organization Name",
    "Organization Name URL",
    "Industries",
    "Headquarters Location", 
    "Description",
    "CB Rank (Company)",
    "recent_funding",
    "recent_founding",
    "revenue_note"
]

FINAL_GROUP_BY = ["Organization Name", "Organization Name URL"]
FINAL_AGG = {
    "Industries":"max",
    "Headquarters Location":"max",
    "Description":"max",
    "CB Rank (Company)":"min",
    "recent_funding":"max",
    "recent_founding":"max",
    "revenue_note":"max"
}

def extract_categories(category_str, delim = "|"):
    '''Meant to be applied along a particular column'''
    if pd.isnull(category_str):
        return []
    return [v.replace(" ","") for v in category_str.split(delim)]


def add_recent_founding_column(df):
    df["recent_founding"] = "FoundedWithinPastYear"
    return df


def add_recent_funding_column(df):
    df["recent_funding"] = "FundedWithinPastYear"
    return df


def add_revenue_note_column(df, over = False):
    rev_note = "OverFiftyMillion" if over else "UnderFiftyMillion"
    df["revenue_note"] = rev_note
    return df


def dedupe_dataframes(list_of_dfs, sort_col_list = CATEGORY_DDUP_COLS,
     sort_bool_list = CATEGORY_DDUP_SORT):
    combined_df = pd.concat(objs = list_of_dfs, axis = 0)
    print(f"{len(combined_df)} rows before deduplication")
    combined_df.sort_values(by = sort_col_list,
         ascending = sort_bool_list,
         inplace = True)
    ddup_cols = sort_col_list[:-1]
    ddup_df = combined_df.drop_duplicates(subset = ddup_cols, keep = "first")
    if len(ddup_df) != len(combined_df):
        print(f"{len(ddup_df)} rows after de-duping by {' '.join(ddup_cols)}")
    return ddup_df


def process_dataframes_by_file_category(tsv_label, 
    base_path = CRUNCHBASE_2020_PATH):
    category_dfs = []
    category_paths = [c for c in base_path.glob(f"*{tsv_label}*.tsv")]
    for path in category_paths:
        df = pd.read_table(path)
        # MAKE ME A HANDLE CB RANK FUNCTION
        df["CB Rank (Company)"] = df["CB Rank (Company)"].fillna("999,999,999")
        try:
            df["CB Rank (Company)"] = df["CB Rank (Company)"].str.replace(",","").astype(int)
        except AttributeError:
            df["CB Rank (Company)"] = df["CB Rank (Company)"].astype(int)
        file_stem = path.stem
        if "Founded" in file_stem:
            print("\tAdding recent_founding note ...")
            df = add_recent_founding_column(df)
        if "Funded" in file_stem:
            print("\tAdding recent_funding note ...")
            df = add_recent_funding_column(df)
        if "$50M" in file_stem:
            print("\tAdding revenue note ...")
            above50 = "$50M+" in file_stem
            df = add_revenue_note_column(df, over = above50)
        if "Everything" in file_stem:
            op_col = "Top 1000, $50M, Founded, Funded"
            print("\tHandling edtech special case ...")
            # I'm sorry XD
            df["recent_founding"] = df[op_col].apply(lambda x: {"Founded Last Year":"FoundedWithinPastYear"}.get(x, None))
            df["recent_funding"] = df[op_col].apply(lambda x: {"Funded in the Last year":"FundedWithinPastYear"}.get(x, None))
            df["revenue_note"] = df[op_col].apply(lambda x: {"$50M+":"OverFiftyMillion"}.get(x, None))
        category_dfs.append(df)
    return dedupe_dataframes(list_of_dfs = category_dfs)


        


if __name__ == "__main__":
    
    #MAIN FUNCTION FOR 2020
    all_the_dfs = []
    for lbl in TSV_LABELS_2020:
        print(lbl)
        cat_df = process_dataframes_by_file_category(tsv_label = lbl)
        all_the_dfs.append(cat_df)
    foo_df = pd.concat(objs = all_the_dfs, axis = 0)
    # ipdb.set_trace()
    foo_df.fillna("", inplace = True)
    # ipdb.set_trace()
    bar_df = foo_df.groupby(by = FINAL_GROUP_BY, as_index = False).agg(FINAL_AGG) 
    # Rename columns
    # boot out china russia and india
    # make document columns  