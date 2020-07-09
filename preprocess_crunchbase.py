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
CATEGORY_DDUP_COLS = ["Organization Name",
    "Organization Name URL",
    "CB Rank (Company)"
]
DROP_COUNTRIES = ["China", "Russia", "India"]
CATEGORY_DDUP_SORT = [True, True, True]
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
FINAL_COLNAMES = {"Organization Name":"org_name",
    "Organization Name URL":"org_url",
    "Industries":"industry_tags",
    "Headquarters Location":"hq_loc", 
    "Description":"description",
    "CB Rank (Company)":"cb_rank"
}
DOCUMENT_COLNAMES = ["clean_tags", 
    "description",
    "recent_founding",
    "recent_funding",
    "revenue_note"
]
PROCESSED_TSV_NAME = "crunchbase2020_processed.tsv"


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


def handle_crunchbase_rank_column(df, fillval = "999,999,999"):
    cb_col = "CB Rank (Company)"
    df[cb_col].fillna(fillval, inplace = True)
    try:
        df[cb_col] = df[cb_col].str.replace(",","").astype(int)
    except AttributeError:
        df[cb_col] = df[cb_col].astype(int)


def remove_country_rows(df, countries):
    df["country"] = df["hq_loc"].apply(lambda s: s.split(", ")[-1])
    return df.loc[~df.country.isin(countries)].drop(labels = "country", axis = 1)


def add_document_column(df, component_col_list):
    df["document"] = df[component_col_list].agg(" ".join, axis = 1)


def clean_category_tags(tags_str, in_delim = ", ", out_delim = " "):
    '''Meant to be applied along a particular column that contains
    the industry/category tags. Returns a string itself

    N.B. for the 2015 data, in_delim is a pipe
    '''
    if pd.isnull(tags_str):
        return ""
    cleaned_tags = []
    for tag in tags_str.split(in_delim):
        tag = tag.replace("&", "_and_")
        for under_punc in "- ":
            tag = tag.replace(under_punc, "_")
        for strip_punc in "/'()":
            tag = tag.replace(strip_punc, "")
        cleaned_tags.append(tag)
    return out_delim.join(cleaned_tags)


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


def process_files_by_category(tsv_label, 
    base_path = CRUNCHBASE_2020_PATH):
    category_dfs = []
    category_paths = [c for c in base_path.glob(f"*{tsv_label}*.tsv")]
    for path in category_paths:
        df = pd.read_table(path)
        # MAKE ME A HANDLE CB RANK FUNCTION
        handle_crunchbase_rank_column(df = df, fillval = "999,999,999")
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


def process_crunchbase2020_data(category_labels = TSV_LABELS_2020,  
    base_path = CRUNCHBASE_2020_PATH, grp_cols = FINAL_GROUP_BY,
    agg_dct = FINAL_AGG, out_names = FINAL_COLNAMES,
    remove_countries = DROP_COUNTRIES, doc_cols = DOCUMENT_COLNAMES,
    write_result = True, save_name = PROCESSED_TSV_NAME):
    crunchbase_dfs = []
    for label in category_labels:
        print(label)
        cb_df = process_files_by_category(tsv_label = label, 
            base_path = base_path)
        crunchbase_dfs.append(cb_df)
    stack_df = pd.concat(objs = crunchbase_dfs, axis = 0)
    print(f"{len(stack_df)} rows before aggregation")
    stack_df.fillna("", inplace = True)
    final_df = stack_df.groupby(by = grp_cols, as_index = False).agg(agg_dct)
    final_df.rename(columns = out_names, inplace = True)
    print(f"{len(final_df)} rows after aggregation")
    final_df["clean_tags"] = final_df["industry_tags"].apply(clean_category_tags)
    final_df = remove_country_rows(df = final_df, countries = remove_countries)
    print(f"{len(final_df)} rows after excluding {', '.join(remove_countries)}")
    final_df["document"] = final_df[doc_cols].apply(" ".join, axis = 1)
    final_df.reset_index(drop = True, inplace = True)
    out_path = base_path / save_name
    if not out_path.exists() and write_result:
        print(f"Saving processed 2020 data to {str(out_path)}")
        final_df.to_csv(path_or_buf = out_path, sep = "\t",
            index = False, header = True)
    return final_df
        

def get_crunchbase2020_data(base_path = CRUNCHBASE_2020_PATH, 
    save_name = PROCESSED_TSV_NAME):
    check_path = base_path / save_name
    if check_path.is_file():
        print("LOADING PRE-PROCESSED DATA")
        return pd.read_table(check_path)
    print("!!! RUNNING 2020 PROCESSING PIPELINE !!!")
    return process_crunchbase2020_data(write_result = True)


if __name__ == "__main__":
    
    #MAIN FUNCTION FOR 2020
    foo_df =  get_crunchbase2020_data()
    ipdb.set_trace() 