import ipdb
import numpy as np
import pathlib
import pandas as pd 
import seaborn as sns

from collections import Counter
from matplotlib import pyplot as plt
from matplotlib import ticker as ticker

from lsh_with_value_props import display_company_info
from preprocess_crunchbase import get_crunchbase2020_data
from settings import CRUNCHBASE_PLOTS_PATH, CRUNCHBASE_2020_PATH



LSH_AGG_PATH = CRUNCHBASE_2020_PATH / "LSH-Runs/100runs-8bit-4vps-seed1024.tsv"

def get_company_to_value_prop_match_df(lsh_run_df):
    agg_dict = {
        "value_prop": lambda x: list(set(x)),
        "org_name": "max",
        "org_url": "count",
        "description": "max",
        "industry_tags": "max",
        "cb_rank": "max" 
    }
    rnm_dict = {
        "org_url": "overall_frequency",
        "value_prop": "vps_matched"
    }
    print("Aggregating LSH results by company ...")
    match_df = lsh_run_df.groupby(by = "company_index").agg(agg_dict)
    match_df.rename(columns = rnm_dict, inplace = True)
    match_df["num_vps_matched"] = match_df.vps_matched.apply(len)
    match_df["vps_matched"] = match_df.vps_matched.apply(lambda y: " ".join(y))
    return match_df


if __name__ == "__main__":
    
    df = pd.read_table(LSH_AGG_PATH)
    base_df = get_crunchbase2020_data()
    affinity_df = get_company_to_value_prop_match_df(lsh_run_df = df)
    
    ipdb.set_trace()
    
    # An example of displaying companies that only matched with one company
    # but matched four times.
    singular_vp_cond = (affinity_df.num_vps_matched == 1)
    four_match_cond = (affinity_df.overall_frequency == 4)
    results_df = affinity_df.loc[singular_vp_cond & four_match_cond]
    for i, j in enumerate(results_df.sort_values(by = "vps_matched").index):
        c = results_df.at[j, "org_name"]
        vp = results_df.at[j, "vps_matched"]
        print(f"{c} only ever matched with {vp} and matched four times\n")
        display_company_info(rank = i + 1, df_index = j, info_df = base_df)
        print()
    ipdb.set_trace()