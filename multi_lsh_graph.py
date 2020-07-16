import ipdb
import pathlib
import pandas as pd 

from lsh_with_value_props import display_company_info
from preprocess_crunchbase import get_crunchbase2020_data
from settings import CRUNCHBASE_2020_PATH


LSH_AGG_PATH = CRUNCHBASE_2020_PATH / "LSH-Runs/100runs-8bit-4vps-seed1024.tsv"
LSH_GRAPH_DIR = CRUNCHBASE_2020_PATH / "Graph-Files"


def get_company_to_value_prop_match_df(lsh_run_df):
    '''May be deprecated ...
    '''
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


def make_lsh_nodes(input_df, save_csv = True, csv_dir = LSH_GRAPH_DIR):
    '''Just needs id and label. Depending on how this ends up looking 
    in Gephi, I may want to add a more detailed label later.
    '''
    unique_vps = input_df.value_prop.unique().tolist()
    company_df = input_df.drop_duplicates(subset = "company_index")
    max_company_node = company_df.company_index.max()
    company_df.set_index(keys = "company_index", drop = True, inplace = True)
    node_rows = []
    for c_index in company_df.index.tolist():
        c_name = company_df.at[c_index, "org_name"]
        node_rows.append({"id": c_index, "label": c_name})
    print(f"Just stored {len(node_rows)} company nodes")
    for i, vp in enumerate(unique_vps):
        v_index = max_company_node + i + 1
        node_rows.append({"id": v_index, "label": vp})
    print(f"Retrieved info for {len(node_rows)} nodes total")
    node_df = pd.DataFrame(data = node_rows)
    if save_csv:
        if not LSH_GRAPH_DIR.exists():
            LSH_GRAPH_DIR.mkdir(exist_ok = True, parents = True)
        print(f"Saving nodes.csv to {LSH_GRAPH_DIR}")
        csv_path = LSH_GRAPH_DIR / "nodes.csv"
        node_df.to_csv(csv_path, sep = ",", index = False, header = True)
    return node_df


def make_lsh_edges(input_df, save_csv = True, csv_dir = LSH_GRAPH_DIR):
    '''Needs columns source and target with the values as node ids. I am assuming you also need to add
    some kind of weight column
    '''
    unique_vps = input_df.value_prop.unique().tolist()
    max_company_node = input_df.company_index.max()
    grp_cols = ["value_prop", "company_index"]
    agg_dict = {"sim_seed": "count"}
    rnm_dict = {
        "sim_seed": "weight",
        "company_index": "source",
        "value_prop": "target"
    }
    edge_df = input_df.groupby(by = grp_cols, as_index = False).agg(agg_dict)
    edge_df.rename(columns = rnm_dict, inplace = True)
    vp_to_id = lambda x: unique_vps.index(x) + max_company_node + 1
    edge_df["target"] = edge_df.target.apply(vp_to_id)
    if edge_df.weight.sum() == len(input_df):
        print("Edge Checksum Passed")
    if save_csv:
        if not LSH_GRAPH_DIR.exists():
            LSH_GRAPH_DIR.mkdir(exist_ok = True, parents = True)
        print(f"Saving edges.csv to {LSH_GRAPH_DIR}")
        csv_path = LSH_GRAPH_DIR / "edges.csv"
        edge_df.to_csv(csv_path, sep = ",", index = False, header = True)
    return edge_df


if __name__ == "__main__":

    lsh_run_df = pd.read_table(LSH_AGG_PATH)
    node_df = make_lsh_nodes(input_df = lsh_run_df, save_csv = True)
    edge_df = make_lsh_edges(input_df = lsh_run_df, save_csv = True)

    # ipdb.set_trace()
    # PROLLY DEPRECATED
    # affinity_df = get_company_to_value_prop_match_df(lsh_run_df)
    # base_df = get_crunchbase2020_data()
    # ipdb.set_trace()
    
    # # An example of displaying companies that only matched with one company
    # # but matched four times.
    # singular_vp_cond = (affinity_df.num_vps_matched == 1)
    # four_match_cond = (affinity_df.overall_frequency == 4)
    # results_df = affinity_df.loc[singular_vp_cond & four_match_cond]
    # for i, j in enumerate(results_df.sort_values(by = "vps_matched").index):
    #     c = results_df.at[j, "org_name"]
    #     vp = results_df.at[j, "vps_matched"]
    #     print(f"{c} only ever matched with {vp} and matched four times\n")
    #     display_company_info(rank = i + 1, df_index = j, info_df = base_df)
    #     print()
    # ipdb.set_trace()