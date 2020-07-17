import ipdb
import pathlib
import pandas as pd 

from lsh_with_value_props import display_company_info
from settings import CRUNCHBASE_2020_PATH, VALUE_PROPS


LSH_AGG_PATH = CRUNCHBASE_2020_PATH / "LSH-Runs/100runs-8bit-4vps-seed1024.tsv"
LSH_GRAPH_DIR = CRUNCHBASE_2020_PATH / "Graph-Files"


def get_value_prop_node_id(vp_index, reference_index):
    '''Company nodes have built-in IDs, because of the way we saved them. For 
    value prop nodes, what we do instead is add the index of the value prop to 
    some constant reference index. Generally, the maximum of all company
    indices.
    '''
    return reference_index + vp_index + 1


def get_value_prop_description(vp_index, vp_descriptions = VALUE_PROPS):
    '''The description for company nodes 
    '''
    return vp_descriptions[vp_index].strip()


def parse_revenue_note(revenue_note):
    if pd.isnull(revenue_note):
        return "Unknown"
    if "Under" in revenue_note:
        return "Less than $50 Million"
    return "Over $50 Million"


def parse_activity_note(funding_note):
    if pd.isnull(funding_note):
        return "Unknown"
    return "Within the past year"


def get_node_info(node_id, info_df, vp_label = None, vp_descrip = None):
    '''Key errors imply that this is a value prop node instead of a company
    node. Company nodes are handled a little different.
    '''
    try:
        node_label = info_df.at[node_id, "org_name"]
        node_dscrip = info_df.at[node_id, "description"]
        sectors = info_df.at[node_id, "industry_tags"]
        cb = info_df.at[node_id, "cb_rank"]
        revenue = info_df.at[node_id, "revenue_note"]
        funded = info_df.at[node_id, "recent_funding"]
        founded = info_df.at[node_id, "recent_founding"]
    except KeyError as e:
        return {
            "id": node_id,
            "label": vp_label,
            "description": vp_descrip, 
            "industries": "N/A",
            "crunchbase rank": -999,
            "revenue category": "N/A",
            "recent funding": "N/A",
            "founded": "N/A"
        }
    return {
        "id": node_id,
        "label": node_label, 
        "description": node_dscrip,
        "industries": "Unknown" if pd.isnull(sectors) else sectors,
        "crunchbase rank": -999 if pd.isnull(cb) else cb, 
        "revenue category": parse_revenue_note(revenue),
        "recent funding": parse_activity_note(funded),
        "founded": parse_activity_note(founded) 
    }


def make_lsh_nodes(input_df, vp_descriptions = VALUE_PROPS,
    save_csv = True, csv_dir = LSH_GRAPH_DIR):
    unique_vps = input_df.value_prop.unique().tolist()
    company_df = input_df.drop_duplicates(subset = "company_index")
    max_company_node = company_df.company_index.max()
    company_df.set_index(keys = "company_index", drop = True, inplace = True)
    node_rows = []
    for c_index in company_df.index.tolist():
        node_info = get_node_info(node_id = c_index, info_df = company_df)
        node_rows.append(node_info)
    print(f"Just stored {len(node_rows)} company nodes")
    for vp_index, vp_label in enumerate(unique_vps):
        vp_node_id = get_value_prop_node_id(vp_index = vp_index,
            reference_index = max_company_node)
        vp_text = get_value_prop_description(vp_index = vp_index,
            vp_descriptions = vp_descriptions)
        node_info = get_node_info(node_id = vp_node_id, info_df = company_df,
            vp_label = vp_label, vp_descrip = vp_text)
        node_rows.append(node_info)
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
    node_df = make_lsh_nodes(input_df = lsh_run_df, vp_descriptions = VALUE_PROPS,
        save_csv = True, csv_dir = LSH_GRAPH_DIR)
    edge_df = make_lsh_edges(input_df = lsh_run_df, save_csv = True,
        csv_dir = LSH_GRAPH_DIR)
    # ipdb.set_trace()
 
    