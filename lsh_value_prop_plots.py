import ipdb
import numpy as np
import pathlib
import pandas as pd 
import seaborn as sns

from collections import Counter
from matplotlib import pyplot as plt
from matplotlib import ticker as ticker
from sklearn.feature_extraction.text import CountVectorizer

# AHAHA These imports
from settings import VALUE_PROPS, CRUNCHBASE_PLOTS_PATH
from preprocess_crunchbase import get_crunchbase2020_data
from lsh_with_value_props import (map_docs_to_hash_table, 
    get_new_document_lookup_key, get_lsh_member_df,
    get_tag_frequency, get_term_prevalence)


def make_cross_hash_set_frequency_df(list_of_hash_keys, hash_table, ref_df, 
    freq_type = "industry", min_freq_cutoff = 0.1, doc_matrix = None,
    feat_array = None):
    '''With a bit more effort this could probably work for terms as well''' 
    df_entries = [] #{"Value Prop": ,"Industry":, "Pct of Companies"}
    for i, key in enumerate(list_of_hash_keys):
        hash_set_df = get_lsh_member_df(key = key, 
            hash_table = hash_table, ref_df = ref_df)
        if freq_type == "industry":
            tag_counts = get_tag_frequency(in_df = hash_set_df, 
                display_top = False)
        elif freq_type == "term":
            if doc_matrix is None:
                raise ValueError("Plotting terms requires vectorized documents.")
            if feat_array is None:
                raise ValueError("Plotting terms requires vocabulary array.")
            tag_counts = get_term_prevalence(in_df = hash_set_df, 
                doc_matrix = doc_matrix, feat_array = feat_array, 
                display_top = False)
        else:
            allowed = "either 'industry' or 'terms'"
            raise ValueError(f"freq_type must be {allowed}, not {freq_type}")
        tag_lbl = f"{freq_type}_tag"
        for tag, count in tag_counts.most_common(len(tag_counts)):
            pct = count / len(hash_set_df)
            if pct < PCT_THRESHOLD:
                print("Tag count too low ... skipping")
                break
            df_entries.append({
                "value_prop": f"Value Prop {i + 1}",
                tag_lbl: tag,
                "count_firms_in_lsh_set": count,
                "pct_firms_in_lsh_set": pct
            })
    return pd.DataFrame(data = df_entries)


def order_tags_by_count(df, tag_col, order_asc = False):
    '''Experimental ... not sure it beats the default ordering'''
    tag_order = df[tag_col].value_counts(ascending = order_asc)
    return tag_order.index.tolist()


def order_tags_by_mean_frequency(df, tag_col, freq_col, order_asc = False):
    '''Experimental ... not sure it beats the default ordering'''
    tag_order = df.groupby(by = tag_col)[freq_col].mean()
    return tag_order.sort_values(ascending = order_asc).index.tolist()


def plot_tag_frequency_across_lsh_sets(df, tag_col, setname_col, freq_col, 
    tag_ordering = None, save_plot = True, save_dir = CRUNCHBASE_PLOTS_PATH):
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize = (16,8))
    x_ax_lbl = tag_col.split("_")[0].title()
    y_ax_lbl = "Percent" if "pct" in freq_col else "Count"
    # Not a huge fan of any of these ordering strategies ... what's
    # probably more important here is to save the SEED that generated this hash
    if tag_ordering == "count":
        ordered_tags = order_tags_by_count(df, tag_col)
        filename = "tag_freq_countOrder.png"
    elif tag_ordering == "average":
        ordered_tags = order_tags_by_mean_frequency(df, tag_col, freq_col)
        filename = "tag_freq_averageOrder.png"
    else:
        ordered_tags = None
        # TODO: ACCESS SEED FROM THIS FUNCTION
        filename = f"{y_ax_lbl.lower()}{x_ax_lbl}FrequencyPlot_seed666.png"
    sns.barplot(x = tag_col, y = freq_col, hue = setname_col, data = df,  
        ci = None, order = ordered_tags, orient = "v", ax = ax)
    sns.despine()
    ax.set_xticklabels(ax.get_xticklabels(), 
        rotation = 60,
        horizontalalignment = "right")
    ax.legend(loc = "upper right", title = "LSH Bucket")
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax = 1.0, decimals = 0))
    ttl_set = " ".join([v.title() for v in setname_col.split("_")])
    plt.title(f"{x_ax_lbl} Frequency across {ttl_set} LSH Buckets")
    ax.set_xlabel(x_ax_lbl)
    ax.set_ylabel(f"{y_ax_lbl} of Firms in LSH Bucket")
    plt.tight_layout()
    if save_plot:
        if not save_dir.exists():
            save_dir.mkdir(parents = True, exist_ok = True)
        print(f"Saving as {str(save_dir / filename)}")
        plt.savefig(save_dir / filename)
    plt.close()


if __name__ == "__main__":

    N_BITS = 8
    SEED = 666
    PCT_THRESHOLD = 0.05 #When using terms, 10% is smarter
    base_df = get_crunchbase2020_data()
    base_vectorizer = CountVectorizer(lowercase = True,
        token_pattern = r"(?u)\b\w\w+\b", min_df = 3,
        stop_words = "english")
    documents = base_df["document"].values.tolist()
    lsh_matrix = base_vectorizer.fit_transform(documents)
    lsh_tbl = map_docs_to_hash_table(lsh_matrix = lsh_matrix, 
        hash_size = N_BITS, proj_seed = SEED)
    vp_matrix = base_vectorizer.transform(VALUE_PROPS)
    vp_keys = get_new_document_lookup_key(doc_matrix = vp_matrix,
        hash_size = N_BITS, proj_seed = SEED)
    termspace_vect = np.array(base_vectorizer.get_feature_names())

    plot_df = make_cross_hash_set_frequency_df(list_of_hash_keys = vp_keys, 
        hash_table = lsh_tbl, ref_df = base_df, freq_type = "industry", 
        doc_matrix = lsh_matrix, feat_array = termspace_vect, 
        min_freq_cutoff = PCT_THRESHOLD)

    plot_tag_frequency_across_lsh_sets(df = plot_df, 
        tag_col = "industry_tag",
        setname_col = "value_prop",
        freq_col = "pct_firms_in_lsh_set",
        tag_ordering = None,
        save_plot = True)

