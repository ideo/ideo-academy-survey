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
VP_COLORS = {
    "Value Prop 1": "blue",
    "Value Prop 2": "orange",
    "Value Prop 3": "green",
    "Value Prop 4": "red"
}
MULTI_LSH_PLOTS_PATH = CRUNCHBASE_PLOTS_PATH / "multiLSH-100runs-seed1024"

def weighted_mean(val_and_weight_tuple):
    '''Explicitly designed to be used as an estimator within seaborn's barplot
    method. See https://github.com/mwaskom/seaborn/issues/722
    '''
    val, weight = map(np.asarray, zip(*val_and_weight_tuple))
    return (val * weight).sum()/weight.sum()


def make_industry_frequency_df(lsh_run_df, focus_valprop):
    simrun_freq_rows = []
    vp_df = lsh_run_df.loc[lsh_run_df.value_prop == focus_valprop]
    for seed in vp_df.sim_seed.unique().tolist():
        seed_df = vp_df.loc[vp_df.sim_seed == seed]
        simrun_weight = len(seed_df)
        ind_freq = Counter()
        ind_vals = seed_df.industry_tags.fillna("").values.tolist()
        for v in ind_vals:
            ind_freq.update(v.split(", "))
        for ind_tag, tag_count in ind_freq.most_common(len(ind_freq)):
            pct = tag_count / simrun_weight
            freq_row = {
                "industry": ind_tag,
                "seed": seed,
                "value_prop": focus_valprop,
                "pct": pct,
                "pct_and_weight": (pct, simrun_weight)
            }
            simrun_freq_rows.append(freq_row)
    return pd.DataFrame(data = simrun_freq_rows)


def filter_industry_freq_df(in_df, filter_dict):
    f_type = filter_dict["type"]
    f_val = filter_dict["val"]
    if f_type not in ["avg", "count"]:
        raise NotImplementedError
    if f_type == "avg":
        return filter_by_industry_average(in_df = in_df, min_val = f_val)
    return filter_by_industry_count(in_df = in_df, min_val = f_val)


def filter_by_industry_average(in_df, min_val = 0.1):
    avg_vals = in_df.groupby(by = "industry")["pct"].mean()
    filtered = avg_vals[avg_vals >= min_val].sort_values(ascending = False)
    print(f"Filtering industries with average frequency below {min_val:.0%}")
    filter_industries = filtered.index.values.tolist()
    filtered_df = in_df.loc[in_df.industry.isin(filter_industries)]
    return filtered_df, filter_industries


def filter_by_industry_count(in_df, min_val = 20):
    count_vals = in_df.industry.value_counts()
    filtered = count_vals[count_vals >= min_val].sort_values(ascending = False)
    print(f"Filtering industries that appeared under {min_val} times")
    filter_industries = filtered.index.values.tolist()
    filtered_df = in_df.loc[in_df.industry.isin(filter_industries)]
    return filtered_df, filter_industries


def get_title_annotation(filter_dict):
    f_type = filter_dict["type"]
    f_val = filter_dict["val"]
    pt1 = "Shows industries with"
    if f_type == "avg":
        pt2 = f">= {f_val:.0%} average frequency"
    else:
        pt2 = f">= {f_val} appearances"
    return f"{pt1} {pt2} across all LSH buckets"


def plot_industry_frequency(plot_df, vp_name, filter_dict, industry_order,
    save_plot = True, base_path = MULTI_LSH_PLOTS_PATH,
    subdir_name = "industry-freq"):
    sns.set_style("whitegrid")
    f_type = filter_dict["type"]
    f_val = filter_dict
    fig, ax = plt.subplots(figsize = (16,8))
    g = sns.barplot(x = "industry", y = "pct_and_weight", data = plot_df,
        order = industry_order, color = VP_COLORS[vp_name], orient = "v",
        alpha = 0.4, ax = ax, ci = 95, estimator = weighted_mean, n_boot = 1000)
    sns.despine()
    ax.set_xlabel("Industry")
    ax.set_ylabel("Proportion of LSH Bucket")
    ax.set_xticklabels(ax.get_xticklabels(), 
        rotation = 60,
        horizontalalignment = "right")
    pct_ax = ticker.PercentFormatter(xmax = 1.0, decimals = 0)
    ax.yaxis.set_major_formatter(pct_ax)
    plt.suptitle(f"Industry Frequency across 100 LSH buckets: {vp_name}")
    plt.title(get_title_annotation(filter_dict), y = 0.9)
    plt.tight_layout()
    if save_plot:
        save_dir = base_path / subdir_name
        if not save_dir.exists():
            save_dir.mkdir(parents = True, exist_ok = True)
        filter_info = f"{f_type}Filter"
        vp_info = vp_name.replace(" ","_")
        full_path = save_dir / f"{filter_info}-{vp_info}.png"
        print(f"Saving plot to {full_path}")
        plt.savefig(full_path)
    plt.close()


def make_industry_frequency_plots(lsh_run_df, list_of_filter_dicts,
    base_path = MULTI_LSH_PLOTS_PATH,
    subdir_name = "industry-freq"):
    value_props = lsh_run_df.value_prop.unique().tolist()
    for vp in value_props:
        industry_df = make_industry_frequency_df(lsh_run_df = lsh_run_df,
            focus_valprop = vp)
        for params in list_of_filter_dicts:
            plot_df, plot_order = filter_industry_freq_df(in_df = industry_df,
                filter_dict = params)
            plot_industry_frequency(plot_df = plot_df, vp_name = vp,
                industry_order = plot_order, filter_dict = params)  


def make_risk_profile_df(lsh_run_df, focus_valprop):
    simrun_risk_rows = []
    risk_cols = ["revenue_note", "recent_funding", "recent_founding"]
    vp_df = lsh_run_df.loc[lsh_run_df.value_prop == focus_valprop]
    for seed in vp_df.sim_seed.unique().tolist():
        seed_df = vp_df.loc[vp_df.sim_seed == seed]
        simrun_weight = len(seed_df)
        for risk in risk_cols:
            tally_df = seed_df[risk].value_counts() / simrun_weight
            for risk_tag, risk_pct in tally_df.to_dict().items():
                risk_row = {
                    "risk": risk_tag,
                    "seed": seed, 
                    "value_prop": focus_valprop,
                    "pct": risk_pct,
                    "pct_and_weight": (risk_pct, simrun_weight)
                }
                simrun_risk_rows.append(risk_row)
    return pd.DataFrame(data = simrun_risk_rows)


def plot_risk_profile(plot_df, vp_name, save_plot = True,
    base_path = MULTI_LSH_PLOTS_PATH,
    subdir_name = "risk-profile"):
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize = (12,8))
    g = sns.barplot(x = "risk", y = "pct_and_weight", data = plot_df,
        color = VP_COLORS[vp_name], orient = "v", alpha = 0.4, ax = ax,
        ci = 95, estimator = weighted_mean, n_boot = 1000)
    sns.despine()
    ax.set_xlabel("Risk Factor")
    ax.set_ylabel("Proportion of LSH Bucket")
    ax.set_xticklabels(ax.get_xticklabels(), 
        rotation = 45,
        horizontalalignment = "right")
    pct_ax = ticker.PercentFormatter(xmax = 1.0, decimals = 0)
    ax.yaxis.set_major_formatter(pct_ax)
    plt.suptitle(f"Industry Frequency across 100 LSH buckets: {vp_name}")
    plt.tight_layout()
    if save_plot:
        save_dir = base_path / subdir_name
        if not save_dir.exists():
            save_dir.mkdir(parents = True, exist_ok = True)
        vp_info = vp_name.replace(" ","_")
        full_path = save_dir / f"{vp_info}.png"
        print(f"Saving plot to {full_path}")
        plt.savefig(full_path)
    plt.close()


def make_risk_profile_plots(lsh_run_df, base_path = MULTI_LSH_PLOTS_PATH,
    subdir_name = "risk-profile"):
    vp_list = lsh_run_df.value_prop.unique().tolist()
    for vp in vp_list:
        risk_df = make_risk_profile_df(lsh_run_df = lsh_run_df, 
            focus_valprop = vp)
        plot_risk_profile(plot_df = risk_df, vp_name = vp)


if __name__ == "__main__":
    
    lsh_run_df = pd.read_table(LSH_AGG_PATH)
    
    # INDUSTRY FREQUENCY PLOTS
    FILTER_DICTS = [{"type": "avg", "val": 0.05}, {"type": "count", "val":40}]
    make_industry_frequency_plots(lsh_run_df = lsh_run_df, 
        list_of_filter_dicts = FILTER_DICTS)

    #RISK PROFILE PLOTS
    make_risk_profile_plots(lsh_run_df = lsh_run_df)






