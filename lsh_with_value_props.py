import numpy as np
import pathlib
import pandas as pd 

from collections import Counter, defaultdict
from pprint import pprint
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_distances
from tqdm import tqdm

from settings import VALUE_PROPS, CRUNCHBASE_2020_PATH
from preprocess_crunchbase import get_crunchbase2020_data


def create_lsh_table(doc_matrix, hash_size = 8, proj_seed = None):
    '''I could probably map this to a broader class that implements this method
    the same time to create several hash tables

    What this ultimately returns is a mapping of N-bit hash codes to a list of 
    document indexes. This means given a hash we can find similar documents.
    '''
    np.random.seed(proj_seed)
    proj_dims = (hash_size, doc_matrix.shape[1])
    rand_proj = np.random.standard_normal(size = proj_dims)
    hashed_docs = (doc_matrix.dot(rand_proj.T) > 0).astype(int)
    hash_table = defaultdict(list)
    for i, int_array in enumerate(hashed_docs):
        hash_table["".join(int_array.astype(str))].append(i)
    return hash_table


def get_document_lsh_keys(doc_matrix, hash_size = 8, proj_seed = None):
    '''Putting all of this in a class would allow us to save the
    projections and hash_size as instance attribute.

    Also, yes, I am aware this is a refactoring dumpster fire right now...
    '''
    np.random.seed(proj_seed)
    proj_dims = (hash_size, doc_matrix.shape[1])
    rand_proj = np.random.standard_normal(size = proj_dims)
    hashed_docs = (doc_matrix.dot(rand_proj.T) > 0).astype(int)
    hash_keys = ["".join(arr.astype(str)) for arr in hashed_docs]
    return hash_keys if len(hash_keys) > 1 else hash_keys[0]


def get_lsh_member_df(key, hash_table, ref_df):
    company_inds = hash_table[key]
    return ref_df.loc[ref_df.index.isin(company_inds)]


def create_company_lsh_inputs(doc_list, doc_vectorizer):
    vectorized_docs = doc_vectorizer.fit_transform(doc_list)
    doc_features = np.array(doc_vectorizer.get_feature_names())
    return vectorized_docs, doc_features


def create_value_prop_lsh_inputs(vp_list, fitted_vectorizer,
    hash_size = 8, proj_seed = None):
    vp_matrix = fitted_vectorizer.transform(vp_list)
    vp_keys = get_document_lsh_keys(doc_matrix = vp_matrix,
        hash_size = hash_size, proj_seed = proj_seed)
    return vp_matrix, vp_keys


def get_tag_frequency(in_df, tag_col = "industry_tags", sep = ", ", 
        display_top = True, top_n = 15):
    tag_count = Counter()
    for tag in in_df[tag_col].values.tolist():
        tag_count.update(tag.split(sep))
    if display_top:
        print("\tTOP TAGS")
        for t, c in tag_count.most_common(top_n):
            print(f"\t\t{t} - matches {c} companies")
    return tag_count


def get_term_prevalence(in_df, doc_matrix, vocab_array, 
    display_top = True, top_n = 15):
    doc_term_count = Counter()
    for i in in_df.index.tolist():
        term_inds = np.nonzero(doc_matrix[i])[1]
        doc_terms = vocab_array[term_inds].tolist()
        doc_term_count.update(doc_terms)
    if display_top:
        print("\tFREQUENTLY USED TERMS")
        for t, c in doc_term_count.most_common(top_n):
            print(f"\t\t{t} - in {c} company descriptions")
    return doc_term_count


def find_term_overlap(index, doc_matrix, vocab_array,
    overlap_inds):
    term_inds = np.nonzero(doc_matrix[index])[1]
    doc_terms = set(vocab_array[term_inds])
    ref_terms = set(vocab_array[overlap_inds])
    overlap_terms = list(doc_terms.intersection(ref_terms))
    return None if not overlap_terms else ", ".join(overlap_terms) 


def get_cosine_neighbors(vp_vector, comparison_vectors,
        n_neighbors=20):
    dist_mat = cosine_distances(vp_vector, comparison_vectors).flatten()
    nearest_inds = np.argsort(dist_mat)[:n_neighbors]
    sim_scores = 1 - dist_mat[nearest_inds]
    return nearest_inds, sim_scores


def add_simrun_annotations(in_df, vp_index, simrun_seed):
    '''Should allow us to keep track of which companies got paired with
    which value props across different runs of LSH.

    value prop index is an INDEX, not the actual integer label.
    '''
    in_df["value_prop"] = f"Value Prop {vp_index + 1}"
    in_df["sim_seed"] = simrun_seed


def add_common_term_columns(in_df, doc_matrix, vocab_array, common_inds):
    common_terms = lambda x: find_term_overlap(x, doc_matrix,
        vocab_array, common_inds)
    count_terms = lambda y: 0 if not y else len(y.split(", "))
    in_df["common_terms"] = in_df.index.map(common_terms)
    in_df["common_terms"].fillna("", inplace = True)
    in_df["n_terms"] = in_df["common_terms"].apply(count_terms)


def add_cosine_similarity(in_df, doc_matrix, vp_vector):
    sim_score = lambda x: (1 - cosine_distances(vp_vector, doc_matrix[x])).item()
    in_df["term_cosine_sim"] = in_df.index.map(sim_score)


def display_company_info(rank, df_index, info_df):
    '''In practice, I would probably only apply this to some subset
    of an entire hash bucket
    '''
    display_cols = ["org_name",
        "description",
        "industry_tags",
        "cb_rank",
        "revenue_note",
        "recent_funding",
        "recent_founding"
    ]
    for col in display_cols:
        display_val = info_df.at[df_index, col]
        if pd.isnull(display_val):
            continue
        if col == "cb_rank":
            display_val = f"{display_val:,}"
        display_name = " ".join([v.title() for v in col.split("_")])
        if col == "org_name":
            display_name = f"{rank}. {display_name}"
        print(f"\t\t{display_name}: {display_val}")


def display_top_cb_rank_companies(in_df, n_display, doc_matrix, vocab_array,
    vp_term_inds):
    display_df = in_df.sort_values(by = "cb_rank").head(n_display)
    print(f"\tDetail on top {n_display} companies by CB Rank")
    for rank, index in enumerate(display_df.index.tolist()):
        display_company_info(rank = rank + 1, df_index = index,
            info_df = display_df)
        common_terms = find_term_overlap(index = index, doc_matrix = doc_matrix,
            vocab_array = vocab_array, overlap_inds=vp_term_inds)
        if common_terms is not None:
            print(f"\t\tCommon Terms: {common_terms}")
        print()


def display_cosine_similar_companies(in_df, n_display, vp_vector,
    doc_matrix, vocab_array, vp_term_inds, compare_hash_keys = True,
    comparison_key_list = None, hash_size = 8, proj_seed = None):
    neighbor_inds, sim_scores = get_cosine_neighbors(vp_vector = vp_vector, 
            comparison_vectors = doc_matrix, n_neighbors = TOP_N)
    print(f"\tDetail on top {TOP_N} companies by Term Similarity")
    for rank, index in enumerate(neighbor_inds):
        display_company_info(rank = rank + 1, df_index = index,
            info_df = in_df)
        print(f"\t\tSimilarity Score: {sim_scores[rank]:.4f}")
        common_terms = find_term_overlap(index = index, doc_matrix = doc_matrix,
            vocab_array = vocab_array, overlap_inds = vp_term_inds)
        if common_terms is not None:
            print(f"\t\tCommon Terms: {common_terms}")
        if compare_hash_keys:
            in_vector = doc_matrix[index]
            neighbor_key = get_document_lsh_keys(doc_matrix = in_vector,
                hash_size = hash_size, proj_seed = proj_seed)
            if neighbor_key in comparison_key_list:
                which_vp = comparison_key_list.index(neighbor_key) + 1
                print(f"\t\tFalls in the same bucket as value prop {which_vp}")
            else:
                print(f"\t\tFalls outside our value prop buckets")
        print()
    

def aggregate_lsh_runs(crunchbase_df, vp_list, lsh_vectorizer, num_runs = 100, 
    hash_size = 8, base_seed = None, write_result = False):
    documents = crunchbase_df["document"].values.tolist()
    doc_matrix, doc_vocab = create_company_lsh_inputs(doc_list = documents, 
        doc_vectorizer = lsh_vectorizer)
    np.random.seed(base_seed)
    lsh_seeds = np.random.choice(a = 5000, size = num_runs, replace = False)

    lsh_dfs = []
    for proj_seed in tqdm(lsh_seeds.tolist()):
        lsh_tbl = create_lsh_table(doc_matrix = doc_matrix,
            hash_size = hash_size, proj_seed = proj_seed)
        vp_matrix, vp_keys = create_value_prop_lsh_inputs(vp_list = vp_list,
            fitted_vectorizer = lsh_vectorizer, hash_size = hash_size, 
            proj_seed = proj_seed)
        for i, key in enumerate(vp_keys):
            vp_nonzero_inds = np.nonzero(vp_matrix[i])[1]
            hash_df = get_lsh_member_df(key = key, hash_table = lsh_tbl,
                ref_df = crunchbase_df)
            add_simrun_annotations(in_df = hash_df, vp_index = i,
                simrun_seed = proj_seed)
            add_common_term_columns(in_df = hash_df, doc_matrix = doc_matrix,
                vocab_array = doc_vocab, common_inds = vp_nonzero_inds)
            add_cosine_similarity(in_df = hash_df, doc_matrix = doc_matrix,
                vp_vector = vp_matrix[i])
            print(f"\tSaving {len(hash_df)} rows-value prop {i + 1}-LSH{proj_seed}")
            lsh_dfs.append(hash_df)
    
    lsh_run_df = pd.concat(objs = lsh_dfs, axis = 0)
    lsh_run_df["company_index"] = lsh_run_df.index
    if write_result:
        save_lsh_runs(write_df = lsh_run_df, num_runs = num_runs,
            hash_size = hash_size, vp_list = vp_list,
            base_seed = base_seed)


def save_lsh_runs(write_df, num_runs, hash_size, vp_list, base_seed,
    base_path = CRUNCHBASE_2020_PATH, subdir_name = "LSH-Runs"):
    save_dir = base_path / subdir_name
    if not save_dir.exists():
        save_dir.mkdir(parents = True, exist_ok = True)
    pt1 = f"{num_runs}runs-{hash_size}bit"
    pt2 = f"{len(vp_list)}vps-seed{'NA' if base_seed is None else base_seed}"
    sv_path = save_dir / f"{pt1}-{pt2}.tsv"
    print(f"Saving {len(write_df)} rows to {sv_path}")
    write_df.to_csv(path_or_buf = sv_path, sep = "\t", index = False,
        header = True)


if __name__ == "__main__":
    
    N_RUNS = 100
    N_BITS = 8
    INIT_SEED = 1024

    base_df = get_crunchbase2020_data()
    base_vectorizer = CountVectorizer(lowercase = True,
        token_pattern = r"(?u)\b\w\w+\b", min_df = 3,
        stop_words = "english")

    aggregate_lsh_runs(crunchbase_df = base_df, vp_list = VALUE_PROPS, 
        lsh_vectorizer = base_vectorizer, num_runs = N_RUNS,
        hash_size = N_BITS, base_seed = INIT_SEED, 
        write_result = True) 



