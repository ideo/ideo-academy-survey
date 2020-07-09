import ipdb
import numpy as np
import pathlib
import pandas as pd 

from collections import Counter, defaultdict
from pprint import pprint
from sklearn.feature_extraction.text import CountVectorizer

from preprocess_crunchbase import get_crunchbase2020_data

VALUE_PROPS = [
    '''
    Get your team’s highest priority job done while learning to work in customer centric ways
    Skill up for success: Solve your team’s toughest challenges with a modern customer centric approach
    ''',
    '''
    Scale new processes to deliver value to customers
    Build a high performing culture: Get teams across your organization to adopt new ways of working and 
    deliver value to your customers
    ''',
    '''
    Lead your organization’s Digital Transformation and create a competitive edge
    Build a future fit organization: Digitally transform your organization by upskilling your teams
    ''',
    '''
    Develop creative leaders while delivering on top strategic goals
    Cultivate future leaders: Develop the next generation of leaders while delivering on your organization’s top strategic goals
    '''
]

def map_docs_to_hash_table(lsh_matrix, hash_size = 8, proj_seed = None):
    '''I could probably map this to a broader class that implements this method
    the same time to create several hash tables

    What this ultimately returns is a mapping of N-bit hash codes to a list of 
    document indexes. This means given a hash we can find similar documents.
    '''
    np.random.seed(proj_seed)
    proj_dims = (hash_size, lsh_matrix.shape[1])
    rand_proj = np.random.standard_normal(size = proj_dims)
    hashed_docs = (lsh_matrix.dot(rand_proj.T) > 0).astype(int)
    hash_table = defaultdict(list)
    for i, int_array in enumerate(hashed_docs):
        hash_table["".join(int_array.astype(str))].append(i)
    return hash_table


def get_new_document_lookup_key(doc_matrix, hash_size = 8, proj_seed = None):
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


def get_term_prevalence(in_df, doc_matrix, feat_array, 
    display_top = True, top_n = 15):
    doc_term_count = Counter()
    for i in in_df.index.tolist():
        term_inds = np.nonzero(doc_matrix[i])[1]
        doc_terms = feat_array[term_inds].tolist()
        doc_term_count.update(doc_terms)
    if display_top:
        print("\tFREQUENTLY USED TERMS")
        for t, c in doc_term_count.most_common(top_n):
            print(f"\t\t{t} - in {c} company descriptions")
    return doc_term_count


def find_term_overlap(index, doc_matrix, feat_array,
    overlap_inds):
    term_inds = np.nonzero(doc_matrix[index])[1]
    doc_terms = set(feat_array[term_inds])
    ref_terms = set(feat_array[overlap_inds])
    overlap_terms = list(doc_terms.intersection(ref_terms))
    return None if not overlap_terms else ", ".join(overlap_terms) 


def display_lsh_member_info(in_df, doc_matrix, feat_array, 
    overlap_inds):
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
    for i, j in enumerate(in_df.index.tolist()):
        for col in display_cols:
            display_val = in_df.at[j, col]
            if pd.isnull(display_val):
                continue
            if col == "cb_rank":
                display_val = f"{display_val:,}"
            display_name = " ".join([v.title() for v in col.split("_")])
            if col == "org_name":
                display_name = f"{i + 1}. {display_name}"
            print(f"\t\t{display_name}: {display_val}")
        common_terms = find_term_overlap(index = j, 
            doc_matrix = doc_matrix, 
            feat_array = feat_array, 
            overlap_inds = overlap_inds)
        if common_terms is not None:
            print(f"\t\tTerms in Common: {common_terms}")
        print()




if __name__ == "__main__":
    
    N_BITS = 8
    SEED = 666
    TOP_N = 20
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
    vectorizer_features = np.array(base_vectorizer.get_feature_names())
    for i, key in enumerate(vp_keys):
        vp_nonzero_inds = np.nonzero(vp_matrix[i])[1]
        print(f"Value Prop {i + 1}: {VALUE_PROPS[i]}")
        hash_set_df = get_lsh_member_df(key = key, 
            hash_table = lsh_tbl, ref_df = base_df)
        print(f"Fell in hash bucket {key} with {len(hash_set_df)} other companies")
        tag_freq = get_tag_frequency(in_df = hash_set_df)
        print("--"*10,"\n")
        term_prev = get_term_prevalence(in_df = hash_set_df, 
            doc_matrix = lsh_matrix, feat_array = vectorizer_features)
        print("--"*10,"\n")
        top_N_df = hash_set_df.sort_values(by = "cb_rank").head(TOP_N)
        print(f"\tDetail on top {TOP_N} companies by CB Rank")
        display_lsh_member_info(in_df = top_N_df, doc_matrix = lsh_matrix, 
            feat_array = vectorizer_features, 
            overlap_inds = vp_nonzero_inds)