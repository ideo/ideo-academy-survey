import ipdb
import numpy as np
import pathlib
import pandas as pd 

from collections import Counter, defaultdict
from pprint import pprint
from sklearn.feature_extraction.text import CountVectorizer

from settings import CRUNCHBASE_2015_PATH


def get_company_df():
    # MOVE TO PREPROCESS CRUNCHBASE
    csv_path = CRUNCHBASE_2015_PATH / "companies.csv"
    df = pd.read_csv(csv_path)
    df = df.loc[pd.notnull(df.category_list)]
    print(f"Returning {len(df)} companies with category info")
    return df.reset_index()


def extract_categories(category_list, delim = "|"):
    if pd.isnull(category_list):
        return []
    return [v.replace(" ","") for v in category_list.split(delim)]


def get_company_location(index, df):
    city = df.at[index, "city"]
    clean_city = city.title() if pd.notnull(city) else "N/A"
    state = df.at[index, "state_code"]
    clean_state = state.upper() if pd.notnull(state) else "N/A"
    country = df.at [index, "country_code"]
    clean_country = country.upper() if pd.notnull(country) else "N/A"
    return f"{clean_city}, {clean_state}, {clean_country}"  


def get_funding_total(index, df):
    fund_usd = df.at[index,"funding_total_usd"]
    if fund_usd == "-":
        return "N/A"
    return f"${float(fund_usd):,.0f}"


def display_company_info(index, df):
    name = df.at[index,"name"]
    cat_list = extract_categories(df.at[index,"category_list"])
    print(f"Summary for {name.title()}")
    print(f"\tLOCATION: {get_company_location(index, df)}")
    print(f"\tFUNDING RAISED: {get_funding_total(index, df)}")
    print(f"\tCATEGORIES: {', '.join(cat_list)}")
    print(f"\tRead more at: {df.at[index,'homepage_url']}")
    print("--"*10,"\n")


def display_random_companies(sample_size, df):
    rand_inds = np.random.choice(a = df.index.tolist(),
                 size = sample_size, replace = False)
    print(f"Testing repr with {sample_size} companies")
    for ind in rand_inds:
        display_company_info(ind, df = df)


def get_vectorizer_vocab(df):
    cat_count = Counter()
    extracted_cats = df["category_list"].apply(extract_categories)
    for cat_list in extracted_cats.values.tolist():
        cat_count.update(cat_list)
    vocab_size = len(cat_count)
    return {term[0]: i for i, term in enumerate(cat_count.most_common(vocab_size))}


def make_lsh_doc_matrix(df):
    lsh_vectorizer = CountVectorizer(lowercase = False, vocabulary = get_vectorizer_vocab(df))
    lsh_lists = df["category_list"].apply(extract_categories)
    lsh_docs = lsh_lists.apply(lambda x: " ".join(x)).values.tolist()
    return lsh_vectorizer.fit_transform(lsh_docs)


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


def choose_random_hash(hash_table, choice_seed = None):
    np.random.seed(choice_seed)
    table_keys = [k for k in hash_table.keys()]
    return np.random.choice(table_keys, size = 1).item(0)


def choose_hashed_companies(hash_table, hash_key = None,
     sample_size = 10, choice_seed = None):
    if hash_key is None:
        hash_key = choose_random_hash(hash_table, choice_seed)
    np.random.seed(choice_seed)
    hash_bucket = hash_table[hash_key]
    return np.random.choice(hash_bucket, size = sample_size, replace = False)


def inspect_lsh_buckets(hash_table, sample_size, df, hash_key = None, 
     key_seed = None, company_seed = None):
    if hash_key is None:
        hash_key = choose_random_hash(hash_table = hash_table, 
            choice_seed = key_seed)
    bucket_size = len(hash_table[hash_key])
    print(f"{bucket_size} companies ended up in bucket {hash_key}")
    h_choices = choose_hashed_companies(hash_table = hash_table,
         hash_key = hash_key, sample_size = sample_size, 
         choice_seed = company_seed)
    print(f"Sample company IDs: {h_choices}")
    print("FIRST COMPANY")
    display_company_info(h_choices[0], df)
    print()
    print("'SIMILAR' COMPANIES")
    for comp in h_choices[1:]:
        display_company_info(comp, df)


def summarize_lsh_bucket():
    pass


if __name__ == "__main__":
    df = get_company_df()
    #display_random_companies(sample_size = 10, df = df)
    lsh_matrix = make_lsh_doc_matrix(df)
    lsh_tbl = map_docs_to_hash_table(lsh_matrix, hash_size = 8, proj_seed = 66)
   
    inspect_lsh_buckets(hash_table = lsh_tbl, sample_size = 9, 
        df = df)
    ipdb.set_trace()   
    
