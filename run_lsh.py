import ipdb
import numpy as np
import pathlib
import pandas as pd 

from collections import Counter
from pprint import pprint

from settings import CRUNCHBASE_2015_PATH


def get_company_df():
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



if __name__ == "__main__":
    df = get_company_df()
    display_random_companies(sample_size = 10, df = df)
    
