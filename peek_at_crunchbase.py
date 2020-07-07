import ipdb
import pathlib
import pandas as pd 

from collections import Counter
from pprint import pprint

from settings import CRUNCHBASE_2015_PATH

files_to_examine = ["acquisitions", "additions",
    "companies", "investments", "rounds"]


def extract_categories(category_list, delim = "|"):
    if pd.isnull(category_list):
        return []
    return [v.replace(" ","") for v in category_list.split(delim)]



if __name__ == "__main__":

    for filename in files_to_examine:
        
        print(f"Here are the columns in {filename}")
        csv_path = CRUNCHBASE_2015_PATH / f"{filename}.csv"
        df = pd.read_csv(csv_path)
        pprint(df.columns.tolist())
        print("-------------")

    company_df = pd.read_csv(CRUNCHBASE_2015_PATH / "companies.csv")
    ipdb.set_trace()




