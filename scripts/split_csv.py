from typing import Sequence
import pandas as pd
import numpy as np
from scripts.text_cleaning import _clean_tweet


def combine_tweets(
    pro_csv_file: str, anti_csv_file: str, use_lockdown: bool, out_file: str
) -> None:
    pro_df = pd.read_csv(pro_csv_file)
    anti_df = pd.read_csv(anti_csv_file)
    if use_lockdown:
        all_tweets = pro_df[["fragment", "lockdown"]].append(
            anti_df[["fragment", "lockdown"]], ignore_index=True
        )
        all_tweets = all_tweets.loc[all_tweets["lockdown"].isin(["y", "n"])]
        all_tweets = all_tweets.rename(columns={"lockdown": "label"})

    else:
        all_tweets = pro_df[["fragment"]].append(anti_df[["fragment"]], ignore_index=True)
        all_tweets["label"] = ["pro"] * len(pro_df) + ["anti"] * len(anti_df)
    all_tweets = all_tweets.rename(columns={"fragment": "text"})
    all_tweets.to_csv(out_file, index=False)


def train_test_split(
    data_csv: str,
    out_files: Sequence = (
        "../data/train.csv",
        "../data/val.csv",
        "../data/test.csv",
        "../data/all.csv",
    ),
    portions: Sequence = (0.8, 0.1, 0.1),
    shuffle: bool = True,
):
    np.random.seed(1024)
    portions = np.cumsum(portions, dtype=np.float32) / np.sum(portions)
    row_indices = [[], [], []]
    data_df = pd.read_csv(data_csv)
    cleaned = []
    rows = data_df.values
    for row in rows:
        tweet = _clean_tweet(row[0])
        if tweet:
            cleaned.append([tweet, row[1]])
    data_df = pd.DataFrame(cleaned, columns=["text", "label"])
    for row_idx in range(len(data_df)):
        i = np.searchsorted(portions, np.random.rand())
        row_indices[i].append(row_idx)

    for r, f_name in zip(row_indices, out_files):
        if shuffle:
            np.random.shuffle(r)
        data_df.iloc[r,].reset_index(
            drop=True
        ).to_csv(f_name, index=True, index_label="index")
    data_df.to_csv(out_files[-1], index=True, index_label="index")


if __name__ == "__main__":
    pro_file = "../data/raw/INSA_ProReopenAmerica.csv"
    anti_file = "../data/raw/INSA_AntiReopenAmerica.csv"
    combine_tweets(
        pro_file,
        anti_file,
        use_lockdown=True,
        out_file="../data/all_tweets_with_lockdown.csv",
    )
    train_test_split("../data/all_tweets_with_lockdown.csv")
