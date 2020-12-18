import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt


def cluster_find_elbow(tweets_vecs, min_k, max_k, standard):
    values = []
    mapping = {}
    K = range(min_k, max_k + 1)

    for k in K:
        print(f"building KMeans on k={k}...")
        kmeanModel = KMeans(n_clusters=k)
        kmeanModel.fit(tweets_vecs)
        if standard == "distortion":
            value = (
                sum(
                    np.min(
                        cdist(tweets_vecs, kmeanModel.cluster_centers_, "euclidean"),
                        axis=1,
                    )
                )
                / tweets_vecs.shape[0]
            )
        elif standard == "inertia":
            value = kmeanModel.inertia_
        else:
            raise ValueError(f"invalid standard: {standard}")
        values.append(value)
        mapping[k] = value
    plt.plot(K, values, "bx-")
    plt.xlabel("Values of K")
    plt.ylabel(f"{standard}")
    plt.title(f"The Elbow Method using {standard}")
    plt.show()


def tfidf_cluster2csv(input_file: str, out_file_prefix, min_df: int, k: int):
    csv_df = pd.read_csv(input_file)
    tweets = csv_df.text
    vectorizer = TfidfVectorizer(lowercase=True, stop_words="english", min_df=min_df)
    tweets_vecs = vectorizer.fit_transform(tweets.values).toarray()
    kmean_model = KMeans(n_clusters=k)
    kmean_model.fit(tweets_vecs)
    csv_df["c_labels"] = kmean_model.labels_
    csv_df.to_csv(f"{out_file_prefix}_{min_df}_{k}.csv", index=False)


def roberta_cluster2csv(input_file, input_pkl: str, out_file_prefix, k: int):
    csv_df = pd.read_csv(input_file)
    with open(input_pkl, "rb") as f:
        tweets_vecs = pickle.load(f)
    kmean_model = KMeans(n_clusters=k)
    kmean_model.fit(tweets_vecs)
    csv_df["c_labels"] = kmean_model.labels_
    csv_df.to_csv(f"{out_file_prefix}_{k}.csv", index=False)


if __name__ == "__main__":
    # csv_df = pd.read_csv("../data/fast_bert_data_clean_lockdown/all.csv")
    # tweets = csv_df.text
    # vectorizer = TfidfVectorizer(lowercase=True, stop_words="english", min_df=5)
    # tweets_vecs = vectorizer.fit_transform(tweets.values).toarray()
    # print(tweets_vecs.shape)

    # with open("../data/rob_tweets_vecs_lockdown.pkl", "rb") as f:
    #     tweets_vecs = pickle.load(f)
    # cluster_find_elbow(tweets_vecs, 2, 10, "inertia")

    all_tweets_file = "../data/fast_bert_data_clean/all.csv"
    all_tweets_rob_vecs = "../data/rob_tweets_vecs_all.pkl"
    all_tweets_lockdown_file = "../data/fast_bert_data_clean_lockdown/all.csv"
    all_tweets_lockdown_rob_vecs = "../data/rob_tweets_vecs_lockdown.pkl"

    result_dir = "../data/stance_clustering_results"

    tfidf_cluster2csv(
        all_tweets_file,
        Path(result_dir).joinpath("all_tweets_clean_tfidf_clustered"),
        min_df=25,
        k=10,
    )
    roberta_cluster2csv(
        all_tweets_file,
        all_tweets_rob_vecs,
        Path(result_dir).joinpath("all_tweets_clean_roberta_clustered"),
        k=10,
    )

    tfidf_cluster2csv(
        all_tweets_lockdown_file,
        Path(result_dir).joinpath("all_tweets_clean_lockdown_tfidf_clustered"),
        min_df=5,
        k=8,
    )
    roberta_cluster2csv(
        all_tweets_lockdown_file,
        all_tweets_lockdown_rob_vecs,
        Path(result_dir).joinpath("all_tweets_clean_lockdown_roberta_clustered"),
        k=8,
    )
