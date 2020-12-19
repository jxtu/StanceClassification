import pickle
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from abc import ABC, abstractmethod
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt


class TwitterClustering(ABC):
    def __init__(self, data_df, tweets_vecs):
        self.data_df = data_df
        self.tweets_vecs = tweets_vecs
        self.c_labels = None

    @classmethod
    @abstractmethod
    def from_files(cls, *args):
        pass

    def fit(self, k: int):
        kmean_model = KMeans(n_clusters=k)
        kmean_model.fit(self.tweets_vecs)
        self.c_labels = kmean_model.labels_

    def draw_elbow(self, min_k, max_k, standard):
        values = []
        mapping = {}
        K = range(min_k, max_k + 1)

        for k in K:
            print(f"building KMeans on k={k}...")
            kmeanModel = KMeans(n_clusters=k)
            kmeanModel.fit(self.tweets_vecs)
            if standard == "distortion":
                value = (
                    sum(
                        np.min(
                            cdist(self.tweets_vecs, kmeanModel.cluster_centers_, "euclidean"),
                            axis=1,
                        )
                    )
                    / self.tweets_vecs.shape[0]
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

    def to_csv(self, out_file):
        self.data_df["c_labels"] = self.c_labels
        self.data_df.to_csv(out_file, index=False)


class TFIDFClustering(TwitterClustering):
    def __init__(self, data_df, tweets_vecs):
        super().__init__(data_df, tweets_vecs)

    @classmethod
    def from_files(cls, input_csv, *, min_df):
        data_df = pd.read_csv(input_csv)
        tweets = data_df.text
        vectorizer = TfidfVectorizer(lowercase=True, stop_words="english", min_df=min_df)
        tweets_vecs = vectorizer.fit_transform(tweets.values).toarray()
        return cls(data_df, tweets_vecs)


class RoBertaClustering(TwitterClustering):
    def __init__(self, data_df, tweets_vecs):
        super().__init__(data_df, tweets_vecs)

    @classmethod
    def from_files(cls, input_csv, *, input_pkl):
        data_df = pd.read_csv(input_csv)
        with open(input_pkl, "rb") as f:
            tweets_vecs = pickle.load(f)
        return cls(data_df, tweets_vecs)


def main():
    parser = argparse.ArgumentParser(description="clustering")
    all_tweets_file = "../data/fast_bert_data_clean/all.csv"
    all_tweets_rob_vecs = "../data/rob_tweets_vecs_all.pkl"
    all_tweets_lockdown_file = "../data/fast_bert_data_clean_lockdown/all.csv"
    all_tweets_lockdown_rob_vecs = "../data/rob_tweets_vecs_lockdown.pkl"
    result_dir = Path("../data/stance_clustering_results")

    parser.add_argument("data", choices=["all", "lockdown"], help="Use all tweets or lockdown only")
    parser.add_argument(
        "vecs", choices=["tfidf", "roberta"], help="Types of vectors to represent tweets"
    )
    parser.add_argument("-c", "--clusters_num", type=int, help="Number of clusters")
    parser.add_argument("-d", "--draw", action="store_true", help="Draw elbow figure")
    parser.add_argument("--min_k", type=int, default=2, help="min_k to be tested")
    parser.add_argument("--max_k", type=int, default=5, help="max_k to be tested")
    args = parser.parse_args()

    if args.vecs == "tfidf":
        if args.data == "lockdown":
            cluster = TFIDFClustering.from_files(all_tweets_lockdown_file, min_df=25)
        else:
            cluster = TFIDFClustering.from_files(all_tweets_file, min_df=25)
    else:
        if args.data == "lockdown":
            cluster = RoBertaClustering.from_files(
                all_tweets_lockdown_file, input_pkl=all_tweets_lockdown_rob_vecs
            )
        else:
            cluster = RoBertaClustering.from_files(all_tweets_file, input_pkl=all_tweets_rob_vecs)
    if args.draw:
        cluster.draw_elbow(args.min_k, args.max_k, "inertia")
    else:
        cluster.fit(args.clusters_num)
        out_path = result_dir.joinpath(f"{args.data}_roberta_{args.clusters_num}.csv")
        cluster.to_csv(out_path)
        print(f"writing output to {out_path}.")


if __name__ == "__main__":
    main()
