import os
from typing import List

import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import KFold, cross_validate

from preprocess import Preprocess


def __make_feature_vec(words: List[str], model: Word2Vec, num_features: int):
    """
    Average the word vectors for a set of words
    """
    feature_vec = np.zeros((num_features,), dtype="float32")
    nwords = 1.0
    if model.wv.index_to_key is None:
        exit(1)
    index_to_key = set(model.wv.index_to_key)
    for word in words:
        if word in index_to_key:
            nwords = nwords + 1
            feature_vec = np.add(feature_vec, model.wv[word])

    feature_vec = np.divide(feature_vec, nwords)
    return feature_vec


def get_sentence(texts: list):
    retval = []
    for text in texts:
        sentences = text.splitlines()
        for sentence in sentences:
            retval.append(sentence.split())
    retval = [x for x in retval if len(x) > 0]
    return retval


def get_avg_feature_vecs(
    sentences: List[List[str]], model: Word2Vec, num_features: int
) -> np.ndarray:
    """
    Calculate average feature vectors for all reviews
    """
    counter = 0
    feature_vecs = np.zeros(
        (len(sentences), num_features), dtype="float32"
    )  # (for speed)
    for sentence in sentences:
        feature_vecs[counter] = __make_feature_vec(sentence, model, num_features)
        counter = counter + 1
    return feature_vecs


def get_feature_vec(x, vectorizer):
    vectorizer_fit = vectorizer.fit(x)
    return vectorizer_fit.transform(x)


def cv_score(X, y):
    clf = RandomForestClassifier()
    kfold = KFold(n_splits=10, shuffle=True, random_state=3)
    score = cross_validate(clf, X, y, cv=kfold, scoring=["f1_macro", "roc_auc"])
    return {
        "f1_macro": round(score["test_f1_macro"].mean(), 3),
        "roc_auc": round(score["test_roc_auc"].mean(), 3),
    }


def load_df():
    if os.path.exists("data/rq1/rq1.pq"):
        return pd.read_parquet("data/rq1/rq1.pq")
    else:
        clean_df = pd.read_pickle("data/rq1/clean.pkl")
        buggy_df = pd.read_pickle("data/rq1/buggy.pkl")
        df = pd.concat([clean_df, buggy_df])[
            ["commit_hash", "label", "message", "body"]
        ]
        prep = Preprocess()
        df["body"] = df["body"].apply(lambda x: prep.lemmatize_sentence(x))
        df["message"] = df["message"].apply(lambda x: prep.lemmatize_sentence(x))
        df["xcm"] = df["message"] + df["body"]
        df = df.drop_duplicates(subset=["message", "body"])
        df.to_parquet("data/rq1/rq1.pq")
        return df


def mean_score(df, vectorizer):
    data_list = []
    for name in ["message", "body", "xcm"]:
        X = get_feature_vec(df[name], vectorizer)
        y = df["label"]
        score = cv_score(X, y)
        data_list.append(score)
    return pd.DataFrame(
        data=data_list,
        columns=["f1_macro", "roc_auc"],
        index=["message", "body", "xcm"],
    )


def mean_score_w2v(df, num_features: int):
    data_list = []
    for name in ["message", "body", "xcm"]:
        sentences = get_sentence(df[name])
        model = Word2Vec(sentences, vector_size=num_features, min_count=3, epochs=20)
        clean_tokens = [x.split() for x in df[df["label"] == 0][name]]
        buggy_tokens = [x.split() for x in df[df["label"] == 1][name]]
        X = get_avg_feature_vecs(
            clean_tokens + buggy_tokens, model=model, num_features=num_features
        )
        y = df["label"]
        score = cv_score(X, y)
        data_list.append(score)
    return pd.DataFrame(
        data=data_list,
        columns=["f1_macro", "roc_auc"],
        index=["message", "body", "xcm"],
    )


def main():
    df = load_df()
    bow_vectorizer: CountVectorizer = CountVectorizer(
        stop_words="english",
        analyzer="word",
        token_pattern=r"(?u)\b[A-Za-z]{3,}\b",
    )
    mean_score(df, bow_vectorizer).to_csv("out/rq1/bow.csv")
    tf_idf_vectorizer: TfidfVectorizer = TfidfVectorizer(
        stop_words="english",
        analyzer="word",
        token_pattern=r"(?u)\b[A-Za-z]{3,}\b",
    )
    mean_score(df, tf_idf_vectorizer).to_csv("out/rq1/tf_idf.csv")
    mean_score_w2v(df, 100).to_csv("out/rq1/w2v.csv")


if __name__ == "__main__":
    main()
