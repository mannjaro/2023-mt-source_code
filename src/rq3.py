import argparse
from collections import Counter
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as imbpipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, label_binarize


def load_data(min_threshold: int):
    df = pd.read_parquet("data/rq3/commit.pq")
    df = df.drop_duplicates(subset=["commit_hash"])
    C = Counter(df["category"])
    category = {
        x: count for x, count in C.items() if count >= min_threshold and x != "other"
    }
    df = df[df["category"].isin(category.keys())]
    df["xcm"] = df["message"] + df["body"]
    df = df.reset_index(drop=True)
    return df


def cv_score(X, y):
    score = {
        "f1_macro": 0,
        "accuracy": 0,
        "roc_auc_ovr": 0,
    }
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)
    if use_smote:
        pipeline = imbpipeline(
            steps=[
                ["smote", SMOTE()],
                ["classifier", RandomForestClassifier()],
            ]
        )
    else:
        pipeline = imbpipeline(
            steps=[
                ["classifier", RandomForestClassifier()],
            ]
        )
    stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True)
    param_grid = {
        "classifier__criterion": ["gini", "entropy"],
        "classifier__max_depth": [3, 5, 7, None],
    }
    for score_name in score.keys():
        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring=score_name,
            cv=stratified_kfold,
            n_jobs=-1,
        )
        grid_search.fit(X_train, y_train)
        score[score_name] = grid_search.score(X_test, y_test)
    return score


def each_category_score(X, y, labels):
    ckf = StratifiedKFold(n_splits=5, shuffle=False)
    each_report = []
    roc_score_list = []
    data = {
        "add_task": {
            "precision": np.array([]),
            "recall": np.array([]),
            "f1-score": np.array([]),
            "roc-auc": np.array([]),
        },
        "condition": {
            "precision": np.array([]),
            "recall": np.array([]),
            "f1-score": np.array([]),
            "roc-auc": np.array([]),
        },
        "edit_var": {
            "precision": np.array([]),
            "recall": np.array([]),
            "f1-score": np.array([]),
            "roc-auc": np.array([]),
        },
        "template": {
            "precision": np.array([]),
            "recall": np.array([]),
            "f1-score": np.array([]),
            "roc-auc": np.array([]),
        },
    }

    smote = SMOTE(random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=45, stratify=y
    )
    for train_fold_index, val_fold_index in ckf.split(X_train, y_train):
        if use_smote:
            X_train_sample, y_train_sample = smote.fit_resample(X_train, y_train)
        else:
            X_train_sample, y_train_sample = (
                X_train[train_fold_index],
                y_train[val_fold_index],
            )
        clf = RandomForestClassifier()
        clf.fit(X_train_sample, y_train_sample)
        y_pred = clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test)

        roc_scores = {}
        y_test_one_hot = label_binarize(y_test, classes=clf.classes_)
        for i in range(4):
            data[labels[i]]["roc-auc"] = np.append(
                data[labels[i]]["roc-auc"],
                roc_auc_score(
                    y_test_one_hot[:, i], y_pred_proba[:, i], multi_class="ovr"
                ),
            )
        roc_score_list.append(roc_scores)
        report = classification_report(
            y_test,
            y_pred,
            target_names=labels,
            zero_division=0,
            digits=4,
            output_dict=True,
        )
        each_report.append(report)
    scores = ["precision", "recall", "f1-score"]
    for report in each_report:
        for label in labels:
            for score in scores:
                data[label][score] = np.append(data[label][score], report[label][score])
    tmp = []
    for label in labels:
        tmp.append(
            [
                label,
                round(data[label][scores[0]].mean(), 4),
                round(data[label][scores[1]].mean(), 4),
                round(data[label][scores[2]].mean(), 4),
                round(data[label]["roc-auc"].mean(), 4),
            ]
        )
    return pd.DataFrame(
        columns=["category", scores[0], scores[1], scores[2], "roc-auc"], data=tmp
    )


def get_sentence(texts: list):
    retval = []
    for text in texts:
        sentences = text.splitlines()
        for sentence in sentences:
            retval.append(sentence.split())
    retval = [x for x in retval if len(x) > 0]
    return retval


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


def overall_score_w2v(df, num_features):
    score_table = []
    for name in ["message", "body", "xcm"]:
        sentences = get_sentence(df[name])
        model = Word2Vec(sentences, vector_size=num_features, min_count=3, epochs=20)
        tokens = [x.split() for x in df[name]]
        X = get_avg_feature_vecs(tokens, model, num_features)
        y = df["category"]
        score_table.append(accumulate(name, X, y))
    return pd.DataFrame(
        columns=["Feature", "f1_macro", "accuracy", "roc_auc_ovr"], data=score_table
    )


def overall_score(df: pd.DataFrame, vectorizer) -> pd.DataFrame:
    score_table = []
    for name in ["message", "body", "xcm"]:
        X = vectorizer.fit_transform(df[name])
        y = df["category"]
        score_table.append(accumulate(name, X, y))
    return pd.DataFrame(
        columns=["Feature", "f1_macro", "accuracy", "roc_auc_ovr"], data=score_table
    )


def accumulate(name, X, y):
    le = LabelEncoder()
    le.fit(y)
    y = le.transform(y)
    scoring = ["f1_macro", "accuracy", "roc_auc_ovr"]
    scores = cv_score(X, y)  # Mean 5-fold cv
    return [
        name,
        round(scores[scoring[0]], 3),
        round(scores[scoring[1]], 3),
        round(scores[scoring[2]], 3),
    ]


def each_score(df, vectorizer):
    retval = {}
    for name in ["message", "body", "xcm"]:
        X = vectorizer.fit_transform(df[name])
        y = df["category"]
        le = LabelEncoder()
        le.fit(y)
        y = le.transform(y)
        retval[name] = each_category_score(X, y, le.classes_)
    return retval


def each_score_w2v(df, num_features: int):
    retval = {}
    for name in ["message", "body", "xcm"]:
        sentences = get_sentence(df[name])
        model = Word2Vec(sentences, vector_size=num_features, min_count=3, epochs=20)
        tokens = [x.split() for x in df[name]]
        X = get_avg_feature_vecs(tokens, model, num_features)
        y = df["category"]
        le = LabelEncoder()
        le.fit(y)
        y = le.transform(y)
        retval[name] = each_category_score(X, y, le.classes_)
    return retval


def save(name: str, overall_df, each_dict, out: str):
    overall_df.to_csv(Path(out) / "{}.csv".format(name), index=False)
    for key, _df in each_dict.items():
        _df.to_csv(Path(out) / "{}_{}.csv".format(name, key), index=False)


def main(out: str):
    df = load_data(10)
    bow_vectorizer = CountVectorizer(
        stop_words="english",
        analyzer="word",
        token_pattern=r"(?u)\b[A-Za-z]{3,15}\b",
    )
    tf_idf_vectorizer = TfidfVectorizer(
        stop_words="english",
        analyzer="word",
        token_pattern=r"(?u)\b[A-Za-z]{3,15}\b",
    )
    print("BOW")
    overall_df = overall_score(df, bow_vectorizer)
    each_dict = each_score(df, bow_vectorizer)
    save("bow", overall_df, each_dict, out)

    print("TF-IDF")
    overall_df = overall_score(df, tf_idf_vectorizer)
    each_dict = each_score(df, tf_idf_vectorizer)
    save("tf_idf", overall_df, each_dict, out)

    print("Word2Vec")
    overall_df = overall_score_w2v(df, 100)
    each_dict = each_score_w2v(df, 100)
    save("w2v", overall_df, each_dict, out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate f1-macro, accuracy, roc-auc"
    )
    parser.add_argument(
        "--no_smote", action="store_false", help="True is using smote(default = true)"
    )
    parser.add_argument(
        "-o",
        "--out",
        type=str,
        default="out/rq3",
        help="Write output to <dir>(default = './out/rq3')",
    )
    args = parser.parse_args()
    use_smote = args.no_smote
    main(args.out)
