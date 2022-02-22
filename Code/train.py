import numpy as np
import pandas as pd
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
import joblib


def stemming(sentence, stemmer):
    stemSentence = ""
    for word in sentence.split():
        stem = stemmer.stem(word)
        stemSentence += stem
        stemSentence += " "
    stemSentence = stemSentence.strip()
    return stemSentence


def prepare_data():
    # https://media.githubusercontent.com/media/Alalalalaki/Econ-Paper-Search/main/Data/papers.csv
    df = pd.read_csv("https://github.com/Alalalalaki/Econ-Paper-Search/blob/main/Data/papers.csv?raw=true")

    # borrow from Econ-Paper-Search
    # drop book reviews (not perfect)
    masks = [~df.title.str.contains(i, case=False, regex=False) for i in ["pp.", " p."]]  # "pages," " pp "
    mask = np.vstack(masks).all(axis=0)
    df = df.loc[mask]
    # drop some duplicates due to weird strings in authors and abstract
    df = df[~df.duplicated(['title', 'url']) | df.url.isna()]

    mask = ~df.jel.isna()
    mask2 = df.year > 2000
    mask3 = ~df.abstract.isna()

    df = df.loc[mask & mask2 & mask3]

    jel_dummy_matrix = df.jel.str.get_dummies(sep="&")
    # remove not 2-digit jel
    jel_dummy_matrix = jel_dummy_matrix.loc[:, ~(jel_dummy_matrix.columns.str.len() < 3)]
    # remove label with too less entry
    label_idx_minor = jel_dummy_matrix.sum(axis=0) <= 1
    label_idx_minor.sum()
    jel_dummy_matrix = jel_dummy_matrix.loc[:, ~label_idx_minor]
    jel_labels = jel_dummy_matrix.columns.values

    # further remove papers with no jel now
    mask = jel_dummy_matrix.sum(axis=1) == 0
    jel_dummy_matrix = jel_dummy_matrix.loc[~mask, :]
    df = df.loc[~mask, :]

    stemmer = SnowballStemmer("english")
    abstract = df.abstract.apply(stemming, args=(stemmer,))

    vectorizer = TfidfVectorizer(stop_words='english', min_df=2,
                                 # strip_accents='unicode',
                                 # ngram_range=(1,2), norm='l2',
                                 )
    token_matrix = vectorizer.fit_transform(abstract)

    # y_train, y_test, X_train, X_test = train_test_split(jel_dummy_matrix, token_matrix,
    #                             random_state=46,
    #                             test_size=0.005, shuffle=True)
    y_train, y_test, X_train, X_test = jel_dummy_matrix, None, token_matrix, None
    return vectorizer, y_train, y_test, X_train, X_test, jel_labels


def find_top_predict(classifier, X_individual, jel_labels, top_n=20):
    predict_proba_ = classifier.predict_proba(X_individual)
    predict_proba_ = np.array([a[0][1] for a in predict_proba_])
    jel_rank = np.argsort(predict_proba_)[::-1]
    if top_n <= 1:
        top_n = (predict_proba_ > top_n).sum()
    y_predict_jel = jel_labels[jel_rank][:top_n]
    y_predict_jel_proba = predict_proba_[jel_rank][:top_n]
    return y_predict_jel, y_predict_jel_proba


def custom_metric(classifier, X_test, y_test, jel_labels, top_n=20):
    find_rates = []
    for i in range(len(y_test)):
        X_test_ = X_test[i, :]
        y_test_ = y_test.iloc[i, :]
        y_test_jel = y_test_[y_test_ > 0].index.values

        y_predict_jel, _ = find_top_predict(classifier, X_test_, jel_labels, top_n=top_n)

        base = len(y_test_jel)
        correct_find = len(set(y_predict_jel) & set(y_test_jel))
        find_rate = correct_find/base
        find_rates.append(find_rate)
    avg_find_rate = np.array(find_rates).mean()
    print(f"custom metric top_n={top_n}: {avg_find_rate}")
    return avg_find_rate


def train_model(X_train, y_train):
    lr = LogisticRegression(solver='saga', max_iter=500, C=1,
                            class_weight="balanced", penalty="l1",  # "elasticnet"
                            )
    classifier = MultiOutputClassifier(lr, n_jobs=-1)

    classifier.fit(X_train, y_train)
    return classifier


def main():
    vectorizer, y_train, y_test, X_train, X_test, jel_labels = prepare_data()
    classifier = train_model(X_train, y_train)
    # custom_metric(classifier, X_test, y_test,  jel_labels, top_n=10)
    # custom_metric(classifier, X_test, y_test,  jel_labels, top_n=20)
    # custom_metric(classifier, X_test, y_test,  jel_labels, top_n=25)

    joblib.dump(jel_labels, "../Data/jel_labels.pkl", protocol=4)
    joblib.dump([vectorizer, classifier], "../Data/model.pkl", protocol=4)

    # load


if __name__ == '__main__':
    main()
