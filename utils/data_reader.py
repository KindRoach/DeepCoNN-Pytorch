from typing import Set

import nltk
import pandas
from pandas import DataFrame
from sklearn.model_selection import train_test_split

from utils.log_hepler import logger
from utils.path_helper import ROOT_DIR


def get_all_data() -> DataFrame:
    return pandas.read_json(ROOT_DIR.joinpath("data/reviews.json"), lines=True)


def get_train_dev_test_data() -> (DataFrame, DataFrame, DataFrame):
    all_data = get_all_data()
    train, test = train_test_split(all_data, test_size=0.2, random_state=42)
    train, dev = train_test_split(train, test_size=0.1, random_state=42)
    return train, dev, test


def get_stop_words() -> Set[str]:
    with open(ROOT_DIR.joinpath("data/stopwords.txt")) as f:
        return set(f.read().splitlines())


def get_punctuations() -> Set[str]:
    with open(ROOT_DIR.joinpath("data/punctuations.txt")) as f:
        return set(f.read().splitlines())


STOP_WORDS = get_stop_words()
PUNCTUATIONS = get_punctuations()
LEMMATIZER = nltk.WordNetLemmatizer()


def clean_review(review: str):
    review = review.lower()
    assert "'" not in PUNCTUATIONS
    for p in PUNCTUATIONS:
        review = review.replace(p, " ")
    tokens = review.split()
    tokens = [word for word in tokens if word not in STOP_WORDS]
    tokens = [LEMMATIZER.lemmatize(word) for word in tokens]
    return " ".join(tokens)


if __name__ == "__main__":
    logger.info("reading raw data...")
    df = pandas.read_json(ROOT_DIR.joinpath("data/Digital_Music_5.json"), lines=True)
    df = df[["reviewerID", "asin", "reviewText", "overall"]]
    df.columns = ["userID", "itemID", "review", "overall"]

    logger.info("cleaning review text...")
    df["review"] = df["review"].apply(clean_review)

    df.to_json(ROOT_DIR.joinpath("data/reviews.json"), orient="records", lines=True)
    logger.info("Processed data saved.")
