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


def get_max_review_length(data: DataFrame, percentile: float = 0.85) -> int:
    """
    We set the max review length to 85% percentile of all data.
    """
    review_lengths = data["review"] \
        .groupby(data["userID"]) \
        .apply(lambda words: len(" ".join(words).split()))
    max_length_user = int(review_lengths.quantile(percentile, interpolation="lower"))

    review_lengths = data["review"] \
        .groupby(data["itemID"]) \
        .apply(lambda words: len(" ".join(words).split()))
    max_length_item = int(review_lengths.quantile(percentile, interpolation="lower"))

    return max(max_length_item, max_length_user)


_STOP_WORDS = get_stop_words()
_PUNCTUATIONS = get_punctuations()
_LEMMATIZER = nltk.WordNetLemmatizer()


def clean_review(review: str):
    review = review.lower()
    assert "'" not in _PUNCTUATIONS
    for p in _PUNCTUATIONS:
        review = review.replace(p, " ")
    tokens = review.split()
    tokens = [word for word in tokens if word not in _STOP_WORDS]
    tokens = [_LEMMATIZER.lemmatize(word) for word in tokens]
    return " ".join(tokens)


if __name__ == "__main__":
    logger.info("reading raw data...")
    df = pandas.read_json(ROOT_DIR.joinpath("data/Digital_Music_5.json"), lines=True)
    df = df[["reviewerID", "asin", "reviewText", "overall"]]
    df.columns = ["userID", "itemID", "review", "rating"]

    logger.info("cleaning review text...")
    df["review"] = df["review"].apply(clean_review)

    df.to_json(ROOT_DIR.joinpath("data/reviews.json"), orient="records", lines=True)
    logger.info("Processed data saved.")

    logger.info(f"max review length is {get_max_review_length(get_all_data())}")
