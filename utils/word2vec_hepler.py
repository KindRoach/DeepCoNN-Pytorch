from typing import Set, List

import numpy as np
import pandas
from gensim.models import KeyedVectors
from gensim.models.keyedvectors import Word2VecKeyedVectors

from utils.log_hepler import logger
from utils.path_helper import ROOT_DIR

PAD_WORD = "<pad>"
WORD_EMBEDDING_SIZE = 300
WORD_VEC: Word2VecKeyedVectors


def review2wid(review: str) -> List[int]:
    wids = []
    for word in review.split():
        if word in _WORD_VEC:
            wid = _WORD_VEC.vocab[word].index
        else:
            wid = _WORD_VEC.vocab[PAD_WORD].index
        wids.append(wid)
    return wids


logger.info("loading word2vec model...")
path = ROOT_DIR.joinpath('data/GoogleNews-vectors-negative300.bin')
_WORD_VEC = KeyedVectors.load_word2vec_format(path, binary=True)
_WORD_VEC.add([PAD_WORD], np.zeros([1, 300]))
logger.info("word2vec model loaded.")

if __name__ == "__main__":
    df = pandas.read_json(ROOT_DIR.joinpath("data/reviews.json"), lines=True)

    unknown_words: Set[str] = set()
    for review in df["review"]:
        for word in review.split():
            if word not in _WORD_VEC:
                unknown_words.add(word)

    logger.warning(f"{len(unknown_words)} unknown words!")
    with open(ROOT_DIR.joinpath("out/UNKs.txt"), "w", encoding="utf-8") as f:
        for word in unknown_words:
            f.write(f"{word}\n")
