from typing import Set, List

import numpy as np
import pandas
import torch
from gensim.models import KeyedVectors
from gensim.models.keyedvectors import Word2VecKeyedVectors

from utils.log_hepler import logger
from utils.path_helper import ROOT_DIR

PAD_WORD = "<pad>"
WORD_EMBEDDING_SIZE = 300


def review2wid(review: str, word_vec: Word2VecKeyedVectors) -> List[int]:
    wids = []
    for word in review.split():
        if word in word_vec:
            wid = word_vec.vocab[word].index
        else:
            wid = word_vec.vocab[PAD_WORD].index
        wids.append(wid)
    return wids


def get_word_vec(path='data/GoogleNews-vectors-negative300.bin'):
    logger.info("loading word2vec model...")
    path = ROOT_DIR.joinpath(path)
    word_vec = KeyedVectors.load_word2vec_format(path, binary=True)
    word_vec.add([PAD_WORD], np.zeros([1, 300]))
    logger.info("word2vec model loaded.")
    return word_vec


def save_embedding_weights(word_vec, out_path="data/embedding_weight.pt"):
    weight = torch.Tensor(word_vec.vectors)
    torch.save(weight, ROOT_DIR.joinpath(out_path))
    logger.info("Word embedding weight saved.")


def load_embedding_weights(path="data/embedding_weight.pt"):
    return torch.load(path)


if __name__ == "__main__":
    df = pandas.read_json(ROOT_DIR.joinpath("data/reviews.json"), lines=True)
    word_vec = get_word_vec()
    save_embedding_weights(word_vec)
    unknown_words: Set[str] = set()
    for review in df["review"]:
        for word in review.split():
            if word not in word_vec:
                unknown_words.add(word)

    logger.warning(f"{len(unknown_words)} unknown words!")
    with open(ROOT_DIR.joinpath("out/UNKs.txt"), "w", encoding="utf-8") as f:
        for word in unknown_words:
            f.write(f"{word}\n")
