from typing import Set

import torch
from gensim.models import KeyedVectors
from gensim.models.keyedvectors import Word2VecKeyedVectors

from utils.data_reader import get_all_data
from utils.log_hepler import logger
from utils.path_helper import ROOT_DIR

_WORD_EMBEDDING_SIZE = 300
_WORD_VEC: Word2VecKeyedVectors


def review2vec(review: str, max_length: int) -> torch.Tensor:
    vecs = []
    for i, word in enumerate(review.split()):
        if i >= max_length:
            break
        if word in _WORD_VEC:
            vecs.append(torch.Tensor(_WORD_VEC[word]))
        else:
            vecs.append(torch.zeros(_WORD_EMBEDDING_SIZE))
    if vecs:
        return torch.stack(vecs)
    else:
        return torch.empty(0, _WORD_EMBEDDING_SIZE)


logger.info("loading word2vec model...")
path = ROOT_DIR.joinpath('data/GoogleNews-vectors-negative300.bin')
_WORD_VEC = KeyedVectors.load_word2vec_format(path, binary=True)
# _WORD_VEC = dict()
logger.info("word2vec model loaded.")

if __name__ == "__main__":
    df = get_all_data()

    unknown_words: Set[str] = set()
    for review in df["review"]:
        for word in review.split():
            if word not in _WORD_VEC:
                unknown_words.add(word)

    logger.warning(f"{len(unknown_words)} unknown words!")
    with open(ROOT_DIR.joinpath("out/UNKs.txt"), "w", encoding="utf-8") as f:
        for word in unknown_words:
            f.write(f"{word}\n")
