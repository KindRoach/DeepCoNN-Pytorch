import pandas
import torch

from model.DeepCoNN import DeepCoNNConfig, DeepCoNN
from utils.data_reader import get_train_dev_test_data
from utils.train_helper import train_model
from utils.word2vec_hepler import WORD_EMBEDDING_SIZE, WORD_VEC

train_data, dev_data, test_data = get_train_dev_test_data()
know_data = pandas.concat([train_data, dev_data])

config = DeepCoNNConfig(
    num_epochs=40,
    batch_size=64,
    learning_rate=1e-3,
    l2_regularization=1e-6,
    learning_rate_decay=0.99,
    device="cuda:0",
    max_review_length=2048,
    word_dim=WORD_EMBEDDING_SIZE,
    kernel_widths=[2, 3, 5, 7],
    kernel_deep=100,
    latent_factors=32,
    fm_k=8
)

model = DeepCoNN(config, torch.Tensor(WORD_VEC.vectors))
train_model(model, train_data)