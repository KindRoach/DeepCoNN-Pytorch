import pandas
import torch

from model.DeepCoNN import DeepCoNNConfig, DeepCoNN
from utils.data_reader import get_train_dev_test_data
from utils.train_helper import train_model
from utils.word2vec_hepler import WORD_EMBEDDING_SIZE, load_embedding_weights

train_data, dev_data, test_data = get_train_dev_test_data()
know_data = pandas.concat([train_data, dev_data])

config = DeepCoNNConfig(
    num_epochs=50,
    batch_size=16,
    learning_rate=1e-3,
    l2_regularization=1e-2,
    learning_rate_decay=0.95,
    device="cuda:0" if torch.cuda.is_available() else "cpu",
    max_review_length=2048,  # Maker sure this value is smaller than max_length in data_reader.py
    word_dim=WORD_EMBEDDING_SIZE,
    kernel_widths=[2, 3, 5, 7],
    kernel_deep=100,
    latent_factors=50,
    fm_k=8
)

model = DeepCoNN(config, load_embedding_weights())
train_model(model, train_data, dev_data)
