from model.DeepCoNN import DeepCoNNConfig, DeepCoNN
from utils.data_reader import get_train_dev_test_data, get_max_review_length, get_all_data
from utils.train_helper import train_model

all_data = get_all_data()
train_data, dev_data, test_data = get_train_dev_test_data()

config = DeepCoNNConfig(
    num_epochs=100,
    batch_size=128,
    learning_rate=1e-3,
    l2_regularization=1e-6,
    learning_rate_decay=0.99,
    use_cuda=False,
    # max_review_length=get_max_review_length(all_data,0.8),
    max_review_length=1024,
    word_dim=300,
    kernel_widths=[2, 3, 5, 7],
    kernel_deep=100,
    latent_factors=32,
    fm_k=8
)

model = DeepCoNN(config)
train_model(model, train_data)
