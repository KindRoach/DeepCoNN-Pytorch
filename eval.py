import torch

from utils.data_reader import get_train_dev_test_data
from utils.train_helper import load_model, eval_model, get_data_loader

train_data, dev_data, test_data = get_train_dev_test_data()
model = load_model("model/checkpoints/DeepCoNN_20200513115255.pt")
model.to(model.config.device)
loss = torch.nn.MSELoss()
data_iter = get_data_loader(test_data, model.config)
print(eval_model(model, data_iter, loss))
