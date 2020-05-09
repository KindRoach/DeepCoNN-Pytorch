from dataclasses import dataclass
from typing import List

import torch

from model.BaseModel import BaseModel, BaseConfig


@dataclass
class DeepCoNNConfig(BaseConfig):
    max_review_length: int
    word_dim: int  # the dimension of word embedding
    kernel_widths: List[int]  # the window sizes of convolutional kernel
    kernel_deep: int  # the number of convolutional kernels
    latent_factors: int
    fm_k: int


class ConvMaxLayer(torch.nn.Module):
    def __init__(self, config: DeepCoNNConfig):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        for width in config.kernel_widths:
            self.Convs.append(torch.nn.Conv1d(
                in_channels=config.word_dim,
                out_channels=config.kernel_deep,
                kernel_size=width))

        self.activation = torch.nn.ReLU()
        self.max_pool = torch.nn.MaxPool1d(kernel_size=config.max_review_length)
        self.full_connect = torch.nn.Linear(config.kernel_deep * len(config.kernel_widths), config.latent_factors)

    def forward(self, review):
        outputs = []
        for conv in self.convs:
            out = self.activation(conv(review))
            max_out = self.max_pool(out)
            outputs.append(max_out)

        conv_out = torch.cat(outputs)
        latent = self.full_connect(conv_out)

        return latent


class FMLayer(torch.nn.Module):
    """
    The implementation of Factorization machine.
    Reference: https://www.kaggle.com/gennadylaptev/factorization-machine-implemented-in-pytorch
    """

    def __init__(self, config: DeepCoNNConfig):
        super().__init__()
        self.V = torch.nn.Parameter(torch.randn(config.latent_factors * 2, config.fm_k), requires_grad=True)
        self.lin = torch.nn.Linear(config.latent_factors * 2, 1)

    def forward(self, x):
        s1_square = torch.matmul(x, self.V).pow(2).sum(1, keepdim=True)
        s2 = torch.matmul(x.pow(2), self.V.pow(2)).sum(1, keepdim=True)

        out_inter = 0.5 * (s1_square - s2)
        out_lin = self.lin(x)
        out = out_inter + out_lin

        return out


class DeepCoNN(BaseModel):
    def __init__(self, config: DeepCoNNConfig):
        assert config is not None
        super().__init__(config)
        self.user_layer = ConvMaxLayer(config)
        self.item_layer = ConvMaxLayer(config)
        self.share_layer = FMLayer(config)

    def forward(self, user_review, item_review):
        user_latent = self.user_layer(user_review)
        item_latent = self.item_layer(item_review)
        latent = torch.cat([user_latent, item_latent])
        predict = self.share_layer(latent)
        return predict
