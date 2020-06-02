import math
import time
from itertools import chain
from typing import Dict, List

import torch
from pandas import DataFrame
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from model.BaseModel import BaseModel, BaseConfig
from utils.data_reader import get_review_dict
from utils.log_hepler import logger, add_log_file, remove_log_file
from utils.path_helper import ROOT_DIR
from utils.word2vec_hepler import PAD_WORD_ID


def save_model(model: torch.nn.Module, train_time: time.struct_time):
    path = "model/checkpoints/%s_%s.pt" % (
        model.__class__.__name__, time.strftime("%Y%m%d%H%M%S", train_time)
    )
    path = ROOT_DIR.joinpath(path)
    torch.save(model, path)
    logger.info(f"model saved: {path}")


def load_model(path: str):
    path = ROOT_DIR.joinpath(path)
    # load model to cpu as default.
    model = torch.load(path, map_location=torch.device('cpu'))
    return model


def load_reviews(review: Dict[str, DataFrame], query_id: str, exclude_id: str, max_length) -> List[int]:
    """
    1. Load review from review dict by userID/itemID
    2. Exclude unknown review by itemID/userID.
    3. Pad review text to max_length

    E.g. get all reviews written by user1 except itemA
         when we predict the rating of itemA marked by user1.

        DataFrame for user1:

            | itemID | review |
            | itemA  | 0,1,2  |
            | itemB  | 1,2,3  |
            | itemC  | 2,3,4  |

        query_id: user1
        exclude_id: itemA
        max_length: 8

        output = [1, 2, 3, 2, 3, 4, PAD_WORD_ID, PAD_WORD_ID]
    """

    reviews = review[query_id]
    key = "userID" if "userID" in reviews.columns else "itemID"
    reviews = reviews["review"][reviews[key] != exclude_id].to_list()
    reviews = list(chain.from_iterable(reviews))

    if len(reviews) >= max_length:
        reviews = reviews[:max_length]
    else:
        reviews = reviews + [PAD_WORD_ID] * (max_length - len(reviews))
    return reviews


def get_data_loader(data: DataFrame, config: BaseConfig):
    logger.info("Generating data iter...")
    review_by_user, review_by_item = get_review_dict()

    user_reviews = [torch.LongTensor(load_reviews(review_by_user, userID, itemID, config.max_review_length))
                    for userID, itemID in zip(data["userID"], data["itemID"])]
    user_reviews = torch.stack(user_reviews)

    item_reviews = [torch.LongTensor(load_reviews(review_by_item, itemID, userID, config.max_review_length))
                    for userID, itemID in zip(data["userID"], data["itemID"])]
    item_reviews = torch.stack(item_reviews)

    ratings = torch.Tensor(data["rating"].to_list()).view(-1, 1)

    dataset = torch.utils.data.TensorDataset(user_reviews, item_reviews, ratings)
    pin_memory = config.device not in ["cpu", "CPU"]
    data_iter = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size, shuffle=True, pin_memory=pin_memory)
    logger.info("Data iter loaded.")
    return data_iter


def eval_model(model: BaseModel, data_iter: DataLoader, loss) -> float:
    model.eval()
    model_name = model.__class__.__name__
    config: BaseConfig = model.config
    logger.debug("Evaluating %s..." % model_name)
    with torch.no_grad():
        predicts = []
        ratings = []
        for batch_id, iter_i in enumerate(data_iter):
            user_review, item_review, rating = iter_i
            user_review = user_review.to(config.device)
            item_review = item_review.to(config.device)
            rating = rating.to(config.device)
            predict = model(user_review, item_review)
            predicts.append(predict)
            ratings.append(rating)

        predicts = torch.cat(predicts)
        ratings = torch.cat(ratings)
        return loss(predicts, ratings).item()


def train_model(model: BaseModel, train_data: DataFrame, dev_data: DataFrame):
    model_name = model.__class__.__name__
    train_time = time.localtime()
    add_log_file(logger, "log/%s_%s.log" % (model_name, time.strftime("%Y%m%d%H%M%S", train_time)))
    logger.info("Training %s..." % model_name)

    config: BaseConfig = model.config
    logger.info(config.__dict__)
    model.to(config.device)

    opt = torch.optim.Adam(model.parameters(), config.learning_rate, weight_decay=config.l2_regularization)
    lr_s = lr_scheduler.ExponentialLR(opt, gamma=config.learning_rate_decay)
    loss = torch.nn.MSELoss()

    last_progress = 0.
    last_loss = float("inf")
    train_data_iter = get_data_loader(train_data, config)
    dev_data_iter = get_data_loader(dev_data, config)
    batches_num = math.ceil(len(train_data) / float(config.batch_size))

    while model.current_epoch < config.num_epochs:

        model.train()

        for batch_id, iter_i in enumerate(train_data_iter):
            user_review, item_review, rating = iter_i
            user_review = user_review.to(config.device)
            item_review = item_review.to(config.device)
            rating = rating.to(config.device)
            opt.zero_grad()
            predict = model(user_review, item_review)
            li = loss(predict, rating)
            li.backward()
            opt.step()

            # log progress
            current_batches = model.current_epoch * batches_num + (batch_id + 1.0)
            total_batches = config.num_epochs * batches_num
            progress = current_batches / total_batches
            if progress - last_progress > 0.001:
                logger.debug("epoch %d, batch %d, loss: %f (%.2f%%)" %
                             (model.current_epoch, batch_id, li.item(), 100.0 * progress))
                last_progress = progress

        # complete one epoch
        train_loss = eval_model(model, train_data_iter, loss)
        dev_loss = eval_model(model, dev_data_iter, loss)
        logger.info("Epoch %d complete. Total loss(train/dev)=%f/%f"
                    % (model.current_epoch, train_loss, dev_loss))

        # save best model
        if train_loss < last_loss:
            last_loss = train_loss
            save_model(model, train_time)

        lr_s.step(model.current_epoch)
        model.current_epoch += 1

    logger.info("%s trained!" % model_name)
    remove_log_file(logger)
