import math
import time

import torch
import numpy as np
from pandas import DataFrame
from torch.nn.utils.rnn import pad_sequence
from torch.optim import lr_scheduler

from model.BaseModel import BaseModel, BaseConfig
from utils.log_hepler import logger, add_log_file, remove_log_file
from utils.path_helper import ROOT_DIR
from utils.word2vec_hepler import review2vec


def save_model(model: torch.nn.Module, train_time: time.struct_time):
    path = "model/checkpoints/%s_%s.pt" % (
        model.__class__.__name__, time.strftime("%Y%m%d%H%M%S", train_time)
    )
    path = ROOT_DIR.joinpath(path)
    torch.save(model, path)


def load_model(path: str):
    path = ROOT_DIR.joinpath(path)
    # load model to cpu as default.
    model = torch.load(path, map_location=torch.device('cpu'))
    return model


def get_x_y_from_data(data: DataFrame, review_by_user: dict, review_by_item: dict, max_length: int):
    user_review = [review_by_user[userID] for userID in data["userID"]]
    user_review = [review2vec(r, max_length) for r in user_review]
    user_review.append(torch.zeros([max_length, 300]))
    user_review = pad_sequence(user_review, batch_first=True)[:-1]

    item_review = [review_by_item[itemID] for itemID in data["itemID"]]
    item_review = [review2vec(r, max_length) for r in item_review]
    item_review.append(torch.zeros([max_length, 300]))
    item_review = pad_sequence(item_review, batch_first=True)[:-1]

    rating = torch.Tensor(data["rating"].to_list()).view(-1, 1)

    return user_review, item_review, rating


def eval_model(model: BaseModel, data: DataFrame, loss, review_by_user: dict, review_by_item: dict) -> float:
    model.eval()
    model_name = model.__class__.__name__
    logger.info("Evaluating %s..." % model_name)
    config: BaseConfig = model.config
    with torch.no_grad():
        predicts = []
        ratings = []
        for batch_id, batch in enumerate(np.array_split(data, config.batch_size)):
            user_review, item_review, rating = \
                get_x_y_from_data(batch, review_by_user, review_by_item, config.max_review_length)
            predict = model(user_review, item_review)
            predicts.append(predict)
            ratings.append(rating)

        predicts = torch.cat(predicts)
        ratings = torch.cat(ratings)
        return loss(predicts, ratings).item()


def train_model(model: BaseModel, data: DataFrame):
    model_name = model.__class__.__name__
    train_time = time.localtime()
    add_log_file(logger, "log/%s_%s.log" % (model_name, time.strftime("%Y%m%d%H%M%S", train_time)))
    logger.info("Training %s..." % model_name)

    config: BaseConfig = model.config
    logger.info(config.__dict__)

    opt = torch.optim.Adam(model.parameters(), config.learning_rate, weight_decay=config.l2_regularization)
    lr_s = lr_scheduler.ExponentialLR(opt, gamma=config.learning_rate_decay)
    loss = torch.nn.MSELoss()

    review_by_user = data["review"].groupby(data["userID"]).apply(lambda reviews: " ".join(reviews)).to_dict()
    review_by_item = data["review"].groupby(data["itemID"]).apply(lambda reviews: " ".join(reviews)).to_dict()

    last_progress = 0.
    last_loss = float("inf")

    batches_num = math.ceil(len(data) / float(config.batch_size))
    batches = list(np.array_split(data, batches_num))
    while model.current_epoch < config.num_epochs:

        model.train()

        for batch_id in range(batches_num):

            logger.debug("get_x_y_from_data...")
            user_review, item_review, rating = \
                get_x_y_from_data(batches[batch_id], review_by_user, review_by_item, config.max_review_length)

            logger.debug("forward...")
            predict = model(user_review, item_review)
            li = loss(predict, rating)

            logger.debug("backward...")
            li.backward()
            opt.step()

            # log progress
            current_batches = model.current_epoch * batches_num + (batch_id + 1.0)
            total_batches = config.num_epochs * batches_num
            progress = current_batches / total_batches
            if progress - last_progress > 0.001:
                logger.debug("epoch %d, batch %d, loss: %f (%.1f%%)" %
                             (model.current_epoch, batch_id, li.item(), 100.0 * progress))
                last_progress = progress

        # complete one epoch
        with torch.no_grad():
            total_loss = eval_model(model, data, loss, review_by_user, review_by_item)
            logger.info("Epoch %d complete. Total loss=%f." % (model.current_epoch, total_loss))

            # save best model
            if total_loss < last_loss:
                last_loss = total_loss
                save_model(model, train_time)

        lr_s.step(model.current_epoch)
        model.current_epoch += 1

    logger.info("%s trained!" % model_name)
    remove_log_file(logger)
