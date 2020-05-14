import math
import pickle
import time

import torch
from pandas import DataFrame
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from model.BaseModel import BaseModel, BaseConfig
from utils.log_hepler import logger, add_log_file, remove_log_file
from utils.path_helper import ROOT_DIR


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


def get_data_loader(data: DataFrame, config: BaseConfig):
    review_by_user = pickle.load(open(ROOT_DIR.joinpath("data/user_review_word_idx.p"), "rb"))
    user_reviews = [torch.LongTensor(review_by_user[userID][:config.max_review_length]) for userID in data["userID"]]
    user_reviews = torch.stack(user_reviews)

    review_by_item = pickle.load(open(ROOT_DIR.joinpath("data/item_review_word_idx.p"), "rb"))
    item_reviews = [torch.LongTensor(review_by_item[itemID][:config.max_review_length]) for itemID in data["itemID"]]
    item_reviews = torch.stack(item_reviews)

    ratings = torch.Tensor(data["rating"].to_list()).view(-1, 1)

    dataset = torch.utils.data.TensorDataset(user_reviews, item_reviews, ratings)
    pin_memory = config.device != "cpu"
    data_iter = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size, shuffle=True, pin_memory=pin_memory)
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
