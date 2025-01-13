import copy
import time
import os

import torch
import numpy as np
from loguru import logger
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_set import DataSet
from utils import metrics


class Trainer(object):
    """
    This class train and evaluate the model
    """
    def __init__(self, model, dataset: DataSet, args):
        self.model = model.to(args.device)
        self.dataset = dataset
        self.behaviors = args.behaviors
        self.topk = args.topk
        self.metrics = args.metrics
        self.learning_rate = args.lr
        self.batch_size = args.batch_size
        self.epochs = args.epochs
        self.model_path = args.model_path
        self.model_name = args.model_name
        self.train_writer = args.train_writer
        self.test_writer = args.test_writer
        self.device = args.device
        self.TIME = args.TIME

        self.optimizer = self.get_optimizer(self.model)

    def get_optimizer(self, model):
        """Returns Adam optimizer"""
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                               lr=self.learning_rate)
        return optimizer

    @logger.catch()
    def train_model(self):
        """Trains the model
        """
        train_dataset_loader = DataLoader(dataset=self.dataset.behavior_dataset(),
                                          batch_size=self.batch_size,
                                          shuffle=True)

        # best_result = np.zeros(len(self.topk) * len(self.metrics))
        best_result = 0
        best_dict = {}
        best_epoch = 0
        best_model = None
        final_test = None
        for epoch in range(self.epochs):

            self.model.train()
            test_metric_dict, validate_metric_dict = self._train_one_epoch(train_dataset_loader, epoch)

            result = validate_metric_dict['hit@20']
            # early stop
            if result - best_result > 0:
                final_test = test_metric_dict
                best_result = result
                best_dict = validate_metric_dict
                best_model = copy.deepcopy(self.model)
                best_epoch = epoch
            if epoch - best_epoch > 20:
                break
        # save the best model
        self.save_model(best_model)
        logger.info(f"training end, best iteration %d, results: %s" %
                    (best_epoch + 1, best_dict.__str__()))

        logger.info(f"final test result is:  %s" % final_test.__str__())

    def _train_one_epoch(self, behavior_dataset_loader, epoch):
        """It trains the model for one epoch using the provided behavior dataset loader. 
        It also computes and returns evaluation metrics for both the test and validation datasets.

        Args:
            behavior_dataset_loader (PyTorch DataLoader)
            epoch (integer)
        """
        start_time = time.time()
        behavior_dataset_iter = (
            tqdm(
                enumerate(behavior_dataset_loader),
                total=len(behavior_dataset_loader),
                desc=f"\033[1;35m Train {epoch + 1:>5}\033[0m"
            )
        )
        total_loss = 0.0
        batch_no = 0
        for batch_index, batch_data in behavior_dataset_iter:
            start = time.time()
            batch_data = batch_data.to(self.device)
            self.optimizer.zero_grad()
            loss = self.model(batch_data)
            loss.backward()
            self.optimizer.step()
            batch_no = batch_index + 1
            total_loss += loss.item()
        total_loss = total_loss / batch_no

        self.train_writer.add_scalar('total Train loss', total_loss, epoch + 1)
        epoch_time = time.time() - start_time
        logger.info('epoch %d %.2fs Train loss is [%.4f] ' % (epoch + 1, epoch_time, total_loss))

        # validate
        start_time = time.time()
        validate_metric_dict = self.evaluate(epoch, self.batch_size, self.dataset.validate_dataset(),
                                             self.dataset.validation_interacts, self.dataset.validation_gt_length,
                                             self.train_writer)
        epoch_time = time.time() - start_time
        logger.info(
            f"validate %d cost time %.2fs, result: %s " % (epoch + 1, epoch_time, validate_metric_dict.__str__()))

        # test
        start_time = time.time()
        test_metric_dict = self.evaluate(epoch, self.batch_size, self.dataset.test_dataset(),
                                         self.dataset.test_interacts, self.dataset.test_gt_length,
                                         self.test_writer)
        epoch_time = time.time() - start_time
        logger.info(
            f"test %d cost time %.2fs, result: %s " % (epoch + 1, epoch_time, test_metric_dict.__str__()))

        return test_metric_dict, validate_metric_dict

    @logger.catch()
    @torch.no_grad()
    def evaluate(self, epoch, batch_size, dataset, gt_interacts, gt_length, writer):
        """It evluates the model's performance on a specified dataset. 
        It calculates and returns evaluation metrics for the given dataset.

        Args:
            epoch (integer)
            batch_size (integer)
            dataset (PyTorch Dataset)
            gt_interacts (dictionary)
            gt_length (numpy array)
            writer (SummaryWriter)
        """
        data_loader = DataLoader(dataset=dataset, batch_size=batch_size)
        self.model.eval()
        iter_data = (
            tqdm(
                enumerate(data_loader),
                total=len(data_loader),
                desc=f"\033[1;35mEvaluate \033[0m"
            )
        )
        topk_list = []
        train_items = self.dataset.train_behavior_dict['buy']
        for batch_index, batch_data in iter_data:
            batch_data = batch_data.to(self.device)
            scores = self.model.full_predict(batch_data)

            for index, user in enumerate(batch_data):
                user_score = scores[index]
                items = train_items.get(str(user.item()), None)
                if items is not None:
                    user_score[items] = -np.inf
                _, topk_idx = torch.topk(user_score, max(self.topk), dim=-1)
                gt_items = gt_interacts[str(user.item())]
                mask = np.isin(topk_idx.to('cpu'), gt_items)
                topk_list.append(mask)

        topk_list = np.array(topk_list)
        metric_dict = self.calculate_result(topk_list, gt_length)
        for key, value in metric_dict.items():
            writer.add_scalar('evaluate ' + key, value, epoch + 1)
        return metric_dict

    def calculate_result(self, topk_list, gt_len):
        """Calculates and returns evaluation metrics based on the provided top-k recommendations and ground truth item lists

        Args:
            topk_list (numpy array)
            gt_len (numpy array)

        Returns:
            metric_dict (dictionary)
        """
        result_list = []
        metrics_dict = metrics()
        for metric in self.metrics:
            metric_fuc = metrics_dict[metric.lower()]
            result = metric_fuc(topk_list, gt_len)
            result_list.append(result)
        result_list = np.stack(result_list, axis=0).mean(axis=1)
        metric_dict = {}
        for topk in self.topk:
            for metric, value in zip(self.metrics, result_list):
                key = '{}@{}'.format(metric, topk)
                metric_dict[key] = np.round(value[topk - 1], 4)

        return metric_dict

    def save_model(self, model):
        """Saves the best-performing model to a file"""
        if not os.path.isdir(self.model_path): 
            os.makedirs(self.model_path) 
        torch.save(model.state_dict(), os.path.join(self.model_path, self.model_name + self.TIME + '.pth'))
