import argparse
import random
import time
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from loguru import logger

from data_set import DataSet

from model import MBA

from trainer import Trainer

SEED = 2024
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


if __name__ == '__main__':
    """
    This is a main function for loading hyper-parameters and starting the program
    """
    parser = argparse.ArgumentParser('Set args', add_help=False)

    parser.add_argument('--embedding_size', type=int, default=64, help='')
    parser.add_argument('--reg_weight', type=float, default=0.001, help='')
    parser.add_argument('--layers', type=int, default=1)
    parser.add_argument('--node_dropout', type=float, default=0.75)

    parser.add_argument('--data_name', type=str, default='beibei', help='')
    parser.add_argument('--behaviors', help='', action='append')
    parser.add_argument('--if_load_model', type=bool, default=False, help='')

    parser.add_argument('--topk', type=list, default=[10, 20, 50, 80], help='')
    parser.add_argument('--metrics', type=list, default=['hit', 'ndcg'], help='')
    parser.add_argument('--lr', type=float, default=0.01, help='')
    parser.add_argument('--batch_size', type=int, default=1024, help='')
    parser.add_argument('--epochs', type=str, default=300, help='')
    parser.add_argument('--model_path', type=str, default='./check_point', help='')
    parser.add_argument('--check_point', type=str, default='', help='')
    parser.add_argument('--device', type=str, default='cuda', help='')

    args = parser.parse_args()
    args.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    if args.data_name == 'tmall':
        args.data_path = './data/tmall'
        args.behaviors = ['click','collect','cart','buy']
        args.layers = [1, 1, 1, 1]
        args.model_name = 'tmall'
    elif args.data_name == 'beibei':
        args.data_path = './data/beibei'
        args.behaviors = ['view', 'cart', 'buy']
        args.layers = [1, 1, 1]
        args.model_name = 'beibei'
    elif args.data_name == 'jdata':
        args.data_path = './data/jdata'
        args.behaviors = ['view','collect','cart','buy']
        args.layers = [1, 1, 1, 1]
        args.model_name = 'jdata'
    else:
        raise Exception('data_name cannot be None')

    TIME = time.strftime("%Y-%m-%d %H_%M_%S", time.localtime())
    args.TIME = TIME
    logfile = '{}_{}'.format(args.model_name, TIME)
    args.train_writer = SummaryWriter('./log/train/' + logfile)
    args.test_writer = SummaryWriter('./log/test/' + logfile)

    dataset = DataSet(args)
    
    model = MBA(args, dataset)
    
    if args.if_load_model is True:
        metric_dict = Trainer(model, dataset, args).evaluate(0, args.batch_size, dataset.test_dataset(), dataset.test_interacts, dataset.test_gt_length, args.test_writer)
        print(metric_dict)
        exit()
    
    start = time.time()
    logger.add('./log/{}/{}.log'.format(args.model_name, logfile), encoding='utf-8')
    logger.info(args.__str__())
    logger.info(model)
    trainer = Trainer(model, dataset, args)
    trainer.train_model()
    logger.info('train end total cost time: {}'.format(time.time() - start))
