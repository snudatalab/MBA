import numpy as np
import os
import random
import json
import torch

from torch.utils.data import Dataset

SEED = 2024
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


class TestData(Dataset):
    """
    This class represents dataloader for test dataset
    """
    def __init__(self, user_count, item_count, samples=None):
        self.user_count = user_count
        self.item_count = item_count
        self.samples = samples

    def __getitem__(self, idx):
        """ Returns the item corresponding to the given index idx from the dataset"""
        return int(self.samples[idx])

    def __len__(self):
        """ Returns the total length of the dataset"""
        return len(self.samples)


class BehaviorData(Dataset):
    """
    This class generates positive and negative (samples) pairs.
    Positive items have a latter behavior than negative items.
    Positive items are sampled by non-uniform distribution of behaviors which follows the the ranking of behaviors.
    (For three behaviors, w_1=1, w_2=2, w_3=3.)
    Given sampled positive items, negative items are sampled by uniform distribution of behaviors.
    """
    def __init__(self, user_count, item_count, behavior_dict=None, behaviors=None):
        self.user_count = user_count
        self.item_count = item_count
        self.behavior_dict = behavior_dict
        self.behaviors = behaviors

    def __getitem__(self, idx):
        """
        Generates and returns data corresponding to the given index idx. 
        This data consists of positive and negative samples (item pairs).
        """
        total = []
        
        pos_b_i = random.choices([i for i in range(len(self.behaviors))], cum_weights=[i+1 for i in range(len(self.behaviors))])[0]
        items = self.behavior_dict[self.behaviors[pos_b_i]].get(str(idx + 1), None)
        for _ in self.behaviors:
            if items is None:
                signal = [0, 0, 0]
            else:
                pos = random.sample(items, 1)[0]
                neg = random.randint(1, self.item_count)
                pos_li = [item for i, item in enumerate(self.behaviors, start=pos_b_i) if i <= len(self.behaviors)]
                while np.isin(neg, pos_li):
                    neg = random.randint(1, self.item_count)
                signal = [idx + 1, pos, neg]
            total.append(signal)
        return np.array(total)

    def __len__(self):
        """Returns the total length of the dataset"""
        return self.user_count


class DataSet():
    """
    This class represents dataloader for all behaviors
    """
    def __init__(self, args):
        """Initializes a dataloader for all behaviors based on provided arguments"""
        self.behaviors = args.behaviors
        self.path = args.data_path

        self.__get_count()
        self.__get_behavior_items()
        self.__get_validation_dict()
        self.__get_test_dict()
        self.__get_sparse_interact_dict()

        self.validation_gt_length = np.array([len(x) for _, x in self.validation_interacts.items()])
        self.test_gt_length = np.array([len(x) for _, x in self.test_interacts.items()])

    def __get_count(self):
        """Counts the number of users and items in the dataset
        """
        with open(os.path.join(self.path, 'count.txt'), encoding='utf-8') as f:
            count = json.load(f)
            self.user_count = count['user']
            self.item_count = count['item']

    def __get_behavior_items(self):
        """
        Loads the list of items corresponding to the user under each behavior
        """
        self.train_behavior_dict = {}
        all_interaction = {}
        for behavior in self.behaviors:
            with open(os.path.join(self.path, behavior + '_dict.txt'), encoding='utf-8') as f:
                b_dict = json.load(f)
                self.train_behavior_dict[behavior] = b_dict
                for k, v in b_dict.items():
                    if all_interaction.get(k, None) is None:
                        all_interaction[k] = v
                    else:
                        all_interaction[k].extend(v)
        for k, v in all_interaction.items():
            all_interaction[k] = sorted(list(set(v)))
        self.train_behavior_dict['all'] = all_interaction
        

    def __get_test_dict(self):
        """
        Loads the list of items that the user has interacted with in the test set
        """
        with open(os.path.join(self.path, 'test_dict.txt'), encoding='utf-8') as f:
            b_dict = json.load(f)
            self.test_interacts = b_dict

    def __get_validation_dict(self):
        """
        Loads the list of items that the user has interacted with in the validation set
        """
        with open(os.path.join(self.path, 'validation_dict.txt'), encoding='utf-8') as f:
            b_dict = json.load(f)
            self.validation_interacts = b_dict

    def __get_sparse_interact_dict(self):
        """
        Loads graphs (edge indices) for different behaviors
        """
        self.edge_index = {}
        self.user_behaviour_degree = []
        all_row = []
        all_col = []
        for behavior in self.behaviors:
            with open(os.path.join(self.path, behavior + '.txt'), encoding='utf-8') as f:
                data = f.readlines()
                row = []
                col = []
                for line in data:
                    line = line.strip('\n').strip().split()
                    row.append(int(line[0]))
                    col.append(int(line[1]))
                indices = np.vstack((row, col))
                indices = torch.LongTensor(indices)

                values = torch.ones(len(row), dtype=torch.float32)
                self.user_behaviour_degree.append(torch.sparse.FloatTensor(indices,
                                                                           values,
                                                                           [self.user_count + 1, self.item_count + 1])
                                                  .to_dense().sum(dim=1).view(-1, 1))
                col = [x + self.user_count + 1 for x in col]
                row, col = [row, col], [col, row]
                row = torch.LongTensor(row).view(-1)
                all_row.append(row)
                col = torch.LongTensor(col).view(-1)
                all_col.append(col)
                edge_index = torch.stack([row, col])
                self.edge_index[behavior] = edge_index
        self.user_behaviour_degree = torch.cat(self.user_behaviour_degree, dim=1)
        all_row = torch.cat(all_row, dim=-1)
        all_col = torch.cat(all_col, dim=-1)
        self.all_edge_index = torch.stack([all_row, all_col])

    def behavior_dataset(self):
        """Loads graphs (edge indices) for different behaviors"""
        return BehaviorData(self.user_count, self.item_count, self.train_behavior_dict, self.behaviors)

    def validate_dataset(self):
        """Returns a validation dataset"""
        return TestData(self.user_count, self.item_count, samples=list(self.validation_interacts.keys()))

    def test_dataset(self):
        """Returns a test dataset"""
        return TestData(self.user_count, self.item_count, samples=list(self.test_interacts.keys()))
