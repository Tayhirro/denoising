
import torch.utils.data as data
from abc import ABC, abstractmethod


class BaseDataset(ABC,data.Dataset):


    def __init__(self, opt):
        self.opt = opt
        self.root = opt.dataset_root #mia


    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, index):
        pass

    @abstractmethod
    def name(self):
        pass

    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser