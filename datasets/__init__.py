from .Denoising_dataset import DenoisingDataset
import importlib
from datasets.Base_dataset import BaseDataset

__all__=('DenosingDataset')


def find_dataset_using_name(dataset_name):
    dataset_filename = "data." + dataset_name + "_dataset"
    datasetlib = importlib.import_module(dataset_filename)
    dataset = None
    target_dataset_name = dataset_name.replace('_', '')+'dataset'   #寻找子类
    for name, cls in datasetlib.__dict__.items():
        if name.lower() == target_dataset_name.lower() and issubclass(cls, BaseDataset):
            dataset = cls

    if dataset is None:
       raise NotImplementedError("In %s.py, there should be a subclass of BaseDataset with class name that matches %s in lowercase." % (dataset_filename, target_dataset_name))
    else:
        return dataset




def get_option_setter(dataset_name):
    """Return the static method <modify_commandline_options> of the dataset class."""
    dataset_class = find_dataset_using_name(dataset_name)
    return dataset_class.modify_commandline_options


