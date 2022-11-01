# Adapted from hugging face examples code https://github.com/huggingface/transformers/blob/main/examples/pytorch/text-classification/run_glue.py
import logging
import sys
from dataclasses import dataclass, field
from types import NoneType
from typing import Optional, Union

import datasets.features
import yaml
from transformers import HfArgumentParser

from .hf_datasets_info import datasets_info, valid_task_names

_logger = logging.getLogger(__name__)

default_args_factory = {
    'data_args': {
        'train_datasets': lambda : ['mnli.train'],
        'eval_datasets': lambda : ['mnli.validation_matched', 'mnli.validation_mismatched', 'ax.test', 'hans.validation'],
        'max_train_samples': lambda : None, # No max sample size
        'max_eval_samples': lambda : None, # No Max sample size
    },
    'model_args': {
        'model_path': lambda : None, # model downloaded from huggingface.co using model_name as id
        'tokenizer_model_name': lambda : None, # same as model_name
        'cache_dir': lambda : None, # default ~/.cache/huggingface/
        'use_fast_tokenizer': lambda : True, 
        'model_revision': lambda : 'main',
    },
    'io_args': {
        'cache_dir': lambda : None,
        'use_auth_token': lambda : False,
        'is_shared_file_system': lambda : False,
    }
}

@dataclass
class IOArguments:
    """
    Arguments pertaining to IO. The following arguments overwrite the options to TrainingArgs
    """
    
    cache_dir: Union[str,NoneType] = field( 
        default_factory=default_args_factory['io_args']['cache_dir'],
        metadata={"help": f"Directory to cache files in. Defaults to ~/.cache/huggingface/datasets or ~/.cache/huggingface/transformers"}
    )

    use_auth_token: Union[str,NoneType] = field( 
        default_factory=default_args_factory['io_args']['use_auth_token'],
        metadata={"help": "Will use the token generated when running `huggingface-cli login` (necessary to use this script with private models)."}
    )

    is_shared_file_system: bool = field(
        default_factory=default_args_factory['io_args']['is_shared_file_system'],
        metadata={"help": "Used only in multi-node distributed training: flag to specify if the different nodes have shared file system. Used for e.g., in caching tokernizers where default behavior is to tokenize once on each node; but setting this to True would invoke tokenizer only on node-0."}
    )


@dataclass
class DatasetArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval. 
    Using `HfArgumentParser` we can turn this class into argparse arguments to be able to specify them on the command line.
    Additional load_dataset arguments can be set in configs.additional_func_kwargs.yaml
    """
    
    train_datasets: Optional[list[str]] = field( 
        default_factory=default_args_factory['data_args']['train_datasets'],
        metadata={"help": f"String or a list of strings of dataset names for training specified in the format `<task_name>.<split>'. Valid task names are {valid_task_names}. See lib.hf_datasets_info.datasets_info for the valid splits for each task name. "}
    )

    eval_datasets: Optional[list[str]] = field(
        default_factory=default_args_factory['data_args']['eval_datasets'],
        metadata={"help": f"String or a list of strings of dataset names for validation specified in the format `<task_name>.<split>'. Valid task names are {valid_task_names}. See lib.hf_datasets_info.datasets_info for the valid splits for each task name. "}
    )

    max_train_samples: Optional[int] = field(
        default_factory=default_args_factory['data_args']['max_train_samples'],
        metadata={"help": "For debugging purposes or quicker training, truncate the number of training examples in each training dataset to this value if set."}
    )
    max_eval_samples: Optional[int] = field(
        default_factory=default_args_factory['data_args']['max_eval_samples'],
        metadata={"help": "For debugging purposes or quicker evaluation, truncate the number of evaluation examples in each eval dataset to this value if set."},
    )

    data_pipeline: dict = field(
        init=False,
        default_factory=dict,
        metadata={"help": "Will be automatically generated after processing command line arguments."},
    )

    def _get_task_info(self,s):
        """
        Checks if a dataset argument (passed as train_datasets or eval_datasets) is valid. If valid, returns appropriately proceessed task_info for the dataset
        """
        dataset_key, split = s.split('.')
        
        assert dataset_key in valid_task_names, f"{dataset_key} is not a valid task name. Please specify the datasets in the format `<task_name>.<split>', where task_names in {valid_task_names}."

        assert split in datasets_info[dataset_key]['splits'], f"{split} is not a valid split for dataset {dataset_key}. Valid splits for {dataset_key} are {datasets_info[dataset_key]['splits']}"

        task_info = datasets_info[dataset_key]
        task_info['load_dataset_kwargs']['split'] = split
        return task_info

    def _check_train_datasets_compatibility(self,train_task_info_list):
        """
        Checks if the datasets passed as train_datasets are compatible to be merged. 
        """
        features_list = [task_info['features'] for task_info in train_task_info_list]
        try:
            datasets.features.features._check_if_features_can_be_aligned(features_list)
        except Exception as e:
            _logger.error(f"The features of the train datasets {self.train_datasets} are not aligned.")
            raise e

    def __post_init__(self):
        
        """
        Creates dataset pipeline
        """
        if isinstance(self.train_datasets, str):
            self.train_datasets = [self.train_datasets]
        if isinstance(self.eval_datasets, str):
            self.eval_datasets = [self.eval_datasets]

        self.data_pipeline = {'train':[], 'eval':[]}

        for ds in self.train_datasets:
            task_info = self._get_task_info(ds)
            self.data_pipeline['train'].append(task_info)
        self._check_train_datasets_compatibility(self.data_pipeline['train'])

        for ds in self.eval_datasets:
            task_info = self._get_task_info(ds)
            self.data_pipeline['eval'].append(task_info)

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model and tokenizer we are going to fine-tune from. Additional tokenizer and model arguments can be set in configs.additional_func_kwargs.yaml
    """

    model_name: str = field(
        metadata={"help": "Model name or identifier. If model_path is specified, this is just as proxy name and can be any string. If model_path is *NOT* specified, then model_name should be a valid *model id* from huggingface.co/models that can be used with AutoModel."}
    )
    model_path: Optional[str] = field(
        default_factory=default_args_factory['model_args']['model_path'],
        metadata={"help": "A path to a directory containing model weights saved using save_pretrained(). If unspecified, model_name is used as the *model id* to load from huggingface.co/models."}
    )
    tokenizer_model_path: Optional[str] = field(
        default_factory=default_args_factory['model_args']['tokenizer_model_name'],
        metadata={"help": "The *model id* of a predefined tokenizer hosted inside a model repo on huggingface.co. If differnet from model_path (which defatults to model_name)"}
    )
    use_fast_tokenizer: bool = field(
        default_factory=default_args_factory['model_args']['use_fast_tokenizer'],
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str =  field(
        default_factory=default_args_factory['model_args']['model_revision'],
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )

    def __post_init__(self):
        if not self.model_path:
            self.model_path = self.model_name
        if not self.tokenizer_model_path:
            self.tokenizer_model_path = self.model_path

class Custom_HfArgumentParser(HfArgumentParser):
    def parse_yaml_first_then_args_into_dataclasses(
        self, args=None, return_remaining_strings=False, yaml_filename=None
    ):
        """
        Parse command-line args into instances of the specified dataclass types.
        This relies on argparse's `ArgumentParser.parse_known_args`. See the doc at:
        docs.python.org/3.7/library/argparse.html#argparse.ArgumentParser.parse_args
        Args:
            args:
                List of strings to parse. The default is taken from sys.argv. (same as argparse.ArgumentParser)
            return_remaining_strings:
                If true, also return a list of remaining argument strings.
            yaml_filename:
                If not None, will first load arguments from yaml file then update with commanline arguments.
        Returns:
            Tuple consisting of:
                - the dataclass instances in the same order as they were passed to the initializer.abspath
                - if applicable, an additional namespace for more (non-dataclass backed) arguments added to the parser
                  after initialization.
                - The potential list of remaining argument strings. (same as argparse.ArgumentParser.parse_known_args)
        """
        if args is  None and len(sys.argv):
            args = sys.argv[1:]
        if yaml_filename is not None:
            args_dict = yaml.full_load(open(yaml_filename,"r"))
        yaml_dict_str = [f'--{k}={v}' for (k,v) in args_dict.items() if v is not None]
        args = yaml_dict_str + args
        return self.parse_args_into_dataclasses(args=args)
