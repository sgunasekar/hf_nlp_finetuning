# Adapted from hugging face examples code https://github.com/huggingface/transformers/blob/main/examples/pytorch/text-classification/run_glue.py
from dataclasses import dataclass, field
from lib.hf_datasets_info import valid_task_names, datasets_info
from typing import Optional, Union


@dataclass
class DatasetArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval. 
    Using `HfArgumentParser` we can turn this class into argparse arguments to be able to specify them on the command line.
    """
    
    train_data: Optional[Union[str,list[str]]] = field( 
        default_factory=list,
        metadata={"help": f"string or a list of strings of dataset names for training specified in the format `<task_name>.<split>'. Valid task names are {valid_task_names}. See lib.hf_datasets_info.datasets_info for the valid splits for each task name. "}
    )

    eval_data: Optional[Union[str,list[str]]] = field(
        default_factory=list,
        metadata={"help": f"string or a list of strings of dataset names for validation specified in the format `<task_name>.<split>'. Valid task names are {valid_task_names}. See lib.hf_datasets_info.datasets_info for the valid splits for each task name. "}
    )

    max_train_samples: Optional[int] = field(
        default=None,
        metadata={"help": "For debugging purposes or quicker training, truncate the number of training examples in each training dataset to this value if set."}
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={"help": "For debugging purposes or quicker training, truncate the number of evaluation examples in each eval dataset to this value if set."},
    )

    def _get_task_info(self,s):
        '''Checks if a dataset argument (passed as train_datasets or eval_datasets) is valid. If valid, returns appropriately proceessed task_info for the dataset'''
        dataset_key, split = s.split('.')
        
        assert dataset_key in valid_task_names, f"{dataset_key} is not a valid task name. Please specify the datasets in the format `<task_name>.<split>', where task_names in {valid_task_names}."
        assert split in datasets_info[dataset_key]['splits'], f"{split} is not a valid split for dataset {dataset_key}. Valid splits for {dataset_key} are {datasets_info[dataset_key]['splits']}"

        task_info = datasets_info[dataset_key]
        task_info['load_dataset_kwargs']['split'] = split
        return task_info

    def __post_init__(self):

        if isinstance(self.train_data, str):
            self.train_datasets = [self.train_datasets]
        if isinstance(self.eval_data, str):
            self.eval_datasets = [self.eval_datasets]

        self.data_pipeline = {'train':[], 'eval':[]}

        for ds in self.train_datasets:
            task_info = self._get_task_info(ds)
            self.data_pipeline['train'].append(task_info)

        for ds in self.eval_datasets:
            task_info = self._get_task_info(ds)
            self.data_pipeline['eval'].append(task_info)

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    ignore_mismatched_sizes: bool = field(
        default=False,
        metadata={"help": "Will enable to load a pretrained model whose head dimensions are different."},
    )