from configs import hf_datasets_info
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from datasets import load_dataset, Dataset
import logging
from typing import Optional
import yaml 
import os
from configs.constants import CUSTOM_FUNC_KWARGS_FILE
ADDITIONAL_KWARGS = yaml.full_load(open(CUSTOM_FUNC_KWARGS_FILE,"r"))

def get_dataset(task_info: dict, max_size:Optional[int]=None)->Dataset:
    load_dataset_kwargs = ADDITIONAL_KWARGS.get(load_dataset.__qualname__, {})
    load_dataset_kwargs.update(task_info["load_dataset_kwargs"])

    dataset = load_dataset(**load_dataset_kwargs)
    # subsampling
    if max_size is not None and max_size < len(dataset):
        dataset = dataset.shuffle().select(range(max_size))
    logging.info(f"Loaded dataset: {str(dataset)}")
    return dataset

def tokenize_dataset(dataset:Dataset, task_info: dict, tokenizer:PreTrainedTokenizerBase)->Dataset:
    """Tokenize a single dataset with a tokenizer."""
    def tokenize_function(examples):
        sentence_keys = task_info["sentence_keys"]
        args = ((examples[key] for key in sentence_keys))
        return tokenizer(*args, padding=True, truncation=True)

    logging.info(f"Tokenizing dataset: {str(dataset)}")
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    if task_info['label_key'] is not None:
        tokenized_dataset = tokenized_dataset.rename_column(task_info['label_key'], "labels")
    if task_info['id_key'] is not None:
        tokenized_dataset = tokenized_dataset.rename_column(task_info['id_key'], "idx")
    
    tokenized_dataset.set_format(
        "torch", columns=["input_ids", "attention_mask", "labels", "idx"])
    
    logging.info(f"\t tokenized dataset: {str(tokenized_dataset)}")
    return tokenized_dataset
    
def get_tokenized_datasets(data_args, model_args):
    """Get tokenized datasets."""
    tokenizer_name = model_args.tokenizer_model_name
    
    tokenizer_kwargs = ADDITIONAL_KWARGS.get(AutoTokenizer.from_pretrained.__qualname__)
    tokenizer_kwargs.update(dict(
        pretrained_model_name_or_path = model_args.tokenizer_model_name,
        use_fast=model_args.use_fast_tokenizer
    ))

    tokenizer = AutoTokenizer.from_pretrained(**tokenizer_kwargs)

    train_task_info_list = data_args.data_pipeline["train"]
    eval_task_info_list = data_args.data_pipeline["eval"]
    # Get raw datasets
    train_datasets = [
        get_dataset(t, data_args.max_train_samples) for t in train_task_info_list
    ]
    eval_datasets = [
        get_dataset(t, data_args.max_train_samples) for t in eval_task_info_list
    ]
    # Tokenize datasets
    tokenized_train_datasets = [
        tokenize_dataset(ds, ti, tokenizer) for ds, ti in zip(train_datasets, train_task_info_list)
    ]
    tokenized_eval_datasets = [
        tokenize_dataset(ds, ti, tokenizer) for ds, ti in zip(eval_datasets, eval_task_info_list)
    ]
    
    return tokenized_train_datasets, tokenized_eval_datasets