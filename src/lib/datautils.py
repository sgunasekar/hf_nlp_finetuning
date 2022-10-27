from transformers import AutoTokenizer, PreTrainedTokenizerBase
from datasets import load_dataset, Dataset
import logging
from typing import Optional

def get_dataset(task_info: dict, max_size:Optional[int]=None)->Dataset:
    load_dataset_kwargs = task_info["load_dataset_kwargs"]
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
    
def get_tokenized_datasets(config):
    """Get tokenized datasets."""
    data_args = config['data_args']
    tokenizer_name = config['model_args'].tokenizer_name
    tokenizer = AutoTokenizer.from_pretrained(
        config['model_args'].tokenizer_name,
        use_fast=config['model_args'].use_fast_tokenizer
    )

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