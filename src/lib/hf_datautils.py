import logging
from typing import Optional

import evaluate
import numpy as np
import yaml
from configs.constants import CUSTOM_FUNC_KWARGS_FILE
from datasets import Dataset, concatenate_datasets, load_dataset
from transformers import AutoTokenizer, EvalPrediction, PreTrainedTokenizerBase

ADDITIONAL_KWARGS = yaml.full_load(open(CUSTOM_FUNC_KWARGS_FILE,"r"))

def get_dataset(task_info: dict, max_size:Optional[int]=None, io_args:Optional[dict]=None)->Dataset:

    load_dataset_kwargs = ADDITIONAL_KWARGS.get(load_dataset.__qualname__, {})
    if io_args and io_args.cache_dir is not None:
        load_dataset_kwargs['cache_dir'] = io_args.cache_dir
    if io_args.use_auth_token:
        load_dataset_kwargs['use_auth_token'] = io_args.use_auth_token
    load_dataset_kwargs.update(task_info['load_dataset_kwargs'])
    
    dataset = load_dataset(**load_dataset_kwargs)
    # subsampling
    if max_size is not None and max_size < len(dataset):
        dataset = dataset.shuffle().select(range(max_size))
    logging.info(f"Loaded dataset: {str(dataset)}")
    return dataset

def tokenize_dataset(dataset:Dataset, task_info: dict, tokenizer:PreTrainedTokenizerBase)->Dataset:
    """Tokenize a single dataset with a tokenizer."""
    def tokenize_function(examples):
        sentence_keys = task_info['sentence_keys']
        args = ((examples[key] for key in sentence_keys[:2])) 
        # TODO: tokenizer  accepts at most two sentences as 'text=args[0]' and 'text_pair=args[1]'. For more columns, preprocess separately. 
        # TODO: add suppport for explicitly additn padding and max_sequence_length. Currently, no max_sequence_length and padding is dynamically performed at run-time        
        # FIXME: handle complicated label2id's -- not an issue for glue 
        return tokenizer(*args, padding=False, truncation=True)

    logging.info(f"Tokenizing dataset: {str(dataset)}")
    # TODO: add support for overwrite cache
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    if task_info['label_key'] is not None and task_info['label_key']!='lables':
        tokenized_dataset = tokenized_dataset.rename_column(task_info['label_key'], "labels")
    if task_info['id_key'] is not None and task_info['id_key']!="idx":
        tokenized_dataset = tokenized_dataset.rename_column(task_info['id_key'], "idx")
    
    tokenized_dataset.set_format(
        "torch", columns=['input_ids', 'attention_mask', 'labels', 'idx'])
    
    logging.info(f"\t tokenized dataset: {str(tokenized_dataset)}")
    return tokenized_dataset
    
def get_tokenized_datasets(data_args, model_args, io_args):
    """Get tokenized datasets."""

    train_task_info_list = data_args.data_pipeline['train']
    eval_task_list = data_args.eval_datasets
    eval_task_info_list = data_args.data_pipeline['eval']

    # Get raw datasets: train datasets will be concatenated as one dataset so they are processed as list while eval datasets are processed as dict
    train_datasets = [
        get_dataset(task_info, data_args.max_train_samples, io_args=io_args) for task_info in train_task_info_list
    ]
    eval_datasets = {
        task_name: get_dataset(task_info, data_args.max_eval_samples, io_args=io_args) for task_name, task_info in zip(eval_task_list,eval_task_info_list)
    }
    # Tokenize datasets
    tokenizer_kwargs = ADDITIONAL_KWARGS.get(AutoTokenizer.from_pretrained.__qualname__)
    tokenizer_kwargs.update(dict(
        pretrained_model_name_or_path = model_args.tokenizer_model_path,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        cache_dir=io_args.cache_dir,
        use_auth_token=True if io_args.use_auth_token else None,
    ))

    tokenizer = AutoTokenizer.from_pretrained(**tokenizer_kwargs)
    tokenized_train_datasets = [
        tokenize_dataset(dset, task_info, tokenizer) for dset, task_info in zip(train_datasets, train_task_info_list)
    ]
    tokenized_train_dataset = concatenate_datasets(tokenized_train_datasets)

    tokenized_eval_datasets = {
        task_name: tokenize_dataset(eval_datasets[task_name], task_info, tokenizer) for task_name, task_info in zip(eval_task_list,eval_task_info_list)
    }

    # You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    # TODO, FIXME: currently using metrics of the first dataset in train_task_list. Add compatibility check 
    metric = evaluate.load(*train_task_info_list[0]['evaluate_load_args']) 
    task_type = train_task_info_list[0]['task_type']

    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.squeeze(preds) if task_type=='regression' else np.argmax(preds, axis=1)
        result = metric.compute(predictions=preds, references=p.label_ids)
        if len(result) > 1:
            result['combined_score'] = np.mean(list(result.values())).item()
        return result

    model_args.num_labels = train_task_info_list[0]['num_outputs']
    return tokenized_train_dataset, tokenized_eval_datasets, tokenizer, compute_metrics
