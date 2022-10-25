## Running this script generates the lib/hf_datasets_info.py file

import json
import os 
import datasets
import sys
from typing import Union
import logging
from datasets import Value

sys.path.insert(0, "..") 

## This scripts creates and updates the ../lib/hf_datasets_info.py file
## Usages: 
# `python build_datasets_info.py`
# `python build_datasets_info.py task1 task2`

## Generated file contains the following:
# valid_task_names: as a list of valid finetuning task names that can be processed by the code
# dataset_info: a dictionary mapping finetuning tasks to the relevant subset of info from DatasetInfo object. 
# For each task/dataset, the dataset_info stores: args for load_dataset(), names of splits, label information, and sentence keys for the input features. 
## Ref: Super GLUE tasks: ['boolq', 'cb', 'copa', 'multirc', 'record', 'rte', 'wic', 'wsc', 'wsc.fixed', 'axb', 'axg']
## Ref: GLUE taks: ['cola', 'sst2', 'mrpc', 'qqp', 'stsb', 'mnli', 'mnli_mismatched', 'mnli_matched', 'qnli', 'rte', 'wnli', 'ax']

default_builders = ["super_glue", "glue", "imdb", "hans", "squad", "squad_v2"]

def create_datasets_info_dict(builder_names):

    datasets_info = {}

    for builder in builder_names:
        info = datasets.get_dataset_infos(builder)
        tasks = list(info.keys())
        # If the builder has only one task, use the builder_name as the key in datasets_info, else use the task name (config_name) as the key
        if len(tasks)==1: 
            TaskInfo = info[tasks[0]]
            datasets_info[builder] = get_task_info_dict(TaskInfo)
        else:
            for task, TaskInfo in info.items():
                datasets_info[task] = get_task_info_dict(TaskInfo)

    return datasets_info


# Extracts relevant info for train/val from a single DatasetInfo object
def get_task_info_dict(TaskInfo: datasets.DatasetInfo) -> dict:
    features = TaskInfo.features
    if "label" in features.keys():
        label = features.pop("label")
    elif "answer" in features.keys():
        label = features.pop("answer")
    else:
        label = None

    if "idx" in features.keys():
        features.pop("idx")
    if "id" in features.keys():
        features.pop("id")
    sentence_keys = list(features.keys())

    splits = list(TaskInfo.splits.keys())
    splits_size = []
    for split in splits:
        splits_size.append(TaskInfo.splits[split].num_examples)

    task_info = dict(
        load_dataset_args=(TaskInfo.builder_name,TaskInfo.config_name),
        sentence_keys=sentence_keys,
        label=label,
        splits=splits,
        splits_num_examples=splits_size
    )

    return task_info


def dump_datasets_info(datasets_info, overwrite = True):

    file_path = os.path.join("..", "lib", "hf_datasets_info.py") 
    if overwrite or not(os.path.exists(file_path)):
        logging.info(f"Creating/overwriting {file_path}")
    else: 
        from lib.hf_datasets_info import datasets_info as old_datasets_info
        datasets_info = {**old_datasets_info, **datasets_info}
        logging.info(f"Updating dataset_info in existing {file_path}")

    with open(file_path, 'w') as finfo:
        finfo.write('# Automatically generated using ../scripts/build_datasets_info.py. \n\n')
        finfo.write('from datasets import ClassLabel, Value\n\n')
        finfo.write(f'valid_task_names={list(datasets_info.keys())}\n\n')
        finfo.write('datasets_info = {\n')
        for key, value in datasets_info.items():
            finfo.write(f'\n\t"{key}": {{\n')
            for subkey,subvalue in datasets_info[key].items():
                finfo.write(f'\t\t"{subkey}": {subvalue},\n')
            finfo.write('\t},\n')
        finfo.write('}\n')

def main(add_tasks=None):
    if add_tasks:
        builder_names = add_tasks
        overwrite = False
    else:
        overwrite = True
        builder_names = default_builders

    dataset_info = create_datasets_info_dict(builder_names)
    dump_datasets_info (dataset_info, overwrite)

if __name__== "__main__":
    add_tasks = sys.argv[1:]
    main(add_tasks)