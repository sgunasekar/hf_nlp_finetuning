## Running this script generates the configs/hf_datasets_info.py file

import json
import os 
import datasets
import sys
from typing import Union
import logging
from datasets import ClassLabel, Value
import evaluate

sys.path.insert(0, "..") 

## This scripts creates and updates the ../configs/hf_datasets_info.py file
## Usages: 
# `python build_datasets_info.py`
# `python build_datasets_info.py task1 task2`

## Generated file contains the following:
# valid_task_names: as a list of valid finetuning task names that can be processed by the code
# datasets_info: a dictionary mapping finetuning tasks to the relevant subset of info from DatasetInfo object. 
# For each task/dataset, the datasets_info stores: args for load_dataset(), names of splits, label information, and sentence keys for the input features. 
## Ref: GLUE tasks: ['cola', 'sst2', 'mrpc', 'qqp', 'stsb', 'mnli', 'mnli_mismatched', 'mnli_matched', 'qnli', 'rte', 'wnli', 'ax']

default_builders = ['glue', 'imdb', 'hans']
'''
By default the sentence_key for each dataset is set as all the "str" type fields in DatasetInfo.features, except for labels/answer/answers and idx/id. 
To override the default behaviour, specify the sentence keys for tasks below
'''
sentence_keys_dict = {
    'hans': ['premise', 'hypothesis']
}
def create_datasets_info_dict(builder_names):

    datasets_info = {}

    for builder in builder_names:
        info = datasets.get_dataset_infos(builder)
        tasks = list(info.keys())
        # If the builder has only one task, use the builder_name as the key in datasets_info, else use the task name (config_name) as the key
        if len(tasks)==1: 
            task_key = builder
            TaskInfo = info[tasks[0]]
            datasets_info[task_key] = get_task_info_dict(TaskInfo, task_key)
        else:
            for task, TaskInfo in info.items():
                task_key = task
                if task in datasets_info.keys():
                    prev_task_info = datasets_info.pop(task)
                    prev_builder = prev_task_info['load_dataset_kwargs']['path']
                    updated_prev_key = f"{task}.{prev_builder}"
                    datasets_info[updated_prev_key] = prev_task_info
                    task_key = f"{task}.{builder}"
                datasets_info[task_key] = get_task_info_dict(TaskInfo, task_key)

    return datasets_info


# Extracts relevant info for train/val from a single DatasetInfo object
def get_task_info_dict(TaskInfo: datasets.DatasetInfo, task_key:str) -> dict:
    features = TaskInfo.features
    
    # Look for features that contains the label/answer
    if "label" in features.keys():
        label_key = "label"
    elif "answer" in features.keys():
        label_key = "answer"
    elif "answers" in features.keys():
        label_key = "answers"
    else:
        label_key = None

    label = features[label_key] if label_key else None
    if isinstance(label, ClassLabel):
        num_outputs = label.num_classes
        # default metric/loss for classification
        evaluate_load_args = ("accuracy")
        task_type = "classification"
    elif label.dtype.startswith(('float','double')):
        num_outputs = 1
        # default metric/loss for regression
        evaluate_load_args = ("mse")
        task_type = "regression"
    else:
        num_outputs = None
        metric = None
        task_type = None
    try:
        # try to load task specific metrics and pass if not
        metric = evaluate.load(TaskInfo.builder_name,TaskInfo.config_name)
        evaluate_load_args = (TaskInfo.builder_name,TaskInfo.config_name)
    except:
        print(f"Unable to load metric for ({TaskInfo.builder_name},{TaskInfo.config_name}) from evaluate. Using default metric for {task_type} task.")
        pass 

    # hueristic for selecting columns for inputs
    # skip features that are not strings or are idx/id
    sentence_keys = []
    id_key = None
    for key in features.keys(): 
        if  key in ["idx","id"]:
            id_key = key
        elif 'string' in features[key].dtype:
            sentence_keys.append(key)
    # overwrite sentence keys if it was explicitly specified
    if task_key in sentence_keys_dict:
        sentence_keys = sentence_keys_dict[task_key]

    splits = list(TaskInfo.splits.keys())
    splits_size = []
    for split in splits:
        splits_size.append(TaskInfo.splits[split].num_examples)

    task_info = dict(
        load_dataset_kwargs=dict(
            path=TaskInfo.builder_name,
            name=TaskInfo.config_name
        ),
        splits=splits,
        splits_num_examples=splits_size,
        sentence_keys=sentence_keys,
        id_key=id_key,
        label_key=label_key,
        evaluate_load_args=evaluate_load_args,
        num_outputs=num_outputs,
        task_type=task_type,
        features=features
    )

    return task_info


def dump_datasets_info(datasets_info, overwrite = True):

    file_path = os.path.join("..", "configs", "hf_datasets_info.py") 
    if overwrite or not(os.path.exists(file_path)):
        logging.info(f"Creating/overwriting {file_path}")
    else: 
        from configs.hf_datasets_info import datasets_info as old_datasets_info
        datasets_info = {**old_datasets_info, **datasets_info}
        logging.info(f"Updating datasets_info in existing {file_path}")

    with open(file_path, 'w') as finfo:
        finfo.write('# Automatically generated using ../scripts/build_datasets_info.py. \n\n')
        finfo.write('from datasets import ClassLabel, Value, Sequence\n\n')
        finfo.write(f'valid_task_names={list(datasets_info.keys())}\n\n')
        finfo.write('datasets_info = {\n')
        for key, value in datasets_info.items():
            finfo.write(f'\n\t"{key}": {{\n')
            for subkey,subvalue in datasets_info[key].items():
                if not(isinstance(subvalue,str)):
                    finfo.write(f'\t\t"{subkey}": {subvalue},\n')
                else:
                    finfo.write(f'\t\t"{subkey}": "{subvalue}",\n')
            finfo.write('\t},\n')
        finfo.write('}\n')

def main(add_tasks=None):
    if add_tasks:
        builder_names = add_tasks
        overwrite = False
    else:
        overwrite = True
        builder_names = default_builders

    datasets_info = create_datasets_info_dict(builder_names)
    dump_datasets_info (datasets_info, overwrite)

if __name__== "__main__":
    add_tasks = sys.argv[1:]
    main(add_tasks)