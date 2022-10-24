import json
import os 
import datasets
from datasets import Value
# Extracts relevant info for training/validation from DatasetInfo object
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
    sentence_keys = list(features.keys())

    task_info = dict(
        load_dataset_args=(TaskInfo.builder_name,TaskInfo.config_name),
        sentence_keys=sentence_keys,
        label=label,
        splits=list(TaskInfo.splits.keys())
    )

    return task_info

## Add new datasets here.
## 'datsets_info' is a dictionary of finetuning datasets with a relevant subset of info from DatasetInfo object. 
# For each task/dataset, the dataset_info stores: args for load_dataset(), names of splits, label information, and sentence keys for the input features. 

def create_datasets_info_dict():

    datasets_info = {}

    ## First we add tasks from data collections "super_glue" and "glue"
    ## Super GLUE tasks
    # ['boolq', 'cb', 'copa', 'multirc', 'record', 'rte', 'wic', 'wsc', 'wsc.fixed', 'axb', 'axg']
    ## GLUE taks
    # ['cola', 'sst2', 'mrpc', 'qqp', 'stsb', 'mnli', 'mnli_mismatched', 'mnli_matched', 'qnli', 'rte', 'wnli', 'ax']    
    datacollection_tasks = {
        "super_glue": datasets.get_dataset_config_names("super_glue"),
        "glue": datasets.get_dataset_config_names("glue")
    }
    datacollection_info = {
        "super_glue": datasets.get_dataset_infos("super_glue"),
        "glue": datasets.get_dataset_infos("glue")
    }

    for builder, info in datacollection_info.items():
        for task, TaskInfo in info.items():
            datasets_info[task] = get_task_info_dict(TaskInfo)

    ## Now we add individual datasets/IMDB and HANS datasets
    for ds in ("imdb","hans"):
        
        TaskInfo = datasets.get_dataset_infos(ds)['plain_text']
        datasets_info[ds] =  get_task_info_dict(TaskInfo)

    return datasets_info

def dump_datasets_info(datasets_info):
    with open(os.path.join("..", "lib", "hf_datasets_info.py"), 'w') as finfo:
        finfo.write('# Automatically generated using ../scripts/build_datasets_info.py. \n\n')
        finfo.write('from datasets import ClassLabel, Value\n\n')
        finfo.write(f'valid_task_names={list(datasets_info.keys())}\n\n')
        finfo.write('datasets_info = {\n')
        for key, value in datasets_info.items():
            finfo.write(f'\n\t"{key}":{{\n')
            for subkey,subvalue in datasets_info[key].items():
                finfo.write(f'\t\t"{subkey}":{subvalue},\n')
            finfo.write('\t},\n')
        finfo.write('}\n')


if __name__== "__main__":
    datasets_info = create_datasets_info_dict()
    dump_datasets_info(datasets_info)
