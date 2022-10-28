import datasets
import transformers
import inspect
import os
import yaml

func_list_skip_and_add_args = [
    (
        datasets.load_dataset, # function name
        ['path','name','split'], # args to skip as it is handled by argparser
        {}, # kwargs with defaults to add 
    ),
    (
        transformers.AutoTokenizer.from_pretrained,['pretrained_model_name_or_path ','use_fast'],
        {
            "config": None,
            "cache_dir": None,
            "force_download": False,
            "resume_download": False,
            "proxies": None,
            "revision": "main",
            "subfolder": None,
            "tokenizer_type": None,
            "trust_remote_code": False,
        },
    ),
]


def main():
    additional_kwargs = {}
    for func, skip_args, add_kwargs in func_list_skip_and_add_args:
        additional_fn_kwargs = {}
        parameters = inspect.signature(func).parameters

        for param, spec in parameters.items():
            if spec.default==inspect._empty:
                continue
            if spec.name in skip_args:
                continue
            additional_fn_kwargs[spec.name] = spec.default

        for kw,default in add_kwargs.items():
            if kw not in additional_fn_kwargs:
                additional_fn_kwargs[kw] = default

        additional_kwargs[func.__qualname__] = additional_fn_kwargs
    
    file_path = os.path.join("..", "configs", "additional_func_kwargs.yaml")
    with open(file_path, "w") as f:
        yaml.dump(additional_kwargs, f, sort_keys=False)

    # additionally create separate training_args.yaml for specifying custom defaults for transformers.TrainingArguments .__init__

    training_kwargs = {}
    parameters = inspect.signature(transformers.TrainingArguments .__init__).parameters

    for param, spec in parameters.items():
        if spec.name=='self':
            continue
        training_kwargs[spec.name] = spec.default

    file_path = os.path.join("..", "configs", "custom_training_args.yaml")
    with open(file_path, "w") as f:
        yaml.dump(training_kwargs, f, sort_keys=False)

if __name__=="__main__":
    main()