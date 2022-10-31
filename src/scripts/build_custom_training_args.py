import inspect
import os

import transformers
import yaml

custom_training_args_defaults = dict(
    output_dir='save',
    overwrite_output_dir=True,
    do_train=True,
    do_eval=True,
    do_predict=False,
    evaluation_strategy='steps',
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    learning_rate=5.0e-05,
    warmup_steps=0,        
    weight_decay=0.0,
    num_train_epochs=6,
    greater_is_better=True,
    lr_scheduler_type='linear',
    logging_strategy='steps',
    save_strategy='steps',
    save_total_limit=5,
    logging_steps=200,
    save_steps=200,
    eval_steps=200,
    seed=-1,
    disable_tqdm=True
)

def main():
    # additionally create separate training_args.yaml for specifying custom defaults for transformers.TrainingArguments .__init__

    training_kwargs = {}
    parameters = inspect.signature(transformers.TrainingArguments .__init__).parameters

    for param, spec in parameters.items():
        if spec.name=='self':
            continue
        training_kwargs[spec.name] = spec.default
    
    training_kwargs.update(custom_training_args_defaults)

    file_path = os.path.join("..", "configs", "custom_training_args.yaml")
    with open(file_path, "w") as f:
        yaml.dump(training_kwargs, f, sort_keys=False)

if __name__ == '__main__':
    main()
