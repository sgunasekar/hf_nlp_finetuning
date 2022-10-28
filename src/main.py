import logging
import sys
import os
import yaml
from GitHub.nlp.src.configs.hf_arguments_utils import (
    DatasetArguments,
    ModelArguments,
    Custom_HfArgumentParser,
)
from transformers import (
    TrainingArguments
)
from configs.constants import CUSTOM_TRAINING_ARGS_FILE

logger = logging.getLogger(__name__)

def main():

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file, let's parse it to get our arguments.
        parser = Custom_HfArgumentParser((DatasetArguments, ModelArguments, TrainingArguments))
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        # When parsing from commandline, we parse DatasetArguments and ModelArguments first. TrainingArguments are by default loaded from configs.custom_training_args.yaml and overwridden by any command line arguments. This effectly changes the default arguments for tranformers.TrainingArguments
        parser = Custom_HfArgumentParser((DatasetArguments, ModelArguments, TrainingArguments))
        model_args, data_args, training_args = parser.parse_yaml_first_then_args_into_dataclasses(yaml_filename=CUSTOM_TRAINING_ARGS_FILE)
    print(data_args)
    print(model_args)
    print(training_args)

if __name__=='__main__':
    main()
