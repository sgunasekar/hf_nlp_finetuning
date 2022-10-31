import logging
import os
import random
import socket
import sys

from transformers import Trainer, TrainingArguments

from configs.constants import CUSTOM_TRAINING_ARGS_FILE
from configs.hf_arguments_utils import (Custom_HfArgumentParser,
                                        DatasetArguments, IOArguments,
                                        ModelArguments)
from lib import hf_datautils, hf_modelutils, utils

logger = logging.getLogger(__name__)
import transformers
import transformers.trainer_utils


def main():

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file, let's parse it to get our arguments.

        parser = Custom_HfArgumentParser((DatasetArguments, ModelArguments, TrainingArguments, IOArguments))
        data_args, model_args, training_args, io_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))

    else:
        # The custom parser takes arguments from custom_training_args.yaml files first and then update with commandline arguments. Commandline arguments take precedence in case of conflict
        
        parser = Custom_HfArgumentParser((DatasetArguments, ModelArguments, TrainingArguments, IOArguments))
        data_args, model_args, training_args, io_args = parser.parse_yaml_first_then_args_into_dataclasses(yaml_filename=CUSTOM_TRAINING_ARGS_FILE)

    # Setup logging and log on each process the small summary:
    # Ref: log_level on non-main gpus is warning
    utils.setup_logging(training_args)
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, hostname: {socket.gethostname()}; "
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}."
    )
    logger.info("=========== args ============")
    logger.info(f"data args: {data_args}\n")
    logger.info(f"model_args: {model_args}\n")
    logger.info(f"training_args: {training_args}\n")
    logger.info(f"io_args: {io_args}\n")
    logger.info("=============================")

    # Set seed before initializing model.
    if training_args.seed>0: 
        transformers.set_seed(training_args.seed)

    # load datasets
    # creates tokenized dataset on only the rank=0 process; the other process read from cache. In multi-node training local=True runs the tokenizer on rank=0 process of each node, else only node=0 & rank=0 process. 
    logger.info("===========data==============")
    with training_args.main_process_first(desc="dataset creation and map pre-processing", local = not io_args.is_shared_file_system):
        tokenized_train_dataset, tokenized_eval_datasets, tokenizer, compute_metrics = hf_datautils.get_tokenized_datasets(data_args, model_args, io_args)

    # Log a few random samples from the training set:
    if training_args.do_train:
        for index in random.sample(range(len(tokenized_train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {tokenized_train_dataset[index]}.")
    
    logger.info("=============================")

    
    model = hf_modelutils.get_model(model_args, io_args)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset if training_args.do_train else None,
        eval_dataset=tokenized_eval_datasets if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer
    )

    # Get resume_checkpoint if applicable, or last_chekpoint, or None
    checkpoint = utils.get_checkpoint(training_args)

    logger.info("===========train=============")
    # Training
    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        metrics["train_samples"] = len(tokenized_train_dataset)

        trainer.save_model()  # Saves the tokenizer too for easy upload
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    logger.info("===========DONE TRAINING===========")
    logger.info("============final eval=============")
    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(eval_dataset=tokenized_eval_datasets)
        metrics["eval_samples"] = {k:len(v) for k,v in tokenized_eval_datasets.items()}

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    logger.info("============DONE=============")

    # Pushing to hub
    kwargs = {'finetuned_from': model_args.model_name, 'tasks': "text-classification"}
    kwargs['dataset_tags'] = data_args.train_datasets[0].load_dataset_kwargs['builder_name']
    kwargs['dataset_args'] = data_args.train_datasets[0].load_dataset_kwargs['builder_name']
    kwargs["dataset"] = f"{kwargs['dataset_tags']}:{kwargs['dataset_args']}"

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)

if __name__=='__main__':
    main()
