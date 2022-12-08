import logging
import os
import sys
import transformers
import datasets 
_logger = logging.getLogger(__name__)

def setup_logging(training_args):

    log_level = training_args.get_process_log_level()
    process_index = training_args.process_index
    log_dir = training_args.logging_dir
    with training_args.main_process_first():
        if not(training_args.process_index):
            os.makedirs(log_dir)
    log_file = os.path.join(log_dir,f"logs{process_index}.log")
    
    # default: INFO on local_rank==0 node and WARNING on other nodes
    # change this behavior using --log_level, --log_level_replica, --log_on_each_node
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    # Setup logging
    logging.basicConfig(
        format='%(asctime)s %(levelname)s [%(name)s %(filename)s:%(lineno)d] >> %(message)s',
        datefmt='%y-%m-%d:%H:%M', 
        handlers= [
            logging.FileHandler(log_file,'w'),
            logging.StreamHandler(sys.stdout),
        ]
    )

    # Set log level for root logger
    logging.getLogger().setLevel(log_level) 
    # Set log levels on hf modules
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)    
    transformers.utils.logging.disable_default_handler()
    transformers.utils.logging.enable_propagation()
    transformers.utils.logging.enable_explicit_format()

    hf_loggers = [
        transformers.utils.logging.get_logger(),
        datasets.utils.logging.get_logger(),
    ]

    # for logger in hf_loggers:
    #     logger.set_verbosity(log_level)
    #     logger.add_handlers(logging.root.handlers)

def get_checkpoint(training_args):
    
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = transformers.trainer_utils.get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            _logger.warning(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint

    return checkpoint

from pynvml import *

def print_gpu_utilization():
    s = get_gpu_utilization()
    print(f"GPU memory occupied: {s}.")

def get_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    return f"{info.used//1024**2} MB"

def print_summary(result):
    print(f"Time: {result.metrics['train_runtime']:.2f}")
    print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
    print_gpu_utilization()