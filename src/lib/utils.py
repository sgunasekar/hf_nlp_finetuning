import logging
import sys
import os
logger = logging.getLogger(__name__)
import datasets, transformers 

def setup_logging(training_args):
    log_level = training_args.get_process_log_level()
    # default: INFO on local_rank==0 node and WARNING on other nodes
    # change this behavior using --log_level, --log_level_replica, --log_on_each_node
    # log_dir
    import socket
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.getLogger().setLevel(logging.INFO)
    logger.handlers = []

    # Setup logging
    logging.basicConfig(
        format='%(asctime)s %(levelname)s [%(name)s %(filename)s:%(lineno)d] >> %(message)s',
        datefmt='%m/%d:%H:%M', 
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    logger.setLevel(log_level)

    print(f'Running on machine {socket.gethostname()}')

    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

def get_checkpoint(training_args):
    
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = transformers.trainer_utils.get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.warning(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint

    return checkpoint
