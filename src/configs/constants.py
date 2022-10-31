import os

CUSTOM_TRAINING_ARGS_FILE=os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    "custom_training_args.yaml"
)

CUSTOM_FUNC_KWARGS_FILE=os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    "custom_func_kwargs.yaml"
)
