
import logging
from typing import Optional

import yaml
from configs.constants import CUSTOM_FUNC_KWARGS_FILE
from datasets import Dataset, load_dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          PreTrainedTokenizerBase)

ADDITIONAL_KWARGS = yaml.full_load(open(CUSTOM_FUNC_KWARGS_FILE,"r"))

def get_model(model_args, io_args):
    model_kwargs = ADDITIONAL_KWARGS.get(AutoModelForSequenceClassification.from_pretrained.__qualname__)

    model_kwargs.update(dict(
        pretrained_model_name_or_path = model_args.model_path,
        revision=model_args.model_revision,
        use_auth_token=True if io_args.use_auth_token else None,
        cache_dir=io_args.cache_dir,
    ))

    model = AutoModelForSequenceClassification(**model_kwargs)
