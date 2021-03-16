"""Train File."""
## Imports
import argparse

# import itertools
import copy
import numpy as np
import torch
import torch.nn as nn

from src.utils.misc import seed, generate_grid_search_configs
from src.utils.configuration import Config

from src.datasets import *
from src.models import *
from src.trainers import *

from src.modules.preprocessors import *
from src.utils.mapper import configmapper
from src.utils.logger import Logger

import os

dirname = os.path.dirname(__file__)  ## For Paths Relative to Current File

## Config
parser = argparse.ArgumentParser(prog="train.py", description="Train a model.")
parser.add_argument(
    "--model",
    type=str,
    action="store",
    help="The configuration for model",
    default=os.path.join(dirname, "./configs/models/forty/default.yaml"),
)
parser.add_argument(
    "--train",
    type=str,
    action="store",
    help="The configuration for model training/evaluation",
    default=os.path.join(dirname, "./configs/trainers/forty/train.yaml"),
)
parser.add_argument(
    "--data",
    type=str,
    action="store",
    help="The configuration for data",
    default=os.path.join(dirname, "./configs/datasets/forty/default.yaml"),
)
parser.add_argument(
    "--grid_search",
    action="store_true",
    help="Whether to do a grid_search",
    default=False,
)
### Update Tips : Can provide more options to the user.
### Can also provide multiple verbosity levels.

args = parser.parse_args()
# print(vars(args))
model_config = Config(path=args.model)
train_config = Config(path=args.train)
data_config = Config(path=args.data)
grid_search = args.grid_search

# verbose = args.verbose

# Preprocessor, Dataset, Model
preprocessor = configmapper.get_object(
    "preprocessors", data_config.main.preprocessor.name
)(data_config)


if grid_search:
    train_configs = generate_grid_search_configs(train_config, train_config.grid_search)
    print(f"Total Configurations Generated: {len(train_configs)}")

    logger = Logger(
        **train_config.grid_search.hyperparams.train.log.logger_params.as_dict()
    )

    for train_config in train_configs:
        print(train_config)

        ## Seed
        seed(train_config.main_config.seed)

        model, train_data, val_data = preprocessor.preprocess(model_config, data_config)
        # Trainer
        trainer = configmapper.get_object("trainers", train_config.trainer_name)(
            train_config
        )

        ## Train
        trainer.train(model, train_data, val_data, logger)

else:
    ## Seed
    seed(train_config.main_config.seed)

    model, train_data, val_data = preprocessor.preprocess(model_config, data_config)

    ## Trainer
    trainer = configmapper.get_object("trainers", train_config.trainer_name)(
        train_config
    )

    ## Train
    trainer.train(model, train_data, val_data)
