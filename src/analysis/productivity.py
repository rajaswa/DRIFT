import json
import operator
import os
from itertools import islice

from ..utils import find_productivity, save_json


def compute_productivity(words, text, save_load_path=None, overwrite=False):
    if (
        save_load_path is not None
        and os.path.isfile(save_load_path)
        and not (overwrite)
    ):
        print(f"Loading Productivity Dictionary from {save_load_path}")
        with open(save_load_path, "r") as prod_dict:
            prod_maps = json.load(prod_dict)
    else:
        prod_maps = find_productivity(words, text, n=2)

        print(f"Saving Productivity Dictionary at {save_load_path}")
        save_json(prod_maps, save_load_path)

    return prod_maps
