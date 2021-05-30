import json
import numpy as np

def intersection(lst1, lst2):
    return list(set(lst1) & set(lst2))

def save_json(dict_obj, save_path):
    if save_path is not None:
        with open(save_path, "w") as json_f:
            json.dump(dict_obj, json_f, indent=4)

def save_npy(arr, save_path):
    if save_path is not None:
        with open(save_path, "wb") as npy_f:
            np.save(npy_f, arr)