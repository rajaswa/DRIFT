"""Miscellaneous utility functions."""

import random
import numpy as np
import torch
import copy
import itertools


def seed(value=42):
    """Set random seed for everything.

    Args:
        value (int): Seed
    """
    np.random.seed(value)
    torch.manual_seed(value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(value)


def map_dict_to_obj(dic):
    result_dic = {}
    if dic is not None:
        for k, v in dic.items():
            if isinstance(v, dict):
                result_dic[k] = map_dict_to_obj(v)
            else:
                try:
                    obj = configmapper.get_object("params", v)
                    result_dic[k] = obj
                except:
                    result_dic[k] = v
    return result_dic


def get_item_in_config(config, path):
    ## config is a dictionary
    curr = config
    if isinstance(config, dict):
        for step in path:
            curr = curr[step]
            if curr is None:
                break
    else:
        for step in path:
            curr = curr.__getattr__(step)
            if curr is None:
                break
    return curr


# init = train_config.grid_search
# curr = get_item_in_config(init,['hyperparams','loader_params'])
# curr.set_value('batch_size',1)
# print(train_config.grid_search)


def generate_grid_search_configs(main_config, grid_config, root="hyperparams"):
    ## DFS
    locations_values_pair = {}
    init = grid_config.as_dict()
    # print(init)
    stack = [root]
    visited = [stack[-1]]

    log_label_path = None
    hparams_path = None

    # root = init[stack[-1]]
    while len(stack) != 0:
        root = get_item_in_config(init, stack)
        flag = 0
        # print(visited)
        # print(stack)
        if (
            not isinstance(root, dict) and "hparams" not in stack
        ):  ## Meaning it is a leaf node
            # print(stack)
            if isinstance(root, list):
                locations_values_pair[
                    tuple(copy.deepcopy(stack))
                ] = root  ## Append the current stack, and the list values
            else:
                locations_values_pair[tuple(copy.deepcopy(stack))] = [
                    root,
                ]  ## Append the current stack, and the list values

            _ = stack.pop()  ## Pop this root because we don't need it.
        else:
            if isinstance(root, list) and "hparams" in stack:
                hparams_path = copy.deepcopy(stack)
                visited.append(".".join(stack))
                stack.pop()
                continue

            if "log_label" in root.keys():
                log_label_path = copy.deepcopy(stack + ["log_label",])

            if "log_label" in root.keys():
                log_label_path = copy.deepcopy(stack + ["log_label",])
            parent = root  ## Otherwise it has children

        for key in parent.keys():  ## For the children
            if (
                ".".join(stack + [key,]) not in visited
            ):  ## Check if I have visited these children
                flag = 1  ## If not, we need to repeat the process for this key
                stack.append(key)  ## Append this key to the stack
                visited.append(".".join(stack))
                break
        if flag == 0:
            stack.pop()

    paths = list(locations_values_pair.keys())
    values = itertools.product(*list(locations_values_pair.values()))

    result_configs = []
    for value in values:
        for item_index in range(len(value)):
            curr_path = paths[item_index]
            curr_item = value[item_index]

            curr_config_item = get_item_in_config(main_config, curr_path[1:-1])
            curr_config_item.set_value(curr_path[-1], curr_item)

            log_item = get_item_in_config(main_config, log_label_path[1:-1])
            log_item.set_value(log_label_path[-1], str(len(result_configs) + 1))

            hparam_item = get_item_in_config(main_config, hparams_path[1:-1])
            hparam_item.set_value(
                hparams_path[-1],
                get_item_in_config(grid_config.hyperparams, hparams_path[1:]),
            )

        result_configs.append(copy.deepcopy(main_config))
    return result_configs
