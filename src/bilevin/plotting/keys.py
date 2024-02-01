import re
from json import load


def all_but_seed_key(run):
    return run.key


def loss_lr_key(run):
    return f"{run['loss_fn']} {run['forward_policy_lr']}"


def opt_loss_group_key(item):
    key, val = item
    words = key.split()
    return f"{words[-1]} {words[1]} {words[3]}"


def lr_mom_group_key(item):
    key, val = item
    words = key.split()
    return f"{words[2]}"


def all_group_key(pth):
    r = re.search(
        "^.*_((?:Bi)?AStar.*)?((?:Bi)?Levin.*)?((?:Bi)?PHS.*)?_e\d+_t\d+.\d+_(lr\d+\.\d+)(?:_)(w\d+\.?\d*)?.*",
        str(pth),
    )
    if r:
        return " ".join([g for g in r.groups() if g is not None])


def phs_test_key(pth):
    r = re.search(
        "^.*(PHS).*_opt(.*)_(lr\d+\.\d+)_(n[tf])_(mn[-+]?\d\.\d+)_(m\d\.\d)_loss(.*)_\d_.*",
        str(pth),
    )
    if r:
        return " ".join([g for g in r.groups() if g is not None])


def alg_sort_key(s: str):
    if "PHS" in s:
        key1 = 3
    elif "Levin" in s:
        key1 = 2
    elif "AStar" in s:
        key1 = 0
    else:
        raise ValueError("Unknown alg")

    if "Bi" in s:
        key2 = 1
    else:
        key2 = 0

    if "Alt" in s:
        key3 = 0
    elif "BFS" in s:
        key3 = 1
    else:
        key3 = 3

    return key1, key2, key3
