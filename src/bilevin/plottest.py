from pathlib import Path
import pickle as pkl

import matplotlib as mpl
from plotting.utils import get_runs_data
from plotting.keys import phs_test_key

# pths = [Path("ccdata/stp3_phs/"), Path("ccdata/stp3_phs2/")]
pths = Path("data/j30/stp4/phs/")
keys = ["loss_fn", "forward_policy_lr", "max_grad_norm", "weight_mse_loss"]
all_runs = get_runs_data(pths, keys, min_valids=5)
f = open("j30stp4.pkl", "wb")
pkl.dump(all_runs, f)
