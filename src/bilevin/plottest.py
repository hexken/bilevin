from pathlib import Path
import pickle as pkl

import matplotlib as mpl
from plotting.utils import get_runs_data
from plotting.keys import phs_test_key

# pths = [Path("ccdata/stp3_phs/"), Path("ccdata/stp3_phs2/")]
pths = list(Path("data/socs/stp4").glob("*/"))
keys = ["agent"]
all_runs = get_runs_data(pths, keys, min_valids=1)
f = open("socs_stp4.pkl", "wb")
pkl.dump(all_runs, f)
