from pathlib import Path
import pickle as pkl

import matplotlib as mpl
from plotting.utils import get_runs_data
from plotting.keys import phs_test_key

# pths = [Path("ccdata/stp3_phs/"), Path("ccdata/stp3_phs2/")]
pths = Path("ccdata/stp3_phs3/")
all_runs = get_runs_data(pths, phs_test_key, min_valids=5)
f = open("all_runs2.pkl", "wb")
pkl.dump(all_runs, f)
