from collections import OrderedDict
from collections.abc import Iterable
import itertools
import json
from pathlib import Path
import pickle as pkl
import re
import subprocess

import matplotlib as mpl
from natsort import natsorted
import pandas as pd
from plot.utils import get_runs_data, phs_test_key
all_runs = get_runs_data(Path("ccdata/stp3_phs2/"), phs_test_key)
f = open("all_runs2.pkl", "wb")
pkl.dump(all_runs, f)
