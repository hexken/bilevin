import utils as putils
from pathlib import Path

all_runs_pth = Path("/home/ken/Projects/bilevin/final_runs_test/").glob("col4*")
all_runs = putils.get_runs_data(all_runs_pth, putils.all_group_key)
