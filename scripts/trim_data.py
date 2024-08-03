from pathlib import Path
import pickle as pkl

import pandas as pd


def main():
    dom_paths = list(Path("/home/ken/Projects/thes_data/").glob("*.pkl"))
    for dom in dom_paths:
        print(f"Trimming {dom.stem}")
        dom_data = pkl.load(dom.open("rb"))
        for agent, runs in dom_data.items():
            for i in range(len(runs)):
                runs[i]["train"] = runs[i]["train"][["exp", "len", "epoch"]]
                runs[i]["valid"] = runs[i]["valid"][["exp", "len", "epoch"]]
            dom_data[agent] = runs
        new_path = dom.parent / f"{dom.stem}_trim.pkl"
        pkl.dump(dom_data, new_path.open("wb"))
        print(f"Trimmed {new_path}")

if __name__ == "__main__":
    main()
