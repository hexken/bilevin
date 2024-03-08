from pathlib import Path
import sys

from natsort import natsorted


def prepare_domain(domdir, model_suffix):
    for agentdir in domdir.glob("*/"):
        if agentdir.name == "astar":
            outfile = domdir / "astar_args_test.txt"
            agents = ("_AStar", "_BiAStarAlt")
        elif agentdir.name == "levin":
            outfile = domdir / "levin_args_test.txt"
            agents = ("_Levin", "_BiLevin")
        elif agentdir.name == "phs":
            outfile = domdir / "phs_args_test.txt"
            agents = ("_PHS", "_BiPHSBFS")
        else:
            continue

        runargs = []
        for rundir in natsorted(agentdir.glob("*train*")):
            model_pth = list(rundir.glob(f"model{model_suffix}"))[0]
            if agents[0] in rundir.name:
                i = 0
            elif agents[1] in rundir.name:
                i = 1
            else:
                print(f"Skipping {rundir.name}")
                continue
            args = f"{agents[i][1:]} {model_pth}"
            runargs.append(args)

        outfile.write_text("\n".join(runargs))


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python prepare_test.py <dom_dir|doms_dir> [all] <best|last>")
        sys.exit(1)

    d = sys.argv[1]
    if sys.argv[2].lower() == "all":
        dom_dirs = list(Path(d).glob("*/"))
    else:
        dom_dirs = [Path(d)]

    if sys.argv[3] == "best":
        model_suffix = "_best_expanded.pt"
    elif sys.argv[4] == "last":
        model_suffix = "_lastest.pt"
    else:
        raise ValueError("Invalid model suffix")

    for domdir in dom_dirs:
        prepare_domain(domdir, model_suffix)
