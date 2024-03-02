from pathlib import Path
import sys

from natsort import natsorted

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python prepare_test.py <domain_dir> <best|last>")
        sys.exit(1)

    if sys.argv[2] == "best":
        model_suffix = "_best_expanded.pt"
    elif sys.argv[2] == "last":
        model_suffix = "_lastest.pt"
    else:
        raise ValueError("Invalid model suffix")

    domdir = Path(sys.argv[1])
    for agentdir in domdir.glob("*/"):
        if agentdir.name == "astar":
            outfile = domdir / "astar_args.txt"
            agents = ("_AStar", "_BiAStarAlt")
        elif agentdir.name == "levin":
            outfile = domdir / "levin_args.txt"
            agents = ("_Levin", "_BiLevin")
        elif agentdir.name == "phs":
            outfile = domdir / "phs_args.txt"
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
                continue
            args = f"{agents[i][1:]} runs/thes/{model_pth}"
            runargs.append(args)

        outfile.write_text("\n".join(runargs))
