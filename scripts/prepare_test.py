from pathlib import Path
import sys

from natsort import natsorted


def prepare_unfinished(domdir, model_suffix):
    for agentdir in domdir.glob("*/"):
        if agentdir.name == "astar":
            outfile = domdir / "astar_args_test_unfinished.txt"
            agents = ("_AStar", "_BiAStar")
        elif agentdir.name == "levin":
            outfile = domdir / "levin_args_test_unfinished.txt"
            agents = ("_Levin", "_BiLevin")
        elif agentdir.name == "phs":
            outfile = domdir / "phs_args_test_unfinished.txt"
            agents = ("_PHS", "_BiPHS")
        else:
            continue

        runargs = []
        for rundir in natsorted(agentdir.glob("*train*")):
            dirs = rundir.glob("f*test_model_{model_suffix}*")
            if len(list(dirs)) == 0:
                models = list(rundir.glob(f"model{model_suffix}"))
                if not models:
                    print(f"Skipping {rundir.name}, no models")
                    continue
                elif len(models) > 1:
                    print(f"Found more than one model {rundir.name}")
                model_pth = models[0]
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


def prepare_new(domdir, model_suffix):
    for agentdir in domdir.glob("*/"):
        if agentdir.name == "astar":
            outfile = domdir / "astar_args_test.txt"
            agents = ("_AStar", "_BiAStar")
        elif agentdir.name == "levin":
            outfile = domdir / "levin_args_test.txt"
            agents = ("_Levin", "_BiLevin")
        elif agentdir.name == "phs":
            outfile = domdir / "phs_args_test.txt"
            agents = ("_PHS", "_BiPHS")
        else:
            continue

        runargs = []
        for rundir in natsorted(agentdir.glob("*train*")):
            models = list(rundir.glob(f"model{model_suffix}"))
            if not models:
                print(f"Skipping {rundir.name}, no models")
                continue
            elif len(models) > 1:
                print(f"Found more than one model {rundir.name}")
            model_pth = models[0]
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
        print(
            "Usage: python prepare_test.py <dom_dir|doms_dir> [all] [unfinished] <best|latest>"
        )
        sys.exit(1)

    d = sys.argv[1]
    if sys.argv[2].lower() == "all":
        dom_dirs = list(Path(d).glob("*/"))
    else:
        dom_dirs = [Path(d)]

    if sys.argv[3] == "unfinished":
        unfinished = True
        model_arg = sys.argv[4]
    else:
        unfinished = False
        model_arg = sys.argv[3]

    if model_arg == "best":
        model_suffix = "_best_expanded.pt"
    elif model_arg == "latest":
        model_suffix = "_lastest.pt"
    else:
        raise ValueError(f"Invalid model suffix: {model_arg}")

    if unfinished:
        for domdir in dom_dirs:
            prepare_unfinished(domdir, model_suffix)
    else:
        for domdir in dom_dirs:
            prepare_new(domdir, model_suffix)
