from pathlib import Path
import json

runs_dir = Path("runs_pancake")
for dom_dir in sorted(runs_dir.iterdir()):
    agent_dirs = [a for a in dom_dir.iterdir()]
    agent_dirs = sorted(agent_dirs, key=lambda x: len(x.name))
    agent_dirs = sorted(agent_dirs, key=lambda x: x.name.split("Bi")[-1])
    for agent_dir in agent_dirs:
        # validate args
        allargs = sorted(list(r / "args.json" for r in agent_dir.iterdir()))
        assert len(allargs) == 5, f"{agent_dir}"
        alllogs = sorted(list(r / "simple_log.txt" for r in agent_dir.iterdir()))
        assert len(alllogs) == 5
        max_time = 0
        successes = 0
        for args, log in zip(allargs, alllogs):
            agentargs = json.load(args.open())
            if "BFS" in agent_dir.name:
                assert "BFS" in agentargs["agent"]
            if "AStar" in agent_dir.name:
                if "w1" in agent_dir.name:
                    assert agentargs["weight_astar"] == 1.0
                else:
                    assert agentargs["weight_astar"] == 2.5

                if "Levin" in agent_dir.name or "PHS" in agent_dir.name:
                    if "_m" in agent_dir.name:
                        assert agentargs["mask_invalid_actions"] == True
                    else:
                        assert agentargs["mask_invalid_actions"] == False
            with log.open() as f:
                for line in f:
                    epoch, _, _, _, _, time = line.split()
                    epoch = int(epoch)
                    time = float(time)
                    if epoch == 10:
                        successes += 1
                        if time > max_time:
                            max_time = time
        if successes != 5:
            print(
                f"Found {successes} successes for {dom_dir.name}/{agent_dir.name} <<<<<<<<<"
            )
        else:
            print(
                f"Max time for {dom_dir.name}/{agent_dir.name}: {max_time/3600:0.2f} hrs"
            )
    print("\n")
