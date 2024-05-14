def plot_medians(ax, run_data, y_label, color):
    xs, central, lower, upper = batch_window_mean(run_data, y_label)
    ax.plot(xs, central, color=color)
    ax.fill_between(
        xs,
        lower,
        upper,
        edgecolor=(*color, 0.1),
        facecolor=(*color, 0.1),
    )

def plot_bi_policy_model(run_data: dict, label: str):
    fig, ax = plt.subplots(2, 2, figsize=(12, 10), sharex=True)
    fl = ax[0, 0]
    fl.set_title(f"Forward")
    fl.set_ylabel("Loss")
    fa = ax[1, 0]
    fa.set_ylabel("Acc")
    fa.set_xlabel("Batch")

    bl = ax[0, 1]
    bl.set_title(f"Backward")
    ba = ax[1, 1]
    ba.set_xlabel("Batch")

    ls_mapper = LineStyleMapper()
    color, _, _ = ls_mapper.get_ls(label)

    plot_medians(fl, run_data["train"], "floss", color)
    plot_medians(fa, run_data["train"], "facc", color)
    plot_medians(bl, run_data["train"], "bloss", color)
    plot_medians(ba, run_data["train"], "bacc", color)

 def plot_uni_policy_model(run_data: dict, label: str):
    fig, ax = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    fl = ax[0]
    fl.set_title(f"Forward")
    fl.set_ylabel("Loss")
    fa = ax[1]
    fa.set_xlabel("Batch")

    ls_mapper = LineStyleMapper()
    color, _, _ = ls_mapper.get_ls(label)

    plot_medians(fl, run_data["train"], "floss", color)
    plot_medians(fa, run_data["train"], "facc", color)
    return fig, ax   return fig, ax

def plot_bi_heuristic_model(run_data: dict, label: str):
    fig, ax = plt.subplots(1, 2, figsize=(12, 10), sharex=True)
    fl = ax[0]
    fl.set_title(f"Forward")
    fl.set_ylabel("Loss")
    fl.set_xlabel("Batch")
    bl = ax[1]
    bl.set_title(f"Backward")
    bl.set_xlabel("Batch")

    ls_mapper = LineStyleMapper()
    color, _, _ = ls_mapper.get_ls(label)

    plot_medians(fl, run_data["train"], "floss", color)
    plot_medians(bl, run_data["train"], "bloss", color)
    return fig, ax

def plot_uni_heuristic_model(run_data: dict, label: str):
    fig, ax = plt.subplots(1, 1, figsize=(12, 10), sharex=True)
    fl = ax
    fl.set_title(f"Forward")
    fl.set_ylabel("Loss")
    fl.set_xlabel("Batch")
    ls_mapper = LineStyleMapper()
    color, _, _ = ls_mapper.get_ls(label)

    plot_medians(fl, run_data["train"], "floss", color)
    return fig, ax

def plot_bi_policy_heuristic_model(run_data: dict, label: str):
    fig, ax = plt.subplots(2, 2, figsize=(12, 10), sharex=True)
    fl = ax[0, 0]
    fl.set_title(f"Forward")
    fl.set_ylabel("Loss")
    fa = ax[1, 0]
    fa.set_ylabel("Acc")
    fa.set_xlabel("Batch")
    # flh = ax[2, 1]

    bl = ax[0, 1]
    bl.set_title(f"Backward")
    ba = ax[1, 1]
    ba.set_xlabel("Batch")
    # blh = ax[2, 1]

    ls_mapper = LineStyleMapper()
    color, _, _ = ls_mapper.get_ls(label)

    # flp.set_ylabel("Policy loss", size=14)
    plot_medians(fl, run_data["train"], "floss", color)
    plot_medians(fa, run_data["train"], "facc", color)
    # plot_medians(flh, run_data["train"], "bloss", color)

    plot_medians(bl, run_data["train"], "bloss", color)
    plot_medians(ba, run_data["train"], "bacc", color)
    # plot_medians(blh, run_data["train"], "bacc", color)
    return fig, ax

def plot_uni_policy_heuristic_model(run_data: dict, label: str):
    fig, ax = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    flp = ax[0]
    flp.set_title(f"Forward")
    flp.set_ylabel("Loss")
    fa = ax[1]
    fa.set_ylabel("Acc")
    fa.set_xlabel("Batch")
    # flh = ax[2]

    ls_mapper = LineStyleMapper()
    color, _, _ = ls_mapper.get_ls(label)

    plot_medians(flp, run_data["train"], "floss", color)
    plot_medians(fa, run_data["train"], "facc", color)
    # plot_medians(flh, run_data["train"], "bloss", color)

    return fig, ax

# fill search vs batch plots
# plot_search_vs_batch(
#     rsdata,
#     "solved",
#     ax=ax2[0],
#     style=style,
#     label=agent,
# )
# plot_search_vs_batch(rsdata, "exp", ax=ax2[1], style=style, label=agent)
# plot_search_vs_batch(rsdata, "len", ax=ax2[2], style=style, label=agent)
# make model train plots
# if "Bi" in agent:
#     if "PHS" in agent:
#         f, a = plot_bi_policy_heuristic_model(rsdata, agent)
#     elif "Levin" in agent:
#         f, a = plot_bi_policy_model(rsdata, agent)
#     elif "AStar" in agent:
#         f, a = plot_bi_heuristic_model(rsdata, agent)
#     else:
#         raise ValueError(f"Unknown agent: {agent}")
# else:
#     if "PHS" in agent:
#         f, a = plot_uni_policy_heuristic_model(rsdata, agent)
#     elif "Levin" in agent:
#         f, a = plot_uni_policy_model(rsdata, agent)
#     elif "AStar" in agent:
#         f, a = plot_uni_heuristic_model(rsdata, agent)
#     else:
#         raise ValueError(f"Unknown agent: {agent}")
# f.savefig(saveroot / f"{agent}_model.pdf", bbox_inches="tight")
