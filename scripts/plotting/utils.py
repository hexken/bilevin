# allowable_domains = {"stp4", "stp5", "tri4", "tri5", "col4", "col5"}
# allowable_domains = {"stp4", "tri4", "tri5", "col4", "col5"}
allowable_domains = {"stp5", "col5"}

y_lims = {
    "tri4": (0, 1900),
    "tri5": (0, 4000),
    "stp4": (500, 4100),
    "stp5": (1000, 7250),
    "col4": (0, 1800),
    "col5": (0, 4100),
}

col4 = (
    "AStar_w1",
    "BiAStar_w1",
    "Levin_nm",
    "BiLevin_nm",
    "PHS_nm",
    "BiPHS_nm",
)

col5 = (
    "AStar_w1",
    "BiAStar_w1",
    "Levin_nm",
    "BiLevin_nm",
    "PHS_nm",
    "BiPHS_nm",
)
tri4 = (
    "AStar_w2.5",
    "BiAStar_w2.5",
    "Levin_nm",
    "BiLevin_nm",
    "PHS_nm",
    "BiPHS_nm",
)

tri5 = (
    "AStar_w1",
    "BiAStar_w1",
    "Levin_nm",
    "BiLevin_nm",
    "PHS_nm",
    "BiPHS_nm",
)
stp4 = (
    "AStar_w2.5",
    "BiAStar_w2.5",
    "Levin_nm",
    "BiLevin_nm",
    "PHS_nm",
    "BiPHS_nm",
)
stp5 = (
    "AStar_w2.5",
    "BiAStar_w2.5",
    "Levin_nm",
    "BiLevin_nm",
    "PHS_nm",
    "BiPHS_nm",
)
bilevin_agents = (
    "BiLevin_m",
    "BiLevin_nm",
    "BiLevinBFS_m",
    "BiLevinBFS_nm",
)

biphs_agents = (
    "BiPHS_m",
    "BiPHS_nm",
    "BiPHSBFS_m",
    "BiPHSBFS_nm",
)

biastar_agents = (
    "BiAStar_w1",
    "BiAStar_w2.5",
    "BiAStarBFS_w1",
    "BiAStarBFS_w2.5",
)

levin_agents = (
    "Levin_m",
    "Levin_nm",
)

phs_agents = (
    "PHS_m",
    "PHS_nm",
)

astar_agents = (
    "AStar_w1",
    "AStar_w2.5",
)


class MixedStyles:
    def __init__(self):
        self.uni_marker = "o"
        self.bi_marker = "x"
        self.uni_ls = "-"
        self.bi_lds = "--"
        self.bibfs_ls = (0, (5, 6))
        self.bialt_ls = ":"
        self.bi_hatch = "|||"
        self.uni_hatch = None
        self.colors = ["#FF0000", "#900000", "#00FF00", "#009000", "#0AA0F5", "#000070"]
        # lighter colors earlier in the list

    def get_ls(self, agent: str, same_color: bool):
        s = agent.split("_")[0]
        if s == "AStar":
            ci = 1
        elif s == "BiAStar":
            ci = 0
        elif s == "Levin":
            ci = 3
        elif s == "BiLevin":
            ci = 2
        elif s == "PHS":
            ci = 5
        elif s == "BiPHS":
            ci = 4
        else:
            raise ValueError(f"Invalid agent {s}")

        # if "Alt" in s:
        #     ls = self.bialt_ls
        # elif "BFS" in s:
        #     ls = self.bibfs_ls
        # else:
        #     ls = self.uni_ls

        if "Bi" in s:
            h = self.bi_hatch
            m = self.bi_marker
        else:
            h = self.uni_hatch
            m = self.uni_marker

        ls = self.uni_ls

        if same_color:
            if "Bi" in s:
                ls = self.bibfs_ls
                ci += 1

        return self.colors[ci], ls, h, m


class SequentialStyles:
    def __init__(self):
        self.marker = ["o", "x"]
        self.ls = ["-", (0, (5, 6))]
        self.hatch = [None, "|||"]
        self.seq_colors = ["r", "g", "b", "c", "m", "y"]
        # lighter colors earlier in the list
        self.mi = 0
        self.lsi = 0
        self.hi = 0
        self.ci = 0

    def get_ls(self, agent: str, same_color: bool):
        color = self.seq_colors[self.ci]
        ls = self.ls[self.lsi]
        hatch = self.hatch[self.hi]
        m = self.marker[self.mi]

        self.mi = (self.mi + 1) % len(self.marker)
        self.lsi = (self.lsi + 1) % len(self.ls)
        self.hi = (self.hi + 1) % len(self.hatch)
        self.ci = (self.ci + 1) % len(self.seq_colors)

        return color, ls, hatch, m
