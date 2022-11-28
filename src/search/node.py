class SearchNode:
    def __init__(self, state, parent, action, g_cost):
        self.state = state
        self.parent = parent
        self.action = action
        self.g_cost = g_cost
        self.reverse_action = state.reverse_action[action]

    def __eq__(self, other):
        """
        Verify if two SearchNodes are identical by verifying the
         state in the nodes.
        """
        return self.state == other.state

    def __lt__(self, other):
        """
        less-than used by the heap
        """
        return self.g_cost < other.g_cost

    def __hash__(self):
        """
        Hash function used in the closed list
        """
        return self.state.__hash__()
