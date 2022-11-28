from models.conv_net import ConvNetSingle, ConvNetDouble, TwoHeadedConvNet, HeuristicConvNet
import torch as to


class ModelWrapper(to.nn.Module):
    def __init__(self, device=to.device("cpu")):
        super().__init__()
        self.model = None
        self.device = device
        self.num_actions = 4

    def initialize(self, in_channels, search_algorithm, two_headed_model=False):
        if (
            search_algorithm == "Levin"
            or search_algorithm == "LevinMult"
            or search_algorithm == "LevinStar"
            or search_algorithm == "PUCT"
            or search_algorithm == "BiLevin"
        ):
            if search_algorithm == "BiLevin":
                self.forward_model = ConvNetDouble(in_channels, (2, 2), 32, self.num_actions)
                self.backward_model = ConvNetDouble(in_channels, (2, 2), 32, self.num_actions)
            if two_headed_model:
                self.model = TwoHeadedConvNet(in_channels, (2, 2), 32, self.num_actions)
            else:
                self.model = ConvNetSingle(in_channels, (2, 2), 32, self.num_actions)
        if search_algorithm == "AStar" or search_algorithm == "GBFS":
            self.model = HeuristicConvNet(in_channels, (2, 2), 32, self.num_actions)

    def forward(self, x):
        if isinstance(x, tuple):
            current_state, goal_state = x
            return self.backward_model(current_state, goal_state)
        return self.forward_model(x)

    def save_weights(self, filepath):
        to.save(self.model.state_dict(), filepath)

    def load_weights(self, filepath, device=to.device("cpu")):
        self.model.load_state_dict(to.load(filepath, map_location=device))
