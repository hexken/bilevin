from models.conv_net import ConvNet, TwoHeadedConvNet, HeuristicConvNet
import torch as to


class ModelWrapper(to.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = None

    def initialize(self, in_channels, search_algorithm, two_headed_model=False):
        if (
            search_algorithm == "Levin"
            or search_algorithm == "LevinMult"
            or search_algorithm == "LevinStar"
            or search_algorithm == "PUCT"
        ):
            if two_headed_model:
                self.model = TwoHeadedConvNet(in_channels, (2, 2), 32, 4)
            else:
                self.model = ConvNet(in_channels, (2, 2), 32, 4)
        if search_algorithm == "AStar" or search_algorithm == "GBFS":
            self.model = HeuristicConvNet(in_channels, (2, 2), 32, 4)

    def forward(self, x):
        return self.model(x)

    def save_weights(self, filepath):
        to.save(self.model.state_dict(), filepath)

    def load_weights(self, filepath):
        self.model.load_state_dict(filepath)
