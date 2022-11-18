from models.conv_net import ConvNet, TwoHeadedConvNet, HeuristicConvNet


class ModelWrapper:
    def __init__(self):
        self.model = None

    def initialize(self, search_algorithm, two_headed_model=False):
        if (
            search_algorithm == "Levin"
            or search_algorithm == "LevinMult"
            or search_algorithm == "LevinStar"
            or search_algorithm == "PUCT"
        ):
            if two_headed_model:
                self.model = TwoHeadedConvNet((2, 2), 32, 4)
            else:
                self.model = ConvNet((2, 2), 32, 4)
        if search_algorithm == "AStar" or search_algorithm == "GBFS":
            self.model = HeuristicConvNet((2, 2), 32, 4)

    def predict(self, x):
        return self.model.predict(x)

    def train_with_memory(self, memory):
        return self.model.train_with_memory(memory)

    def save_weights(self, filepath):
        self.model.save_weights(filepath)

    def load_weights(self, filepath):
        self.model.load_weights(filepath).expect_partial()
