import torch as to
import torch.nn.functional as F


def levin_loss(trajectory, model):
    actions = to.tensor(trajectory.actions)

    states = to.stack(trajectory.states)
    logits = model(states)

    loss = F.cross_entropy(logits, actions)
    loss *= to.tensor(trajectory.num_expanded)

    return loss


def improved_levin_loss(trajectory, model):
    actions_one_hot = F.one_hot(trajectory.actions, model.actions)

    states = to.tensor(trajectory.states)
    logits = model(states)

    loss = F.cross_entropy(actions_one_hot, logits)

    d = len(trajectory.actions) + 1
    pi = trajectory.get_solution_pi()
    num_expanded = trajectory.non_normalized_num_expanded + 1

    a = 0
    if pi < 1.0:
        a = (to.log((d + 1) / num_expanded)) / to.log(pi)
    if a < 0:
        a = 0

    loss *= to.tensor(num_expanded * a)

    return loss


def mse_loss(trajectory, model):
    states = to.tensor(trajectory.states)
    h = model(states)
    cost_to_gos = to.tensor(trajectory.cost_to_gos).unsqueeze(1)
    loss = F.mse_loss(h, cost_to_gos)

    return loss


def cross_entropy_loss(trajectory, model):
    actions_one_hot = F.one_hot(trajectory.actions, model.actions)

    states = to.tensor(trajectory.states)
    logits = model(states)
    loss = F.cross_entropy(actions_one_hot, logits)

    return loss


# class LossFunction(ABC):
#     def compute_loss(self, trajectory, model):
#         pass


# class ImprovedLevinMSELoss(LossFunction):
#     def __init__(self):
#         self.cross_entropy_loss = tf.keras.losses.CategoricalCrossentropy(
#             from_logits=True
#         )
#         self.mse = tf.keras.losses.MeanSquaredError()

#     def compute_loss(self, trajectory, model):
#         states = [s.state_tensor() for s in trajectory.states]
#         actions_one_hot = tf.one_hot(
#             trajectory.actions, model.actions
#         )
#         _, _, logits_pi, logits_h = model(np.array(states))

#         weights = model.get_weights()
#         weights_l2_norm = 0
#         for w in weights:
#             weights_l2_norm += tf.norm(w, ord=2)

#         loss = self.cross_entropy_loss(actions_one_hot, logits_pi)

#         d = len(trajectory.actions) + 1
#         pi = trajectory.get_solution_pi()
#         expanded = trajectory.non_normalized_num_expanded + 1

#         a = 0
#         if pi < 1.0:
#             a = (math.log((d + 1) / expanded)) / math.log(pi)
#         if a < 0:
#             a = 0

#         loss *= tf.convert_to_tensor(expanded * a, dtype=tf.float64)

#         solution_costs_tf = tf.expand_dims(
#             tf.convert_to_tensor(trajectory.get_solution_costs(), dtype=tf.float64), 1
#         )
#         loss += (
#             self.mse(solution_costs_tf, logits_h) + model._reg_const * weights_l2_norm
#         )

#         return loss


# class RegLevinMSELoss(LossFunction):
#     def __init__(self):
#         self.mse = tf.keras.losses.MeanSquaredError()

#     def compute_loss(self, trajectory, model):
#         states = [s.state_tensor() for s in trajectory.states]
#         actions_one_hot = tf.one_hot(
#             trajectory.actions, model.actions
#         )
#         _, probs_softmax, _, logits_h = model(np.array(states))

#         probs_used_on_path = tf.math.multiply(
#             tf.cast(actions_one_hot, dtype=tf.float64), probs_softmax
#         )
#         probs_used_on_path = tf.math.reduce_sum(probs_used_on_path, axis=1)
#         solution_prob = tf.math.reduce_prod(probs_used_on_path)

#         weights = model.get_weights()
#         weights_l2_norm = 0
#         for w in weights:
#             weights_l2_norm += tf.norm(w, ord=2)

#         solution_costs = trajectory.get_solution_costs()
#         solution_cost = solution_costs[len(solution_costs) - 1]

#         loss = tf.math.divide(
#             tf.convert_to_tensor(solution_cost, dtype=tf.float64), solution_prob
#         )

#         solution_costs_tf = tf.expand_dims(
#             tf.convert_to_tensor(solution_costs, dtype=tf.float64), 1
#         )
#         loss += (
#             self.mse(solution_costs_tf, logits_h) + model._reg_const * weights_l2_norm
#         )

#         return loss


# class Reglevin_loss(LossFunction):
#     def compute_loss(self, trajectory, model):
#         states = [s.state_tensor() for s in trajectory.states]
#         actions_one_hot = tf.one_hot(
#             trajectory.actions, model.actions
#         )
#         _, probs_softmax, _ = model(np.array(states))

#         probs_used_on_path = tf.math.multiply(
#             tf.cast(actions_one_hot, dtype=tf.float64), probs_softmax
#         )
#         probs_used_on_path = tf.math.reduce_sum(probs_used_on_path, axis=1)
#         solution_prob = tf.math.reduce_prod(probs_used_on_path)

#         weights = model.get_weights()
#         weights_l2_norm = 0
#         for w in weights:
#             weights_l2_norm += tf.norm(w, ord=2)

#         solution_costs = trajectory.get_solution_costs()
#         solution_cost = solution_costs[len(solution_costs) - 1]

#         loss = tf.math.divide(
#             tf.convert_to_tensor(solution_cost, dtype=tf.float64), solution_prob
#         )
#         loss += model._reg_const * weights_l2_norm

#         return loss


# class CrossEntropyMSELoss(LossFunction):
#     def __init__(self):
#         self.cross_entropy_loss = tf.keras.losses.CategoricalCrossentropy(
#             from_logits=True
#         )
#         self.mse = tf.keras.losses.MeanSquaredError()

#     def compute_loss(self, trajectory, model):
#         states = [s.state_tensor() for s in trajectory.states]
#         actions_one_hot = tf.one_hot(
#             trajectory.actions, model.actions
#         )
#         _, _, logits_pi, logits_h = model(np.array(states))

#         weights = model.get_weights()
#         weights_l2_norm = 0
#         for w in weights:
#             weights_l2_norm += tf.norm(w, ord=2)

#         loss = self.cross_entropy_loss(actions_one_hot, logits_pi)

#         solution_costs_tf = tf.expand_dims(
#             tf.convert_to_tensor(trajectory.get_solution_costs(), dtype=tf.float64), 1
#         )
#         loss += (
#             self.mse(solution_costs_tf, logits_h) + model._reg_const * weights_l2_norm
#         )

#         return loss


# class LevinMSELoss(LossFunction):
#     def __init__(self):
#         self.cross_entropy_loss = tf.keras.losses.CategoricalCrossentropy(
#             from_logits=True
#         )
#         self.mse = tf.keras.losses.MeanSquaredError()

#     def compute_loss(self, trajectory, model):
#         states = [s.state_tensor() for s in trajectory.states]
#         actions_one_hot = tf.one_hot(
#             trajectory.actions, model.actions
#         )
#         _, _, logits_pi, logits_h = model(np.array(states))

#         weights = model.get_weights()
#         weights_l2_norm = 0
#         for w in weights:
#             weights_l2_norm += tf.norm(w, ord=2)

#         loss = self.cross_entropy_loss(actions_one_hot, logits_pi)

#         loss *= tf.convert_to_tensor(
#             trajectory.non_normalized_num_expanded, dtype=tf.float64
#         )

#         solution_costs_tf = tf.expand_dims(
#             tf.convert_to_tensor(trajectory.get_solution_costs(), dtype=tf.float64), 1
#         )
#         loss += (
#             self.mse(solution_costs_tf, logits_h) + model._reg_const * weights_l2_norm
#         )

#         return loss
