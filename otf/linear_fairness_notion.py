import torch
from torch.nn.functional import one_hot


class LinearFairnessNotion:
    def __init__(self, cond_on_target=False):
        self.cond_on_target = cond_on_target  # If True, this is basically Equalized Odds.

    def compute_G(self, prot_input, target):
        nb_constraints = prot_input.shape[0]
        if self.cond_on_target:
            target = one_hot(target, num_classes=2)
            inner_variables = torch.einsum("nl,nk->nlk", target, prot_input)
            # TODO: clean up reshape here
            overall_mean_term = (target / torch.sum(target, dim=0)).reshape(target.shape[0], target.shape[1], 1)
            nb_constraints *= 2
        else:
            inner_variables = prot_input
            overall_mean_term = 1 / prot_input.shape[0]

        inner_variable_means = torch.sum(inner_variables, dim=0)
        G = inner_variables / inner_variable_means - overall_mean_term
        G = G.reshape(-1, nb_constraints) # TODO is this reshape necessary?
        return G

