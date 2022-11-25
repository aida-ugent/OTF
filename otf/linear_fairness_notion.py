import torch
from torch.nn.functional import one_hot


class LinearFairnessNotion:
    def compute_G(self, *args):
        raise NotImplementedError


class ProbabilisticDemographicParity(LinearFairnessNotion):
    def compute_G(self, prot_input, *args):
        """
        Compute constraints matrix for Probabilistic Demographic Parity.
        :param prot_input: (N, prot_dim) tensor of protected/sensitive values where categorical sensitive values are
        one-hot encoded.
        """
        inner_variable_means = torch.sum(prot_input, dim=0)
        G = prot_input / inner_variable_means - 1 / prot_input.shape[0]
        return G


class ProbabilisticEqualizedOdds(LinearFairnessNotion):
    def compute_G(self, prot_input, target, *args):
        """
        Compute constraints matrix for Probabilistic Demographic Parity.
        :param prot_input: (N, prot_dim) tensor of protected/sensitive values where categorical sensitive values are
        one-hot encoded.
        :param target: (N,) target label for each prediction.
        """

        target = one_hot(target, num_classes=2)
        inner_variables = torch.einsum("nl,nk->nlk", target, prot_input)

        overall_mean_term = (target / torch.sum(target, dim=0)).unsqueeze(-1)
        inner_variable_means = torch.sum(inner_variables, dim=0)
        G = inner_variables / inner_variable_means - overall_mean_term

        # Put all constraints on the same axis.
        G = G.reshape(prot_input.shape[0], -1)
        return G

