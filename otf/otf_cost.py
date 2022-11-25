import numpy as np
import torch
from torch.nn.functional import logsigmoid
from scipy.special import logsumexp
from scipy.optimize import minimize
from scipy.spatial.distance import cdist


class OTFCost:
    def __init__(self,
                 fairness_notion,
                 nb_epochs=100,
                 margin_tol=1e-3,
                 constraint_tol=1e-3,
                 reg_strength=1e-3):
        self.fairness_notion = fairness_notion
        self.nb_epochs = nb_epochs
        self.margin_tol = margin_tol
        self.constraint_tol = constraint_tol
        self.reg_strength = reg_strength

    def collate_fn(self, batch):
        unprot_input, prot_input, target = zip(*batch)
        unprot_input = torch.stack(unprot_input)
        prot_input = torch.stack(prot_input)
        target = torch.stack(target)

        C = cdist(unprot_input.numpy(), unprot_input.numpy())
        matrix_min = np.min(C)
        matrix_max = np.max(C)
        C = (C - matrix_min) / (matrix_max - matrix_min)
        return unprot_input, prot_input, target, C

    def forward(self, logits, prot_input, target, C):
        probs = torch.sigmoid(logits)
        probs_detached = probs.detach()
        log_probs = logsigmoid(logits)
        log_probs_detached = log_probs.detach()

        G = self.fairness_notion.compute_G(prot_input, target).numpy()

        # If any columns have nan vals (e.g. if a group is not represented in this batch) then ignore those columns.
        G_not_nan_cols = np.logical_not(np.any(np.isnan(G), axis=0))
        G = G[:, G_not_nan_cols]

        final_lambdas, final_mus = self._fit(log_probs_detached.numpy(), C, G)

        # Tight lower bound:
        adj_lambdas, adj_mus, adj_nus = self._fit_adjustment(probs_detached.numpy(), log_probs_detached.numpy(), C, G)

        loss = (torch.from_numpy(final_lambdas - adj_lambdas).squeeze() * probs).mean()

        if self.reg_strength <= 0.01:
            # For higher reg_strength, the total moved mass is higher, which increases the magnitude of the loss.
            loss /= self.reg_strength
        else:
            # However, this extra mass is increasingly used just for smoothing, so we do not decrease the scale from a
            # certain threshold onwards.
            loss *= 100
        loss *= 100

        return loss

    def _fit(self, log_probs, C, G):
        n = log_probs.shape[0]
        nb_constraints = G.shape[1]

        lambdas = np.zeros((n, 1))
        mus = np.zeros((nb_constraints, 1))

        def mu_obj(new_mu, constraint_idx, C_lambdas_col_sums):
            mus_copy = mus.copy()
            mus_copy[constraint_idx] = new_mu

            P_col_sums = np.exp(1 / self.reg_strength * self._compute_G_terms(mus_copy, G)) * C_lambdas_col_sums

            obj = self.reg_strength * np.sum(P_col_sums)
            grad = np.sum(P_col_sums * G[:, constraint_idx])
            return obj, grad

        log_probs = log_probs[:, np.newaxis]
        prev_lambdas = lambdas.copy()
        for epoch in range(self.nb_epochs):
            # Update lambdas.
            new_lambdas = self._optimal_lambdas(mus, C, G, log_probs)
            lambdas = new_lambdas

            # Update mus one by one.
            for i in range(nb_constraints):
                col_sums = np.sum(np.exp(1 / self.reg_strength * (-C + lambdas)), axis=0, keepdims=True)
                result = minimize(mu_obj, jac=True, x0=0, args=(i, col_sums), tol=self.constraint_tol)
                if result.success:
                    mu_res = result.x
                    mus[i] = mu_res

            # Stopping criterion: L1 norm of the updates.
            # From: https://github.com/jeanfeydy/global-divergences/blob/master/common/sinkhorn_balanced.py
            if epoch > 0 and self.reg_strength * np.mean(np.abs((lambdas - prev_lambdas))) < self.margin_tol:
                break
            prev_lambdas = lambdas.copy()

        return lambdas, mus

    @staticmethod
    def _compute_G_terms(mus, G, nus=None):
        if nus is None:
            return np.dot(G, mus).T
        else:
            return np.dot(G, mus - nus).T

    def _compute_P(self, lambdas, mus, C, G, nus=False):
        C_lambdas_terms = -C + lambdas
        G_terms = self._compute_G_terms(mus, G, nus)
        P = np.exp(1 / self.reg_strength * (C_lambdas_terms + G_terms))
        return P

    def _optimal_lambdas(self, mus, C, G, log_probs, nus=None):
        using_torch = isinstance(log_probs, torch.Tensor)
        if len(log_probs.shape) == 1:
            if using_torch:
                log_probs = log_probs.reshape(-1 , 1)
            else:
                log_probs = log_probs[:, np.newaxis]

        G_sum = self._compute_G_terms(mus, G, nus)
        exponent = 1 / self.reg_strength * (-C + G_sum)

        if using_torch:
            exponent = torch.from_numpy(exponent).float()
            lse = torch.logsumexp(exponent, dim=1, keepdim=True)
        else:
            lse = logsumexp(exponent, axis=1, keepdims=True)
        lambdas = self.reg_strength * (log_probs - lse)
        return lambdas

    def _otf_cost(self, lambdas, probs, P):
        constraints_term = (lambdas.squeeze() * probs).sum()
        otf = constraints_term - self.reg_strength * np.sum(P)
        return otf

    def _fit_adjustment(self, probs, log_probs, C, G):
        gamma = np.abs(np.dot(probs, G))
        n = log_probs.shape[0]
        nb_constraints = G.shape[1]

        lambdas = np.zeros((n, 1))
        mus = np.zeros((nb_constraints, 1))
        nus = np.zeros((nb_constraints, 1))

        def constraint_obj(new_val, constraint_idx, mu_phase, C_lambdas_col_sums):
            if mu_phase:
                mus_copy = mus.copy()
                mus_copy[constraint_idx] = new_val
                nus_copy = nus
                grad_sign = -1
            else:
                nus_copy = nus.copy()
                nus_copy[constraint_idx] = new_val
                mus_copy = mus
                grad_sign = 1

            P_col_sums = np.exp(1 / self.reg_strength * self._compute_G_terms(mus_copy, G, nus_copy)) * C_lambdas_col_sums

            obj = gamma[constraint_idx] * new_val - self.reg_strength * np.sum(P_col_sums)
            grad = gamma[constraint_idx] + grad_sign * np.sum(P_col_sums * G[:, constraint_idx])
            return -obj, -grad  # Negative because we are maximizing

        log_probs = log_probs[:, np.newaxis]
        prev_lambdas = lambdas.copy()
        for epoch in range(self.nb_epochs):
            # Update lambdas.
            new_lambdas = self._optimal_lambdas(mus, C, G, log_probs, nus)
            lambdas = new_lambdas

            # Update mus and nus one by one.
            for optimizing_mu in [True, False]:
                for i in range(nb_constraints):
                    col_sums = np.sum(np.exp(1 / self.reg_strength * (-C + lambdas)), axis=0, keepdims=True)
                    result = minimize(constraint_obj, jac=True, x0=-0.001, args=(i, optimizing_mu, col_sums),
                                      tol=self.constraint_tol, bounds=[(None, 0)])
                    if result.success:
                        res = result.x
                        if optimizing_mu:
                            mus[i] = res
                        else:
                            nus[i] = res
                    else:
                        print("Mu iterations did not converge!")

            # Stopping criterion: L1 norm of the updates.
            # From: https://github.com/jeanfeydy/global-divergences/blob/master/common/sinkhorn_balanced.py
            if epoch > 0 and self.reg_strength * np.mean(np.abs((lambdas - prev_lambdas))) < self.margin_tol:
                break
            prev_lambdas = lambdas.copy()

        return lambdas, mus, nus

