import torch
import torch.nn as nn


class SOT(nn.Module):

    def __init__(
            self,
            sinkhorn_regularization: float = 0.1,
            sinkhorn_iterations: int = 10,
    ):

        super().__init__()

        self.sinkhorn_iterations = sinkhorn_iterations
        self.sinkhorn_regularization = sinkhorn_regularization
        self._diagonal_value = 1e3
        self._eps = 1e-8

    def __call__(self, X: torch.Tensor, return_cost=False) -> torch.Tensor:

        if X.dim() > 2:
            X = X.transpose(0, -2)

        cost = self.compute_cost(X)
        cost_masked = self.mask_diagonal(cost, self._diagonal_value)

        sinkhorn = self.sinkhorn(cost_masked)
        sinkhorn = sinkhorn / sinkhorn.max(dim=-1, keepdim=True)[0]

        output = self.mask_diagonal(sinkhorn, 1)

        if output.dim() > 2:
            output = output.transpose(0, -2)

        if return_cost:
            return output, cost

        return output

    def sinkhorn(self, X):

        row_marginals = torch.ones(X.shape[:-1], requires_grad=False) / X.shape[-2]
        col_marginals = torch.ones(X.shape[:-1], requires_grad=False) / X.shape[-2]

        rows = torch.zeros_like(row_marginals, requires_grad=False)
        cols = torch.zeros_like(col_marginals, requires_grad=False)

        # Sinkhorn iterations
        for i in range(self.sinkhorn_iterations):
            rows_cost = (-X + rows.unsqueeze(-1) + cols.unsqueeze(-2)) / self.sinkhorn_regularization

            new_rows = torch.log(row_marginals + self._eps)
            new_rows -= torch.logsumexp(rows_cost, dim=-1)
            rows += self.sinkhorn_regularization * new_rows

            cols_cost = (-X + rows.unsqueeze(-1) + cols.unsqueeze(-2)) / self.sinkhorn_regularization

            new_cols = torch.log(col_marginals + self._eps)
            new_cols -= torch.logsumexp(cols_cost.transpose(-2, -1), dim=-1)
            cols += self.sinkhorn_regularization * new_cols

        transport_plan_log = (-X + rows.unsqueeze(-1) + cols.unsqueeze(-2)) / self.sinkhorn_regularization
        return torch.exp(transport_plan_log)

    @staticmethod
    def mask_diagonal(X, value):
        return X + value * torch.eye(X.shape[-1])

    @staticmethod
    def compute_cost(X):
        X = X / X.norm(dim=-1, keepdim=True)
        return 1 - X @ X.transpose(-1, -2)
