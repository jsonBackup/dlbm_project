import torch
import torch.nn as nn


class SOTTransform(nn.Module):
    def __init__(self, iter: int = 100, lamb: float = 0.1, alpha: float = 1e6):
        super().__init__()
        self.iter = iter
        self.lamb = lamb
        self.alpha = alpha

        # Doesn't matter, here just to make things consistent
        self.e = None

    def iterate_through_a_step(self, u_k: torch.tensor, v_k: torch.tensor, k_iter: torch.tensor):
        r_k = torch.matmul(torch.matmul(u_k, k_iter),
                           torch.matmul(v_k, self.e))

        # Now we diagonalize this for the next iteration
        u_k = torch.diag(1./r_k)

        # In theory it's e transposed, but it shouldn't matter with how matrix
        # multiplications are handled here with a vector... I don't think
        c_k = torch.matmul(torch.matmul(self.e, u_k),
                           torch.matmul(k_iter, v_k))

        v_k = torch.diag(1./c_k)

        return u_k, v_k

    def forward(self, feature_matrix: torch.tensor) -> torch.Tensor:

        if feature_matrix.dim() > 2:
            # If there are more than two dimensions, I just flatten it to be 2d
            temp_matrix = feature_matrix.flatten(1)
            magnitude = torch.norm(temp_matrix)
            # Unit normalize the vector
            normalized_feature = temp_matrix / magnitude
        else:

            magnitude = torch.norm(feature_matrix)
            # Unit normalize the vector
            normalized_feature = feature_matrix / magnitude

        d = 2*(1-torch.matmul(normalized_feature, torch.t(normalized_feature)))
        d_inf = d + self.alpha*torch.eye(d.shape[0])

        # Now we get the K that we optimize
        # W
        k_iter = torch.exp(-self.lamb * d_inf)
        #k_iter.retain_grad()

        # Here this should be upgraded to have an extra stopping criteria, but
        # this is good enough!

        # We need a version per iteration, this is good enough

        # Simplest explanation of code:
        # https://strathprints.strath.ac.uk/19685/1/skapp.pdf
        # https://fulkast.medium.com/the-sinkhorn-knopp-algorithm-without-proof-697c9af7df7

        u_k = torch.eye(d.shape[0])
        v_k = torch.eye(d.shape[0])

        # e being a tensor of one should be okay...
        # The multiplication should work in all cases

        # used for other steps, to save space internally
        self.e = torch.ones(d.shape[0])

        k_step = k_iter.clone()
        for i in range(self.iter):
            u_k, v_k = self.iterate_through_a_step(
                u_k, v_k, k_iter
            )
            # column_sums = k_step.sum(axis=0)
            # row_sums = k_step.sum(axis=1)
            # if (i % 2 )== 0:
            #     u_k *= torch.diag(1./ row_sums)
            # else:
            #     v_k *= torch.diag(1./ column_sums)

            # k_step = torch.matmul(u_k, torch.matmul(k_iter, v_k))

        res = torch.matmul(u_k, torch.matmul(k_iter, v_k))
        #res.retain_grad()

        # An attempt to clean up, because we end up using too much memory somewhere...
        #u_k, v_k = [], []

        res += torch.eye(res.shape[0])
        return res
