# This code is modified from https://github.com/jakesnell/prototypical-networks 

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

from methods.meta_template import MetaTemplate

from sot_implementations.sot_1 import SOTTransform
from sot_implementations.sot_2 import SOT


class ProtoNetSOT(MetaTemplate):
    def __init__(self, backbone, n_way, n_support, sot_type: int = 1, iter_count: int = 100, lamb: float = 0.1):
        super(ProtoNetSOT, self).__init__(backbone, n_way, n_support)
        self.loss_fn = nn.CrossEntropyLoss()

        self.sot = None

        if sot_type == 1:
            self.sot = SOTTransform(iter=iter_count, lamb=lamb)
        elif sot_type == 2:
            self.sot = SOT(sinkhorn_regularization=lamb, sinkhorn_iterations=iter_count)

    def set_forward(self, x, is_feature=False):
        z_all = self.parse_feature_together(x, is_feature)

        # The system does SOT on support and query... which like why????
        # That feels like you're getting part of the result???? So like... no???/
        if self.sot is not None:
            # We join the two back up, and then split them again!
            z_all = self.sot(z_all)

        z_support, z_query = z_all[:, :self.n_support], z_all[:, self.n_support:]

        z_support = z_support.contiguous()
        z_proto = z_support.view(self.n_way, self.n_support, -1).mean(1)  # the shape of z is [n_data, n_dim]
        z_query = z_query.contiguous().view(self.n_way * self.n_query, -1)

        dists = euclidean_dist(z_query, z_proto)
        scores = -dists
        return scores


    def set_forward_loss(self, x):
        y_query = torch.from_numpy(np.repeat(range( self.n_way ), self.n_query ))
        y_query = Variable(y_query.cuda())

        scores = self.set_forward(x)

        return self.loss_fn(scores, y_query )


def euclidean_dist( x, y):
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)
