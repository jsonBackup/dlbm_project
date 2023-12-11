# This code is modified from https://github.com/jakesnell/prototypical-networks 

import torch.nn as nn
from torch.autograd import Variable

from methods.meta_template_sot import MetaTemplateSOT
from methods.protonet import ProtoNet



class ProtoNetSOT(MetaTemplateSOT, ProtoNet):
    def __init__(self,
        backbone,
        n_way,
        n_support,
        sinkhorn_iterations=10,
        sinkhorn_regularization=0.1
    ):
        MetaTemplateSOT.__init__(self, backbone, n_way, n_support, sinkhorn_iterations=sinkhorn_iterations, sinkhorn_regularization=sinkhorn_regularization)
        self.loss_fn = nn.CrossEntropyLoss()

    