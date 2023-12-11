# This code is modified from https://github.com/jakesnell/prototypical-networks 

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
        ProtoNet.__init__(self, backbone, n_way, n_support)
        MetaTemplateSOT.__init__(self, backbone, n_way, n_support, sinkhorn_iterations, sinkhorn_regularization)

    