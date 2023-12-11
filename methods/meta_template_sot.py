from torch.autograd import Variable

from methods.meta_template import MetaTemplate
from methods.sot import SOT


class MetaTemplateSOT(MetaTemplate):
    def __init__(
        self,
        backbone,
        n_way,
        n_support,
        change_way=True,
        sinkhorn_iterations=10,
        sinkhorn_regularization=0.1
    ):
        super(MetaTemplateSOT, self).__init__(backbone, n_way, n_support, change_way)
        
        self.sot = SOT(sinkhorn_regularization=sinkhorn_regularization, sinkhorn_iterations=sinkhorn_iterations)

    def parse_feature(self, x, is_feature):
        if isinstance(x, list):
            x = [Variable(obj.to(self.device)) for obj in x]
        else: x = Variable(x.to(self.device))
        if is_feature:
            z_all = x
        else:
            if isinstance(x, list):
                x = [obj.contiguous().view(self.n_way * (self.n_support + self.n_query), *obj.size()[2:]) for obj in x]
            else: x = x.contiguous().view(self.n_way * (self.n_support + self.n_query), *x.size()[2:])
            
            x = self.feature.forward(x)
            x = self.sot(x)

            z_all = x.view(self.n_way, self.n_support + self.n_query, -1)
        z_support = z_all[:, :self.n_support]
        z_query = z_all[:, self.n_support:]

        return z_support, z_query

    def parse_feature_together(self, x, is_feature):
        if isinstance(x, list):
            x = [Variable(obj.to(self.device)) for obj in x]
        else: x = Variable(x.to(self.device))
        if is_feature:
            z_all = x
        else:
            if isinstance(x, list):
                x = [obj.contiguous().view(self.n_way * (self.n_support + self.n_query), *obj.size()[2:]) for obj in x]
            else: x = x.contiguous().view(self.n_way * (self.n_support + self.n_query), *x.size()[2:])

            x = self.feature.forward(x)
            x = self.sot(x)

            z_all = x.view(self.n_way, self.n_support + self.n_query, -1)

        return z_all
