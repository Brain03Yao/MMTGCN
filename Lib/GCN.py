import torch
import math
from torch.nn.modules import Module
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix

class Chebyshev_GL(Module):
    """
    GCN k-hop Layers
    x' = Sigma^k-1 (Z^k * w0^k), Z^k= polynomial
    """
    def __init__(self, in_features, out_features, k_hop, bias=True):
        super(Chebyshev_GL, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.k_hop = k_hop
        self.weight = Parameter(torch.cuda.FloatTensor(k_hop, in_features, out_features))

        if bias:
            self.bias = Parameter(torch.cuda.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        # BN operations
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, laplacian):
        # Polynomial Achieved
        # Laplacian must be rescale
        tx_0 = input
        out = torch.matmul(input, self.weight[0])

        if self.weight.size(0) > 1:
            tx_1 = torch.matmul(laplacian, input)
            out = out + torch.matmul(tx_1, self.weight[1])

        for k in range(2, self.weight.size(0)):
            tx_2 = 2 * torch.matmul(laplacian, tx_1) - tx_0
            out = out + torch.matmul(tx_2, self.weight[k])

        if self.bias is not None:
            out = out + self.bias

        return out

    def __repr__(self):
        # print layer's structure
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + 'with K={' + str(self.weight.size(0)) + '})'


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    Z = f(X, A) = softmax(A` * ReLU(A` * X * W0)* W1)
    A` = D'^(-0.5) * A * D'^(-0.5)
    """

    def __init__(self, in_features, out_features, bias=True):
        # input
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.cuda.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.cuda.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        # BN operations
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        # A` * X * W
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        # print layer's structure
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN(Module):
    """
    simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, nfeat, dropout):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nfeat)
        self.gc2 = GraphConvolution(nfeat, nfeat)
        self.gc3 = GraphConvolution(nfeat, nfeat)
        self.dropout = dropout
        # self.classify = nn.Linear(nfeat, 1)

    def forward(self, x, adj):
        batch_size = adj.size(0)
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc2(x, adj))
        x = self.gc3(x, adj)
        out = x.view(batch_size, -1)
        return out


class TripletGCN(Module):
    """
    Triplet network with GCN
    This GCN without any FC layer
    """
    def __init__(self, nfeat, dropout):
        super(TripletGCN, self).__init__()
        self.gcn = GCN(nfeat, dropout)

    def forward_once(self, x, adj):
        # 单个NN
        return self.gcn(x, adj)

    def forward(self, x1, adj1, x2, adj2, x3, adj3):
        out1 = self.forward_once(x1, adj1)
        out2 = self.forward_once(x2, adj2)
        out3 = self.forward_once(x3, adj3)
        # 输出三个embedding
        dist_a = F.pairwise_distance(out1, out2, 2)
        dist_b = F.pairwise_distance(out1, out3, 2)
        # return out1, out2, out3
        return dist_a, dist_b


class Weighted_Margin(Module):
     # weights for each TGCN
    def __init__(self, w1=0.25, w2=0.25, w3=0.2, w4=0.3):
        super(Weighted_Margin, self).__init__()
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.w3 = w4

    def forward(self, outputa1, outputa2, outputb1, outputb2,
                outputc1, outputc2, outputd1, outputd2, tgcn_func):

        target = torch.cuda.FloatTensor(outputa1.size()).fill_(-1)
        loss = self.w1 * tgcn_func(outputa1, outputa2, target) + self.w2 * tgcn_func(outputb1, outputb2, target) \
            + self.w3 * tgcn_func(outputc1, outputc2, target) + self.w4 * tgcn_func(outputd1, outputd2, target)

        return loss


class Combine_MTGCN(Module):
    # combined multi TripletGCN
    def __init__(self, dropout=0.5):
        super(Combine_MTGCN, self).__init__()
        # ROI numbers on each template
        self.triplet1 = TripletGCN(116, dropout)
        self.triplet2 = TripletGCN(200, dropout)
        self.triplet3 = TripletGCN(264, dropout)
        self.triplet4 = TripletGCN(325, dropout)

    def forward(self, sub1a, adj1a, sub1b, adj1b, sub1c, adj1c, sub1d, adj1d,
                sub2a, adj2a, sub2b, adj2b, sub2c, adj2c, sub2d, adj2d,
                sub3a, adj3a, sub3b, adj3b, sub3c, adj3c, sub3d, adj3d):
        dist_a1, dist_a2 = self.triplet1(sub1a, adj1a, sub2a, adj2a, sub3a, adj3a)
        dist_b1, dist_b2 = self.triplet2(sub1b, adj1b, sub2b, adj2b, sub3b, adj3b)
        dist_c1, dist_c2 = self.triplet3(sub1c, adj1c, sub2c, adj2c, sub3c, adj3c)
        dist_d1, dist_d2 = self.triplet3(sub1d, adj1d, sub2d, adj2d, sub3d, adj3d)

        return dist_a1, dist_a2, dist_b1, dist_b2, dist_c1, dist_c2, dist_d1, dist_d2


def accuracy_single(dista, distb, margin=5.0):
    # distb represent the distance between anchor and negative sample
    pred = (distb - dista - margin).cpu().data
    pred = pred.view(pred.numel())
    pred = (pred > 0).float()
    return pred.sum()*1.0/pred.numel()


def accuracy_single_acc(dista, distb, margin, label_ndarray):
    # distb represent the distance between anchor and negative sample
    pred = (distb - dista - margin).cpu().data
    pred = pred.view(pred.numel())
    pred = (pred > 0).float()
    pred_dist = pred.tolist()
    pred_list = []
    for i, v in enumerate(pred_dist):
        if v:
            pred_list.append(label_ndarray[i])
        else:
            pred_list.append(1 - label_ndarray[i])
    cm = confusion_matrix(label_ndarray, pred_list)
    # ASD=1, HC=0
    sen = cm[1, 1] / (cm[1, 0] + cm[1, 1])
    spe = cm[0, 0] / (cm[0, 0] + cm[0, 1])

    return pred.sum()*1.0/pred.numel(), sen, spe


def accuracy_fusion(dista1, dista2, distb1, distb2, distc1, distc2, distd1, distd2, margin=5.0):
    # distb represent the distance between anchor and negative sample
    pred = (weighted1 * (dista2 - dista1 - margin) + weighted2 * (distb2 - distb1 - margin) +
            weighted3 * (distc2 - distc1 - margin) + weighted4 * (distd2 - distd1 - margin)).cpu().data
    pred = pred.view(pred.numel())
    pred = (pred > 0).float()
    return pred.sum()*1.0/pred.numel()


def accuracy_fusion_acc(dista1, dista2, distb1, distb2, distc1, distc2, distd1, distd2, label_ndarray, margin=5.0):
    # distb represent the distance between anchor and negative sample
    pred = (weighted1 * (dista2 - dista1 - margin) + weighted2 * (distb2 - distb1 - margin) +
            weighted3 * (distc2 - distc1 - margin) + weighted4 * (distd2 - distd1 - margin)).cpu().data
    pred = pred.view(pred.numel())
    pred = (pred > 0).float()
    pred_dist = pred.tolist()
    pred_list = []
    for i, v in enumerate(pred_dist):
        if v:
            pred_list.append(label_ndarray[i])
        else:
            pred_list.append(1 - label_ndarray[i])
    cm = confusion_matrix(label_ndarray, pred_list)
    # ASD=1, HC=0
    sen = cm[1, 1] / (cm[1, 0] + cm[1, 1])
    spe = cm[0, 0] / (cm[0, 0] + cm[0, 1])
    return pred.sum() * 1.0 / pred.numel(), sen, spe
