import numpy as np
import scipy
import pickle
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from mlxtend.feature_selection import ColumnSelector
from mlxtend.classifier import StackingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
import torch.nn as nn
from torch.nn.modules import Module
import torch.nn.functional as F
import argparse
import networkx as nx
from GCN import GCN

class TripletCNN(Module):
    """
    Triplet network with CNN
    """
    def __init__(self):
        super(TripletCNN, self).__init__()
        self.cnn1 = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=5, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(4),
            nn.Dropout2d(p=.2),

            nn.Conv2d(4, 8, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),
            nn.Dropout2d(p=.2),
        )
        self.fc1 = nn.Sequential(
            nn.Linear(8 * 30 * 30, 500),
            nn.ReLU(inplace=True),

            nn.Linear(500, 100),
            nn.ReLU(inplace=True),

            nn.Linear(100, 10)
        )

    def forward_once(self, x):
        batch_size = x.size(0)
        output = self.cnn1(x)
        output = output.view(batch_size, -1)
        output = self.fc1(output)

        return output

    def forward(self, x1, x2, x3):

        out1 = self.forward_once(x1)
        out2 = self.forward_once(x2)
        out3 = self.forward_once(x3)
        # output three embeddings
        dist_a = F.pairwise_distance(out1, out2, 2)
        dist_b = F.pairwise_distance(out1, out3, 2)
        # return out1, out2, out3
        return dist_a, dist_b


class SiameseGCN(Module):
    """
    Siamese network with GCN
    """
    def __init__(self, nfeat):
        super(SiameseGCN, self).__init__()
        self.gcn = GCN(nfeat)

    def forward_once(self, x, adj):
        # 单个NN
        return self.gcn(x, adj)

    def forward(self, x1, adj1, x2, adj2):
        out1 = self.forward_once(x1, adj1)
        out2 = self.forward_once(x2, adj2)
        dist_a = F.pairwise_distance(out1, out2, 2)
        return dist_a


class combined_MSGCN(Module):
    # combined multi SiameseGCN
    def __init__(self, dropout=0.5):
        super(combined_MSGCN, self).__init__()
        # ROI numbers on each template
        self.siamese1 = TripletGCN(116, dropout)
        self.siamese2 = TripletGCN(200, dropout)
        self.siamese3 = TripletGCN(264, dropout)
        self.siamese4 = TripletGCN(325, dropout)
        
    def forward(self, sub1a, adj1a, sub1b, adj1b, sub1c, adj1c, sub1d, adj1d,
                sub2a, adj2a, sub2b, adj2b, sub2c, adj2c, sub2d, adj2d):

        dist_a1 = self.siamese1(sub1a, adj1a, sub2a, adj2a)
        dist_b1 = self.siamese2(sub1b, adj1b, sub2b, adj2b)
        dist_c1 = self.siamese3(sub1c, adj1c, sub2c, adj2c)
        dist_d1 = self.siamese4(sub1d, adj1d, sub2d, adj2d)

        return dist_a1, dist_b1, dist_c1, dist_d1


class ContrastiveLoss(Module):
    def __init__(self, margin=3):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.loss = nn.BCELoss()

    def forward(self, output, label):
        label = label.view(label.size()[0])
        loss_same = label * torch.pow(output, 2)
        loss_diff = (1 - label) * torch.pow(torch.clamp(self.margin - output, min=0.0), 2)
        loss_contrastive = torch.mean(loss_same + loss_diff)

        return loss_contrastive


class Weighted_Siamese_Margin(Module):
    # weights for each TGCN
    def __init__(self, w1=0.25, w2=0.25, w3=0.2, w4=0.3):
        super(Weighted_Siamese_Margin, self).__init__()
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.w3 = w4

    def forward(self, outputa, outputb, outputc, outputd, sgcn_func):
        target = torch.cuda.FloatTensor(outputa1.size()).fill_(-1)
        loss = self.w1 * sgcn_func(outputa, target) + self.w2 * sgcn_func(outputb, target) \
               + self.w3 * sgcn_func(outputc, target) + self.w4 * sgcn_func(outputd, target)

        return loss



def train_whole_lap(file_path, anchor_train_index, k_neighbour):
    # loading CC features
    adj_template = without_random_walk_train_whole(anchor_train_index, k_neighbour, file_path)
    adj_template_g = nx.from_scipy_sparse_matrix(adj_template)
    adj_template_cc_features = list(nx.clustering(adj_template_g).values())

    return adj_template_cc_features


def adjacency_part(idx):
    # 根据索引(I,J)和值V构造全节点的临界矩阵，大于k近邻的节点连接为0
    # 返回扩展的K近邻(无连接权重)矩阵
    M, k = idx.shape
    I = np.arange(0, M).repeat(k)
    J = idx.reshape(M * k)
    V = np.ones(M * k)
    W = scipy.sparse.coo_matrix((V, (I, J)), shape=(M, M))

    # No self-connections.
    W.setdiag(0)

    # Non-directed graph, symmetric matrix
    bigger = W.T > W
    W = W - W.multiply(bigger) + W.T.multiply(bigger)

    assert W.nnz % 2 == 0
    assert np.abs(W - W.T).mean() < 1e-10
    assert type(W) is scipy.sparse.csr.csr_matrix
    return W


def load_features_pca(file_path, sub_index):
    # loading PCA_features

    features_sub = pickle.load(open(file_path, 'rb'))
    sub_features = []
    for tmp_index in sub_index:
        iu = np.triu_indices(273)
        print(features_sub[tmp_index][iu].shape)
        sub_features.append(features_sub[tmp_index][iu])

    return sub_features


def without_random_walk_train_whole(train_index, k_value, file_path):
    # Mean Functional Connectivity without Random Walk

    matrix_pickle = pickle.load(open(file_path, 'rb'))
    whole_mean_matrix = matrix_pickle[train_index]
    d = metrics.pairwise.pairwise_distances(whole_mean_matrix, metric='euclidean', n_jobs=-2)
    prob_d = np.exp(d * -1)
    idx = np.argsort(prob_d)[:, ::-1]
    idx_mean = idx[:, 1:k_value + 1]

    mean_matrix_adj = adjacency_part(idx_mean)

    return mean_matrix_adj


def cc_performance(file_list, id_train, id_test, label_train, label_test):
    # CC+RF performance
    top_list = []
    top_index = []
    for k_value in range(2, 100):
        train_features = []
        train_features1 = []
        train_features2 = []
        train_features3 = []
        test_features = []
        test_features1 = []
        test_features2 = []
        test_features3 = []

        file_path, file_path1, file_path2, file_path3 =\
            file_list[0], file_list[1], file_list[2], file_list[3]

        for tmp_id in id_train:
            # print(tmp_id, end="")
            tmp_feaures = train_whole_lap(file_path, tmp_id, k_value)
            tmp_feaures1 = train_whole_lap(file_path1, tmp_id, k_value)
            tmp_feaures2 = train_whole_lap(file_path2, tmp_id, k_value)
            tmp_feaures3 = train_whole_lap(file_path3, tmp_id, k_value)
            train_features.append(tmp_feaures)
            train_features1.append(tmp_feaures1)
            train_features2.append(tmp_feaures2)
            train_features3.append(tmp_feaures3)

        for tmp_id in id_test:
            # print(tmp_id, end="")
            tmp_feaures = train_whole_lap(file_path, tmp_id, k_value)
            tmp_feaures1 = train_whole_lap(file_path1, tmp_id, k_value)
            tmp_feaures2 = train_whole_lap(file_path2, tmp_id, k_value)
            tmp_feaures3 = train_whole_lap(file_path3, tmp_id, k_value)
            test_features.append(tmp_feaures)
            test_features1.append(tmp_feaures1)
            test_features2.append(tmp_feaures2)
            test_features3.append(tmp_feaures3)

        top = 0
        feature_len = np.shape(train_features)[1]
        feature_len1 = np.shape(train_features1)[1]
        feature_len2 = np.shape(train_features2)[1]
        feature_len3 = np.shape(train_features3)[1]
        train_features_nd  = np.concatenate((train_features, train_features1, train_features2, train_features3), axis=1)
        train_features = scale(train_features_nd)
        test_features_nd = np.concatenate((test_features, test_features1, test_features2, test_features3), axis=1)
        test_features = scale(test_features_nd)

        for c_value in np.arange(50, 200, 10):
            for tmp_depth in np.arange(1, 6):
                print("======================PCA component {}, N_estimators {} and Depth {}======================"
                      .format(k_value, c_value, tmp_depth))

                pipe1 = make_pipeline(ColumnSelector(cols=range(0, feature_len)),
                                      RandomForestClassifier(random_state=42, n_estimators=c_value, max_depth=tmp_depth))
                pipe2 = make_pipeline(ColumnSelector(cols=range(feature_len, feature_len+feature_len1)),
                                      RandomForestClassifier(random_state=42, n_estimators=c_value, max_depth=tmp_depth))
                pipe3 = make_pipeline(ColumnSelector(cols=range(feature_len+feature_len1,
                                                                feature_len+feature_len1+feature_len2)),
                                      RandomForestClassifier(random_state=42, n_estimators=c_value, max_depth=tmp_depth))
                pipe4 = make_pipeline(ColumnSelector(cols=range(feature_len+feature_len1+feature_len2,
                                                                feature_len+feature_len1+feature_len2+feature_len3)),
                                      RandomForestClassifier(random_state=42, n_estimators=c_value,
                                                             max_depth=tmp_depth))

                eclf = StackingClassifier(classifiers=[pipe1, pipe2, pipe3, pipe4], meta_classifier=LogisticRegression())

                eclf.fit(train_features, label_train)
                new_pred = eclf.predict(test_features)

                cm = confusion_matrix(label_test, new_pred)
                print(cm)
                tmp_acc = (cm[0, 0] + cm[1, 1]) / (cm[0, 0] + cm[1, 1] + cm[0, 1] + cm[1, 0])
                tmp_spe = cm[0, 0] / (cm[0, 0] + cm[0, 1])
                tmp_sen = cm[1, 1] / (cm[1, 1] + cm[1, 0])

                print('Acc: {}'.format(tmp_acc))
                print('Spe: {}'.format(tmp_spe))
                print('Sen: {}'.format(tmp_sen))

                if tmp_acc > top:
                    top = tmp_acc
                    top_list.append((tmp_acc, tmp_spe, tmp_sen))
                    top_index.append((k_value, c_value, tmp_depth))

    print('Top: {}'.format(top_list))
    print('Top: {}'.format(top_index))


def pca_performance(file_list, id_train, id_test, label_train, label_test, k_value):
    # PCA+RF performance
    file_path, file_path1, file_path2, file_path3 = \
        file_list[0], file_list[1], file_list[2], file_list[3]

    train_features = []
    train_features1 = []
    train_features2 = []
    train_features3 = []
    test_features = []
    test_features1 = []
    test_features2 = []
    test_features3 = []

    for tmp_id in id_train:
        # print(tmp_id, end="")
        tmp_feaures = load_features_pca(file_path, tmp_id)
        tmp_feaures1 = load_features_pca(file_path1, tmp_id)
        tmp_feaures2 = load_features_pca(file_path2, tmp_id)
        tmp_feaures3 = load_features_pca(file_path3, tmp_id)
        train_features.append(tmp_feaures)
        train_features1.append(tmp_feaures1)
        train_features2.append(tmp_feaures2)
        train_features3.append(tmp_feaures3)

    for tmp_id in id_test:
        # print(tmp_id, end="")
        tmp_feaures = load_features_pca(file_path, tmp_id)
        tmp_feaures1 = load_features_pca(file_path1, tmp_id)
        tmp_feaures2 = load_features_pca(file_path2, tmp_id)
        tmp_feaures3 = load_features_pca(file_path3, tmp_id)
        test_features.append(tmp_feaures)
        test_features1.append(tmp_feaures1)
        test_features2.append(tmp_feaures2)
        test_features3.append(tmp_feaures3)


    for k_value in range(1, 20):
        for c_value in np.arange(50, 200, 10):
            for tmp_depth in np.arange(1, 6, 2):
                print("======================PCA component {}, N_estimators {} and Depth {}======================"
                      .format(k_value, c_value, tmp_depth))
                pca = PCA(n_components=k_value)
                pca.fit(train_features)
                pca1 = PCA(n_components=k_value)
                pca1.fit(train_features1)
                pca2 = PCA(n_components=k_value)
                pca2.fit(train_features1)
                pca3 = PCA(n_components=k_value)
                pca3.fit(train_features1)
                train_t_features = pca.fit_transform(train_features)
                train_t_features1 = pca.fit_transform(train_features1)
                train_t_features2 = pca.fit_transform(train_features2)
                train_t_features3 = pca.fit_transform(train_features3)
                test_t_features = pca.fit_transform(test_features)
                test_t_features1 = pca.fit_transform(test_features1)
                test_t_features2 = pca.fit_transform(test_features2)
                test_t_features3 = pca.fit_transform(test_features3)

                clf = RandomForestClassifier(random_state=42, n_estimators=c_value, max_depth=tmp_depth)
                clf1 = RandomForestClassifier(random_state=42, n_estimators=c_value, max_depth=tmp_depth)
                clf2 = RandomForestClassifier(random_state=42, n_estimators=c_value, max_depth=tmp_depth)
                clf3 = RandomForestClassifier(random_state=42, n_estimators=c_value, max_depth=tmp_depth)

                clf.fit(train_t_features, label_train)
                clf1.fit(train_t_features1, label_train)
                clf2.fit(train_t_features2, label_train)
                clf3.fit(train_t_features3, label_train)

                pred_list = clf.predict_proba(test_t_features)
                pred_list1 = clf1.predict_proba(test_t_features1)
                pred_list2 = clf2.predict_proba(test_t_features2)
                pred_list3 = clf3.predict_proba(test_t_features3)
                new_pred_p = 0.25 * pred_list + 0.25 * pred_list1 + 0.2 * pred_list2 + 0.3 * pred_list3
                new_pred = [0 if i[0] > i[1] else 1 for i in new_pred_p]
                cm = confusion_matrix(label_test, new_pred)
                print(cm)
                tmp_acc = (cm[0, 0] + cm[1, 1]) / (cm[0, 0] + cm[1, 1] + cm[0, 1] + cm[1, 0])
                tmp_spe = cm[0, 0] / (cm[0, 0] + cm[0, 1])
                tmp_sen = cm[1, 1] / (cm[1, 1] + cm[1, 0])
                print('Acc: {}'.format(tmp_acc))
                print('Spe: {}'.format(tmp_spe))
                print('Sen: {}'.format(tmp_sen))
                if tmp_acc > top and tmp_spe > 0.25:
                    top = tmp_acc
                    top_list = []
                    top_index = []
                    top_list.append(tmp_acc)
                    top_list.append(tmp_spe)
                    top_list.append(tmp_sen)
                    top_index.append((k_value, c_value))
    print('Top: {}'.format(top_list))
    print('Top: {}'.format(top_index))



if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    args = parser.parse_args()
    np.random.seed(args.seed)

    # loading_data: {ID:Label}
    file_path = 'ADHD/AAL116_correlation_new.pkl'
    file_path1 = 'ADHD/CC200_correlation_new.pkl'
    file_path2 = 'ADHD/BN274_correlation_new.pkl'
    file_path3 = 'ADHD/BASC327_correlation_new.pkl'
    class_sub = pickle.load(open('ADHD/label/id_class.pkl', 'rb'))

    data_id = np.asarray(list(class_sub.keys()))
    data_label = np.asarray(list(class_sub.values()))
    # train(80%) test(20%) split
    id_train, id_test, label_train, label_test = train_test_split(data_id, data_label, test_size=0.2)

    print(id_test)
    print(label_test)




