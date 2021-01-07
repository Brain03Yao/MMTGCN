import pickle
import numpy as np
import scipy
import sklearn
from sklearn.metrics import confusion_matrix
import torch


def without_random_walk_train_whole(train_index, k_value, file_path):
    # calculate group-level adj matrix  from training set
    idx_mean, prob_mean = read_train_pkl_k_whole(train_index, k_value, file_path)
    mean_matrix_adj = adjacency_part(idx_mean)

    return mean_matrix_adj




def laplacian(W, normalized=True, normalized_track=True):
    # Return the Laplacian of the weighted matrix

    # Degree matrix.
    d = W.sum(axis=0)
    # print(d)
    # Laplacian matrix.
    if not normalized:
        print('L = D - W')
        # L = D - W
        D = scipy.sparse.diags(d.A.squeeze(), 0)
        L = D - W
    elif not normalized_track:
        # L = I - D^(-0.5)*W*D^(-0.5)
        # print('L = I - D^(-0.5)*W*D^(-0.5)')
        d += np.spacing(np.array(0, W.dtype))
        d = 1 / np.sqrt(d)
        D = scipy.sparse.diags(d.A.squeeze(), 0)
        I = scipy.sparse.identity(d.size, dtype=W.dtype)
        L = I - D * W * D
    else:
        # W` = W + I； D`(ii) = Sigma(j)(W`(ij))
        print('W` = W + I； D`(ii) = Sigma(j)(W`(ij))')
        I = scipy.sparse.identity(d.size, dtype=W.dtype)
        W_trick = W + I
        d_trick = W_trick.sum(axis=0)
        # try with or without this operation
        d_trick += np.spacing(np.array(0, W.dtype))
        d_trick = 1 / np.sqrt(d_trick)
        D = scipy.sparse.diags(d_trick.A.squeeze(), 0)
        L = D * W_trick * D

    # assert np.abs(L - L.T).mean() < 1e-9
    assert type(L) is scipy.sparse.csr.csr_matrix
    L = L.toarray()
    return L


def read_train_pkl_k_whole(train_index, k, file_path):
    # calculate each node's k-nearest neighbor
    label_path = '/data/label/id_class.pkl'
    matrix_pickle = pickle.load(open(file_path, 'rb'))
    pickle_label = pickle.load(open(label_path, 'rb'))

    asd_tmp_matrix = 0
    hc_tmp_matrix = 0
    asd_num = 0
    hc_num = 0
    for tmp_index in train_index:
        if pickle_label[tmp_index] == np.float64(1.0):
            if np.shape(asd_tmp_matrix) == ():
                asd_tmp_matrix = matrix_pickle[tmp_index]
            else:
                asd_tmp_matrix += matrix_pickle[tmp_index]
            asd_num += 1
        elif pickle_label[tmp_index] == np.float64(0.0):
            if np.shape(hc_tmp_matrix) == ():
                hc_tmp_matrix = matrix_pickle[tmp_index]
            else:
                hc_tmp_matrix += matrix_pickle[tmp_index]
            hc_num += 1

    hc_mean_matrix = hc_tmp_matrix / hc_num
    asd_mean_matrix = asd_tmp_matrix / asd_num
    whole_mean_matrix = (hc_mean_matrix + asd_mean_matrix) / 2
    d = sklearn.metrics.pairwise.pairwise_distances(whole_mean_matrix, metric='euclidean', n_jobs=-2)
    prob_d = np.exp(d*-1)

    idx = np.argsort(prob_d)[:, ::-1]
    prob_d.sort()
    prob_d = prob_d[:, ::-1]

    return idx[:, 1:k+1], prob_d[:, 1:k+1]


def read_test_pkl_k(test_index, k=10):

    file_path = 'data/BASC325.pkl'

    matrix_pickle = pickle.load(open(file_path, 'rb'))
    sub_matrix = matrix_pickle[test_index]

    d = sklearn.metrics.pairwise.pairwise_distances(sub_matrix, metric='euclidean', n_jobs=-2)
    prob_d = np.exp(d*-1)


    idx = np.argsort(prob_d)[:, ::-1]
    prob_d.sort()
    prob_d = prob_d[:, ::-1]

    return idx[:, 1:k+1], prob_d[:, 1:k+1]


def adjacency_part(idx):
    # Generate symmetric adjacent matrix
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



def without_random_walk_test(test_index):

    sub_idx_mean, sub_prob_mean = read_test_pkl_k(test_index)
    sub_matrix_adj = adjacency_part(sub_idx_mean)
    sub_matrix_lap = laplacian(sub_matrix_adj)

    return sub_matrix_lap


def loading_test_dic(sub_index, features_dic, lap_dic):
    # loading test features and adj matrix
    sub_features = []
    sub_lap = []
    for tmp_index in sub_index:
        sub_features.append(features_dic[tmp_index])
        sub_lap.append(lap_dic[tmp_index])

    return torch.Tensor(sub_features), torch.Tensor(sub_lap)


def train_whole_lap(patient_index, hc_index, k_neighbour):
    # generate group-level adj matrix with different atlas
    file_path = '/data/ABIDE_AAL116.pkl'
    file_path1 = '/data/ABIDE_CC200.pkl'
    file_path2 = '/data/ABIDE_BN273.pkl'
    file_path3 = '/data/ABIDE_BASC325.pkl'
    anchor_train_index = np.concatenate((patient_index, hc_index))
    adj_aal116 = without_random_walk_train_whole(anchor_train_index, k_neighbour, file_path)
    adj_cc200 = without_random_walk_train_whole(anchor_train_index, k_neighbour, file_path1)
    adj_bn273 = without_random_walk_train_whole(anchor_train_index, k_neighbour, file_path2)
    adj_basc325 = without_random_walk_train_whole(anchor_train_index, k_neighbour, file_path3)

    test_matrix_lap = laplacian(adj_aal116)
    test_matrix_lap1 = laplacian(adj_cc200)
    test_matrix_lap2 = laplacian(adj_bn273)
    test_matrix_lap3 = laplacian(adj_basc325)

    return test_matrix_lap, test_matrix_lap1, test_matrix_lap2, test_matrix_lap3


def load_features(sub_index, asd_adj, hc_adj):

    features_sub = pickle.load(open('/data/BASC325.pkl', 'rb'))
    hc_sub = pickle.load(open('/data/label/HC_ID.pkl', 'rb'))
    asd_sub = pickle.load(open('data/label/ASD_ID.pkl', 'rb'))

    sub_features = []
    sub_adj = []
    for tmp_index in sub_index:
        sub_features.append(features_sub[tmp_index])
        if tmp_index in hc_sub.keys():
            sub_adj.append(hc_adj)
        if tmp_index in asd_sub.keys():
            sub_adj.append(asd_adj)

    return torch.Tensor(sub_features), torch.Tensor(sub_adj)


def load_features_pkl(sub_index, whole_adj, pkl_path):
    # return features and adj matrix based on one template
    features_sub = pickle.load(open(pkl_path, 'rb'))

    sub_features = []
    for tmp_index in sub_index:
        tmp = torch.Tensor(features_sub[tmp_index])
        sub_features.append(tmp)
    sub_features = torch.stack(sub_features)
    sub_adj = torch.Tensor(whole_adj)
    tmp_len = len(sub_index)
    sub_adj = sub_adj.repeat(tmp_len, 1, 1)
    return sub_features, sub_adj


def load_features_whole(sub_index, whole_adj):
    # return features and adj matrix based on different template
    tmp_prefix = '/data/'
    file_path = tmp_prefix + 'AAL116.pkl'
    file_path1 = tmp_prefix + 'CC200.pkl'
    file_path2 = tmp_prefix + 'BN273.pkl'
    file_path3 = tmp_prefix + 'BASC325.pkl'
    features_sub = pickle.load(open(file_path, 'rb'))
    features_sub1 = pickle.load(open(file_path1, 'rb'))
    features_sub2 = pickle.load(open(file_path2, 'rb'))
    features_sub3 = pickle.load(open(file_path3, 'rb'))
    sub_features = []
    sub_features1 = []
    sub_features2 = []
    sub_features3 = []

    for tmp_index in sub_index:
        tmp = torch.Tensor(features_sub[tmp_index])
        sub_features.append(tmp)
        tmp1 = torch.Tensor(features_sub1[tmp_index])
        sub_features1.append(tmp1)
        tmp2 = torch.Tensor(features_sub2[tmp_index])
        sub_features2.append(tmp2)
        tmp3 = torch.Tensor(features_sub3[tmp_index])
        sub_features3.append(tmp3)

    sub_features = torch.stack(sub_features)
    sub_features1 = torch.stack(sub_features1)
    sub_features2 = torch.stack(sub_features2)
    sub_features3 = torch.stack(sub_features3)

    sub_adj, sub_adj1, sub_adj2, sub_adj3 \
        = torch.Tensor(whole_adj[0]), torch.Tensor(whole_adj[1]), \
                         torch.Tensor(whole_adj[2]), torch.Tensor(whole_adj[3])
    tmp_len = len(sub_index)
    sub_adj = sub_adj.repeat(tmp_len, 1, 1)
    sub_adj1 = sub_adj1.repeat(tmp_len, 1, 1)
    sub_adj2 = sub_adj2.repeat(tmp_len, 1, 1)
    sub_adj3 = sub_adj3.repeat(tmp_len, 1, 1)

    return sub_features, sub_features1, sub_features2, sub_features3, \
           sub_adj, sub_adj1, sub_adj2, sub_adj3



