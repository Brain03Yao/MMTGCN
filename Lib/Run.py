import pickle
import argparse
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from GCN import Combine_MTGCN
from Triplet_Strategy import train_cv_whole_combine, test_cv_whole
from Graph_Construction import train_whole_lap


if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disables CUDA training.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=80,
                        help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--margin', type=float, default=5.0,
                        help='Margin of Loss.')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate (1 - keep probability).')
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    # Cross-validation
    # cv = StratifiedKFold(n_splits=5)

    # loading_data: {ID:Label}
    class_path = 'ASD/label/id_class.pkl'
    class_sub = pickle.load(open(class_path, 'rb'))

    data_id = np.asarray(list(class_sub.keys()))
    data_label = np.asarray(list(class_sub.values()))
    # train(80%) test(20%) split
    id_train, id_test, label_train, label_test = train_test_split(data_id, data_label, test_size=0.2)

    k_value = 10
    # network anchitecture
    device_ids = [1, 2]
    net = Combine_MTGCN(args.dropout)
    # net = net.cuda(device_ids[0])
    net = nn.DataParallel(net)
    criterion = torch.nn.MarginRankingLoss(margin=args.margin)
    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # split train and test data under their label
    train_patient = np.asarray(id_train)[[i for i, v in enumerate(label_train) if v]]
    train_hc = np.asarray(id_train)[[i for i, v in enumerate(label_train) if not v]]
    test_patient = np.asarray(id_test)[[i for i, v in enumerate(label_test) if v]]
    test_hc = np.asarray(id_test)[[i for i, v in enumerate(label_test) if not v]]

    # saved results
    train_loss_list = []
    train_acc_list = []
    test_loss_list = []
    test_acc_list = []
    test_sen_top = 0
    test_spe_top = 0
    test_acc_top = 0
    for epoch in range(1, args.epochs+1):
        train_whole_mat, train_whole_mat1, train_whole_mat2, train_whole_mat3\
            = train_whole_lap(train_patient, train_hc, k_value)
        len_train_patient = 0
        len_train_hc = 0
        batch_part = 16
        for tmp_batch in range(batch_part):
            tmp_patient = len_train_patient
            tmp_hc = len_train_hc

            len_train_patient += int(train_patient.shape[0] / batch_part)
            len_train_hc += int(train_hc.shape[0] / batch_part)
            new_train_patient = train_patient[tmp_patient:len_train_patient]
            new_train_hc = train_hc[tmp_hc:len_train_hc]

            train_loss, train_acc = train_cv_whole_combine(new_train_patient, new_train_hc,
                                                           net, criterion, optimizer, epoch,
                                                           test_matrix_lap=[train_whole_mat, train_whole_mat1,
                                                                            train_whole_mat2, train_whole_mat3],
                                                           part_flag=tmp_batch)

        train_loss_list.append(train_loss.data)
        train_acc_list.append(train_acc.data)

        len_test_patient = 0
        len_test_hc = 0
        batch_test = 20
        tmp_acc = 0
        tmp_sen = 0
        tmp_spe = 0
        tmp_loss = 0
        for tmp_batch in range(batch_test):
            tmp_patient = len_test_patient
            tmp_hc = len_test_hc

            len_test_patient += int(test_patient.shape[0] / batch_test)
            len_test_hc += int(test_hc.shape[0] / batch_test)
            new_test_patient = test_patient[tmp_patient:len_test_patient]
            new_test_hc = test_hc[tmp_hc:len_test_hc]
            test_loss, test_acc, test_sen, test_spe = test_cv_whole(
                new_test_patient, new_test_hc, net, [train_whole_mat, train_whole_mat1,
                                                     train_whole_mat2,train_whole_mat3],
                train_patient, train_hc, tmp_batch+1)
            tmp_acc += test_acc.data
            tmp_sen += test_sen
            tmp_spe += test_spe
            tmp_loss += test_loss.data
        print("=============The Epoch {}, test accuracy:{}, sen :{} and spe :{}============="
              .format(epoch, tmp_acc/batch_test, tmp_sen/batch_test, tmp_spe/batch_test))

        test_loss_list.append(tmp_loss/batch_test)
        test_acc_list.append(tmp_acc/batch_test)

        if test_acc_top == 0:
            test_acc_top = tmp_acc / batch_test
        else:
            if test_acc_top < tmp_acc / batch_test:
                test_acc_top = tmp_acc / batch_test
                test_sen_top = tmp_sen / batch_test
                test_spe_top = tmp_spe / batch_test

        x_range = range(1, args.epochs + 1)

