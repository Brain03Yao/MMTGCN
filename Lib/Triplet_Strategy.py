import numpy as np
import torch
import copy
import argparse
from GCN import accuracy_single, accuracy_single_acc, accuracy_new, accuracy_new_add, Weighted_Margin
from Graph_Construction import load_features_whole, load_features_pkl
import torch.nn as nn


def triplet_template_cv(patients, hc):
    # sampling numbers
    triplet_array_training = np.random.permutation(patients + hc)
    return triplet_array_training


def triplet_template_cv_m(patients, hc, sampling_number, train_data=True):
    # based on sampling, we will generate anchor, positive and negative subjects
    triplet_list_training = []
    triplet_list_test = []
    if train_data:
        # sampling training_triplet
        for sub_anchor in patients:
            positive_list = copy.deepcopy(patients.tolist())
            # without self
            positive_list.remove(sub_anchor)
            negative_list = hc
            pos_select = np.random.choice(positive_list, size=sampling_number)
            neg_select = np.random.choice(negative_list, size=sampling_number)
            tmp_paired = [(sub_anchor, p, n)for p, n in zip(pos_select, neg_select)]
            triplet_list_training.extend(tmp_paired)

        for sub_anchor in hc:
            positive_list = copy.deepcopy(hc.tolist())
            # without self
            positive_list.remove(sub_anchor)
            negative_list = patients
            pos_select = np.random.choice(positive_list, size=sampling_number)
            neg_select = np.random.choice(negative_list, size=sampling_number)
            tmp_paired = [(sub_anchor, p, n)for p, n in zip(pos_select, neg_select)]
            triplet_list_training.extend(tmp_paired)

        triplet_array_training = np.asarray(triplet_list_training)

        # anchor, positive and negative
        return triplet_array_training[:, 0], triplet_array_training[:, 1], triplet_array_training[:, 2]
    else:
        # sampling test_triplet
        for sub_anchor in patients:
            positive_list = copy.deepcopy(patients.tolist())
            # without self
            positive_list.remove(sub_anchor)
            negative_list = hc
            pos_select = np.random.choice(positive_list, size=sampling_number)
            neg_select = np.random.choice(negative_list, size=sampling_number)
            tmp_paired = [(sub_anchor, p, n)for p, n in zip(pos_select, neg_select)]
            triplet_list_test.extend(tmp_paired)

        for sub_anchor in hc:
            positive_list = copy.deepcopy(hc.tolist())
            # without self
            positive_list.remove(sub_anchor)
            negative_list = patients
            pos_select = np.random.choice(positive_list, size=sampling_number)
            neg_select = np.random.choice(negative_list, size=sampling_number)
            tmp_paired = [(sub_anchor, p, n)for p, n in zip(pos_select, neg_select)]
            triplet_list_test.extend(tmp_paired)

        triplet_array_test = np.asarray(triplet_list_test)
        # anchor, positive and negative
        return triplet_array_test[:, 0], triplet_array_test[:, 1], triplet_array_test[:, 2]


def triplet_template_cv_test_m(patients, hc, sampling_number, patients_train, hc_train):
    # positive and negative subjects for test data

    triplet_list_test = []
    # sampling test_triplet
    for sub_anchor in patients:
        positive_list = copy.deepcopy(patients_train.tolist())
        negative_list = hc_train
        pos_select = np.random.choice(positive_list, size=sampling_number)
        neg_select = np.random.choice(negative_list, size=sampling_number)
        tmp_paired = [(sub_anchor, p, n)for p, n in zip(pos_select, neg_select)]
        triplet_list_test.extend(tmp_paired)

    for sub_anchor in hc:
        positive_list = copy.deepcopy(hc_train.tolist())
        negative_list = patients_train
        pos_select = np.random.choice(positive_list, size=sampling_number)
        neg_select = np.random.choice(negative_list, size=sampling_number)
        tmp_paired = [(sub_anchor, p, n)for p, n in zip(pos_select, neg_select)]
        triplet_list_test.extend(tmp_paired)

    triplet_array_test = np.asarray(triplet_list_test)
    # anchor, positive and negative
    return triplet_array_test[:, 0], triplet_array_test[:, 1], triplet_array_test[:, 2]


def triplet_template_cv_newtest(patients, hc, sampling_number, patients_train, hc_train):

    triplet_list_test = []
    # sampling test_triplet
    for sub_anchor in patients:
        positive_list = copy.deepcopy(patients_train.tolist())
        negative_list = hc_train
        pos_select = np.random.choice(positive_list, size=sampling_number)
        neg_select = np.random.choice(negative_list, size=sampling_number)
        tmp_paired = [(sub_anchor, p, n)for p, n in zip(pos_select, neg_select)]
        triplet_list_test.extend(tmp_paired)

    for sub_anchor in hc:
        positive_list = copy.deepcopy(hc_train.tolist())
        negative_list = patients_train
        pos_select = np.random.choice(positive_list, size=sampling_number)
        neg_select = np.random.choice(negative_list, size=sampling_number)
        tmp_paired = [(sub_anchor, p, n)for p, n in zip(pos_select, neg_select)]
        triplet_list_test.extend(tmp_paired)

    triplet_array_test = np.asarray(triplet_list_test)
    # anchor, positive and negative
    return triplet_array_test[:, 0], triplet_array_test[:, 1], triplet_array_test[:, 2]


def train_cv_whole_combine(patient_index, hc_index, model, criterion, optimizer, epoch,
                           test_matrix_lap, part_flag=True):
    # 1. Index Confirmed
    anchor_train_index = triplet_template_cv(patient_index, hc_index)
    if not part_flag:
        print('---------The Train Set Size is {}------------'.format(np.shape(anchor_train_index)))

    # 2. Load features for each sub
    anchor_train_features, anchor_train_features1, anchor_train_features2, anchor_train_features3,\
    anchor_train_adj, anchor_train_adj1, anchor_train_adj2, anchor_train_adj3 \
        = load_features_whole(anchor_train_index, test_matrix_lap)

    if not part_flag:
        print('Train shape: {}'.format(anchor_train_features.shape))
    if torch.cuda.is_available():
        if not part_flag:
            print('CUDA is avaliable!')
        model.cuda()
        anchor_train_features, anchor_train_features1, anchor_train_features2,  anchor_train_features3,\
        anchor_train_adj, anchor_train_adj1, anchor_train_adj2, anchor_train_adj3 = \
            map(lambda x: x.cuda(),
                [anchor_train_features, anchor_train_features1, anchor_train_features2, anchor_train_features3,
                 anchor_train_adj, anchor_train_adj1, anchor_train_adj2, anchor_train_adj3])



    # claculate Loss
    outa1, outa2, outb1, outb2, \
    outc1, outc2, outd1, outd2 = model(anchor_train_features, anchor_train_adj, anchor_train_features1,
                                anchor_train_adj1, anchor_train_features2, anchor_train_adj2)

    my_criterion = Weighted_Margin()
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    loss = my_criterion(outa1, outa2, outb1, outb2, outc1, outc2, outd1, outd2, criterion)
    acc = accuracy_new(outa1, outa2, outb1, outb2, outc1, outc2, outd1, outd2, args.margin)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print("~~~~~Epoch number {0} and Batch Part_{1} ~~~~~".format(epoch, part_flag))
    print("Train loss {} and train accuracy:{}".format(loss, acc))

    return loss.data, acc.data


def test_cv_whole(patient_index, hc_index, model, whole_mean_lap,
                  patient_train_index, hc_train_index, batch_number):
    # 1. Index Confirmed
    sampling_number = 5
    anchor_test_index, positive_test_index, negative_test_index \
        = triplet_template_cv_newtest(patient_index, hc_index, sampling_number, patient_train_index, hc_train_index)
    print('---------The Test Set Size is {}------------'.format(np.shape(anchor_test_index)))

    # 2. Load features for each sub
    anchor_test_features, anchor_test_features1, anchor_test_features2, anchor_test_features3, \
    anchor_test_adj, anchor_test_adj1, anchor_test_adj2, anchor_test_adj3 \
        = load_features_whole(anchor_test_index, whole_mean_lap)
    positive_test_features, positive_test_features1, positive_test_features2, positive_test_features3, \
    positive_test_adj, positive_test_adj1, positive_test_adj2, positive_test_adj3, \
        = load_features_whole(positive_test_index, whole_mean_lap)
    negative_test_features, negative_test_features1, negative_test_features2, negative_test_features3, \
    negative_test_adj, negative_test_adj1, negative_test_adj2, negative_test_adj3 \
        = load_features_whole(negative_test_index, whole_mean_lap)

    print('Test shape: {}'.format(anchor_test_features.shape))
    if torch.cuda.is_available():
        print('CUDA is avaliable!')
        model.cuda()
        anchor_test_features, positive_test_features, negative_test_features, \
        anchor_test_adj, positive_test_adj, negative_test_adj = \
            map(lambda x: x.cuda(),
                [anchor_test_features, positive_test_features, negative_test_features,
                 anchor_test_adj, positive_test_adj, negative_test_adj])

        anchor_test_features1, positive_test_features1, negative_test_features1, \
        anchor_test_adj1, positive_test_adj1, negative_test_adj1 = \
            map(lambda x: x.cuda(),
                [anchor_test_features1, positive_test_features1, negative_test_features1,
                 anchor_test_adj1, positive_test_adj1, negative_test_adj1])

        anchor_test_features2, positive_test_features2, negative_test_features2, \
        anchor_test_adj2, positive_test_adj2, negative_test_adj2 = \
            map(lambda x: x.cuda(),
                [anchor_test_features2, positive_test_features2, negative_test_features2,
                 anchor_test_adj2, positive_test_adj2, negative_test_adj2])

        anchor_test_features3, positive_test_features3, negative_test_features3, \
        anchor_test_adj3, positive_test_adj3, negative_test_adj3 = \
            map(lambda x: x.cuda(),
                [anchor_test_features3, positive_test_features3, negative_test_features3,
                 anchor_test_adj3, positive_test_adj3, negative_test_adj3])

    outa1, outa2, outb1, outb2,\
    outc1, outc2, outd1, outd2 = model(anchor_test_features, anchor_test_adj, anchor_test_features1, anchor_test_adj1,
                                anchor_test_features2, anchor_test_adj2, anchor_test_features3, anchor_test_adj3,
                                positive_test_features, positive_test_adj, positive_test_features1, positive_test_adj1,
                                positive_test_features2, positive_test_adj2, positive_test_features3,positive_test_adj3,
                                negative_test_features, negative_test_adj, negative_test_features1, negative_test_adj1,
                                negative_test_features2, negative_test_adj2, negative_test_features3, negative_test_adj3)

    new_outa1 = []
    new_outa2 = []
    new_outb1 = []
    new_outb2 = []
    new_outc1 = []
    new_outc2 = []
    new_outd1 = []
    new_outd2 = []

    for i in range(anchor_test_index.shape[0]):
        if i % sampling_number == 0:
            acc_outa1 = outa1[i]
            acc_outa2 = outa2[i]
            acc_outb1 = outb1[i]
            acc_outb2 = outb2[i]
            acc_outc1 = outc1[i]
            acc_outc2 = outc2[i]
            acc_outd1 = outd1[i]
            acc_outd2 = outd2[i]

        elif i % sampling_number != sampling_number - 1:
            acc_outa1 += outa1[i]
            acc_outa2 += outa2[i]
            acc_outb1 += outb1[i]
            acc_outb2 += outb2[i]
            acc_outc1 += outc1[i]
            acc_outc2 += outc2[i]
            acc_outd1 += outd1[i]
            acc_outd2 += outd2[i]
        else:
            acc_outa1 += outa1[i]
            acc_outb1 += outb1[i]
            acc_outa2 += outa2[i]
            acc_outb2 += outb2[i]
            acc_outc1 += outc1[i]
            acc_outc2 += outc2[i]
            acc_outd1 += outd1[i]
            acc_outd2 += outd2[i]

            new_outa1.append(acc_outa1)
            new_outa2.append(acc_outa2)
            new_outb1.append(acc_outb1)
            new_outb2.append(acc_outb2)
            new_outc1.append(acc_outc1)
            new_outc2.append(acc_outc2)
            new_outd1.append(acc_outd1)
            new_outd2.append(acc_outd2)

    new_outa1 = torch.Tensor(new_outa1).cuda()
    new_outa2 = torch.Tensor(new_outa2).cuda()
    new_outb1 = torch.Tensor(new_outb1).cuda()
    new_outb2 = torch.Tensor(new_outb2).cuda()
    new_outc1 = torch.Tensor(new_outc1).cuda()
    new_outc2 = torch.Tensor(new_outc2).cuda()
    new_outd1 = torch.Tensor(new_outd1).cuda()
    new_outd2 = torch.Tensor(new_outd2).cuda()

    my_criterion = Weighted_Margin()
    loss = my_criterion(new_outa1, new_outa2, new_outb1, new_outb2, new_outc1, new_outc2,
                        new_outd1, new_outd2, my_criterion)

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    asd_num = np.shape(patient_index)[0]
    hc_num = np.shape(hc_index)[0]
    label_list = np.concatenate((np.ones(asd_num), np.zeros(hc_num)))

    acc, sen, spe = accuracy_new_add(new_outa1, new_outa2, new_outb1, new_outb2,
                                     new_outc1, new_outc2, new_outd1, new_outd2, label_list, args.margin)

    return loss.data, acc.data, sen, spe


def mutual_kl(x, lx, func):
    # x is target loss，lx represent other losses，func is KL function
    kl_loss = 0
    for tmp_loss in lx:
        kl_loss += func(x, tmp_loss)

    return kl_loss / len(lx)
    # return np.average(map(func, [x, x], lx))
    # return (func(x, lx[0]) + func(x, lx[1])) / len(lx)


def train_cv_whole_combine(patient_index, hc_index, model_list, criterion, optimizer, epoch,
                             batch, test_matrix_lap):
    # 1. Index Confirmed
    sampling_number = 30
    anchor_train_index, positive_train_index, negative_train_index \
        = triplet_template_cv_m(patient_index, hc_index, sampling_number=sampling_number, train_data=True)
    # if not part_flag:
    #     print('---------The Train Set Size is {}------------'.format(np.shape(anchor_train_index)))

    # 2. Load features for each sub with one specific template
    # set-up fixed model number
    model_num = len(model_list)
    tmp_pkl = 'AAL116.pkl'
    tmp_pkl1 = 'CC200.pkl'
    tmp_pkl2 = 'BN273.pkl'
    tmp_pkl3 = 'BN273.pkl'
    anchor_train_features, anchor_train_adj \
        = load_features_pkl(anchor_train_index, test_matrix_lap[0], tmp_pkl)
    positive_train_features, positive_train_adj \
        = load_features_pkl(positive_train_index, test_matrix_lap[0], tmp_pkl)
    negative_train_features, negative_train_adj \
        = load_features_pkl(negative_train_index, test_matrix_lap[0], tmp_pkl)

    anchor_train_features1, anchor_train_adj1 \
        = load_features_pkl(anchor_train_index, test_matrix_lap[1], tmp_pkl1)
    positive_train_features1, positive_train_adj1 \
        = load_features_pkl(positive_train_index, test_matrix_lap[1], tmp_pkl1)
    negative_train_features1, negative_train_adj1 \
        = load_features_pkl(negative_train_index, test_matrix_lap[1], tmp_pkl1)

    anchor_train_features2, anchor_train_adj2 \
        = load_features_pkl(anchor_train_index, test_matrix_lap[2], tmp_pkl2)
    positive_train_features2, positive_train_adj2 \
        = load_features_pkl(positive_train_index, test_matrix_lap[2], tmp_pkl2)
    negative_train_features2, negative_train_adj2 \
        = load_features_pkl(negative_train_index, test_matrix_lap[2], tmp_pkl2)

    anchor_train_features3, anchor_train_adj3 \
        = load_features_pkl(anchor_train_index, test_matrix_lap[2], tmp_pkl3)
    positive_train_features3, positive_train_adj3 \
        = load_features_pkl(positive_train_index, test_matrix_lap[2], tmp_pkl3)
    negative_train_features3, negative_train_adj3 \
        = load_features_pkl(negative_train_index, test_matrix_lap[2], tmp_pkl3)


    if torch.cuda.is_available():
        # print('CUDA is avaliable!')
        for i, tmp_model in enumerate(model_list):
                model_list[i] = tmp_model.cuda()

        anchor_train_features, positive_train_features, negative_train_features, \
        anchor_train_adj, positive_train_adj, negative_train_adj = \
            map(lambda x: x.cuda(),
                [anchor_train_features, positive_train_features, negative_train_features,
                 anchor_train_adj, positive_train_adj, negative_train_adj])

        anchor_train_features1, positive_train_features1, negative_train_features1, \
            anchor_train_adj1, positive_train_adj1, negative_train_adj1 = \
            map(lambda x: x.cuda(),
                [anchor_train_features1, positive_train_features1, negative_train_features1,
                 anchor_train_adj1, positive_train_adj1, negative_train_adj1])

        anchor_train_features2, positive_train_features2, negative_train_features2, \
        anchor_train_adj2, positive_train_adj2, negative_train_adj2 = \
            map(lambda x: x.cuda(),
                [anchor_train_features2, positive_train_features2, negative_train_features2,
                 anchor_train_adj2, positive_train_adj2, negative_train_adj2])

        anchor_train_features3, positive_train_features3, negative_train_features3, \
        anchor_train_adj3, positive_train_adj3, negative_train_adj3 = \
            map(lambda x: x.cuda(),
                [anchor_train_features3, positive_train_features3, negative_train_features3,
                 anchor_train_adj3, positive_train_adj3, negative_train_adj3])


    # claculate Loss
    out1, out2 = model_list[0](anchor_train_features, anchor_train_adj,
                               positive_train_features, positive_train_adj,
                               negative_train_features, negative_train_adj)

    outb1, outb2 = model_list[1](anchor_train_features1, anchor_train_adj1,
                                 positive_train_features1, positive_train_adj1,
                                 negative_train_features1, negative_train_adj1)

    outc1, outc2 = model_list[2](anchor_train_features2, anchor_train_adj2,
                                 positive_train_features2, positive_train_adj2,
                                 negative_train_features2, negative_train_adj2)

    outd1, outd2 = model_list[3](anchor_train_features3, anchor_train_adj3,
                                 positive_train_features3, positive_train_adj3,
                                 negative_train_features3, negative_train_adj3)

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    # KL based models
    target = torch.cuda.FloatTensor(out1.size()).fill_(-1)
    loss = criterion(out1, out2, target)
    loss1 = criterion(outb1, outb2, target)
    loss2 = criterion(outc1, outc2, target)
    loss3 = criterion(outd1, outd2, target)

    # KL_div
    kl_function = nn.KLDivLoss()
    loss_norm = torch.autograd.Variable(loss / args.margin * -1, requires_grad=True)
    loss1_norm = torch.autograd.Variable(loss1 / args.margin * -1, requires_grad=True)
    loss2_norm = torch.autograd.Variable(loss2 / args.margin * -1, requires_grad=True)
    loss3_norm = torch.autograd.Variable(loss3 / args.margin * -1, requires_grad=True)

    KL_loss = mutual_kl(loss_norm, [loss1_norm, loss2_norm, loss3_norm], kl_function)
    KL_loss1 = mutual_kl(loss1_norm, [loss_norm, loss2_norm, loss3_norm], kl_function)
    KL_loss2 = mutual_kl(loss2_norm, [loss_norm, loss1_norm, loss3_norm], kl_function)
    KL_loss3 = mutual_kl(loss3_norm, [loss_norm, loss1_norm, loss2_norm], kl_function)

    new_loss = loss + 0.5 * KL_loss
    new_loss1 = loss1 + 0.5 * KL_loss1
    new_loss2 = loss1 + 0.5 * KL_loss2
    new_loss3 = loss3 + 0.5 * KL_loss3


    # new_loss = torch.autograd.Variable(new_loss, requires_grad=True)
    # new_loss1 = torch.autograd.Variable(new_loss1, requires_grad=True)

    total_loss_list = [new_loss, new_loss1, new_loss2, new_loss3]
    acc = accuracy_single(out1, out2, args.margin)
    acc1 = accuracy_single(outb1, outb2, args.margin)
    acc2 = accuracy_single(outc1, outc2, args.margin)
    acc3 = accuracy_single(outd1, outd2, args.margin)
    acc_list = [acc.data, acc1.data, acc2.data, acc3.data]

    for i in range(model_num):
        optimizer[i].zero_grad()
        total_loss_list[i].backward()
        optimizer[i].step()

    print("~~~~~Epoch number {0} and Batch Part_{1} ~~~~~".format(epoch, batch))
    print("Train loss {} and train accuracy:{}".format(loss, acc))

    return total_loss_list, acc_list


def test_cv_whole_mutual(patient_index, hc_index, model_list, criterion, whole_mean_lap,
                  patient_train_index, hc_train_index):
    # 1. Index Confirmed
    sampling_number = 20
    anchor_test_index, positive_test_index, negative_test_index \
        = triplet_template_cv_test_m(patient_index, hc_index, sampling_number, patient_train_index, hc_train_index)
    print('---------The Test Set Size is {}------------'.format(np.shape(anchor_test_index)))

    tmp_pkl = 'AAL116.pkl'
    tmp_pkl1 = 'CC200.pkl'
    tmp_pkl2 = 'BN273.pkl'
    tmp_pkl3 = 'BN273.pkl'
    # 2. Load features for each sub
    anchor_test_features, anchor_test_adj \
        = load_features_pkl(anchor_test_index, whole_mean_lap[0], tmp_pkl)
    positive_test_features, positive_test_adj \
        = load_features_pkl(positive_test_index, whole_mean_lap[0], tmp_pkl)
    negative_test_features, negative_test_adj \
        = load_features_pkl(negative_test_index, whole_mean_lap[0], tmp_pkl)

    anchor_test_features1, anchor_test_adj1 \
        = load_features_pkl(anchor_test_index, whole_mean_lap[1], tmp_pkl1)
    positive_test_features1, positive_test_adj1 \
        = load_features_pkl(positive_test_index, whole_mean_lap[1], tmp_pkl1)
    negative_test_features1, negative_test_adj1 \
        = load_features_pkl(negative_test_index, whole_mean_lap[1], tmp_pkl1)

    anchor_test_features2, anchor_test_adj2 \
        = load_features_pkl(anchor_test_index, whole_mean_lap[2], tmp_pkl2)
    positive_test_features2, positive_test_adj2 \
        = load_features_pkl(positive_test_index, whole_mean_lap[2], tmp_pkl2)
    negative_test_features2, negative_test_adj2 \
        = load_features_pkl(negative_test_index, whole_mean_lap[2], tmp_pkl2)

    anchor_test_features3, anchor_test_adj3 \
        = load_features_pkl(anchor_test_index, whole_mean_lap[3], tmp_pkl3)
    positive_test_features3, positive_test_adj3 \
        = load_features_pkl(positive_test_index, whole_mean_lap[3], tmp_pkl3)
    negative_test_features3, negative_test_adj3 \
        = load_features_pkl(negative_test_index, whole_mean_lap[3], tmp_pkl3)


    if torch.cuda.is_available():
        # print('CUDA is avaliable!')
        for i, tmp_model in enumerate(model_list):
            model_list[i] = tmp_model.cuda()

        anchor_test_features, positive_test_features, negative_test_features, \
        anchor_test_adj, positive_test_adj, negative_test_adj = \
            map(lambda x: x.cuda(),
                [anchor_test_features, positive_test_features, negative_test_features,
                 anchor_test_adj, positive_test_adj, negative_test_adj])

        anchor_test_features1, positive_test_features1, negative_test_features1, \
        anchor_test_adj1, positive_test_adj1, negative_test_adj1 = \
            map(lambda x: x.cuda(),
                [anchor_test_features1, positive_test_features1, negative_test_features1,
                 anchor_test_adj1, positive_test_adj1, negative_test_adj1])

        anchor_test_features2, positive_test_features2, negative_test_features2, \
        anchor_test_adj2, positive_test_adj2, negative_test_adj2 = \
            map(lambda x: x.cuda(),
                [anchor_test_features2, positive_test_features2, negative_test_features2,
                 anchor_test_adj2, positive_test_adj2, negative_test_adj2])

        anchor_test_features3, positive_test_features3, negative_test_features3, \
        anchor_test_adj3, positive_test_adj3, negative_test_adj3 = \
            map(lambda x: x.cuda(),
                [anchor_test_features3, positive_test_features3, negative_test_features3,
                 anchor_test_adj3, positive_test_adj3, negative_test_adj3])

    out1, out2 = model_list[0](anchor_test_features, anchor_test_adj,
                               positive_test_features, positive_test_adj,
                               negative_test_features, negative_test_adj)

    outb1, outb2 = model_list[1](anchor_test_features1, anchor_test_adj1,
                                 positive_test_features1, positive_test_adj1,
                                 negative_test_features1, negative_test_adj1)

    outc1, outc2 = model_list[2](anchor_test_features2, anchor_test_adj2,
                                 positive_test_features2, positive_test_adj2,
                                 negative_test_features2, negative_test_adj2)

    outd1, outd2 = model_list[3](anchor_test_features3, anchor_test_adj3,
                                 positive_test_features3, positive_test_adj3,
                                 negative_test_features3, negative_test_adj3)

    new_out1 = []
    new_out2 = []
    new_outb1 = []
    new_outb2 = []
    new_outc1 = []
    new_outc2 = []
    new_outd1 = []
    new_outd2 = []

    for i in range(anchor_test_index.shape[0]):
        if i % sampling_number == 0:
            acc_out1 = out1[i]
            acc_outb1 = outb1[i]
            acc_outc1 = outc1[i]
            acc_outd1 = outd1[i]
            acc_out2 = out2[i]
            acc_outb2 = outb2[i]
            acc_outc2 = outc2[i]
            acc_outd2 = outd2[i]
        elif i % sampling_number != sampling_number - 1:
            acc_out1 += out1[i]
            acc_outb1 += outb1[i]
            acc_outc1 += outc1[i]
            acc_outd1 += outd1[i]
            acc_out2 += out2[i]
            acc_outb2 += outb2[i]
            acc_outc2 += outc2[i]
            acc_outd2 += outd2[i]
        else:
            acc_out1 += out1[i]
            acc_outb1 += outb1[i]
            acc_outc1 += outc1[i]
            acc_outd1 += outd1[i]
            acc_out2 += out2[i]
            acc_outb2 += outb2[i]
            acc_outc2 += outc2[i]
            acc_outd2 += outd2[i]
            new_out1.append(acc_out1)
            new_outb1.append(acc_outb1)
            new_outc1.append(acc_outc1)
            new_outd1.append(acc_outd1)
            new_out2.append(acc_out2)
            new_outb2.append(acc_outb2)
            new_outc2.append(acc_outc2)
            new_outd2.append(acc_outd2)

    new_out1 = torch.Tensor(new_out1).cuda()
    new_out2 = torch.Tensor(new_out2).cuda()

    new_outb1 = torch.Tensor(new_outb1).cuda()
    new_outb2 = torch.Tensor(new_outb2).cuda()

    new_outc1 = torch.Tensor(new_outc1).cuda()
    new_outc2 = torch.Tensor(new_outc2).cuda()

    new_outd1 = torch.Tensor(new_outd1).cuda()
    new_outd2 = torch.Tensor(new_outd2).cuda()

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    # KL based models
    target = torch.cuda.FloatTensor(new_out1.size()).fill_(-1)
    loss = criterion(new_out1, new_out2, target) / sampling_number
    loss1 = criterion(new_outb1, new_outb2, target) / sampling_number
    loss2 = criterion(new_outc1, new_outc2, target) / sampling_number
    loss3 = criterion(new_outd1, new_outd2, target) / sampling_number

    # KL_div
    kl_function = nn.KLDivLoss()

    loss_norm = torch.autograd.Variable(loss / args.margin * -1, requires_grad=True)
    loss1_norm = torch.autograd.Variable(loss1 / args.margin * -1, requires_grad=True)
    loss2_norm = torch.autograd.Variable(loss2 / args.margin * -1, requires_grad=True)
    loss3_norm = torch.autograd.Variable(loss3 / args.margin * -1, requires_grad=True)

    KL_loss = mutual_kl(loss_norm, [loss1_norm, loss2_norm, loss3_norm], kl_function)
    KL_loss1 = mutual_kl(loss1_norm, [loss_norm, loss2_norm, loss3_norm], kl_function)
    KL_loss2 = mutual_kl(loss2_norm, [loss_norm, loss1_norm, loss3_norm], kl_function)
    KL_loss3 = mutual_kl(loss3_norm, [loss_norm, loss1_norm, loss2_norm], kl_function)
    
    new_loss = loss + 0.5 * KL_loss
    new_loss1 = loss1 + 0.5 * KL_loss1
    new_loss2 = loss2 + 0.5 * KL_loss2
    new_loss3 = loss3 + 0.5 * KL_loss3

    asd_num = np.shape(patient_index)[0]
    hc_num = np.shape(hc_index)[0]
    label_list = np.concatenate((np.ones(asd_num), np.zeros(hc_num)))
    acc, sen, spe = accuracy_single_acc(new_out1, new_out2, args.margin, label_list)
    acc1, sen1, spe1 = accuracy_single_acc(new_outb1, new_outb2, args.margin, label_list)
    acc2, sen2, spe2 = accuracy_single_acc(new_outc1, new_outc2, args.margin, label_list)
    acc3, sen3, spe3 = accuracy_single_acc(new_outd1, new_outd2, args.margin, label_list)

    loss_list = [new_loss.data, new_loss1.data, new_loss2.data, new_loss3.data]
    acc_list = [acc.data, acc1.data, acc2.data, acc3.data]
    sen_list = [sen, sen1, sen2, sen3]
    spe_list = [spe, spe1, spe2, spe3]
    return loss_list, acc_list, sen_list, spe_list