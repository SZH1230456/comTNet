import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import dill
import time
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
import os
from collections import defaultdict
from util import llprint, multi_label_metric, ddi_rate_score_, get_n_params, test_test, ddi_rate_score
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score, roc_curve
from bayes_opt import BayesianOptimization
import sys
from torch.nn.parameter import Parameter
import math

model_name = 'GAMENet'
resume_name = ''


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.mm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN(nn.Module):
    def __init__(self, voc_size, emb_dims, adj, device):
        super(GCN, self).__init__()
        self.voc_size = voc_size
        self.emb_dims = emb_dims
        self.device = device

        adj = self.normalize(adj + np.eye(adj.shape[0]))
        self.adj = torch.FloatTensor(adj).to(self.device)

        self.x = torch.eye(voc_size).to(self.device)
        self.gcn1 = GraphConvolution(voc_size, emb_dims)
        self.dropout = nn.Dropout(p=0.3)
        self.gcn2 = GraphConvolution(emb_dims, emb_dims)

    def forward(self):
        node_embedding = self.gcn1(self.x, self.adj)  # (voc_size, emb_dims)
        node_embedding = F.relu(node_embedding)
        node_embedding = self.dropout(node_embedding)
        node_embedding = self.gcn2(node_embedding, self.adj)  # (voc_size, emb_dims)
        return node_embedding

    def normalize(self, mx):
        """Row-normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = np.diagflat(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx


class GAMENet(nn.Module):
    def __init__(self, emb_dims, voc_size, ehr_adj, ddi_adj, device=torch.device('cuda:0')):
        super(GAMENet, self).__init__()
        self.emb_dims = emb_dims
        self.voc_size = voc_size
        self.device = device
        self.tensor_ddi_adj = torch.FloatTensor(ddi_adj).to(device)
        self.beta = nn.Parameter(torch.FloatTensor(1))

        self.embedding_diagnosis = nn.Embedding(self.voc_size[0], self.emb_dims)

        self.embedding_procedure = nn.Embedding(self.voc_size[1], self.emb_dims)

        self.dropout = nn.Dropout(p=0.4)

        self.query = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.emb_dims*4, emb_dims)
        )
        self.encoder_diagnosis = nn.GRU(self.emb_dims, self.emb_dims*2, batch_first=True)
        self.encoder_procedure = nn.GRU(self.emb_dims, self.emb_dims*2, batch_first=True)

        self.ehr_gcn = GCN(voc_size=voc_size[2], emb_dims=emb_dims, adj=ehr_adj, device=self.device)
        self.ddi_gcn = GCN(voc_size=voc_size[2], emb_dims=emb_dims, adj=ddi_adj, device=self.device)

        self.output = nn.Sequential(
            nn.ReLU(),
            nn.Linear(emb_dims * 3, emb_dims * 2),
            nn.ReLU(),
            nn.Linear(emb_dims*2, voc_size[2])
        )
        self.init_weights()

    def forward(self, input):
        diagnosis_seq = []
        procedure_seq = []
        for adm in input:
            diagnosis = self.dropout(self.embedding_diagnosis(torch.LongTensor(adm[0]).unsqueeze(dim=0).to(self.device)))
            # (1, len, dim)
            procedure = self.dropout(self.embedding_procedure(torch.LongTensor(adm[1]).unsqueeze(dim=0).to(self.device)))
            # (1, len, dim)
            diagnosis_seq.append(diagnosis.mean(dim=1).unsqueeze(dim=0))  # (1, dim)
            procedure_seq.append(procedure.mean(dim=1).unsqueeze(dim=0))  # (1, dim)
        diagnosis_seq = torch.cat(diagnosis_seq, dim=1)  # (1, visit, dim)
        procedure_seq = torch.cat(procedure_seq, dim=1)  # (1, visit, dim)

        # o: (1, visit, 2*emb_dims)  h:(1, 1, emb_dims*2)
        o_diagnosis, h_diagnosis = self.encoder_diagnosis(diagnosis_seq)
        o_procedure, h_procedure = self.encoder_procedure(procedure_seq)

        queries = torch.cat((o_diagnosis, o_procedure), dim=2).squeeze(dim=0)  # (visit, 4*emb_dims)
        queries_ = self.query(queries)
        query = queries_[-1:]  # (1, emb_dims)

        drug_memory = self.ehr_gcn() - self.beta * self.ddi_gcn()  # (voc_size, emb_dims)

        if len(input) > 1:
            history_keys = queries_[:queries_.size(0)-1]

            history_values = np.zeros(shape=[len(input)-1, self.voc_size[2]])
            for idx, adm in enumerate(input):
                if idx != len(input)-1:
                    history_values[idx, adm[2]] = 1
            history_values = torch.FloatTensor(history_values).to(self.device)

        a_tc = F.softmax(torch.mm(query, drug_memory.t()), dim=-1)  # (1, voc_size)
        o_tb = torch.mm(a_tc, drug_memory)  # (1, emb_dims)

        if len(input) > 1:
            a_tm = F.softmax(torch.mm(query, history_keys.t()), dim=-1)  # (1, visit-1)
            a_tm = torch.mm(a_tm, history_values)  # (1, size)
            o_td = torch.mm(a_tm, drug_memory)  # (1, dim)
        else:
            o_td = o_tb
        output = self.output(torch.cat((query, o_tb, o_td), dim=-1))

        if self.training:
            neg_pred_prob = F.sigmoid(output)
            neg_pred_prob = neg_pred_prob.t() * neg_pred_prob  # (voc_size, voc_size)
            batch_neg = neg_pred_prob.mul(self.tensor_ddi_adj).mean()

            return output, batch_neg
        else:
            return F.sigmoid(output)

    def init_weights(self):
        """Initialize weights."""
        initrange = 0.1
        self.embedding_diagnosis.weight.data.uniform_(-initrange, initrange)

        self.embedding_procedure.weight.data.uniform_(-initrange, initrange)

        self.beta.data.uniform_(-initrange, initrange)


def eval(model, data_eval, voc_size, epoch):
    with torch.no_grad():
        print('')
        model.eval()
        ja, prauc, avg_p, avg_r, avg_f1 = [[] for _ in range(5)]
        y_pred_label_all_for_ddi = np.zeros(shape=[0, voc_size[2]])
        for step, input in enumerate(data_eval):
            if len(input) < 2:
                continue
            else:
                y_pred_prob_patient = np.zeros(shape=[0, voc_size[2]])
                y_pred_label_patient = np.zeros(shape=[0, voc_size[2]])
                y_label_patient = np.zeros(shape=[0, voc_size[2]])
                for i in range(1, len(input)):
                    label_for_patient = np.zeros(voc_size[2])
                    label_for_patient[input[i][2]] = 1
                    y_label_patient = np.concatenate((y_label_patient, label_for_patient.reshape(1, -1)), axis=0)

                    adm = input[:i]
                    prediction = model(adm)
                    prediction = prediction.detach().cpu().numpy()
                    y_pred_prob_patient = np.concatenate((y_pred_prob_patient, prediction.reshape(1, -1)), axis=0)

                    y_pred_temp = prediction.copy()
                    y_pred_temp[y_pred_temp >= 0.5] = 1
                    y_pred_temp[y_pred_temp < 0.5] = 0

                    y_pred_label_patient = np.concatenate((y_pred_label_patient, np.array(y_pred_temp).reshape(1, -1)), axis=0)
                    for idx, value in enumerate(y_pred_temp[0, :]):
                        if value == 1:
                            y_pred_temp[0, idx] = idx

                    y_pred_label_all_for_ddi = np.concatenate((y_pred_label_all_for_ddi, np.array(y_pred_temp).reshape(1, -1)), axis=0)

                adm_ja, adm_prauc, adm_avg_p, adm_avg_r, adm_avg_f1 = multi_label_metric(y_label_patient,
                                                                                         y_pred_label_patient,
                                                                                         y_pred_prob_patient)
                ja.append(adm_ja)
                prauc.append(adm_prauc)
                avg_p.append(adm_avg_p)
                avg_r.append(adm_avg_r)
                avg_f1.append(adm_avg_f1)

        # ddi rate
        ddi_rate = ddi_rate_score_(y_pred_label_all_for_ddi)
        print('\t epoch: %.2f, DDI Rate: %.4f, Jaccard: %.4f,  PRAUC: %.4f, '
              'AVG_PRC: %.4f, AVG_RECALL: %.4f, AVG_F1: %.4f\n' % (
                    epoch, ddi_rate, np.mean(ja), np.mean(prauc), np.mean(avg_p), np.mean(avg_r),
                    np.mean(avg_f1)))

        return ddi_rate, np.mean(ja), np.mean(prauc), np.mean(avg_p), np.mean(avg_r), np.mean(avg_f1)


def train(LR, emb_dims, l2_regularization):
    LR = 10 ** LR
    emb_dims = 2 ** int(emb_dims)
    l2_regularization = 10 ** l2_regularization
    print('LR____{}____emb_dims____{}____l2_regularization___{}'.format(LR, emb_dims, l2_regularization))

    data_path = '../../data/records_final.pkl'
    voc_path = '../../data/voc_final.pkl'

    ehr_adj_path = '../../data/ehr_adj_final.pkl'
    ddi_adj_path = '../../data/ddi_A_final.pkl'
    device = torch.device('cuda:0')

    ehr_adj = dill.load(open(ehr_adj_path, 'rb'))
    ddi_adj = dill.load(open(ddi_adj_path, 'rb'))
    data = dill.load(open(data_path, 'rb'))
    voc = dill.load(open(voc_path, 'rb'))
    diag_voc, pro_voc, med_voc = voc['diag_voc'], voc['pro_voc'], voc['med_voc']

    split_point = int(len(data) * 2 / 3)
    data_train = data[:split_point]
    eval_len = int(len(data[split_point:]) / 2)
    data_test = data[split_point:split_point + eval_len]
    data_eval = data[split_point + eval_len:]
    voc_size = (len(diag_voc.idx2word), len(pro_voc.idx2word), len(med_voc.idx2word))

    EPOCH = 10
    TARGET_DDI = 0.05
    T = 0.5
    decay_weight = 0.85
    ddi_rate, ja, prauc, avg_p, avg_r, avg_f1 = [0 for i in range(6)]

    model = GAMENet(emb_dims=emb_dims, voc_size=voc_size, ehr_adj=ehr_adj, ddi_adj=ddi_adj, device=device)
    weight_decay_list = (param for name, param in model.named_parameters() if name[-4:] != 'bias' and "bn" not in name)
    no_decay_list = (param for name, param in model.named_parameters() if name[-4:] == 'bias' or "bn" in name)
    parameters = [{'params': weight_decay_list},
                  {'params': no_decay_list, 'weight_decay': 0.}]

    optimizer = Adam(parameters, lr=LR, weight_decay=l2_regularization)

    model.to(device)

    for epoch in range(EPOCH):
        start_time = time.time()
        loss_record = []
        neg_loss_cnt = 0
        prediction_loss_cnt = 0
        model.train()
        for step, input in enumerate(data_train):
            if len(input) < 2:
                continue
            else:
                for idx in range(1, len(input)):
                    seq_input = input[:idx]
                    loss_1_target = np.zeros([1, voc_size[2]])
                    loss_1_target[:, input[idx][2]] = 1

                    loss_2_target = np.full((1, voc_size[2]), -1)
                    for idx, item in enumerate(input[idx][2]):
                        loss_2_target[0][idx] = item

                    prediction, loss_ddi = model(seq_input)
                    loss_1 = F.binary_cross_entropy_with_logits(prediction, torch.FloatTensor(loss_1_target).to(device))
                    loss_2 = F.multilabel_margin_loss(prediction, torch.LongTensor(loss_2_target).to(device))

                    prediction = prediction.detach().cpu().numpy()[0]
                    prediction[prediction >= 0.5] = 1
                    prediction[prediction < 0.5] = 0
                    y_label = np.where(prediction == 1)[0]
                    current_ddi_rate = ddi_rate_score([[y_label]])

                    if current_ddi_rate < TARGET_DDI:
                        loss = 0.9 * loss_1 + 0.1 * loss_2
                        prediction_loss_cnt += 1
                    else:
                        rnd = np.exp((TARGET_DDI - current_ddi_rate)/T)

                        if np.random.rand(1) < rnd:
                            loss = loss_ddi
                            neg_loss_cnt += 1
                        else:
                            loss = 0.9 * loss_1 + 0.1 * loss_2
                            prediction_loss_cnt += 1

                    optimizer.zero_grad()
                    loss.backward(retain_graph=True)
                    optimizer.step()

                    loss_record.append(loss.item())

        T *= decay_weight

        ddi_rate, ja, prauc, avg_p, avg_r, avg_f1 = eval(model, data_eval, voc_size, epoch)
        end_time = time.time()
        elapsed_time = (end_time - start_time) / 60
        llprint('\tEpoch: %d, Loss1: %.4f, '
                'One Epoch Time: %.2fm, Appro Left Time: %.2fh\n' % (epoch,
                                                                     np.mean(loss_record),
                                                                     elapsed_time,
                                                                     elapsed_time * (EPOCH - epoch - 1) / 60))
    return ja
    # return ddi_rate, ja, prauc, avg_p, avg_r, avg_f1


if __name__ == '__main__':
    test_test('GAMENet_7_4_train_all_input.txt')
    Encode_Decode_Time_BO = BayesianOptimization(
            train, {
                'emb_dims': (5, 8),
                'LR': (-5, 0),
                'l2_regularization': (-8, -3),
            }
        )
    Encode_Decode_Time_BO.maximize()
    print(Encode_Decode_Time_BO.max)
    # ddi_rate_all, ja_all, prauc_all, avg_p_all, avg_r_all, avg_f1_all = [[] for _ in range(6)]
    # for i in range(10):
    #     ddi_rate, ja, prauc, avg_p, avg_r, avg_f1 = train(LR=9.505530998929784e-05, emb_dims=256,
    #                                                       l2_regularization=8.01377484816919e-05)
    #     ddi_rate_all.append(ddi_rate)
    #     ja_all.append(ja)
    #     prauc_all.append(prauc)
    #     avg_p_all.append(avg_p)
    #     avg_r_all.append(avg_r)
    #     avg_f1_all.append(avg_f1)
    # print('ddi_rate{}---ja{}--prauc{}---avg_p{}---avg_r---{}--avg_f1--{}'.format(np.mean(ddi_rate_all),
    #                                                                              np.mean(ja_all),
    #                                                                              np.mean(prauc_all),
    #                                                                              np.mean(avg_p_all),
    #                                                                              np.mean(avg_r_all),
    #                                                                              np.mean(avg_f1_all)))





















