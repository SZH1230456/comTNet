import torch
import torch.nn as nn
import numpy as np
import dill
import torch.nn.functional as F
import time
from torch.optim import Adam
from itertools import chain
import os
from collections import defaultdict
from util import llprint, multi_label_metric, ddi_rate_score_, get_n_params, test_test, ddi_rate_score
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score, roc_curve
from bayes_opt import BayesianOptimization
import math
from sklearn.metrics import log_loss
from sklearn.metrics.pairwise import cosine_similarity

model_name = 'Proposed'
resume_name = ''


class ProposedModel(nn.Module):
    def __init__(self, vocab_size, emb_dim=64, device=torch.device('cuda:0')):
        super(ProposedModel, self).__init__()
        self.K = len(vocab_size)
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.device = device
        self.input_len = vocab_size[0] + vocab_size[1] + vocab_size[2]

        self.embedding = nn.Sequential(
            nn.Embedding(self.input_len + 1, emb_dim, padding_idx=self.input_len),
            nn.Dropout(p=0.3))

        self.encoder_1 = nn.GRU(emb_dim, emb_dim, batch_first=True)
        self.encoder_2 = nn.GRU(emb_dim, emb_dim, batch_first=True)

        self.attention_1 = nn.Linear(emb_dim, 1)
        self.attention_2 = nn.Linear(emb_dim, emb_dim)

        self.output = nn.Sequential(
            nn.Linear(self.emb_dim, self.vocab_size[2]))

    def get_top_k_patient_similarity(self, all_visit, k):
        similarity_matrix = nn.CosineSimilarity(all_visit, all_visit)
        similar_patients = np.zeros(shape=[0, all_visit.shape[1]])

        for i in range(all_visit.shape[0]):
            one_patient_similarity = similarity_matrix[i, :]
            top_k = np.argsort(-one_patient_similarity)[:k + 1]
            top_k_similarity = one_patient_similarity[top_k]
            top_k_similarity = np.transpose(np.tile(top_k_similarity, (all_visit.shape[1], 1)))
            similar_one_patient = all_visit[top_k, :]
            similar_one_patient = similar_one_patient * top_k_similarity
            similar_one_patient = similar_one_patient.reshape(-1, all_visit.shape[1])
            similar_one_patient = np.sum(similar_one_patient, axis=0)
            similar_patients = np.concatenate((similar_patients, similar_one_patient.reshape(-1, all_visit.shape[1])),
                                              axis=0)
        return similar_patients

    def attention(self, input):
        input_seq = []
        max_len = self.input_len
        for i in range(0, len(input)):
            visit = input[i]
            input_tmp = []
            input_tmp.extend(visit[0])
            input_tmp.extend(list(np.array(visit[1]) + self.vocab_size[0]))
            if i != len(input) - 1:
                input_tmp.extend(list(np.array(visit[2]) + self.vocab_size[0] + self.vocab_size[1]))
            if len(input_tmp) < max_len:
                input_tmp.extend([self.input_len] * (max_len - len(input_tmp)))

            input_seq.append(input_tmp)

        visit_emb = self.embedding(torch.LongTensor(input_seq[:-1]).to(self.device))  # (visit-1, max_len, dim_emb)
        visit_emb = torch.sum(visit_emb, dim=1)  # (visit-1, dim_emb)

        g, _ = self.encoder_1(visit_emb.unsqueeze(dim=0))  # (1, visit-1, dim_emb)
        h, _ = self.encoder_2(visit_emb.unsqueeze(dim=0))

        g = g.squeeze(dim=0)  # (visit, dim_emb)
        h = h.squeeze(dim=0)  # (visit, dim_emb)

        attn_g = F.softmax(self.attention_1(g), dim=-1)  # (visit-1, 1)
        attn_h = F.tanh(self.attention_2(h))  # (visit-1, emb)

        c = attn_g * attn_h * visit_emb  # (visit-1, emb)
        c = torch.sum(c, dim=0).unsqueeze(dim=0)  # (1, dim_emb)

        current_visit = self.embedding(torch.LongTensor(input_seq[-1]).to(self.device))
        current_visit = torch.sum(current_visit, dim=0).unsqueeze(dim=0)  # (1, dim_emb)
        all_input = c + current_visit
        return all_input

    def forward(self, input):
        all_visit_encoder = torch.zeros(size=[0, self.emb_dim]).to(self.device)
        all_patient_loss_1 = torch.zeros(size=[0, self.vocab_size[2]]).to(self.device)
        all_patient_loss_2 = torch.zeros(size=[0, self.vocab_size[2]]).to(self.device)
        all_patient_loss_2 = all_patient_loss_2.long()

        loss_1 = 0.0
        loss_2 = 0.0
        prediction = torch.zeros(size=[0, self.vocab_size[2]]).to(self.device)
        for i in range(1, len(input)):
            loss_1_target = np.zeros(shape=[1, self.vocab_size[2]])
            loss_1_target[:, input[i][2]] = 1

            loss_2_target = np.full((1, self.vocab_size[2]), -1)
            for idx, item in enumerate(input[i][2]):
                loss_2_target[:, idx] = item

            all_patient_loss_1 = torch.cat((all_patient_loss_1, torch.FloatTensor(loss_1_target).to(self.device)), dim=0)
            all_patient_loss_2 = torch.cat((all_patient_loss_2, torch.LongTensor(loss_2_target).to(self.device)), dim=0)

            encoder_visit = self.attention(input[:i+1])
            prediction_one_visit = torch.sigmoid(self.output(encoder_visit))
            prediction = torch.cat((prediction, prediction_one_visit), dim=0)

            loss_1 += F.binary_cross_entropy(prediction_one_visit, torch.FloatTensor(loss_1_target).to(self.device))
            loss_2 += F.multilabel_margin_loss(prediction_one_visit, torch.LongTensor(loss_2_target).to(self.device))
        loss = loss_1

        return all_patient_loss_1, all_patient_loss_2, prediction, loss


def eval(model, data_eval, voc_size, epoch, k, emb_dims, device):
    with torch.no_grad():
        print('')
        model.eval()
        ja, prauc, avg_p, avg_r, avg_f1 = [[] for i in range(5)]
        y_pred_label_all_patients_for_ddi = np.zeros(shape=[0, voc_size[2]])
        for step, input in enumerate(data_eval):
            if len(input) < 2:
                continue
            one_patient_loss_1, one_patient_loss_2, one_patient_prediction, one_patient_loss = model(input)
            one_patient_loss_1 = one_patient_loss_1.detach().cpu().numpy()
            one_patient_prediction = one_patient_prediction.detach().cpu().numpy()
            y_pred_label = one_patient_prediction.copy().reshape(-1,)
            y_pred_label[y_pred_label >= 0.3] = 1
            y_pred_label[y_pred_label < 0.3] = 0

            adm_ja, adm_prauc, adm_avg_p, adm_avg_r, adm_avg_f1 = multi_label_metric(np.array(one_patient_loss_1),
                                                                                     np.array(y_pred_label).reshape(-1, voc_size[2]),
                                                                                     np.array(one_patient_prediction))
            one_label_ddi = y_pred_label.copy().reshape(-1, voc_size[2])
            for i in range(one_label_ddi.shape[0]):
                y_pred_ = one_label_ddi[i, :]
                for idx, value in enumerate(y_pred_):
                    if value == 1:
                        one_label_ddi[i, idx] = idx

            y_pred_label_all_patients_for_ddi = np.concatenate((y_pred_label_all_patients_for_ddi, one_label_ddi.reshape(-1, voc_size[2])), axis=0)

            ja.append(adm_ja)
            prauc.append(adm_prauc)
            avg_p.append(adm_avg_p)
            avg_r.append(adm_avg_r)
            avg_f1.append(adm_avg_f1)

        ddi_rate = ddi_rate_score_(y_pred_label_all_patients_for_ddi)

        llprint('\t epoch: %.2f, DDI Rate: %.4f, Jaccard: %.4f, '
                ' PRAUC: %.4f, AVG_PRC: %.4f, AVG_RECALL: %.4f, AVG_F1: %.4f\n' % (
                epoch, ddi_rate, np.mean(ja), np.mean(prauc), np.mean(avg_p), np.mean(avg_r), np.mean(avg_f1)))
        return ddi_rate, np.mean(ja), np.mean(prauc), np.mean(avg_p), np.mean(avg_r), np.mean(avg_f1)


def train(emb_dims, LR, l2_regularization):
    emb_dims = 2 ** int(emb_dims)
    LR = 10 ** LR
    l2_regularization = 10 ** l2_regularization
    print('emb_dims___{}__LR__{}__l2_regularization___{}'.format(emb_dims, LR, l2_regularization))

    data_path = '../../data/records_final.pkl'
    voc_path = '../../data/voc_final.pkl'
    device = torch.device('cuda:0')

    data = dill.load(open(data_path, 'rb'))
    voc = dill.load(open(voc_path, 'rb'))
    diag_voc, pro_voc, med_voc = voc['diag_voc'], voc['pro_voc'], voc['med_voc']

    split_point = int(len(data) * 2 / 3)
    data_train = data[:split_point]
    eval_len = int(len(data[split_point:]) / 2)
    data_eval = data[split_point + eval_len:]
    data_test = data[split_point:split_point + eval_len]
    voc_size = (len(diag_voc.idx2word), len(pro_voc.idx2word), len(med_voc.idx2word))

    EPOCH = 10
    k = 2
    batch_size = 10

    model = ProposedModel(voc_size, emb_dims, device)
    model.to(device)

    weight_decay_list = (param for name, param in model.named_parameters()
                                 if name[-4:] != 'bias' and "bn" not in name)
    no_decay_list = (param for name, param in model.named_parameters()
                             if name[-4:] == 'bias' or "bn" in name)
    parameters = [{'params': weight_decay_list},
                  {'params': no_decay_list, 'weight_decay': 0.}]
    optimizer = Adam(parameters, lr=LR, weight_decay=l2_regularization)

    for epoch in range(EPOCH):
        start_time = time.time()
        model.train()
        for step, input in enumerate(data_train[:10]):
            if len(input) < 2:
                continue
            all_patient_loss_1, all_patient_loss_2, prediction_, loss = model(input)
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
        ddi_rate, ja, prauc, avg_p, avg_r, avg_f1 = eval(model,
                                                         data_eval[:10], voc_size,
                                                         epoch, k, emb_dims, device)
        end_time = time.time()
        elapsed_time = (end_time - start_time) / 60
        llprint('\tEpoch: %d, Loss1: %.4f, '
                'One Epoch Time: %.2fm, Appro Left Time: %.2fh\n' % (epoch,
                                                                     loss.item(),
                                                                     elapsed_time,
                                                                     elapsed_time * (EPOCH - epoch - 1) / 60))
    return ja
    # return ddi_rate, ja, prauc, avg_p, avg_r, avg_f1


if __name__ == '__main__':
    # train(emb_dims=128, LR=0.0001, l2_regularization=1e-5)
    # test_test('proposed_model_6_24.txt')
    Encode_Decode_Time_BO = BayesianOptimization(
            train, {
                'emb_dims': (5, 8),
                'LR': (-5, 0),
                'l2_regularization': (-8, -3),
            }
        )
    Encode_Decode_Time_BO.maximize()
    print(Encode_Decode_Time_BO.max)
