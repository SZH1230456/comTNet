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
from util import llprint, multi_label_metric, ddi_rate_score_, get_n_params, test_test
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score, roc_curve
from bayes_opt import BayesianOptimization
import sys

model_name = 'Retain'
resume_name = ''


class Retrive_Treat(nn.Module):
    def __init__(self, vocab_size, emb_dim=64, device=torch.device('cuda:0')):
        super(Retrive_Treat, self).__init__()
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

        self.output = nn.Linear(emb_dim, vocab_size[2])

    def forward(self, input):
        input_seq = []
        max_len = self.input_len
        for i in range(0, len(input)):
            visit = input[i]
            input_tmp = []
            input_tmp.extend(visit[0])
            input_tmp.extend(list(np.array(visit[1]) + self.vocab_size[0]))
            if i != len(input)-1:
                input_tmp.extend(list(np.array(visit[2]) + self.vocab_size[0] + self.vocab_size[1]))
            if len(input_tmp) < max_len:
                input_tmp.extend([self.input_len]*(max_len-len(input_tmp)))

            input_seq.append(input_tmp)

        visit_emb = self.embedding(torch.LongTensor(input_seq[:-1]).to(self.device))  # (visit, max_len, dim_emb)
        visit_emb = torch.sum(visit_emb, dim=1)  # (visit, dim_emb)

        g, _ = self.encoder_1(visit_emb.unsqueeze(dim=0))  # (1, visit, dim_emb)
        h, _ = self.encoder_2(visit_emb.unsqueeze(dim=0))

        g = g.squeeze(dim=0)  # (visit, dim_emb)
        h = h.squeeze(dim=0)  # (visit, dim_emb)

        attn_g = F.softmax(self.attention_1(g), dim=-1)  # (visit, 1)
        attn_h = F.tanh(self.attention_2(h))  # (visit, emb)

        c = attn_g * attn_h * visit_emb  # (visit, emb)
        c = torch.sum(c, dim=0).unsqueeze(dim=0)  # (1, dim_emb)

        current_visit = self.embedding(torch.LongTensor(input_seq[-1]).to(self.device))
        current_visit = torch.sum(current_visit, dim=0).unsqueeze(dim=0)
        all_input = c + current_visit
        output_ = self.output(all_input)
        return torch.sigmoid(output_)


def eval(model, data_eval, voc_size, epoch, training=True):
    with torch.no_grad():
        print('')
        model.eval()
        ja, prauc, avg_p, avg_r, avg_f1 = [[] for _ in range(5)]
        y_pred_label_all_for_ddi = np.zeros(shape=[0, voc_size[2]])

        for step, input in enumerate(data_eval):
            if len(input) < 2:  # visit > 2
                continue
            y_label_patient = np.zeros(shape=[0, voc_size[2]])
            y_pred_prob_patient = np.zeros(shape=[0, voc_size[2]])
            y_pred_label_patient = np.zeros(shape=[0, voc_size[2]])
            for i in range(1, len(input)):
                y = np.zeros(voc_size[2])
                y[input[i][2]] = 1
                y_label_patient = np.concatenate((y_label_patient, np.array(y).reshape(1, -1)), axis=0)

                pre = model(input[:i+1])
                pre = pre.detach().cpu().numpy()
                y_pred_prob_patient = np.concatenate((y_pred_prob_patient, np.array(pre).reshape(1, -1)), axis=0)
                y_pred_label_tmp = pre.copy()
                y_pred_label_tmp[y_pred_label_tmp >= 0.3] = 1
                y_pred_label_tmp[y_pred_label_tmp < 0.3] = 0
                y_pred_label_patient = np.concatenate((y_pred_label_patient, np.array(y_pred_label_tmp).reshape(1, -1)),axis=0)

                for idx, value in enumerate(y_pred_label_tmp[0,:]):
                    if value == 1:
                        y_pred_label_tmp[0, idx] = idx
                y_pred_label_all_for_ddi = np.concatenate((y_pred_label_all_for_ddi, np.array(y_pred_label_tmp).reshape(1, -1)), axis=0)

            adm_ja, adm_prauc, adm_avg_p, adm_avg_r, adm_avg_f1 = multi_label_metric(np.array(y_label_patient),
                                                                                     np.array(y_pred_label_patient),
                                                                                     np.array(y_pred_prob_patient))
            # if training:
            #     llprint('\rTraining--Epoch: %d, Step: %d/%d' % (epoch, step, len(data_eval)))
            # else:
            #     llprint('\rEval--Epoch: %d, Step: %d/%d' % (epoch, step, len(data_eval)))

            ja.append(adm_ja)
            prauc.append(adm_prauc)
            avg_p.append(adm_avg_p)
            avg_r.append(adm_avg_r)
            avg_f1.append(adm_avg_f1)

        # ddi rate
        ddi_rate = ddi_rate_score_(y_pred_label_all_for_ddi)
        if training:
            llprint('\t epoch: %.2f, DDI Rate: %.4f, Jaccard: %.4f,  PRAUC: %.4f, AVG_PRC: %.4f, AVG_RECALL: %.4f, AVG_F1: %.4f\n' % (
                epoch, ddi_rate, np.mean(ja), np.mean(prauc), np.mean(avg_p), np.mean(avg_r), np.mean(avg_f1)
            ))
        else:
            llprint('\tepoch: %.2f, DDI Rate: %.4f, Jaccard: %.4f,  PRAUC: %.4f, AVG_PRC: %.4f, AVG_RECALL: %.4f, AVG_F1: %.4f\n' % (
                epoch, ddi_rate, np.mean(ja), np.mean(prauc), np.mean(avg_p), np.mean(avg_r), np.mean(avg_f1)
            ))

        return ddi_rate, np.mean(ja), np.mean(prauc), np.mean(avg_p), np.mean(avg_r), np.mean(avg_f1)


# (diagnosis, producdure, drug)_{0, t-1}  ---> drug_t
def train(LR, emb_dims, l2_regularization):
    LR = 10 ** LR
    emb_dims = 2 ** int(emb_dims)
    l2_regularization = 10 ** l2_regularization
    print('learning_rate--{}---emb_dims---{}---l2_regularization--{}'.format(LR, emb_dims, l2_regularization))

    data_path = '../../data/records_final.pkl'
    voc_path = '../../data/voc_final.pkl'
    device = torch.device('cuda:0')

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
    TEST = False
    ddi_rate, ja, prauc, avg_p, avg_r, avg_f1 = [0 for i in range(6)]

    model = Retrive_Treat(voc_size, emb_dim=emb_dims, device=device)
    if TEST:
        model.load_state_dict(torch.load(open(os.path.join("saved", model_name, resume_name), 'rb')))
    model.to(device=device)

    weight_decay_list = (param for name, param in model.named_parameters() if name[-4:] != 'bias' and "bn" not in name)
    no_decay_list = (param for name, param in model.named_parameters() if name[-4:] == 'bias' or "bn" in name)
    parameters = [{'params': weight_decay_list},
                  {'params': no_decay_list, 'weight_decay': 0.}]

    optimizer = Adam(parameters, lr=LR, weight_decay=l2_regularization)

    if TEST:
        eval(model, data_test, voc_size, 0)
    else:
        for epoch in range(EPOCH):
            start_time = time.time()
            model.train()
            loss_record = []
            for step, input in enumerate(data_train[:10]):
                if len(input) < 2:
                    continue

                loss_1 = 0
                loss_2 = 0
                for i in range(1, len(input)):
                    loss_1_target = np.zeros([1, voc_size[2]])
                    loss_1_target[:, input[i][2]] = 1

                    loss_2_target = np.full((1, voc_size[2]), -1)
                    for idx, item in enumerate(input[i][2]):
                        loss_2_target[0][idx] = item

                    output = model(input[:i+1])
                    loss_1 += F.binary_cross_entropy(output, torch.FloatTensor(loss_1_target).to(device))
                    loss_2 += F.multilabel_margin_loss(output, torch.LongTensor(loss_2_target).to(device))

                    loss = 0.9 * loss_1 + 0.01 * loss_2

                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()
                loss_record.append(loss.item())

                # llprint('\rTrain--Epoch: %d, Step: %d/%d' % (epoch, step, len(data_train)))

            ddi_rate, ja, prauc, avg_p, avg_r, avg_f1 = eval(model, data_test[:10], voc_size, epoch, training=False)

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
    # test_test('Retain_6_10_test_all_input.txt')
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
    #     ddi_rate, ja, prauc, avg_p, avg_r, avg_f1 = train(LR=0.0008107578622821, emb_dims=128, l2_regularization=5.981726129026116e-05)
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









