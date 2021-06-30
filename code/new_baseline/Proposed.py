import torch
import torch.nn as nn
import numpy as np
import dill
import torch.nn.functional as F
import time
from torch.optim import Adam
import os
from collections import defaultdict
from util import llprint, multi_label_metric, ddi_rate_score_, get_n_params, test_test, ddi_rate_score
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score, roc_curve
from bayes_opt import BayesianOptimization
import math
from sklearn.metrics import log_loss

model_name = 'Proposed'
resume_name = ''


class Seq2Seq(nn.Module):
    def __init__(self, emb_dims, voc_size, device):
        super(Seq2Seq, self).__init__()
        self.emb_dims = emb_dims
        self.voc_size = voc_size
        self.device = device
        self.input_len = self.voc_size[0] + self.voc_size[1] + self.voc_size[2]

        self.embedding = nn.Sequential(
            nn.Embedding(self.input_len + 1, self.emb_dims),
            nn.Dropout(p=0.3))

        self.encoder = nn.GRU(emb_dims, emb_dims, batch_first=True)

        self.decoder = nn.GRU(emb_dims, emb_dims, batch_first=True)

        self.output = nn.Sequential(
            nn.Linear(emb_dims, self.input_len),
            nn.Linear(self.input_len, self.input_len),
            nn.Linear(self.input_len, self.input_len))

    def forward(self, admissions):
        input_seq = []
        for i in range(0, len(admissions)):
            adm = admissions[i]
            input_adm = []
            input_adm.extend(adm[0])
            input_adm.extend(list(np.array(adm[1])+self.voc_size[0]))
            input_adm.extend(list(np.array(adm[2])+self.voc_size[0]+self.voc_size[1]))

            if len(input_adm) < self.input_len:
                input_adm.extend([self.input_len]*(self.input_len-len(input_adm)))

            input_seq.append(input_adm)

        admissions_embedding = self.embedding(torch.LongTensor(input_seq).to(self.device))  # (visit,input_len,emb_dim)
        admissions_embedding = torch.sum(admissions_embedding, dim=1)  # (visit, emb_dim)
        encoder_embedding, _ = self.encoder(admissions_embedding.unsqueeze(dim=0))  # (1, visit, emb_dim)
        decoder_embedding, _ = self.decoder(encoder_embedding)  # (1, visit, emb_dim)
        decoder_embedding = decoder_embedding.squeeze(dim=0)  # (visit, emb_dim)

        output = self.output(decoder_embedding)
        output = F.sigmoid(output)
        return input_seq, output


def mask(input_seq, voc_size):
    input_seq = np.array(input_seq)
    padding_matrix = np.ones_like(input_seq) * voc_size
    mask_index = (input_seq == padding_matrix)
    non_mask_index = 1 - mask_index
    return non_mask_index


def get_patient_result(input_seq, prediction, non_mask_index, max_size):
    input_seq = input_seq.detach().cpu().numpy()
    prediction = prediction.detach().cpu().numpy()

    jaccard_all = []
    for patient in range(input_seq.shape[0]):
        input_seq_ = input_seq[patient, :]
        prediction_ = prediction[patient, :]
        non_mask_index_ = non_mask_index[patient, :]
        real = input_seq_[np.where(non_mask_index_ == 1)]
        pred = prediction_[np.where(non_mask_index_ == 1)]

        inter = set(real) & set(pred)
        union = set(real) | set(pred)

        jaccard_score = 0 if union == 0 else len(inter) / len(union)
        jaccard_all.append(jaccard_score)

    return np.mean(jaccard_all)


def eval_pre_train_seq2seq(model, data_eval, voc_size, device):
    with torch.no_grad():
        model.eval()

        loss_record = []
        jaccard_all = []
        for step, input in enumerate(data_eval):
            if len(input) < 2:
                continue

            input_seq, prediction = model(input)
            non_mask_index = mask(input_seq, voc_size[0] + voc_size[1] + voc_size[2])

            input_seq = torch.FloatTensor(np.array(input_seq))*torch.FloatTensor(non_mask_index)
            input_seq = input_seq.to(device)
            prediction = prediction*torch.FloatTensor(non_mask_index).to(device)

            jaccard = get_patient_result(input_seq,
                                               prediction,
                                               non_mask_index,
                                               voc_size[0] + voc_size[1] + voc_size[2])

            jaccard_all.append(jaccard)
        return np.mean(jaccard_all)


def pre_train_seq2seq(emb_dims, LR, l2_regularization):
    emb_dims = 2 ** int(emb_dims)
    LR = 10 ** LR
    l2_regularization = 10 ** l2_regularization
    print('emb_dims____{}___LR___{}____l2_regularization___{}'.format(emb_dims, LR, l2_regularization))

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
    model = Seq2Seq(emb_dims=emb_dims, voc_size=voc_size, device=device)
    model.to(device)

    weight_decay_list = (param for name, param in model.named_parameters() if name[-4:] != 'bias' and "bn" not in name)
    no_decay_list = (param for name, param in model.named_parameters() if name[-4:] == 'bias' or "bn" in name)
    parameters = [{'params': weight_decay_list},
                  {'params': no_decay_list, 'weight_decay': 0.}]

    optimizer = Adam(parameters, lr=LR, weight_decay=l2_regularization)

    for epoch in range(EPOCH):
        model.train()
        loss_record = []
        for step, input in enumerate(data_train):
            if len(input) < 2:
                continue

            input_seq, prediction = model(input)
            non_mask_index = mask(input_seq, voc_size[0] + voc_size[1] + voc_size[2])

            input_seq = torch.FloatTensor(np.array(input_seq))*torch.FloatTensor(non_mask_index)
            input_seq = input_seq.to(device)
            non_mask_index_ = torch.FloatTensor(non_mask_index).to(device)
            prediction = prediction*non_mask_index_

            loss = F.binary_cross_entropy_with_logits(prediction, input_seq)

            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

            loss_record.append(loss.item())

        if (epoch +1) % 1 == 0:
            jaccard = eval_pre_train_seq2seq(model, data_eval, voc_size, device)
            print('epoch__{}__loss_train___{}___ja_eval__{}'.format(int(epoch),
                                                                    np.mean(loss_record),
                                                                    jaccard))

    return jaccard


if __name__ == '__main__':
    test_test('pre_train_seq2seq.txt')
    Encode_Decode_Time_BO = BayesianOptimization(
            pre_train_seq2seq, {
                'emb_dims': (5, 8),
                'LR': (-5, 0),
                'l2_regularization': (-8, -3),
            }
        )
    Encode_Decode_Time_BO.maximize()
    print(Encode_Decode_Time_BO.max)

