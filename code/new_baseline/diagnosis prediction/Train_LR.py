import dill
import numpy as np
from util import multi_label_metric
import torch.nn as nn
import torch
from torch.optim import Adam
from util import sequence_output_process, sequence_metric, ddi_rate_score_, llprint, test_test
import time
from bayes_opt import BayesianOptimization
import torch.nn.functional as F


class LogisticRegression(nn.Module):
    def __init__(self, voc_size, emb_dim, device=torch.device('cuda:0')):
        super(LogisticRegression, self).__init__()
        self.voc_size = voc_size
        self.input_dim = voc_size[0] + voc_size[1] + voc_size[2]
        self.output_dim = voc_size[0]
        self.emb_dim = emb_dim

        self.embedding = nn.Sequential(
            nn.Embedding(self.input_dim + 1, emb_dim, padding_idx=self.input_dim),
            nn.Dropout(p=0.3))

        self.linear = nn.Sequential(
            nn.Linear(self.input_dim, self.output_dim),
            nn.Sigmoid())
        self.device = device

    def forward(self, input):
        input_seq = []
        max_len = self.input_dim

        input_seq.extend(input[0])
        input_seq.extend(list(np.array(input[1]) + self.voc_size[0]))
        input_seq.extend(list(np.array(input[2]) + self.voc_size[0] + self.voc_size[1]))

        if len(input_seq) < max_len:
            input_seq.extend([self.input_dim] * (max_len - len(input_seq)))

        input_emb = self.embedding(torch.LongTensor(input_seq).to(self.device))  # (max_len, dim_emb)
        input_emb = torch.sum(input_emb, dim=1)
        output = self.linear(input_emb)
        return output


def eval(model, data_eval, voc_size, epoch):
    with torch.no_grad():
        print('')
        model.eval()

        ja, prauc, avg_p, avg_r, avg_f1 = [[] for _ in range(5)]
        for step, input in enumerate(data_eval):
            y_label_patient = np.zeros(shape=[0, voc_size[0]])
            y_pred_patient = np.zeros(shape=[0, voc_size[0]])
            y_pred_prob_patient = np.zeros(shape=[0, voc_size[0]])

            if len(input) < 2:
                continue
            for i in range(1, len(input)):
                adm = input[i]
                target = np.zeros(shape=[voc_size[0]])
                target[adm[0]] = 1
                y_label_patient = np.concatenate((y_label_patient, target.reshape(1, -1)), axis=0)

                pred_prob = model(input[i-1])
                pred_prob = pred_prob.detach().cpu().numpy()
                y_pred_prob_patient = np.concatenate((y_pred_prob_patient, pred_prob.reshape(1, -1)), axis=0)

                y_pred = pred_prob.copy()
                y_pred[y_pred >= 0.3] = 1
                y_pred[y_pred < 0.3] = 0

                y_pred_patient = np.concatenate((y_pred_patient, y_pred.reshape(1, -1)), axis=0)

            adm_ja, adm_prauc, adm_avg_p, adm_avg_r, adm_avg_f1 = multi_label_metric(np.array(y_label_patient),
                                                                                     np.array(y_pred_patient),
                                                                                     np.array(y_pred_prob_patient))
            ja.append(adm_ja)
            prauc.append(adm_prauc)
            avg_p.append(adm_avg_p)
            avg_r.append(adm_avg_r)
            avg_f1.append(adm_avg_f1)
        llprint(
            '\t epoch: %.2f, Jaccard: %.4f,  '
            'PRAUC: %.4f, AVG_PRC: %.4f, AVG_RECALL: %.4f, AVG_F1: %.4f\n' % (
                epoch, np.mean(ja), np.mean(prauc), np.mean(avg_p), np.mean(avg_r), np.mean(avg_f1)
            ))
        return np.mean(ja), np.mean(prauc), np.mean(avg_p), np.mean(avg_r), np.mean(avg_f1)


def train(LR, l2_regularization, emb_dim):
    # LR = 10 ** LR
    # l2_regularization = 10 ** l2_regularization
    # emb_dim = 2 ** int(emb_dim)
    print('LR_____{}___l2_regularization____{}____emb_dim{}'.format(LR, l2_regularization, emb_dim))

    data_path = '../../../data/records_final_icd_2.pkl'
    voc_path = '../../../data/voc_final_icd_2.pkl'
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

    model = LogisticRegression(voc_size=voc_size, emb_dim=emb_dim)
    model.to(device)
    weight_decay_list = (param for name, param in model.named_parameters() if name[-4:] != 'bias' and "bn" not in name)
    no_decay_list = (param for name, param in model.named_parameters() if name[-4:] == 'bias' or "bn" in name)
    parameters = [{'params': weight_decay_list},
                  {'params': no_decay_list, 'weight_decay': 0.}]
    optimizer = Adam(parameters, lr=LR, weight_decay=l2_regularization)

    EPOCH = 10
    ddi_rate, ja, prauc, avg_p, avg_r, avg_f1 = [0 for i in range(6)]

    for epoch in range(EPOCH):
        start_time = time.time()
        loss_record = []
        for step, input in enumerate(data_train):
            if len(input) < 2:
                continue
            model.train()
            for i in range(1, len(input)):
                adm = input[i]
                target = np.zeros([voc_size[0]])
                target[adm[0]] = 1

                prediction = model(input[i-1])
                loss = F.binary_cross_entropy(prediction, torch.FloatTensor(target).to(device))

                loss_record.append(loss.item())
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()

        ja, prauc, avg_p, avg_r, avg_f1 = eval(model, data_test, voc_size, epoch)
        end_time = time.time()
        elapsed_time = (end_time - start_time) / 60
        llprint('\tEpoch: %d, Loss1: %.4f, '
                'One Epoch Time: %.2fm, Appro Left Time: %.2fh\n' % (epoch,
                                                                     np.mean(loss_record),
                                                                     elapsed_time,
                                                                     elapsed_time * (EPOCH - epoch - 1) / 60))
    return ja, prauc, avg_p, avg_r, avg_f1
    # return ja


if __name__ == '__main__':
    test_test('LR_12_6_诊断_0.3_test_前一次记录_99个分类.txt')
    # Encode_Decode_Time_BO = BayesianOptimization(
    #         train, {
    #             'emb_dim': (5, 8),
    #             'LR': (-5, 0),
    #             'l2_regularization': (-8, -3),
    #         }
    #     )
    # Encode_Decode_Time_BO.maximize()
    # print(Encode_Decode_Time_BO.max)

    ddi_rate_all, ja_all, prauc_all, avg_p_all, avg_r_all, avg_f1_all = [[] for _ in range(6)]
    for i in range(10):
        ja, prauc, avg_p, avg_r, avg_f1 = train(LR=0.00029392811696937185,
                                                l2_regularization=1e-8, emb_dim=256)
        ja_all.append(ja)
        prauc_all.append(prauc)
        avg_p_all.append(avg_p)
        avg_r_all.append(avg_r)
        avg_f1_all.append(avg_f1)
    print('ddi_rate{}---ja{}--prauc{}---avg_p{}---avg_r---{}--avg_f1--{}'.format(np.mean(ddi_rate_all),
                                                                                 np.mean(ja_all),
                                                                                 np.mean(prauc_all),
                                                                                 np.mean(avg_p_all),
                                                                                 np.mean(avg_r_all),
                                                                                 np.mean(avg_f1_all)))
