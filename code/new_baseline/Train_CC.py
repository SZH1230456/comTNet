import numpy as np
from sklearn.ensemble import RandomForestClassifier
import dill
from torch.optim import Adam
from util import sequence_output_process, sequence_metric, ddi_rate_score_, llprint, test_test, multi_label_metric
import time
from bayes_opt import BayesianOptimization
import torch.nn.functional as F
import torch.nn as nn
import torch


class LogisticRegression(nn.Module):
    def __init__(self, emb_dims, input_dim, device=torch.device('cuda:0')):
        super(LogisticRegression, self).__init__()
        self.emb_dims = emb_dims
        self.input_dim = input_dim
        self.output_dim = 1
        self.device = device
        self.linear = nn.Sequential(
            nn.Linear(self.input_dim, self.output_dim),
            nn.Sigmoid())

    def forward(self, input):
        output = self.linear(input)
        return output


class ClassifierChain(nn.Module):
    def __init__(self, emb_dims, voc_size, device=torch.device('cuda:0')):
        super(ClassifierChain, self).__init__()
        self.emb_dims = emb_dims
        self.voc_size = voc_size
        self.input_dim = voc_size[0] + voc_size[1]
        self.num_model = voc_size[2]
        self.device = device
        self.embedding = nn.Sequential(
            nn.Embedding(self.input_dim+1, self.emb_dims, padding_idx=self.input_dim),
            nn.Dropout(p=0.3))
        self.lrs = [[] for i in range(self.num_model)]
        for i in range(self.num_model):
            self.lrs[i] = LogisticRegression(emb_dims=self.emb_dims+i, input_dim=i+self.input_dim, device=self.device)
            self.lrs[i].to(device)

    def forward(self, input):
        input_ = []
        max_len = self.input_dim
        input_.extend(input[0])
        input_.extend(list(np.array(input[1])+self.voc_size[0]))
        if len(input_) < max_len:
            input_.extend([max_len]*(max_len-len(input_)))

        input_all = self.embedding(torch.LongTensor(input_).to(self.device))  # (input_dim, emb_dim)
        input_all = torch.sum(input_all, dim=1)  # (input_dim, 1)

        prediction_adm = torch.zeros(size=[0]).to(self.device)
        for i in range(self.num_model):
            y = torch.Tensor(np.array(input[2][0, :i]).reshape(-1)).to(self.device)
            input_all_ = input_all.clone()
            input_all_ = torch.cat((input_all_, y), dim=0)
            prediction = self.lrs[i](input_all_)
            prediction_adm = torch.cat((prediction_adm, prediction), dim=0)
        return prediction_adm

    def predict(self, input):
        input_ = []
        max_len = self.input_dim
        input_.extend(input[0])
        input_.extend(list(np.array(input[1]) + self.voc_size[0]))
        if len(input_) < max_len:
            input_.extend([max_len] * (max_len - len(input_)))

        input_all = self.embedding(torch.LongTensor(input_).to(self.device))  # (input_dim, emb_dim)
        input_all = torch.sum(input_all, dim=1)  # (input_dim, 1)

        prediction_adm = torch.zeros(size=[0]).to(self.device)
        for i in range(self.num_model):
            y = prediction_adm[:i]
            input_all_ = input_all.clone()
            input_all_ = torch.cat((input_all_, y), dim=0)
            prediction = self.lrs[i](input_all_)
            prediction_adm = torch.cat((prediction_adm, prediction), dim=0)
        return prediction_adm


def eval(model, data_eval, voc_size, epoch):
    with torch.no_grad():
        print('')
        model.eval()

        ja, prauc, avg_p, avg_r, avg_f1 = [[] for _ in range(5)]
        y_pred_label_all_for_ddi = np.zeros(shape=[0, voc_size[2]])
        for step, input in enumerate(data_eval):
            y_label_patient = np.zeros(shape=[0, voc_size[2]])
            y_pred_patient = np.zeros(shape=[0, voc_size[2]])
            y_pred_prob_patient = np.zeros(shape=[0, voc_size[2]])

            if len(input) < 2:
                continue
            for i in range(1, len(input)):
                adm = input[i]
                target = np.zeros(shape=[voc_size[2]])
                target[adm[2]] = 1
                y_label_patient = np.concatenate((y_label_patient, target.reshape(1, -1)), axis=0)

                pred_prob = model.predict(adm)
                pred_prob = pred_prob.detach().cpu().numpy()
                y_pred_prob_patient = np.concatenate((y_pred_prob_patient, pred_prob.reshape(1, -1)), axis=0)

                y_pred = pred_prob.copy()
                y_pred[y_pred >= 0.3] = 1
                y_pred[y_pred < 0.3] = 0

                y_pred_patient = np.concatenate((y_pred_patient, y_pred.reshape(1, -1)), axis=0)
                y_pred = y_pred.reshape(1, -1)
                for idx, value in enumerate(y_pred[0, :]):
                    if value == 1:
                        y_pred[0, idx] = idx
                y_pred_label_all_for_ddi = np.concatenate((y_pred_label_all_for_ddi, y_pred), axis=0)

            adm_ja, adm_prauc, adm_avg_p, adm_avg_r, adm_avg_f1 = multi_label_metric(np.array(y_label_patient),
                                                                                     np.array(y_pred_patient),
                                                                                     np.array(y_pred_prob_patient))
            ja.append(adm_ja)
            prauc.append(adm_prauc)
            avg_p.append(adm_avg_p)
            avg_r.append(adm_avg_r)
            avg_f1.append(adm_avg_f1)
        ddi_rate = ddi_rate_score_(y_pred_label_all_for_ddi)
        llprint(
            '\t epoch: %.2f, DDI Rate: %.4f, Jaccard: %.4f,  '
            'PRAUC: %.4f, AVG_PRC: %.4f, AVG_RECALL: %.4f, AVG_F1: %.4f\n' % (
                epoch, ddi_rate, np.mean(ja), np.mean(prauc), np.mean(avg_p), np.mean(avg_r), np.mean(avg_f1)
            ))
        return ddi_rate, np.mean(ja), np.mean(prauc), np.mean(avg_p), np.mean(avg_r), np.mean(avg_f1)


def train(LR, l2_regularization, emb_dim):
    LR = 10 ** LR
    l2_regularization = 10 ** l2_regularization
    emb_dim = 2 ** int(emb_dim)
    print('LR_____{}___l2_regularization____{}____emb_dim{}'.format(LR, l2_regularization, emb_dim))

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

    model = ClassifierChain(emb_dims=emb_dim, voc_size=voc_size)
    model.to(device)

    weight_decay_list = (param for name, param in model.named_parameters() if name[-4:] != 'bias' and "bn" not in name)
    no_decay_list = (param for name, param in model.named_parameters() if name[-4:] == 'bias' or "bn" in name)
    parameters = [{'params': weight_decay_list},
                  {'params': no_decay_list, 'weight_decay': 0.}]
    optimizer = Adam(parameters, lr=LR, weight_decay=l2_regularization)
    # optimizer = Adam(model.parameters(), lr=LR)

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
                if epoch == 0:
                    adm = input[i]
                    target = np.zeros([1, voc_size[2]])

                    target[:, adm[2]] = 1
                    adm[2] = target

                prediction = model(adm)
                loss = F.binary_cross_entropy_with_logits(prediction.unsqueeze(dim=0), torch.FloatTensor(target).to(device))

                loss_record.append(loss.item())
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()
        ddi_rate, ja, prauc, avg_p, avg_r, avg_f1 = eval(model, data_eval, voc_size, epoch)
        end_time = time.time()
        elapsed_time = (end_time - start_time) / 60
        llprint('\tEpoch: %d, Loss1: %.4f, '
                'One Epoch Time: %.2fm, Appro Left Time: %.2fh\n' % (epoch,
                                                                     np.mean(loss_record),
                                                                     elapsed_time,
                                                                     elapsed_time * (EPOCH - epoch - 1) / 60))
    # return ddi_rate, ja, prauc, avg_p, avg_r, avg_f1

    return ja


if __name__ == '__main__':
    test_test('cc_6_18_train.txt')
    Encode_Decode_Time_BO = BayesianOptimization(
            train, {
                'emb_dim': (5, 8),
                'LR': (-5, 0),
                'l2_regularization': (-8, -3),
            }
        )
    Encode_Decode_Time_BO.maximize()
    print(Encode_Decode_Time_BO.max)