import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import dill
import os
from torch.optim import Adam
from util import sequence_output_process, sequence_metric, ddi_rate_score, llprint, test_test
import time
from bayes_opt import BayesianOptimization

model_name = 'LEAP'
resume_name = ''


class LEAP(nn.Module):
    def __init__(self,  vocab_size, emb_dim=64, device=torch.device('cuda:0')):
        super(LEAP, self).__init__()
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.device = device

        self.SOS_TOKEN = vocab_size[0]
        self.END_TOKEN = vocab_size[0] + 1
        
        self.encode_embedding = nn.Sequential(
            nn.Embedding(vocab_size[0] + vocab_size[1] + vocab_size[2] + 1, emb_dim,),
            nn.Dropout(0.3)
        )
        
        self.decode_embedding = nn.Sequential(
            nn.Embedding(vocab_size[0] + 2, emb_dim,),
            nn.Dropout(0.3)
        )

        self.attention_layer = nn.Linear(emb_dim*2, 1)

        self.gru = nn.GRU(emb_dim*2, emb_dim, batch_first=True)
        self.output = nn.Linear(emb_dim, vocab_size[0] + 2)

    def forward(self, input, input_, max_len=20):
        device = self.device
        input_diagnosis = torch.LongTensor(input[0]).to(device)
        input_procedure = torch.LongTensor(input[1]).to(device)
        input_medicine = torch.LongTensor(input[2]).to(device)
        
        diagnosis_embedding = self.encode_embedding(input_diagnosis.unsqueeze(dim=0)).squeeze(dim=0)
        procedure_embedding = self.encode_embedding(input_procedure.unsqueeze(dim=0)).squeeze(dim=0)
        medicine_embedding = self.encode_embedding(input_medicine.unsqueeze(dim=0)).squeeze(dim=0)
        diagnosis_embedding = torch.cat((diagnosis_embedding, procedure_embedding, medicine_embedding), dim=0)
        
        outputs = []
        hidden_state = None
        
        if self.training:
            for med_code in [self.SOS_TOKEN] + input_[0]:
                dec_input = torch.LongTensor([med_code]).unsqueeze(dim=0).to(device)
                dec_input = self.decode_embedding(dec_input).squeeze(dim=0)
                
                if hidden_state is None:
                    hidden_state = dec_input
                hidden_state_repeat = hidden_state.repeat(diagnosis_embedding.size(0), 1)  # (len, dim)
                combined_weight = torch.cat([hidden_state_repeat, diagnosis_embedding], dim=-1)  # (len, 2*dim)
                attention_weight = F.softmax(self.attention_layer(combined_weight).t(), dim=-1)  # (1, len)
                attention_weight_ = attention_weight.repeat(diagnosis_embedding.size(1), 1).t()  # (len, dim)
                diagnosis_embedding_ = attention_weight_.mul(diagnosis_embedding)  # (len, dim)
                diagnosis_embedding_ = diagnosis_embedding_.sum(0).unsqueeze(dim=0)  # (1, dim)
                _, hidden_state = self.gru(torch.cat([diagnosis_embedding_, dec_input], dim=-1).unsqueeze(dim=0), hidden_state.unsqueeze(dim=0))
                hidden_state = hidden_state.squeeze(dim=0)
                outputs.append(self.output(F.relu(hidden_state)))
            return torch.cat(outputs, dim=0)
        else:
            for di in range(max_len):
                if di == 0:
                    dec_input = torch.LongTensor([[self.SOS_TOKEN]]).to(device)
                dec_input = self.decode_embedding(dec_input).squeeze(dim=0)
                if hidden_state is None:
                    hidden_state = dec_input
                hidden_state_repeat = hidden_state.repeat(diagnosis_embedding.size(0), 1)
                combined_weight = torch.cat([hidden_state_repeat, diagnosis_embedding], dim=-1)
                attention_weight = F.softmax(self.attention_layer(combined_weight).t(), dim=-1)
                attention_weight_ = attention_weight.repeat(diagnosis_embedding.size(1), 1).t()
                diagnosis_embedding_ = attention_weight_.mul(diagnosis_embedding)
                diagnosis_embedding_ = diagnosis_embedding_.sum(0).unsqueeze(dim=0)
                _, hidden_state = self.gru(torch.cat([diagnosis_embedding_, dec_input], dim=-1).unsqueeze(dim=0), hidden_state.unsqueeze(dim=0))
                hidden_state = hidden_state.squeeze(dim=0)
                output = self.output(F.relu(hidden_state))
                topv, topi = output.data.topk(1)
                outputs.append(F.softmax(output, dim=-1))
                dec_input = topi.detach()
            return torch.cat(outputs, dim=0)


def eval(model, data_eval, voc_size, epoch):
    with torch.no_grad():
        print('')
        model.eval()

        ja, prauc, avg_p, avg_r, avg_f1 = [[] for _ in range(5)]
        y_pred_label_all_for_ddi = []
        for step, input in enumerate(data_eval):
            if len(input) < 2:  # visit > 2
                continue
            y_label_patient = []
            y_pred_prob_patient = []
            y_pred_label_patient = []
            for i in range(1, len(input)):
                y = np.zeros(voc_size[0])
                y[input[i][0]] = 1
                y_label_patient.append(y)  # 获取标签

                prediction = model(input[i-1], input[i])
                prediction = prediction.detach().cpu().numpy()
                out_list, sorted_prediction = sequence_output_process(prediction, [voc_size[0], voc_size[0] + 1])

                y_pred_label_patient.append(sorted(sorted_prediction))
                y_pred_prob_patient.append(np.mean(prediction[:, :-2], axis=0))

            y_pred_label_all_for_ddi.append(y_pred_label_patient)

            adm_ja, adm_prauc, adm_avg_p, adm_avg_r, adm_avg_f1 = sequence_metric(np.array(y_label_patient),
                                                                                  0, np.array(y_pred_prob_patient), np.array(y_pred_label_patient))
            ja.append(adm_ja)
            prauc.append(adm_prauc)
            avg_p.append(adm_avg_p)
            avg_r.append(adm_avg_r)
            avg_f1.append(adm_avg_f1)

        llprint(
            '\t Eval---epoch: %.2f, Jaccard: %.4f,  PRAUC: %.4f, AVG_PRC: %.4f, '
            'AVG_RECALL: %.4f, AVG_F1: %.4f\n' % (
                epoch, np.mean(ja), np.mean(prauc), np.mean(avg_p), np.mean(avg_r), np.mean(avg_f1)
            ))
        return np.mean(ja), np.mean(prauc), np.mean(avg_p), np.mean(avg_r), np.mean(avg_f1)


# diagnosis_t ---> drug_t
def train(LR, emb_dims, l2_regularization):
    # LR = 10 ** LR
    # emb_dims = 2 ** int(emb_dims)
    # l2_regularization = 10 ** l2_regularization
    print('learning_rate--{}---emb_dims---{}---l2_regularization--{}'.format(LR, emb_dims, l2_regularization))

    data_path = '../../../data/records_final_icd_2.pkl'
    voc_path = '../../../data/voc_final_icd_2.pkl'
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

    model = LEAP(vocab_size=voc_size, emb_dim=emb_dims, device=device)
    model.to(device)
    weight_decay_list = (param for name, param in model.named_parameters() if name[-4:] != 'bias' and "bn" not in name)
    no_decay_list = (param for name, param in model.named_parameters() if name[-4:] == 'bias' or "bn" in name)
    parameters = [{'params': weight_decay_list},
                  {'params': no_decay_list, 'weight_decay': 0.}]

    optimizer = Adam(parameters, lr=LR, weight_decay=l2_regularization)

    EPOCH = 10
    TEST = False
    END_TOKEN = voc_size[0] + 1
    ddi_rate, ja, prauc, avg_p, avg_r, avg_f1 = [0 for i in range(6)]

    for epoch in range(EPOCH):
        start_time = time.time()
        loss_record = []
        for step, input in enumerate(data_train):
            if len(input) < 2:  # visit >= 2
                continue
            model.train()
            for adm in range(1, len(input)):
                target = input[adm][0] + [END_TOKEN]
                predictions = model(input[adm-1], input[adm])
                loss = F.cross_entropy(predictions, torch.LongTensor(target).to(device))

                loss_record.append(loss.item())
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()

        ja, prauc, aug_p, avg_r, avg_f1 = eval(model, data_test, voc_size, epoch)
        end_time = time.time()
        elapsed_time = (end_time - start_time) / 60
        llprint('\tEpoch: %d, Loss1: %.4f, '
                'One Epoch Time: %.2fm, Appro Left Time: %.2fh\n' % (epoch,
                                                                     np.mean(loss_record),
                                                                     elapsed_time,
                                                                     elapsed_time * (EPOCH - epoch - 1) / 60))


    # return ja
    return ddi_rate, ja, prauc, avg_p, avg_r, avg_f1


if __name__ == '__main__':
    test_test('LEAP_12_6_test_0.3_99分类.txt')
    # Encode_Decode_Time_BO = BayesianOptimization(
    #         train, {
    #             'emb_dims': (5, 8),
    #             'LR': (-5, 0),
    #             'l2_regularization': (-8, -3),
    #         }
    #     )
    # Encode_Decode_Time_BO.maximize()
    # print(Encode_Decode_Time_BO.max)
    ddi_rate_all, ja_all, prauc_all, avg_p_all, avg_r_all, avg_f1_all = [[] for _ in range(6)]
    for i in range(10):
        ddi_rate, ja, prauc, avg_p, avg_r, avg_f1 = train(LR=0.0013058395603257739, emb_dims=64,
                                                          l2_regularization=3.275471783064622e-07)
        ddi_rate_all.append(ddi_rate)
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
