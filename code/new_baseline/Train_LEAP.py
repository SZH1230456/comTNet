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

        self.SOS_TOKEN = vocab_size[2]
        self.END_TOKEN = vocab_size[2] + 1
        
        self.encode_embedding = nn.Sequential(
            nn.Embedding(vocab_size[0] + vocab_size[1], emb_dim,),
            nn.Dropout(0.3)
        )
        
        self.decode_embedding = nn.Sequential(
            nn.Embedding(vocab_size[2] + 2, emb_dim,),
            nn.Dropout(0.3)
        )

        self.attention_layer = nn.Linear(emb_dim*2, 1)

        self.gru = nn.GRU(emb_dim*2, emb_dim, batch_first=True)
        self.output = nn.Linear(emb_dim, vocab_size[2] + 2)

    def forward(self, input, max_len=20):
        device = self.device
        input_diagnosis = torch.LongTensor(input[0]).to(device)
        input_procedure = torch.LongTensor(input[1]).to(device)
        
        diagnosis_embedding = self.encode_embedding(input_diagnosis.unsqueeze(dim=0)).squeeze(dim=0)
        procedure_embedding = self.encode_embedding(input_procedure.unsqueeze(dim=0)).squeeze(dim=0)
        diagnosis_embedding = torch.cat((diagnosis_embedding, procedure_embedding), dim=0)
        
        outputs = []
        hidden_state = None
        
        if self.training:
            for med_code in [self.SOS_TOKEN] + input[2]:
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
                y = np.zeros(voc_size[2])
                y[input[i][2]] = 1
                y_label_patient.append(y)  # 获取标签

                prediction = model(input[i])
                prediction = prediction.detach().cpu().numpy()
                out_list, sorted_prediction = sequence_output_process(prediction, [voc_size[2], voc_size[2] + 1])

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
            # llprint('\rEval--Epoch: %d, Step: %d/%d' % (epoch, step, len(data_eval)))

        ddi_rate = ddi_rate_score(y_pred_label_all_for_ddi)
        llprint(
            '\t Eval---epoch: %.2f, DDI Rate: %.4f, Jaccard: %.4f,  PRAUC: %.4f, AVG_PRC: %.4f, '
            'AVG_RECALL: %.4f, AVG_F1: %.4f\n' % (
                epoch, ddi_rate, np.mean(ja), np.mean(prauc), np.mean(avg_p), np.mean(avg_r), np.mean(avg_f1)
            ))
        return ddi_rate, np.mean(ja), np.mean(prauc), np.mean(avg_p), np.mean(avg_r), np.mean(avg_f1)


# diagnosis_t ---> drug_t
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

    model = LEAP(vocab_size=voc_size, emb_dim=emb_dims, device=device)
    model.to(device)
    weight_decay_list = (param for name, param in model.named_parameters() if name[-4:] != 'bias' and "bn" not in name)
    no_decay_list = (param for name, param in model.named_parameters() if name[-4:] == 'bias' or "bn" in name)
    parameters = [{'params': weight_decay_list},
                  {'params': no_decay_list, 'weight_decay': 0.}]

    optimizer = Adam(parameters, lr=LR, weight_decay=l2_regularization)

    EPOCH = 10
    TEST = False
    END_TOKEN = voc_size[2] + 1
    ddi_rate, ja, prauc, avg_p, avg_r, avg_f1 = [0 for i in range(6)]

    if TEST:
        model.load_state_dict(torch.load(open(os.path.join("saved", model_name, resume_name), 'rb')))

    for epoch in range(EPOCH):
        start_time = time.time()
        loss_record = []
        for step, input in enumerate(data_train):
            if len(input) < 2:  # visit > 2
                continue
            model.train()
            for adm in range(1, len(input)):
                target = input[adm][2] + [END_TOKEN]
                predictions = model(input[adm])
                loss = F.cross_entropy(predictions, torch.LongTensor(target).to(device))

                loss_record.append(loss.item())
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()
                # llprint('\rTrain--Epoch: %d, Step: %d/%d' % (epoch, step, len(data_train)))

        ddi_rate, ja, prauc, aug_p, avg_r, avg_f1 = eval(model, data_eval, voc_size, epoch)
        end_time = time.time()
        elapsed_time = (end_time - start_time) / 60
        llprint('\tEpoch: %d, Loss1: %.4f, '
                'One Epoch Time: %.2fm, Appro Left Time: %.2fh\n' % (epoch,
                                                                     np.mean(loss_record),
                                                                     elapsed_time,
                                                                     elapsed_time * (EPOCH - epoch - 1) / 60))
    torch.save(model.state_dict(),
               open(os.path.join('saved', model_name, 'fine_JA_%.4f_DDI_%.4f.model' % (ja, ddi_rate)),
                    'wb'))
    print('')

    return ja


def fine_tuning_LEAP(pre_training_model_name, LR, l2_regularization):
    data_path = '../../data/records_final.pkl'
    voc_path = '../../data/voc_final.pkl'
    device = torch.device('cuda:0')

    data = dill.load(open(data_path, 'rb'))
    voc = dill.load(open(voc_path, 'rb'))
    diag_voc, pro_voc, med_voc = voc['diag_voc'], voc['pro_voc'], voc['med_voc']
    ddi_A = dill.load(open('../../data/ddi_A_final.pkl', 'rb'))

    split_point = int(len(data) * 2 / 3)
    data_train = data[:split_point]
    eval_len = int(len(data[split_point:]) / 2)
    data_test = data[split_point:split_point + eval_len]
    data_eval = data[split_point + eval_len:]
    voc_size = (len(diag_voc.idx2word), len(pro_voc.idx2word), len(med_voc.idx2word))

    # TODO: 修改emd_dims
    model = LEAP(vocab_size=voc_size, emb_dim=64, device=device)
    model.load_state_dict(torch.load(open(os.path.join("saved", model_name, pre_training_model_name), 'rb')))
    model.to(device)
    EPOCH = 10
    weight_decay_list = (param for name, param in model.named_parameters() if name[-4:] != 'bias' and "bn" not in name)
    no_decay_list = (param for name, param in model.named_parameters() if name[-4:] == 'bias' or "bn" in name)
    parameters = [{'params': weight_decay_list},
                  {'params': no_decay_list, 'weight_decay': 0.}]

    optimizer = Adam(parameters, lr=LR, weight_decay=l2_regularization)
    for epoch in range(EPOCH):
        start_time = time.time()
        loss_record = []
        for step, input in enumerate(data_train):
            if len(input) < 2:
                continue
            model.train()
            for i in range(1, len(input)):
                target = input[i][2]
                prediction = model(input[i])
                out_list, sorted_prediction = sequence_output_process(prediction.detach().cpu().numpy(), [voc_size[2], voc_size[2]+1])

                inter = set(out_list) & set(target)
                union = set(out_list) | set(target)
                jaccard = 0 if union == 0 else len(inter) / len(union)
                k = 1
                for j in out_list:
                    if k == 0:
                        k_flag = True
                        break
                    for p in out_list:
                        if ddi_A[j][p] == 1:
                            k = 0
                            break
                loss = - jaccard * k * torch.mean(F.log_softmax(prediction, dim=-1))

                loss_record.append(loss.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        ddi_rate, ja, prauc, avg_p, avg_r, avg_f1 = eval(model, data_test, voc_size, epoch)
        end_time = time.time()
        elapsed_time = (end_time - start_time) / 60
        llprint('\tEpoch: %d, Loss1: %.4f, '
                'One Epoch Time: %.2fm, Appro Left Time: %.2fh\n' % (epoch,
                                                                     np.mean(loss_record),
                                                                     elapsed_time,
                                                                     elapsed_time * (EPOCH - epoch - 1) / 60))
        return ja


if __name__ == '__main__':
    test_test('LEAP_6_16_train_all_input.txt')
    Encode_Decode_Time_BO = BayesianOptimization(
            train, {
                'emb_dims': (5, 8),
                'LR': (-5, 0),
                'l2_regularization': (-8, -3),
            }
        )
    Encode_Decode_Time_BO.maximize()
    print(Encode_Decode_Time_BO.max)
    # ja = train(LR=8.715253429438384e-05, emb_dims=256, l2_regularization=1e-8)