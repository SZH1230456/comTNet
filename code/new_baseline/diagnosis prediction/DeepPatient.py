import torch
import torch.nn as nn
import torch.nn.functional as F
import dill
from torch.optim import Adam
import time
import numpy as np
from util import llprint
from bayes_opt import BayesianOptimization
from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from util import llprint, multi_label_metric, ddi_rate_score_, get_n_params, test_test, ddi_rate_score


class DeNosingAutoEncoder(nn.Module):
    def __init__(self, encode_dim, input_dim, vocab_size):
        super(DeNosingAutoEncoder, self).__init__()
        self.diagnosis_length = vocab_size[0]
        self.procedure_length = vocab_size[1]
        self.medicine_length = vocab_size[2]

        self.encode1 = nn.Linear(input_dim, encode_dim)
        self.encode2 = nn.Linear(encode_dim, encode_dim)
        self.encode3 = nn.Linear(encode_dim, encode_dim)

        self.decode1 = nn.Linear(encode_dim, encode_dim)
        self.decode2 = nn.Linear(encode_dim, encode_dim)
        self.decode3 = nn.Linear(encode_dim, input_dim)

        self.input_dim = input_dim
        self.encode_dim = encode_dim

    def encode(self, input):
        input_diagnosis, input_procedure, input_medicine = input[0], input[1], input[2]
        input_procedure = [i + self.diagnosis_length for i in input_procedure]
        input_medicine = [i + self.diagnosis_length + self.procedure_length for i in input_medicine]

        input_diagnosis = torch.LongTensor(input_diagnosis)
        input_procedure = torch.LongTensor(input_procedure)
        input_medicine = torch.LongTensor(input_medicine)

        input_data = torch.cat((input_diagnosis, input_procedure, input_medicine), dim=0)
        input = F.one_hot(input_data, num_classes=self.diagnosis_length + self.procedure_length + self.medicine_length)
        input = input.float()
        input = torch.sum(input, dim=0)
        input = torch.unsqueeze(input, dim=0)
        hidden = self.encode1(input)
        hidden = self.encode2(torch.sigmoid(hidden))
        hidden = self.encode3(torch.sigmoid(hidden))
        return input_data, torch.sigmoid(hidden)

    def decode(self, encode_hidden):
        hidden = self.decode1(encode_hidden)
        hidden = self.decode2(torch.sigmoid(hidden))
        hidden = self.decode3(torch.sigmoid(hidden))
        return torch.sigmoid(hidden)

    def forward(self, input):
        input_, encode_hidden = self.encode(input)
        reconstruct_input = self.decode(encode_hidden)
        return input_, reconstruct_input


def train(learning_rate, hidden_size, l2_regularization):
    # learning_rate = 10 ** learning_rate
    # hidden_size = 2 ** int(hidden_size)
    # l2_regularization = 10 ** l2_regularization
    print(f"learning_rate: {learning_rate}, hidden_size: {hidden_size}, l2_regularization: {l2_regularization}")

    data_path = '../../../data/records_final_icd_2.pkl'
    voc_path = '../../../data/voc_final_icd_2.pkl'
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    data = dill.load(open(data_path, 'rb'))
    voc = dill.load(open(voc_path, 'rb'))
    diag_voc, pro_voc, med_voc = voc['diag_voc'], voc['pro_voc'], voc['med_voc']

    split_point = int(len(data) * 2 / 3)
    data_train = data[:split_point]
    eval_len = int(len(data[split_point:]) / 2)
    data_test = data[split_point:split_point + eval_len]
    data_eval = data[split_point + eval_len:]
    voc_size = (len(diag_voc.idx2word), len(pro_voc.idx2word), len(med_voc.idx2word))

    model = DeNosingAutoEncoder(encode_dim=hidden_size, input_dim=voc_size[0] + voc_size[1] + voc_size[2],
                                vocab_size=voc_size)
    model.to(device)
    weight_decay_list = (param for name, param in model.named_parameters() if name[-4:] != 'bias' and "bn" not in name)
    no_decay_list = (param for name, param in model.named_parameters() if name[-4:] == 'bias' or "bn" in name)
    parameters = [{'params': weight_decay_list},
                  {'params': no_decay_list, 'weight_decay': 0.}]

    optimizer = Adam(parameters, lr=learning_rate, weight_decay=l2_regularization)
    EPOCH = 10
    ddi_rate, ja, prauc, avg_p, avg_r, avg_f1 = [0 for i in range(6)]
    for epoch in range(EPOCH):
        start_time = time.time()
        loss_record = []
        for step, input in enumerate(data_train):
            if len(input) < 2:
                continue
            model.train()

            for adm in input:
                input_, reconstruct_x = model(adm)
                loss_1_target = np.zeros(shape=[1, voc_size[0] + voc_size[1] + voc_size[2]])
                loss_1_target[0, input_] = 1
                loss = F.binary_cross_entropy(reconstruct_x, torch.FloatTensor(loss_1_target).to(device))

                loss_record.append(loss.item())
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()
        eval_loss = eval(model, data_test, epoch, voc_size, device)
        if eval_loss <= 0.0383:
            torch.save({'model': model.state_dict()}, 'de_noising_auto_encoder.pth')
            print('保存成功！')

        end_time = time.time()
        elapsed_time = (end_time - start_time) / 60
        llprint('\tEpoch: %d, Loss_train: %.4f, '
                'One Epoch Time: %.2fm, Appro Left Time: %.2fh\n' % (epoch,
                                                                     np.mean(loss_record),
                                                                     elapsed_time,
                                                                     elapsed_time * (EPOCH - epoch - 1) / 60))

    return - eval_loss


def eval(model, data_eval, epoch, voc_size, device):
    with torch.no_grad():
        model.eval()
        loss_record = []
        for step, input in enumerate(data_eval):
            if len(input) < 2:
                continue
            for adm in input:
                input_, reconstruct_x = model(adm)
                loss_1_target = np.zeros(shape=[1, voc_size[0] + voc_size[1] + voc_size[2]])
                loss_1_target[0, input_] = 1
                loss = F.binary_cross_entropy(reconstruct_x, torch.FloatTensor(loss_1_target).to(device))
                loss_record.append(loss.item())
    llprint('\t epoch: %.2f, eval_loss: %.4f, ' % (
        epoch, np.mean(loss_record)))
    return np.mean(loss_record)


def get_dataset_representations_and_labels(model, data_set, voc_size, hidden_size):
    all_hidden_representation = np.zeros(shape=[0, hidden_size])
    all_labels = np.zeros(shape=[0, voc_size[0]])
    for step, input in enumerate(data_set):
        if len(input) < 2:
            continue

        for adm in input:
            _, hidden_representation = model.encode(adm)
            label = np.zeros(shape=[1, voc_size[0]])
            label[0, adm[0]] = 1

            all_hidden_representation = np.concatenate((all_hidden_representation,
                                                        hidden_representation.detach().cpu().numpy().reshape(-1,
                                                                                                             hidden_size)),
                                                       axis=0)
            all_labels = np.concatenate((all_labels, label), axis=0)
    return all_hidden_representation, all_labels


def get_diagnosis_performance():
    hidden_size = 32

    data_path = '../../../data/records_final_icd_2.pkl'
    voc_path = '../../../data/voc_final_icd_2.pkl'
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    data = dill.load(open(data_path, 'rb'))
    voc = dill.load(open(voc_path, 'rb'))
    diag_voc, pro_voc, med_voc = voc['diag_voc'], voc['pro_voc'], voc['med_voc']

    split_point = int(len(data) * 2 / 3)
    data_train = data[:split_point]
    eval_len = int(len(data[split_point:]) / 2)
    data_test = data[split_point:split_point + eval_len]
    data_eval = data[split_point + eval_len:]
    voc_size = (len(diag_voc.idx2word), len(pro_voc.idx2word), len(med_voc.idx2word))

    model = DeNosingAutoEncoder(encode_dim=hidden_size, input_dim=voc_size[0] + voc_size[1] + voc_size[2],
                                vocab_size=voc_size)
    model.to(device)
    model.load_state_dict(torch.load('de_noising_auto_encoder.pth')['model'])
    forest = RandomForestClassifier(random_state=1)
    multi_target_forest = MultiOutputClassifier(forest)

    train_hidden, train_label = get_dataset_representations_and_labels(model, data_train, voc_size, hidden_size)
    multi_target_forest.fit(train_hidden[:-1, :], train_label[1:, :])

    ja, prauc, avg_p, avg_r, avg_f1 = [[] for _ in range(5)]
    # test random forest classifier
    for step, input in enumerate(data_test):
        if len(input) < 2:
            continue

        y_label_patient = np.zeros(shape=[0, voc_size[0]])
        y_pred_patient = np.zeros(shape=[0, voc_size[0]])
        y_pred_prob_patient = np.zeros(shape=[0, voc_size[0]])

        for i in range(len(input)):
            adm = input[i]
            # 获得标签
            label = np.zeros(shape=[voc_size[0]])
            label[adm[0]] = 1
            y_label_patient = np.concatenate((y_label_patient, label.reshape(1, -1)), axis=0)

            _, hidden = model.encode(input[i-1])
            pred_prob = multi_target_forest.predict_proba(hidden.detach().cpu().numpy())
            # print(pred_prob)
            pred_prob = np.array([1-i[0][0] for i in pred_prob])
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
        print(f'step: {step} 完成！')
    llprint(
        '\t Jaccard: %.4f,  '
        'PRAUC: %.4f, AVG_PRC: %.4f, AVG_RECALL: %.4f, AVG_F1: %.4f\n' % (
            np.mean(ja), np.mean(prauc), np.mean(avg_p), np.mean(avg_r), np.mean(avg_f1)
        ))
    return np.mean(ja), np.mean(prauc), np.mean(avg_p), np.mean(avg_r), np.mean(avg_f1)


if __name__ == '__main__':
    # step 1: pre-training by de-nosing auto-encoders
    # Encode_Decode_Time_BO = BayesianOptimization(
    #     train, {
    #         'hidden_size': (5, 8),
    #         'learning_rate': (-5, 0),
    #         'l2_regularization': (-8, -3),
    #     }
    # )
    # Encode_Decode_Time_BO.maximize()
    # print(Encode_Decode_Time_BO.max)

    # step 2: save model and hidden representations
    # for i in range(100):
    #     train(learning_rate=0.0011532834253559377,
    #           hidden_size=32, l2_regularization=1e-8)

    get_diagnosis_performance()
