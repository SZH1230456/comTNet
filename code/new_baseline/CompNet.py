import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import dill
import collections
import scipy.sparse as sp
import time
import os
from collections import defaultdict
from util import llprint, multi_label_metric, ddi_rate_score_, get_n_params, test_test, sequence_metric
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score, roc_curve
from bayes_opt import BayesianOptimization
import sys
import random
import gc


class Agent(object):
    def __init__(self, LR, state_dims, action_size, num_layers, voc_size, target_update_iter, memory, device):
        super(Agent, self).__init__()
        self.LR = LR
        self.state_dims = state_dims
        self.action_size = action_size
        self.num_layers = num_layers
        self.gamma = 0.9
        self.epsilon = 0.9
        self.epsilon_min = 0.05  # agent控制随机探索的阈值
        self.epsilon_decay = 0.995  # 随着agent做选择越来越好，降低探索率
        self.device = device
        self.state_dims = state_dims
        self.memory = memory

        self.model = DQN(state_dims, action_size, device).to(device)
        self.target_model = DQN(state_dims, action_size, device).to(device)

        self.learn_step_counter = 0
        self.target_update_iter = target_update_iter

        self.cnn_diagnosis = CNN(state_dims, device, voc_size[0], 32).to(device)
        self.cnn_procedure = CNN(state_dims, device, voc_size[1], 32).to(device)

        self.rgcn = RGCN(num_layers, voc_size[2], device).to(device)

        self.model_params = list(self.cnn_diagnosis.parameters()) + list(self.cnn_procedure.parameters()) + \
                            list(self.rgcn.parameters()) + list(self.model.parameters())
        self.optimizer = torch.optim.Adam(self.model_params, lr=LR, betas=(0.9, 0.999), weight_decay=5.0)
        self.loss = nn.MSELoss()

    def patient_representation(self, input):
        """
        :param input: m_d, m_p
        :return: z= z_d + z_p in Eq.(15)
        """
        diagnosis, procedure = input[0], input[1]
        diagnosis = torch.LongTensor(diagnosis).to(self.device)
        procedure = torch.LongTensor(procedure).to(self.device)
        diagnosis = self.cnn_diagnosis(diagnosis)  # (1, state_dims)
        procedure = self.cnn_procedure(procedure)  # (1, state_dims)
        z = torch.cat((diagnosis, procedure), dim=0)
        return z

    def act(self, x_t, h_t_1, selected_actions):
        """
        :return: a_t, h_t
        """
        if np.random.rand() < self.epsilon:
            while True:
                action = random.randrange(self.action_size)
                if action not in selected_actions:
                    return action, h_t_1
        next_h, output = self.model((x_t, h_t_1))
        while True:
            with torch.no_grad():
                action = torch.max(output, 1)[1]
                if action not in selected_actions:
                    return action, next_h
                else:
                    output[0][action] = -999999

    def next_state(self, z, g):
        """
        :param z: z=z_d + z_p
        :param g:
        :return: x_t in Eq.(7)
        """
        a = F.softmax(torch.mm(z, g.t()))
        z_t = torch.mm(a.t(), z)  # z_t in Eq.(13)
        x_t = z_t + g  # x_t in Eq.(7)
        return x_t

    def step(self, action, selected_action, y):
        if int(action) in y and int(action) not in selected_action:
            reward = 1
        else:
            reward = -1
        return reward

    def replay(self, batch_size):
        if self.learn_step_counter % self.target_update_iter == 0:
            self.update_target_model()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        self.learn_step_counter += 1
        batch_idx = np.random.choice(len(self.memory), batch_size)
        b_x, b_h, b_action, b_reward, b_next_x, b_next_h = [[] for i in range(6)]
        for idx in batch_idx:
            b_x.append(self.memory[idx][0])
            b_h.append(self.memory[idx][1])
            b_action.append(self.memory[idx][2])
            b_reward.append(self.memory[idx][3])
            b_next_x.append(self.memory[idx][4])
            b_next_h.append(self.memory[idx][5])

        b_x = torch.cat(b_x, 0).to(self.device)
        b_h = torch.cat(b_h, 0).to(self.device)
        b_action = torch.LongTensor(b_action).to(self.device)
        b_reward = torch.FloatTensor(b_reward).to(self.device)
        b_next_x = torch.cat(b_next_x, 0).to(self.device)
        b_next_h = torch.cat(b_next_h, 0).to(self.device)

        return b_x, b_h, b_action, b_reward, b_next_x, b_next_h

    def update_model(self, b_x, b_h, b_action, b_reward, b_next_x, b_next_h):
        """
        :param b_x:
        :param b_h:
        :param b_action:
        :param b_reward:
        :param b_next_x:
        :param b_next_h:
        :return: loss
        """
        b_action = b_action.unsqueeze(1).to(self.device)
        _, q_value = self.model((b_x, b_h))
        q_value = q_value.gather(1, b_action)
        _, q_next = self.target_model((b_next_x, b_next_h))

        q_target = (b_reward + self.gamma * q_next.max(1)[0]).unsqueeze(1)
        loss = self.loss(q_value, q_target)

        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        self.optimizer.step()
        return loss

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())


class CNN(nn.Module):
    def __init__(self, state_dims, device, voc_size, num_channels):
        super(CNN, self).__init__()
        self.state_dims = state_dims
        self.device = device

        self.embedding = nn.Embedding(voc_size, state_dims)

        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=state_dims,  # input height
                      out_channels=num_channels,  # n_filters
                      kernel_size=1,  # filter size
                      stride=2,),
            nn.Tanh(),
            nn.Conv1d(in_channels=num_channels,  # input height
                      out_channels=num_channels,  # n_filters
                      kernel_size=1,  # filter size
                      stride=2,),
            nn.Tanh(),
            nn.Conv1d(in_channels=num_channels,  # input height
                      out_channels=num_channels,  # n_filters
                      kernel_size=1,  # filter size
                      stride=2),
        )
        self.output = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_channels, state_dims, bias=True))

    def forward(self, input):
        input_embedding = self.embedding(input.unsqueeze(dim=0))
        input_embedding = input_embedding.permute(0, 2, 1)
        x = self.conv(input_embedding)
        remaining_size = x.size(2)
        features = F.max_pool1d(x, remaining_size).squeeze(dim=2)
        return self.output(features)


class DQN(nn.Module):
    def __init__(self, state_dims, action_size, device):
        super(DQN, self).__init__()
        self.state_dims = state_dims
        self.action_size = action_size
        self.device = device
        self.state_dims = state_dims
        self.W = nn.Parameter(torch.FloatTensor(state_dims, state_dims))
        self.U = nn.Parameter(torch.FloatTensor(state_dims, state_dims))

        self.fc1 = nn.Sequential(nn.Linear(state_dims, state_dims),
                                 nn.ReLU(inplace=False))

        self.fc2 = nn.Linear(state_dims, action_size)

    def forward(self, input):
        """
        :param input: x_t, h_t_1
        :return: h_t in Eq.(6); Q functions
        """
        x_t, h_t_1 = input[0], input[1]
        h_t_1 = h_t_1.to(self.device)
        h_t = F.sigmoid(torch.mm(self.W, x_t.t()) + torch.mm(self.U, h_t_1.t()))
        fc1 = self.fc1(h_t.t())
        s_t = self.fc2(fc1)
        return h_t.t(), s_t


class RGCN(nn.Module):
    def __init__(self, num_layers, voc_size, device):
        super(RGCN, self).__init__()
        self.num_layers = num_layers
        self.voc_size = voc_size
        self.device = device

        self.node_init = None
        self.layers = nn.ModuleList()
        for i in self.num_layers:
            self.layers.append(GCN(voc_size, i))
            voc_size = i

    def forward(self, adj):
        """
        :param adj:
        :return: H^L:(action_size, 50), g_t:(1, 100)
        """
        out = self.node_init
        final_rep = []
        for i, layer in enumerate(self.layers):
            if i != 0:
                out = F.relu(out)
            out = layer(out, adj)
            final_rep.append(out)
        final_rep = torch.cat(final_rep, dim=1)  # (action_size, 100)
        final_rep = torch.sum(final_rep, dim=0).view(-1, final_rep.size(1)) # (1, 100)
        return out, final_rep


class GCN(nn.Module):
    def __init__(self, in_size, output_size, total_rel=2, n_basis=2, device='cuda:0'):
        super(GCN, self).__init__()
        self.in_size = in_size
        self.output_size = output_size
        self.total_rel = total_rel
        self.n_basis = n_basis
        self.device = device

        self.basis_weight = nn.Parameter(torch.FloatTensor(self.n_basis, self.in_size, self.output_size))
        self.basis_coeff = nn.Parameter(torch.FloatTensor(self.total_rel, self.n_basis))

        self.register_parameter('bias', None)
        self.reset_parameters()

    def forward(self, inp, adj):
        """
        :param inp:
        :param adj: [[action_size, action_size], [action_size, action_size]]
        :return:
        """
        rel_weights = torch.einsum('ij,jmn->imn', [self.basis_coeff, self.basis_weight])  # (2,action_size,output_size)
        weights = rel_weights.view(rel_weights.shape[0] * rel_weights.shape[1],
                                   rel_weights.shape[2])  # (2*action_size, output_size)
        emb_acc = []
        if inp is not None:
            for mat in adj:
                emb_acc.append(torch.mm(mat, inp))
            tmp = torch.cat(emb_acc, 1)
        else:
            tmp = torch.cat([item.to_dense() for item in adj], 1)  # (action_size, 2*action_size)

        out = torch.matmul(tmp, weights)  # (action_size, output_size)

        if self.bias is not None:
            out += self.bias.unsqueeze(0)
        return out

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.basis_weight.data)
        nn.init.xavier_uniform_(self.basis_coeff.data)

        if self.bias is not None:
            nn.init.xavier_uniform_(self.bias.data)


def init_adj(action_size):
    adj_list = []
    edges = np.empty((action_size, 2), dtype=np.int32)
    for j in range(action_size):
        edges[j] = np.array([j, j])
    row, col = np.transpose(edges)
    data = np.zeros(len(row))
    adj = sp.csr_matrix((data, (row, col)), shape=(action_size, action_size), dtype=np.uint8)
    adj_list.append(adj)
    adj_list.append(adj)
    return adj_list


def update_adj(action, selected_actions, voc_size, ddi):
    adj_list = []
    adj_shape = (voc_size, voc_size)
    spo_list_rel0, spo_list_rel1 = get_spo(action, selected_actions, ddi)
    edges = np.empty((len(spo_list_rel0), 2), dtype=np.int32)
    for j, (s, p, o) in enumerate(spo_list_rel0):
        edges[j] = np.array([s, o])
    row, col = np.transpose(edges)
    data = np.ones(len(row))
    adj = sp.csr_matrix((data, (row, col)), shape=adj_shape, dtype=np.uint8)
    adj_list.append(adj)

    if len(spo_list_rel1) == 0:
        edges = np.empty((voc_size, 2), dtype=np.int32)
        for j in range(voc_size):
            edges[j] = np.array([j, j])
        row, col = np.transpose(edges)
        data = np.zeros(len(row))
        adj = sp.csr_matrix((data, (row, col)), shape=adj_shape, dtype=np.uint8)
    else:
        edges = np.empty((len(spo_list_rel1), 2), dtype=np.int32)
        for j, (s, p, o) in enumerate(spo_list_rel1):
            # print('s-p-o:',(s,p,o))
            edges[j] = np.array([s, o])
        row, col = np.transpose(edges)
        data = np.ones(len(row))
        adj = sp.csr_matrix((data, (row, col)), shape=adj_shape, dtype=np.uint8)
    adj_list.append(adj)
    del adj, edges, adj_shape, data, row, col, spo_list_rel0, spo_list_rel1
    gc.collect()
    return adj_list


def get_spo(action, seletectedActions, ddiIDS):
    spo_list0 = [[action, '组合', action]]
    if len(seletectedActions) != 0:
        for id in seletectedActions:
            spo_list0.append([action, '组合', id])
            spo_list0.append([id, '组合', action])

    spo_list1 = []
    for row in ddiIDS:
        if action == row[0]:
            spo_list1.append([action, '对抗', row[1]])
        if action == row[1]:
            spo_list1.append([row[0], '对抗', action])
    return spo_list0, spo_list1


def get_torch_sparse_matrix(adj_list, dev):
    new_adj_list = []
    for row in adj_list:
        idx = torch.LongTensor([row.tocoo().row, row.tocoo().col])
        dat = torch.FloatTensor(row.tocoo().data)
        new_adj_list.append(torch.sparse.FloatTensor(idx, dat, torch.Size([row.shape[0], row.shape[1]])).to(dev))
    return new_adj_list


def eval(agent, data_test, voc_size, epoch, state_dims, ddi, device):
    adj = init_adj(voc_size[2])
    adj = get_torch_sparse_matrix(adj, device)
    _, init_g = agent.rgcn(adj)
    ja, prauc, avg_p, avg_r, avg_f1 = [[] for _ in range(5)]
    y_pred_label_all_for_ddi = np.zeros(shape=[0, voc_size[2]])
    for step, input in enumerate(data_test):
        if len(input) < 2:
            continue
        y_label_patient = np.zeros(shape=[0, voc_size[2]])
        y_pred_label_patient = np.zeros(shape=[0, voc_size[2]])
        for idx in range(1, len(input)):
            adm = input[idx]
            target_label_patient = adm[2]
            y_label_adm = np.zeros(shape=[voc_size[2]])
            y_label_adm[target_label_patient] = 1
            y_label_patient = np.concatenate((y_label_patient, y_label_adm.reshape(1, -1)), axis=0)

            g_t = init_g
            z = agent.patient_representation(adm)
            h_t_1 = np.zeros((1, state_dims))
            h_t_1 = torch.FloatTensor(h_t_1)

            selected_actions = []
            for t in range(len(target_label_patient)):
                x_t = agent.next_state(z, g_t)
                a_t, h_t = agent.act(x_t, h_t_1, selected_actions)
                adj_next = update_adj(a_t, selected_actions, voc_size[2], ddi)
                adj_next = get_torch_sparse_matrix(adj_next, device)
                _, g_t_next = agent.rgcn(adj_next)
                h_t_1 = h_t
                g_t = g_t_next
                selected_actions.append(int(a_t))

            y_pred_label_adm = np.zeros(shape=[voc_size[2]])
            y_pred_label_adm[selected_actions] = 1
            y_pred_label_patient = np.concatenate((y_pred_label_patient, y_pred_label_adm.reshape(1, -1)), axis=0)
            y_pred_label_all_for_ddi = np.concatenate((y_pred_label_all_for_ddi, y_pred_label_patient), axis=0)

        adm_ja, adm_prauc, adm_avg_p, adm_avg_r, adm_avg_f1 = sequence_metric(y_label_patient, 0,
                                                                             y_pred_label_patient,
                                                                             y_pred_label_patient)
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


def format_ddi(ddi_adj):
    ddi_list = []
    for i in range(len(ddi_adj)):
        for j in range(len(ddi_adj)):
            if ddi_adj[i][j] == 1:
                ddi_list.append([i, j])
    return ddi_list[:int(len(ddi_list)/2)]


def train(state_dims, LR):
    state_dims = 2 ** int(state_dims)
    LR = 10 ** LR
    print('state_dims____{}____LR___{}'.format(state_dims, LR))

    data_path = '../../data/records_final.pkl'
    voc_path = '../../data/voc_final.pkl'
    ddi_adj_path = '../../data/ddi_A_final.pkl'
    device = torch.device('cpu')

    ddi_adj = dill.load(open(ddi_adj_path, 'rb'))
    ddi = format_ddi(ddi_adj)

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
    batch_size = 64
    num_layers = [int(state_dims/2), int(state_dims/2)]
    target_update_iter = 10
    memory = collections.deque(maxlen=100)
    ddi_rate, ja, prauc, avg_p, avg_r, avg_f1 = [0 for i in range(6)]

    adj = init_adj(voc_size[2])
    adj = get_torch_sparse_matrix(adj, device)

    agent = Agent(LR, state_dims, voc_size[2], num_layers, voc_size, target_update_iter, memory, device)
    _, init_g = agent.rgcn(adj)
    for epoch in range(EPOCH):
        start_time = time.time()
        loss = []
        for step, input in enumerate(data_train[:2]):
            if len(input) < 2:
                continue
            for idx in range(1, len(input)):
                selected_actions = []
                adm = input[idx]
                target = adm[2]
                g_t = init_g  # (1, 100)
                z = agent.patient_representation(adm)  # (2, state_dims)
                h_t_1 = np.zeros((1, state_dims))  # （1, state_dims）
                h_t_1 = torch.FloatTensor(h_t_1)
                for t in range(len(target)):
                    x_t = agent.next_state(z, g_t)  # (1, state_dims)
                    a_t, h_t = agent.act(x_t, h_t_1, selected_actions)
                    r_t = agent.step(a_t, selected_actions, target)
                    adj_next = update_adj(a_t, selected_actions, voc_size[2], ddi)
                    adj_next = get_torch_sparse_matrix(adj_next, device)
                    _, g_t_next = agent.rgcn(adj_next)
                    selected_actions.append(a_t)
                    x_t_next = agent.next_state(z, g_t_next)
                    h_t_1 = h_t_1.to(device)
                    h_t = h_t.to(device)
                    memory.append((x_t, h_t_1, a_t, r_t, x_t_next, h_t))
                    g_t = g_t_next
                    h_t = h_t.to('cpu')
                    h_t_1 = h_t

                if len(memory) > batch_size:
                    b_x, b_h, b_action, b_reward, b_next_x, b_next_h = agent.replay(batch_size)
                    loss_ = agent.update_model(b_x, b_h, b_action, b_reward, b_next_x, b_next_h)
                    loss.append(loss_)

            print("进度:{0}%".format(round((step + 1) * 100 / len(data_train))), end="\r")

        ddi_rate, ja, prauc, avg_p, avg_r, avg_f1 = eval(agent, data_eval, voc_size, epoch, state_dims, ddi, device)
        end_time = time.time()
        elapsed_time = (end_time - start_time) / 60
        llprint('\tEpoch: %d, Loss1: %.4f, '
                'One Epoch Time: %.2fm, Appro Left Time: %.2fh\n' % (epoch,
                                                                     loss[-1],
                                                                     elapsed_time,
                                                                     elapsed_time * (EPOCH - epoch - 1) / 60))
    return ja
    # return ddi_rate, ja, prauc, avg_p, avg_r, avg_f1


if __name__ == '__main__':
    test_test('CompNet_6_22_train_all_input.txt')
    Encode_Decode_Time_BO = BayesianOptimization(
            train, {
                'state_dims': (5, 8),
                'LR': (-5, 0),
            }
        )
    Encode_Decode_Time_BO.maximize()
    print(Encode_Decode_Time_BO.max)
    # train(state_dims=32, LR=0.01)