import dill
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
from util import multi_label_metric


def statistics():
    data_path = '../../data/records_final.pkl'
    data = dill.load(open(data_path, 'rb'))
    num_patients = 0
    num_visits = 0

    num_diagnosis_for_each_patient = []
    num_procedure_for_each_patient = []
    num_medication_for_each_patient = []

    for step, input in enumerate(data):
        if len(input) < 2:
            continue
        else:
            num_patients += 1
            num_visits += len(input)
            for idx in range(0, len(input)):
                adm = input[idx]
                num_diagnosis_for_each_patient.append(len(adm[0]))
                num_procedure_for_each_patient.append(len(adm[1]))
                num_medication_for_each_patient.append(len(adm[2]))

    print('num_patients  {}  num_visit  {}  '
          'diagnosis {} procedure  {}  medication {} '.format(num_patients,
                                                              num_visits,
                                                              np.sum(np.array(num_diagnosis_for_each_patient)),
                                                              np.sum(np.array(num_procedure_for_each_patient)),
                                                              np.sum(np.array(num_medication_for_each_patient))))


def statistical_top_15_cid_code():
    data_path = '../../data/records_final.pkl'
    voc_path = '../../data/voc_final.pkl'

    data = dill.load(open(data_path, 'rb'))
    voc = dill.load(open(voc_path, 'rb'))
    diag_voc, pro_voc, med_voc = voc['diag_voc'].word2idx, voc['pro_voc'].word2idx, voc['med_voc'].word2idx

    diag_dict = {}
    for diag in diag_voc:
        diag_dict[diag] = 0

    diag_voc = voc['diag_voc'].idx2word

    for i, input_ in enumerate(data):
        if len(input_) < 2:
            continue
        for j in range(len(input_)):
            for k in range(len(input_[j][0])):
                n = input_[j][0][k]
                m = diag_voc[n]
                diag_dict[m] += 1

    L = sorted(diag_dict.items(), key=lambda item: item[1], reverse=True)
    data = L[:15]
    x_names = []
    y_values = []
    for i in range(len(data)):
        x_names.append(data[i][0])
        y_values.append(data[i][1])
    return x_names, y_values


def statistical_top_15_pro_code():
    data_path = '../../data/records_final.pkl'
    voc_path = '../../data/voc_final.pkl'

    data = dill.load(open(data_path, 'rb'))
    voc = dill.load(open(voc_path, 'rb'))
    diag_voc, pro_voc, med_voc = voc['diag_voc'].word2idx, voc['pro_voc'].word2idx, voc['med_voc'].word2idx

    diag_dict = {}
    for diag in pro_voc:
        diag_dict[diag] = 0

    diag_voc = voc['pro_voc'].idx2word

    for i, input_ in enumerate(data):
        if len(input_) < 2:
            continue
        for j in range(len(input_)):
            for k in range(len(input_[j][1])):
                n = input_[j][1][k]
                m = diag_voc[n]
                diag_dict[m] += 1

    L = sorted(diag_dict.items(), key=lambda item: item[1], reverse=True)
    data = L[:15]
    x_names = []
    y_values = []
    for i in range(len(data)):
        x_names.append(data[i][0])
        y_values.append(data[i][1])
    return x_names, y_values


def statistical_top_15_drug_code():
    data_path = '../../data/records_final.pkl'
    voc_path = '../../data/voc_final.pkl'

    data = dill.load(open(data_path, 'rb'))
    voc = dill.load(open(voc_path, 'rb'))
    diag_voc, pro_voc, med_voc = voc['diag_voc'].word2idx, voc['pro_voc'].word2idx, voc['med_voc'].word2idx

    diag_dict = {}
    for diag in med_voc:
        diag_dict[diag] = 0

    diag_voc = voc['med_voc'].idx2word

    for i, input_ in enumerate(data):
        if len(input_) < 2:
            continue
        for j in range(len(input_)):
            for k in range(len(input_[j][2])):
                n = input_[j][2][k]
                m = diag_voc[n]
                diag_dict[m] += 1

    L = sorted(diag_dict.items(), key=lambda item: item[1], reverse=True)
    data = L[:15]
    x_names = []
    y_values = []
    for i in range(len(data)):
        x_names.append(data[i][0])
        y_values.append(data[i][1])
    return x_names, y_values


def plot_top_k_statistical():
    plt.rcParams['font.family'] = "Times"
    icd_code_name, icd_values = statistical_top_15_cid_code()
    pro_code_name, pro_values = statistical_top_15_pro_code()
    drug_code_name, drug_values = statistical_top_15_drug_code()

    plt.figure(figsize=(20, 5), dpi=1200)
    plt.subplot(131)  # 将一个画板分割成两行一列共两个绘图区域
    plt.bar(icd_code_name[:10], icd_values[:10], color='mediumpurple')
    plt.xlabel('ICD-9 Code')
    plt.ylabel('Count')
    plt.title('Diagnosis Distribution')

    plt.subplot(132)
    plt.bar(pro_code_name[:10], pro_values[:10], color='salmon')
    plt.xlabel('ICD-9 Code')
    plt.ylabel('Count')
    plt.title('Procedure Distribution')

    plt.subplot(133)
    plt.bar(drug_code_name[:10], drug_values[:10], color='sienna')
    plt.xlabel('NDC Code')
    plt.ylabel('Count')
    plt.title('Medication Distribution')

    plt.savefig('统计前10个ICD.png', dpi=1200, bbox_inches='tight')
    plt.show()


def plot_figure():
    plt.rcParams['font.family'] = "Times"
    x = [1, 3, 5, 8, 10]
    Jacard = [0.387053343, 0.392214684373899, 0.406960044143724, 0.405960352957997, 0.40071274043793]
    Recall = [0.756057880017882, 0.769399976479146, 0.783217405065719, 0.777966769547776, 0.777305384722024]
    F1 = [0.54735221, 0.552723512396513, 0.567180037, 0.566312664, 0.561122792]
    PR_AUC = [0.630897999, 0.651219255868442, 0.6682184995664, 0.661182113460844, 0.655763761]

    Jacard_leap = [[0.367] for i in range(5)]
    Recall_leap = [[0.5266] for i in range(5)]
    F1_leap = [[0.5407] for i in range(5)]
    AUC_LEAP = [[0.5288] for i in range(5)]
    # plt.figure(figsize=(15, 2.5))
    plt.figure()
    ax1 = plt.subplot(221)
    ax2 = plt.subplot(222)
    ax3 = plt.subplot(223)
    ax4 = plt.subplot(224)
    plt.subplots_adjust(wspace=0.5, hspace=0.5)  # 调整子图间距

    plt.sca(ax1)
    plt.plot(x, PR_AUC, 'ro--', label='Proposed')
    plt.plot(x, AUC_LEAP, color='#808080', linestyle='--', label='LEAP')
    plt.xlim(1, 10)
    plt.xticks(x)
    plt.ylim(0.2, 1.0)
    plt.xlabel('$n$')
    plt.ylabel('PR-AUC')
    plt.legend()

    plt.sca(ax2)
    plt.plot(x, Jacard, 'ro--', label='Proposed')
    plt.plot(x, Jacard_leap, color='#808080', linestyle='--', label='LEAP')
    plt.xlim(1, 10)
    plt.xticks(x)
    plt.ylim(0.2, 0.8)
    plt.xlabel('$n$')
    plt.ylabel('Jaccard')
    plt.legend()

    plt.sca(ax3)
    plt.plot(x, Recall, 'ro--', label='Proposed')
    plt.plot(x, Recall_leap, color='#808080', linestyle='--', label='LEAP')
    plt.xlim(1, 10)
    plt.xticks(x)
    plt.ylim(0.3, 1.1)
    plt.xlabel('$n$')
    plt.ylabel('Recall')
    plt.legend()

    plt.sca(ax4)
    plt.plot(x, F1, 'ro--', label='Proposed')
    plt.plot(x, F1_leap, color='#808080', linestyle='--', label='LEAP')
    plt.xlim(1, 10)
    plt.xticks(x)
    plt.ylim(0.2, 0.8)
    plt.xlabel('$n$')
    plt.ylabel('F1')
    plt.legend()

    plt.suptitle('Medication recommendation performance')
    # plt.gcf().savefig('size_performance.eps', dpi=600, format='eps', bbox_inches='tight')
    plt.savefig('size_performance_5_26.png', dpi=1200, bbox_inches='tight')
    plt.show()

    # x = [0, 1, 3, 5, 8, 10]
    # Jacard = [0.43222, 0.422983333, 0.4434, 0.4920, 0.4095, 0.49154]
    # Precision = [0.53906, 0.58908, 0.5722, 0.6729, 0.5004, 0.61164]
    # Recall = [0.70807, 0.61712, 0.6691, 0.6548, 0.7028, 0.71216]
    # F1 = [0.59244, 0.59023, 0.612, 0.6569, 0.5766, 0.64702]
    #
    # Jacard_leap = [[0.3933] for i in range(6)]
    # Precision_leap = [[0.5689] for i in range(6)]
    # Recall_leap = [[0.5761] for i in range(6)]
    # F1_leap = [[0.5498] for i in range(6)]
    # plt.figure(figsize=(15, 2.5))
    # ax1 = plt.subplot(1, 4, 1)
    # ax2 = plt.subplot(1, 4, 2)
    # ax3 = plt.subplot(1, 4, 3)
    # ax4 = plt.subplot(1, 4, 4)
    # plt.subplots_adjust(wspace=0.5, hspace=0)  # 调整子图间距
    #
    # plt.sca(ax1)
    # plt.plot(x, F1, 'ro--', label='Proposed')
    # plt.plot(x, F1_leap, color='#808080', linestyle='--', label='LEAP')
    # plt.xlim(0, 10)
    # plt.xticks(x)
    # plt.ylim(0.2, 1.0)
    # plt.xlabel('$|n|$')
    # plt.ylabel('F1')
    # plt.legend()
    #
    # plt.sca(ax2)
    # plt.plot(x, Recall, 'ro--', label='Proposed')
    # plt.plot(x, Recall_leap, color='#808080', linestyle='--', label='LEAP')
    # plt.xlim(0, 10)
    # plt.xticks(x)
    # plt.ylim(0.2, 1.2)
    # plt.xlabel('$|n|$')
    # plt.ylabel('Recall')
    # plt.legend()
    #
    # plt.sca(ax3)
    # plt.plot(x, Jacard, 'ro--', label='Proposed')
    # plt.plot(x, Jacard_leap, color='#808080', linestyle='--', label='LEAP')
    # plt.xlim(0, 10)
    # plt.xticks(x)
    # plt.ylim(0.0, 1.0)
    # plt.xlabel('$|n|$')
    # plt.ylabel('Jaccard')
    # plt.legend()
    #
    # plt.sca(ax4)
    # plt.plot(x, Precision, 'ro--', label='Proposed')
    # plt.plot(x, Precision_leap, color='#808080', linestyle='--', label='LEAP')
    # plt.xlim(0, 10)
    # plt.xticks(x)
    # plt.ylim(0.2, 1.0)
    # plt.xlabel('$|n|$')
    # plt.ylabel('Precision')
    # plt.legend()
    # plt.suptitle('Medication recommendation performance')
    # # plt.gcf().savefig('size_performance.eps', dpi=600, format='eps', bbox_inches='tight')
    # plt.savefig('size_performance_5_11.png', dpi=600, bbox_inches='tight')
    # plt.show()


def analysis_results():
    k_nearest_results = np.load(
        'H:\\IJMEDI\\GAMENet\code\\new_baseline\\medication recommendation\\save_results\\k-nearest\\k_nearest_predicted_medication_0.07911120730079717_.npy',
        allow_pickle=True)
    k_frequency_results = np.load(
        'H:\\IJMEDI\\GAMENet\code\\new_baseline\\medication recommendation\\save_results\\\k-frequency\\k_frequency_predicted_medication_0.07728866740494647_9.npy',
        allow_pickle=True)
    lr_results = np.load(
        'H:\\IJMEDI\\GAMENet\code\\new_baseline\\medication recommendation\\save_results\\LR\\predicted_medication_0.010188420463079646_2.npy',
        allow_pickle=True)
    leap_results = np.load(
        'H:\\IJMEDI\\GAMENet\code\\new_baseline\\medication recommendation\\save_results\\LEAP\\LEAP_predicted_medication_0.06086239927875134_8.npy',
        allow_pickle=True)

    lr_results[lr_results > 0] = 1
    k_frequency_results_format = np.zeros(shape=[len(k_frequency_results), 145])
    for i in range(len(k_frequency_results)):
        for j in range(k_frequency_results.shape[1]):
            m = k_frequency_results[i, j]
            k_frequency_results_format[i, int(m)] = 1

    np.savetxt('k_nearest_results.csv', k_nearest_results, delimiter=',')
    np.savetxt('k_frequency_results.csv', k_frequency_results_format, delimiter=',')
    np.savetxt('lr_results.csv', lr_results, delimiter=',')
    # np.savetxt('leap_results.csv', leap_results, delimiter=',')
    print('保存成功！')


def save_labels():
    data_path = '../../data/records_final.pkl'
    voc_path = '../../data/voc_final.pkl'

    ddi_adj_path = '../../data/ddi_A_final.pkl'
    ddi_adj = dill.load(open(ddi_adj_path, 'rb'))

    data = dill.load(open(data_path, 'rb'))
    voc = dill.load(open(voc_path, 'rb'))
    diag_voc, pro_voc, med_voc = voc['diag_voc'], voc['pro_voc'], voc['med_voc']

    split_point = int(len(data) * 2 / 3)
    data_train = data[:split_point]
    eval_len = int(len(data[split_point:]) / 2)
    data_test = data[split_point:split_point + eval_len]

    labels = np.zeros(shape=[0, 145])
    for step, input in enumerate(data_test):
        if len(input) < 2:
            continue
        for i in range(1, len(input)):
            loss_1_target = np.zeros(shape=[1, 145])
            loss_1_target[0, input[i][2]] = 1

            labels = np.concatenate((labels, loss_1_target), axis=0)

    np.savetxt('labels.csv', labels, delimiter=',')
    print('保存成功！')


def ddi_score(y_pred):
    ddi_adj_path = '../../data/ddi_A_final.pkl'
    ddi_A = dill.load(open(ddi_adj_path, 'rb'))
    score = []
    for adm in y_pred:
        all_cnt = 0
        dd_cnt = 0
        med_code_set = np.where(adm == 1)[0]
        for i, med_i in enumerate(med_code_set):
            for j, med_j in enumerate(med_code_set):
                if j <= i:
                    continue
                all_cnt += 1
                if ddi_A[med_i, med_j] == 1 or ddi_A[med_j, med_i] == 1:
                    dd_cnt += 1
                    print(med_i, med_j)
        if all_cnt == 0:
            return 0
        else:
            score.append(dd_cnt / all_cnt)
    return score


def plot_results():
    # 构造数据
    labels = ['Ground-truth', 'K-frequency', 'LR', 'LEAP', 'GAMENet', 'Proposed']
    data_a = [0.079, 0.11, 0.089, 0.064, 0.057, 0.060]

    x = np.arange(len(labels))
    x = x + 1.5
    width = 0.4

    # "#e74c3c"
    flatui = sns.color_palette("Blues_r", n_colors=6)
    # flatui = ['#34495e'] * 6
    # flatui = ["#9b59b6", "#3498db", "#95a5a6", "#2ecc71", "#34495e", "#948279"]
    plt.rcParams['font.family'] = "Times"
    # plots
    fig, ax = plt.subplots(figsize=(5, 3.5), dpi=200)
    ax.grid(alpha=0.1)
    ax.bar(x - width / 2, data_a, width, color=flatui, ec='black', lw=.5)
    # 定制化设计
    ax.tick_params(axis='x', direction='in', bottom=False)
    ax.tick_params(axis='y', direction='out', labelsize=8, length=3)
    ax.set_xticks(x - 0.2)
    ax.set_xticklabels(labels, size=8)
    # ax.set_ylim(bottom=0, top=0.75)
    # ax.set_yticks(np.arange(0, 0.8, step=0.1))

    # for spine in ['top', 'right']:
    #     ax.spines[spine].set_color('none')

    ax.legend(fontsize=7, frameon=False)
    plt.xlabel('Model')
    plt.ylabel('DDI Rate')

    plt.savefig("DDI_case_study_5_26.png", dpi=1200, bbox_inches='tight', pad_inches=0.0)
    plt.show()


if __name__ == '__main__':
    plot_top_k_statistical()
    # plot_figure()
    plot_results()
    # ground-truth: 0
    y_pred = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 15, 19, 20, 26,
              27, 31, 34, 37, 45, 81]

    # k-nearest: 0
    # y_pred = [1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 26, 27, 31, 38, 43, 44]

    # k_frequency: 0.1111111111111111
    # y_pred = [0, 1, 2, 3, 4, 6, 12, 19, 26]

    # lr: 0.08947368421052632
    # y_pred = [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 15, 19, 20, 25, 26, 28, 43, 44, 87]

    # leap: 0.06432748538011696
    # y_pred = [0, 1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 19, 20, 26, 27, 28, 43, 44, 87]

    # gamenet: 0.05714285714285714
    # y_pred = [4, 6, 7, 8, 9, 11, 25, 27, 28, 32, 43, 44, 80, 84, 87]

    # proposed: 0.06159420289855073
    # y_pred = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 15, 17, 19, 20, 22, 26, 27, 28, 31, 34, 37]
    y_pred_ = np.zeros(shape=[1, 145])
    for i in y_pred:
        y_pred_[0, i] = 1
    score = ddi_score(y_pred_)
    print(score)
    # plot_figure()
