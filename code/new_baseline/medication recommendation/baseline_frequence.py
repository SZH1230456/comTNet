import numpy as np
import dill
from util import ddi_rate_score, test_test, multi_label_metric


def sequence_metric(y_gt, y_pred, y_prob, y_label):
    def jaccard(y_gt, y_label):
        score = []
        for b in range(y_gt.shape[0]):
            target = np.where(y_gt[b] == 1)[0]
            out_list = y_label[b]
            inter = set(out_list) & set(target)
            union = set(out_list) | set(target)
            jaccard_score = 0 if union == 0 else len(inter) / len(union)
            score.append(jaccard_score)
        return np.mean(score)

    def average_prc(y_gt, y_label):
        score = []
        for b in range(y_gt.shape[0]):
            target = np.where(y_gt[b]==1)[0]
            out_list = y_label[b]
            inter = set(out_list) & set(target)
            prc_score = 0 if len(out_list) == 0 else len(inter) / len(out_list)
            score.append(prc_score)
        return score

    def average_recall(y_gt, y_label):
        score = []
        for b in range(y_gt.shape[0]):
            target = np.where(y_gt[b] == 1)[0]
            out_list = y_label[b]
            inter = set(out_list) & set(target)
            recall_score = 0 if len(target) == 0 else len(inter) / len(target)
            score.append(recall_score)
        return score

    def average_f1(average_prc, average_recall):
        score = []
        for idx in range(len(average_prc)):
            if (average_prc[idx] + average_recall[idx]) == 0:
                score.append(0)
            else:
                score.append(2*average_prc[idx]*average_recall[idx] / (average_prc[idx] + average_recall[idx]))
        return score

    # prauc = precision_auc(y_gt, y_prob)
    ja = jaccard(y_gt, y_label)
    avg_prc = average_prc(y_gt, y_label)
    avg_recall = average_recall(y_gt, y_label)
    avg_f1 = average_f1(avg_prc, avg_recall)

    return ja, 0, np.mean(avg_prc), np.mean(avg_recall), np.mean(avg_f1)


def train():
    data_path = '../../../data/records_final.pkl'
    voc_path = '../../../data/voc_final.pkl'

    data = dill.load(open(data_path, 'rb'))
    voc = dill.load(open(voc_path, 'rb'))
    diag_voc, pro_voc, med_voc = voc['diag_voc'], voc['pro_voc'], voc['med_voc']

    split_point = int(len(data) * 2 / 3)
    data_train = data[:split_point]
    eval_len = int(len(data[split_point:]) / 2)
    data_eval = data[split_point + eval_len:]
    data_test = data[split_point:split_point + eval_len]
    voc_size = (len(diag_voc.idx2word), len(pro_voc.idx2word), len(med_voc.idx2word))

    frequency_matrix = np.zeros(shape=[voc_size[0], voc_size[2]])
    for step, input in enumerate(data_train):
        if len(input) < 2:
            continue
        for visit in range(1, len(input)):
            diagnosis = input[visit][0]
            drug = input[visit][2]
            for diag_ in diagnosis:
                frequency_matrix[diag_, drug] += 1

    for k in range(2, 10):
        ddi, adm_ja, adm_prauc, adm_avg_p, adm_avg_r, adm_avg_f1 = eval(frequency_matrix, data_test, k)
        print('k__{}__ddi__{}__ja__{}__auc__{}__pre_{}__recall__{}__f1___{}'.format(k,
                                                                                    ddi,
                                                                                    adm_ja,
                                                                                    adm_prauc,
                                                                                    adm_avg_p,
                                                                                    adm_avg_r,
                                                                                    adm_avg_f1))


def eval(frequencey, data_eval, k):
    ja, prauc, avg_p, avg_r, avg_f1 = [[] for _ in range(5)]
    y_pred_label = np.zeros(shape=[0, k])
    y_pred_label_all_for_ddi = []
    for step, input in enumerate(data_eval):
        if len(input) < 2:
            continue
        predictions_one_patient = []
        labels_one_patient = np.zeros(shape=[0, 145])
        for visit in range(1, len(input)):
            diagnosis = input[visit][0]
            drugs = frequencey[diagnosis]
            drugs = np.sum(drugs, axis=0)
            drugs = np.argsort(-drugs)
            drugs = list(drugs[:k])

            y_pred_label = np.concatenate((y_pred_label, np.array(drugs).reshape(-1, k)), axis=0)

            predictions_one_patient.append(drugs)
            label = np.zeros(shape=[1, 145])
            label[:, input[visit][2]] = 1
            labels_one_patient = np.concatenate((labels_one_patient, label), axis=0)

        y_pred_label_all_for_ddi.append(predictions_one_patient)

        y_pred = np.zeros((labels_one_patient.shape[0], 145))
        for idx, item in enumerate(predictions_one_patient):
            y_pred[idx, item] = 1

        adm_ja, adm_prauc, adm_avg_p, adm_avg_r, adm_avg_f1 = multi_label_metric(labels_one_patient, y_pred, y_pred)
        ja.append(adm_ja)
        prauc.append(adm_prauc)
        avg_p.append(adm_avg_p)
        avg_r.append(adm_avg_r)
        avg_f1.append(adm_avg_f1)
    ddi_rate = ddi_rate_score(y_pred_label_all_for_ddi)
    np.save('k_frequency_predicted_medication_' + str(ddi_rate) + '_' + str(k) + '.npy', y_pred_label)
    print('保存成功！')

    print(
        '\t Eval-- DDI Rate: %.4f, '
        'Jaccard: %.4f,  PRAUC: %.4f, AVG_PRC: %.4f, '
        'AVG_RECALL: %.4f, AVG_F1: %.4f\n' % (ddi_rate, np.mean(ja),
                                              np.mean(prauc), np.mean(avg_p),
                                              np.mean(avg_r), np.mean(avg_f1)))
    return ddi_rate, np.mean(ja), np.mean(prauc), np.mean(avg_p), np.mean(avg_r), np.mean(avg_f1)


if __name__ == '__main__':
    train()