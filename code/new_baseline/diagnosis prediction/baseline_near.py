import dill
import numpy as np

import sys
sys.path.append("..")
from util import multi_label_metric

data_path = '../../../data/records_final_icd_3.pkl'
voc_path = '../../../data/voc_final_icd_3.pkl'

data = dill.load(open(data_path, 'rb'))
voc = dill.load(open(voc_path, 'rb'))
diag_voc, pro_voc, med_voc = voc['diag_voc'], voc['pro_voc'], voc['med_voc']
split_point = int(len(data) * 2 / 3)
data_train = data[:split_point]
eval_len = int(len(data[split_point:]) / 2)
data_test = data[split_point:split_point + eval_len]
data_eval = data[split_point+eval_len:]


def main():
    gt = []
    pred = []
    for patient in data_test:
        if len(patient) < 2:
            continue
        for adm_idx, adm in enumerate(patient):
            if adm_idx < len(patient)-1:
                gt.append(patient[adm_idx+1][0])
                pred.append(adm[0])
    diag_voc_size = len(diag_voc.idx2word)
    y_gt = np.zeros((len(gt), diag_voc_size))
    y_pred = np.zeros((len(gt), diag_voc_size))
    for idx, item in enumerate(gt):
        y_gt[idx, item] = 1
    for idx, item in enumerate(pred):
        y_pred[idx, item] = 1



    ja, prauc, avg_p, avg_r, avg_f1 = multi_label_metric(y_gt, y_pred, y_pred)

    print('\tJaccard: %.4f,  PRAUC: %.4f, AVG_PRC: %.4f, AVG_RECALL: %.4f, AVG_F1: %.4f\n' % (
        ja, prauc, avg_p, avg_r, avg_f1
    ))


if __name__ == '__main__':
    main()