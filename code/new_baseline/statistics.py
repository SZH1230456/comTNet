import dill
import numpy as np
import matplotlib.pyplot as plt

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


def plot_figure():
    x = [0, 1, 3, 5, 8, 10]
    Jacard = [0.43222, 0.422983333, 0.4434, 0.4920, 0.4095, 0.49154]
    Precision = [0.53906, 0.58908, 0.5722, 0.6729, 0.5004, 0.61164]
    Recall = [0.70807, 0.61712, 0.6691, 0.6548, 0.7028, 0.71216]
    F1 = [0.59244, 0.59023, 0.612, 0.6569, 0.5766, 0.64702]

    Jacard_leap = [[0.3933] for i in range(6)]
    Precision_leap = [[0.5689] for i in range(6)]
    Recall_leap = [[0.5761] for i in range(6)]
    F1_leap = [[0.5498] for i in range(6)]
    plt.figure(figsize=(15, 2.5))
    ax1 = plt.subplot(1, 4, 1)
    ax2 = plt.subplot(1, 4, 2)
    ax3 = plt.subplot(1, 4, 3)
    ax4 = plt.subplot(1, 4, 4)
    plt.subplots_adjust(wspace=0.5, hspace=0)  # 调整子图间距

    plt.sca(ax1)
    plt.plot(x, F1, 'ro--', label='ComTNet')
    plt.plot(x, F1_leap, color='#808080', linestyle='--', label='LEAP')
    plt.xlim(0, 10)
    plt.xticks(x)
    plt.ylim(0.2, 1.0)
    plt.xlabel('$|\mathcal{R}^i|$')
    plt.ylabel('F1')
    plt.legend()

    plt.sca(ax2)
    plt.plot(x, Recall, 'ro--', label='ComTNet')
    plt.plot(x, Recall_leap, color='#808080', linestyle='--', label='LEAP')
    plt.xlim(0, 10)
    plt.xticks(x)
    plt.ylim(0.2, 1.2)
    plt.xlabel('$|\mathcal{R}^i|$')
    plt.ylabel('Recall')
    plt.legend()

    plt.sca(ax3)
    plt.plot(x, Jacard, 'ro--', label='ComTNet')
    plt.plot(x, Jacard_leap, color='#808080', linestyle='--', label='LEAP')
    plt.xlim(0, 10)
    plt.xticks(x)
    plt.ylim(0.0, 1.0)
    plt.xlabel('$|\mathcal{R}^i|$')
    plt.ylabel('Jaccard')
    plt.legend()

    plt.sca(ax4)
    plt.plot(x, Precision, 'ro--', label='ComTNet')
    plt.plot(x, Precision_leap, color='#808080', linestyle='--', label='LEAP')
    plt.xlim(0, 10)
    plt.xticks(x)
    plt.ylim(0.2, 1.0)
    plt.xlabel('$|\mathcal{R}^i|$')
    plt.ylabel('Precision')
    plt.legend()

    # plt.gcf().savefig('size_performance.eps', dpi=600, format='eps', bbox_inches='tight')
    plt.savefig('size_performance.png',  dpi=600, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    plot_figure()