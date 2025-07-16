import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_curve, auc, recall_score
import time
import os
import matplotlib.pyplot as plt

def calculate_metrics(labels, predictions, scores):
    """
    计算准确率、特异性、F1 分数和 AUC
    :param labels: 真实标签
    :param predictions: 预测标签
    :param scores: 预测得分
    :return: 准确率、特异性、F1 分数、AUC
    """
    accuracy = accuracy_score(labels, predictions)
    sensitivity = recall_score(labels, predictions)
    specificity = recall_score(labels, predictions, pos_label=0)
    f1 = f1_score(labels, predictions)
    fpr, tpr, _ = roc_curve(labels, scores)
    auc_score = auc(fpr, tpr)
    return accuracy, sensitivity, specificity, f1, auc_score

def evaluate_k_fold(fold_metrics):
    """
    计算 k 折交叉验证的平均值和标准差
    :param fold_metrics: 每一折的指标列表
    :return: 平均值和标准差
    """
    accuracy_list = [metrics[0] for metrics in fold_metrics]
    sensitivity_list = [metrics[1] for metrics in fold_metrics]
    specificity_list = [metrics[2] for metrics in fold_metrics]
    f1_list = [metrics[3] for metrics in fold_metrics]
    auc_list = [metrics[4] for metrics in fold_metrics]

    accuracy_mean = np.mean(accuracy_list)
    accuracy_std = np.std(accuracy_list)
    sensitivity_mean = np.mean(sensitivity_list)
    sensitivity_std = np.std(sensitivity_list)
    specificity_mean = np.mean(specificity_list)
    specificity_std = np.std(specificity_list)
    f1_mean = np.mean(f1_list)
    f1_std = np.std(f1_list)
    auc_mean = np.mean(auc_list)
    auc_std = np.std(auc_list)

    return accuracy_mean, accuracy_std, sensitivity_mean, sensitivity_std, specificity_mean, specificity_std, f1_mean, f1_std, auc_mean, auc_std

def save_results_to_file(results, output_dir='results'):
    """
    将结果保存到文件
    :param results: 结果字符串
    :param output_dir: 输出文件夹
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    file_path = os.path.join(output_dir, f'results_{timestamp}.txt')
    with open(file_path, 'w') as f:
        f.write(results)

def plot_roc_curve(fold_roc_info, output_dir='roc_curves'):
    """
    绘制 ROC 曲线并保存到文件夹
    :param fold_roc_info: 每一折的 ROC 信息 (fpr, tpr, auc_score)
    :param output_dir: 输出文件夹
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plt.figure()
    for fold, (fpr, tpr, auc_score) in enumerate(fold_roc_info):
        plt.plot(fpr, tpr, label=f'Fold {fold + 1} (AUC = {auc_score:.2f})')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    file_path = os.path.join(output_dir, f'roc_curve_{timestamp}.png')
    plt.savefig(file_path)
    plt.close()