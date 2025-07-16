import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Subset
import torch

# 确定数据集标签
def decide_dataset():
    label_list = ['HC', 'MDD']
    label_set = {'HC': 0, 'MDD': 1}
    return label_list, label_set

# 读取数据集
def read_dataset_npy(path, file):
    file_path = os.path.join(path, file)
    data = np.load(file_path).astype(np.float32)[:, :90]  # 仅保留前90列
    return data

def get_Pearson_fc(sub_region_series, threshold):

    # 计算原始 FC 矩阵
    fc_matrix = np.corrcoef(np.transpose(sub_region_series))  # 形状为 [num_nodes, num_nodes]
    fc_matrix = np.nan_to_num(fc_matrix)  # 将 NaN 值替换为 0

    # 去除对角线的值
    #np.fill_diagonal(fc_matrix, 0)

    # 生成边索引
    n_nodes = fc_matrix.shape[0]
    triu_indices = np.triu_indices(n_nodes, k=1)  # 获取上三角矩阵的索引
    fc_values = fc_matrix[triu_indices]  # 提取上三角矩阵的值

    # 根据阈值选择边
    thindex = int(threshold * len(fc_values))
    threshold_value = np.sort(fc_values)[-thindex] if thindex > 0 else 1.0

    # 生成邻接矩阵
    adj_matrix = np.zeros_like(fc_matrix, dtype=np.float32)
    adj_matrix[fc_matrix >= threshold_value] = 1.0

    return adj_matrix,fc_matrix# 返回邻接矩阵

# 计算节点度矩阵
def get_fc_degree(subj_fc_adj):
    rowsum = np.array(subj_fc_adj.sum(1))
    N = np.diag(rowsum)
    return N

# 最大 - 最小归一化
def max_min_norm(sub_region_series):
    subj_fc_mat_list = sub_region_series.reshape((-1))
    subj_fc_feature = (sub_region_series - np.min(subj_fc_mat_list)) / (
            np.max(subj_fc_mat_list) - np.min(subj_fc_mat_list))
    return subj_fc_feature

# Z 分数标准化
def z_score_norm(data):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    return (data - mean) / std

class MyDynamicDataSet(Dataset):
    def __init__(self, data_dir, threshold=0.2, window_size=80, step=19, feature='FC'):
        self.labels = []
        self.each_sub_adj = []
        self.each_sub_feature = []
        self.each_sub_fc = []

        label_list, label_set = decide_dataset()

        for file in os.listdir(data_dir):
            fc_adj = []
            fc_features = []
            sub_fc = []

            data = read_dataset_npy(data_dir, file)
            total_window_size = data.shape[0]
            # 使用滑动窗口遍历数据
            for j in range(0, total_window_size - window_size + 1, step):
                sub_region_series = data[j:j + window_size, :]

                subj_fc_adj, subj_fc = get_Pearson_fc(sub_region_series, threshold)
                fc_adj.append(subj_fc_adj)
                sub_fc.append(subj_fc)

                if feature == 'BOLD':
                    subj_fc_feature = max_min_norm(sub_region_series)
                    fc_features.append(np.transpose(subj_fc_feature))
                elif feature == 'BoldCatDegree':
                    subj_fc_feature = max_min_norm(sub_region_series)
                    fc_degree = get_fc_degree(subj_fc_adj)
                    bold_C_degree = np.concatenate((subj_fc_feature, fc_degree), 0)
                    fc_features.append(np.transpose(bold_C_degree))
                elif feature == 'FC':
                    fc_features.append(subj_fc)

            self.each_sub_adj.append(np.array(fc_adj))
            self.each_sub_feature.append(np.array(fc_features))
            self.each_sub_fc.append(np.array(sub_fc))

            if file.startswith('ROISignals_S20-1-'):
                self.labels.append(label_set['MDD'])
            elif file.startswith('ROISignals_S20-2-'):
                self.labels.append(label_set['HC'])

        self.length = len(self.labels)
        print('The size of this dataset is %d' % self.length)

    def __getitem__(self, index):
        label = self.labels[index]
        fc_adj = self.each_sub_adj[index]
        fc_features = self.each_sub_feature[index]
        subj_fc = self.each_sub_fc[index]

        #fc_features = add_random_noise(fc_features)
        data = {
            'labels': label,
            'fc_adj': fc_adj,
            'fc_features': fc_features,
            'subj_fc': subj_fc
        }
        return data

    def __len__(self):
        return self.length

def get_k_fold_dataloaders(data_dir, k=5, batch_size=8, threshold=0.2, window_size=100, step=4, feature='FC'):
    dataset = MyDynamicDataSet(data_dir, threshold, window_size, step, feature)
    labels = np.array(dataset.labels)
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=25)
    fold_dataloaders = []

    for train_index, test_index in skf.split(np.zeros(len(labels)), labels):
        train_subset = Subset(dataset, train_index)
        test_subset = Subset(dataset, test_index)

        train_dataloader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        test_dataloader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)

        fold_dataloaders.append((train_dataloader, test_dataloader))

    return fold_dataloaders

'''
data_dir = 'E:\\pycharm\\data\\s20'  # 替换为你的数据目录路径
fold_dataloaders = get_k_fold_dataloaders(data_dir, k=5, batch_size=32, threshold=0.2, window_size=230, step=1, feature='FC')

# 获取第一折的训练和测试数据加载器
train_dataloader, test_dataloader = fold_dataloaders[0]

# 从训练数据加载器中获取一个批次的数据
for batch in train_dataloader:
    print("Batch data shapes and types:")
    print("Labels shape:", batch['labels'].shape, "type:", type(batch['labels']))
    print("fc_adj shape:", batch['fc_adj'].shape, "type:", type(batch['fc_adj']))
    print("fc_features shape:", batch['fc_features'].shape, "type:", type(batch['fc_features']))
    print("subj_fc shape:", batch['subj_fc'].shape, "type:", type(batch['subj_fc']))
    print("Adjacency Matrices (fc_adj):")
    for i, adj_matrix in enumerate(batch['fc_adj']):
        print(f"Adjacency Matrix {i + 1}:\n{adj_matrix}")
    break
'''