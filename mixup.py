import torch


def structured_mixup(features, adj_matrices, labels, alpha=0.4):

    if alpha <= 0:
        return features, adj_matrices, labels, labels, 1.0

    # 生成混合系数
    lam = torch.distributions.Beta(alpha, alpha).sample().to(features.device)

    # 批次维度混洗
    batch_size = features.size(0)
    index = torch.randperm(batch_size, device=features.device)

    # 结构化混合
    mixed_features = lam * features + (1 - lam) * features[index]
    mixed_adj = lam * adj_matrices + (1 - lam) * adj_matrices[index]

    return mixed_features, mixed_adj, labels, labels[index], lam

def graph_mixup_loss(criterion, pred, labels_a, labels_b, lam):

    loss = lam * criterion(pred, labels_a)
    loss += (1 - lam) * criterion(pred, labels_b)
    return loss