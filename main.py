import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
import yaml
import logging
from sklearn.metrics import roc_curve
import os
import random
from model.transformer import TransformerClassifier
from dataload import get_k_fold_dataloaders
from metrics import calculate_metrics, save_results_to_file, plot_roc_curve, evaluate_k_fold
from mixup import structured_mixup,graph_mixup_loss
# 配置日志记录
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler()
    ]
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train(model, train_dataloader, criterion, optimizer, config, writer, epoch, fold):
    model.train()
    train_total_loss = 0
    train_correct = 0
    train_total = 0
    flood_level = 0.1  # 洪水水位，即最低损失
    for data in train_dataloader:
        labels = data['labels'].long().to(device)
        fc_features = data['fc_features'].float().to(device)
        fc_adj = data['fc_adj'].float().to(device)
        subj_fc = data['subj_fc'].float().to(device)

        optimizer.zero_grad()
        outputs = model(fc_features, fc_adj, subj_fc)
        _, predicted = torch.max(outputs.data, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()

        loss = criterion(outputs, labels)
        # 洪水正则化
        #loss = (loss - flood_level).abs() + flood_level

        loss.backward()
        optimizer.step()

        train_total_loss += loss.item()

    train_accuracy = 100 * train_correct / train_total
    train_avg_loss = train_total_loss / len(train_dataloader)

    writer.add_scalars(f'Fold{fold + 1}/Loss', {'train': train_avg_loss}, epoch)
    writer.add_scalars(f'Fold{fold + 1}/Accuracy', {'train': train_accuracy}, epoch)

    return train_avg_loss, train_accuracy


def test(model, test_dataloader, criterion, config, writer, epoch, fold):
    model.eval()
    val_total_loss = 0
    val_correct = 0
    val_total = 0
    all_labels = []
    all_predictions = []
    all_scores = []

    with torch.no_grad():
        for data in test_dataloader:
            labels = data['labels'].long().to(device)
            fc_features = data['fc_features'].float().to(device)
            fc_adj = data['fc_adj'].float().to(device)
            subj_fc = data['subj_fc'].float().to(device)

            outputs = model(fc_features, fc_adj, subj_fc)
            scores = torch.exp(outputs)[:, 1]  # 直接取指数得到概率
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

            loss = criterion(outputs, labels)
            val_total_loss += loss.item()

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            all_scores.extend(scores.cpu().numpy())

    val_accuracy = 100 * val_correct / val_total
    val_avg_loss = val_total_loss / len(test_dataloader)

    writer.add_scalars(f'Fold{fold + 1}/Loss', {'val': val_avg_loss}, epoch)
    writer.add_scalars(f'Fold{fold + 1}/Accuracy', {'val': val_accuracy}, epoch)

    accuracy, sensitivity, specificity, f1, auc_score = calculate_metrics(all_labels, all_predictions, all_scores)
    fpr, tpr, _ = roc_curve(all_labels, all_scores)

    return val_avg_loss, val_accuracy, accuracy, sensitivity, specificity, f1, auc_score, fpr, tpr


def main():
    set_seed(1121)

    with open('config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    writer = SummaryWriter()

    data_dir = config['data_dir']
    k = config['k_fold']
    batch_size = config['batch_size']
    threshold = config['threshold']
    window_size = config['window_size']
    step = config['step']
    feature = config['feature']
    input_dim = config['input_dim']
    d_model = config['d_model']
    nhead = config['nhead']
    num_layers = config['num_layers']
    dim_feedforward = config['dim_feedforward']
    learning_rate = config['learning_rate']
    weight_decay = config['weight_decay']
    early_stopping_patience = config['early_stopping_patience']
    early_stopping_min_delta = config['early_stopping_min_delta']

    model_save_dir = 'saved_models'
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)

    criterion = nn.NLLLoss()
    criterion = criterion.to(device)

    fold_dataloaders = get_k_fold_dataloaders(data_dir, k, batch_size, threshold, window_size, step, feature)

    fold_metrics = []
    fold_roc_info = []

    for fold, (train_dataloader, test_dataloader) in enumerate(fold_dataloaders):
        logging.info(f"Starting Fold {fold + 1}/{k}")

        best_val_accuracy = 0
        patience_counter = 0

        # 从数据中计算 num_fc
        for data in train_dataloader:
            fc_features = data['fc_features']
            num_win = fc_features.shape[1]
            break

        model = TransformerClassifier(input_dim, d_model, nhead, num_layers, dim_feedforward, num_win)
        model = model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        # 设置学习率调度器，周期为 config['epochs']
        scheduler = CosineAnnealingLR(optimizer, T_max=config['epochs'], eta_min=1e-5)

        for epoch in range(config['epochs']):
            train_avg_loss, train_accuracy = train(model, train_dataloader, criterion, optimizer, config, writer, epoch,
                                                   fold)
            val_avg_loss, val_accuracy, accuracy, sensitivity, specificity, f1, auc_score, fpr, tpr = test(model,
                                                                                                           test_dataloader,
                                                                                                           criterion,
                                                                                                           config,
                                                                                                           writer,
                                                                                                           epoch, fold)

            logging.info(
                f"Fold {fold + 1} - Epoch {epoch + 1} - Train Loss: {train_avg_loss:.4f}, Train Acc: {train_accuracy:.2f}%, Val Loss: {val_avg_loss:.4f}, Val Acc: {val_accuracy:.2f}%")

            # 更新学习率
            scheduler.step()

            # 打印当前学习率
            current_lr = optimizer.param_groups[0]['lr']
            logging.info(f"Fold {fold + 1} - Epoch {epoch + 1} - Current Learning Rate: {current_lr:.6f}")

            # 早停
            if val_accuracy > best_val_accuracy + early_stopping_min_delta:
                best_val_accuracy = val_accuracy
                patience_counter = 0
                # 保存最佳模型
                model_path = os.path.join(model_save_dir, f'best_model_fold_{fold + 1}.pth')
                torch.save(model.state_dict(), model_path)
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    logging.info(f"Early stopping at epoch {epoch + 1} for fold {fold + 1}")
                    break

        fold_metrics.append((accuracy, sensitivity, specificity, f1, auc_score))
        fold_roc_info.append((fpr, tpr, auc_score))

    writer.close()

    accuracy_mean, accuracy_std, sensitivity_mean, sensitivity_std, specificity_mean, specificity_std, f1_mean, f1_std, auc_mean, auc_std = evaluate_k_fold(
        fold_metrics)

    results = f"K-Fold Cross Validation Results:\n"
    results += f"Accuracy: {accuracy_mean:.4f} ± {accuracy_std:.4f}\n"
    results += f"Sensitivity: {sensitivity_mean:.4f} ± {sensitivity_std:.4f}\n"
    results += f"Specificity: {specificity_mean:.4f} ± {specificity_std:.4f}\n"
    results += f"F1 Score: {f1_mean:.4f} ± {f1_std:.4f}\n"
    results += f"AUC: {auc_mean:.4f} ± {auc_std:.4f}\n"

    # save_results_to_file(results)
    plot_roc_curve(fold_roc_info)

    logging.info(results)


if __name__ == "__main__":
    main()