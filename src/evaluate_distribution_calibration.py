import logging
import os
import json
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import f1_score
from tqdm import tqdm

from src.model_and_dataset import ClassificationModelDC


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RANDOM_SEED = 99


def setup_seed():
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)


def distribution_calibration(feature_vector: np.array,
                             base_means: List,
                             base_cov: List,
                             top_k: int,
                             alpha: float):
    dist = [np.linalg.norm(feature_vector - base_mean) for base_mean in base_means]
    indexes = np.argpartition(dist, top_k)[:top_k]
    selected_means = np.concatenate([np.array(base_means)[indexes], feature_vector[np.newaxis, :]])
    calibrated_mean = np.mean(selected_means, axis=0)
    calibrated_cov = np.mean(np.array(base_cov)[indexes], axis=0) + alpha
    return calibrated_mean, calibrated_cov


def evaluate_distribution_calibration(config: Dict):
    logger.info('###################### Load base data and do statistics ######################')
    base_means, base_cov = [], []
    base_features_path = os.path.join(config.get('feature_vectors_path'), 'base_train_features.pkl')
    base_features = torch.load(base_features_path)
    for feature_base_list in base_features.values():
        feature_base_array = np.array(feature_base_list)
        mean = np.mean(feature_base_array, axis=0)
        cov = np.cov(feature_base_array.T)
        base_means.append(mean)
        base_cov.append(cov)

    # Tukey's transform
    novel_features_path_support = os.path.join(config.get('feature_vectors_path'), 'novel_train_features.pkl')
    support_data = torch.load(novel_features_path_support)
    novel_features_path_query = os.path.join(config.get('feature_vectors_path'), 'novel_test_features.pkl')
    query_data = torch.load(novel_features_path_query)
    if len(support_data) != len(query_data):
        raise IndexError('Number of support data is not equal to number of query data class, check data please')
    for c in support_data.keys():
        support_data[c] = np.power(support_data[c], config.get('lambda'))
        query_data[c] = np.power(query_data[c], config.get('lambda'))

    # distribution calibration
    logger.info('####################### Distribution calibration start #######################')
    sampled_data, sampled_label = [], []
    num_sampled = config.get('generated_feature_num') // config.get('n_shot')
    for c, features in support_data.items():
        if config.get('use_calibration'):
            for feature in features:
                calibrated_mean, calibrated_cov = distribution_calibration(
                    feature, base_means, base_cov, config.get('top_k'), config.get('alpha')
                )
                sampled_data.append(np.random.multivariate_normal(mean=calibrated_mean, cov=calibrated_cov,
                                                                  size=num_sampled))
            # add support data
            sampled_data.append(features)
            sampled_label.extend([c] * (len(features) * num_sampled + len(features)))
        else:
            sampled_data.append(features)
            sampled_label.extend([c] * len(features))
    x_aug = np.concatenate(sampled_data)
    y_aug = np.array(sampled_label)
    dataset = TensorDataset(
        torch.from_numpy(x_aug).to(torch.float32),
        torch.from_numpy(y_aug).to(torch.int64)
    )
    data_loader = DataLoader(dataset, batch_size=config.get('batch_size_train'), shuffle=True, num_workers=0)

    # train classifier
    logger.info('##################### Classifier training start ######################')
    losses = []
    model = ClassificationModelDC(hidden_size=x_aug.shape[-1], label_num=config.get('label_num_novel'))
    device = config.get('device')
    model.to(device)
    loss_obj = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.get('lr_scratch_layer'))
    model.train()
    for i in range(config.get('classifier_epoch')):
        for j, (x, y) in enumerate(tqdm(data_loader)):
            optimizer.zero_grad()
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = loss_obj(logits, y)
            loss.backward()
            optimizer.step()
            if j % 100 == 0:
                logger.info(f'Epoch {i} batch {j}, train loss is: {loss.item()}')
        losses.append(loss.item())
        torch.save(model.state_dict(), os.path.join(config.get('checkpoint_save_path'), f'classifier_checkpoint_{i}'))
        logger.info(f'Save checkpoint of classifier at epoch {i}')
    # evaluate classifier
    x_query, y_query = [], []
    for c, d in query_data.items():
        x_query.append(d)
        y_query.extend([c] * len(d))
    x_query = np.concatenate(x_query)
    y_query = np.array(y_query)
    dataset_query = TensorDataset(
        torch.from_numpy(x_query).to(torch.float32),
        torch.from_numpy(y_query).to(torch.int64)
    )
    data_loader_query = DataLoader(dataset_query, batch_size=config.get('batch_size_test'),
                                   shuffle=False, num_workers=0)
    pred_all, label_all = [], []
    best_weights = torch.load(os.path.join(config.get('checkpoint_save_path'),
                                           f'classifier_checkpoint_{np.argmin(losses)}'), map_location=device)
    model.load_state_dict(best_weights)
    model.eval()
    with torch.no_grad():
        for x_q, y_q in tqdm(data_loader_query):
            x_q = x_q.to(device)
            pred = model(x_q)
            pred = np.argmax(F.softmax(pred.cpu(), dim=1).numpy(), axis=1)
            pred_all.extend(pred.tolist())
            label_all.extend(y_q.numpy().tolist())
    f1 = f1_score(label_all, pred_all, average='macro')
    logger.info(f'F1-score on query set is : {f1:.4f}')


if __name__ == '__main__':
    setup_seed()
    with open(r'config.json', 'r') as file:
        configs = json.load(file)
    if not os.path.exists(configs.get('checkpoint_save_path')):
        os.makedirs(configs.get('checkpoint_save_path'))
    device_ = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    configs['device'] = device_
    logger.info(f'Using device {device_}')
    evaluate_distribution_calibration(configs)

