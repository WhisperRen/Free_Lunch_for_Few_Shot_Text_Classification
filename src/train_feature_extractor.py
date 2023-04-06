import logging
import os
import json
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import BertTokenizer, AlbertModel
from sklearn.metrics import f1_score
from tqdm import tqdm

from src.model_and_dataset import MyDataSet, AlbertClassificationModel


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_feature_extractor(config: Dict):
    bert_tokenizer = BertTokenizer.from_pretrained(config.get('pretrained_weights_path'))
    albert_model = AlbertModel.from_pretrained(config.get('pretrained_weights_path'))
    model = AlbertClassificationModel(pretrained_model=albert_model,
                                      hidden_size=albert_model.config.hidden_size,
                                      label_num=config.get('label_num_base'))
    train_set = MyDataSet(tokenizer=bert_tokenizer, config=config, base_or_novel='base', mode='train')
    train_loader = DataLoader(train_set, batch_size=config.get('batch_size_train'), shuffle=True, num_workers=0)
    val_set = MyDataSet(tokenizer=bert_tokenizer, config=config, base_or_novel='base', mode='val')
    val_loader = DataLoader(val_set, batch_size=config.get('batch_size_val'), shuffle=False, num_workers=0)
    loss_obj = nn.CrossEntropyLoss()
    device = config.get('device')
    model.to(device)
    optimizer = optim.AdamW([{'params': model.pretrained_model.parameters(), 'lr': config.get('lr_pretrained_layer')},
                             {'params': model.post_layer.parameters(), 'lr': config.get('lr_scratch_layer')}])
    f1_scores = []
    max_text_length = config.get('max_text_length')
    logger.info('#################### Extractor fine-tuning start ######################')
    for i in range(config.get('epoch')):
        model.train()
        for j, (x, y) in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()
            inputs_dict = {'input_ids': x.get('input_ids').squeeze().reshape(-1, max_text_length).to(device),
                           'token_type_ids': x.get('token_type_ids').squeeze().reshape(-1, max_text_length).to(device),
                           'attention_mask': x.get('attention_mask').squeeze().reshape(-1, max_text_length).to(device),
                           'return_dict': True}
            y = y.to(device)
            logits, _ = model(inputs_dict)
            loss = loss_obj(logits, y)
            loss.backward()
            optimizer.step()
            if j % 100 == 0:
                logger.info(f'Epoch {i} Batch {j}, Train loss is: {loss.item()}')

        logger.info(f'Epoch {i} validation start')
        pred_all, label_all = [], []
        model.eval()
        with torch.no_grad():
            for x_val, y_val in tqdm(val_loader):
                inputs_dict = {'input_ids': x_val.get('input_ids').squeeze().reshape(-1, max_text_length).to(device),
                               'token_type_ids': x_val.get('token_type_ids').squeeze().reshape(-1, max_text_length).to(
                                   device),
                               'attention_mask': x_val.get('attention_mask').squeeze().reshape(-1, max_text_length).to(
                                   device),
                               'return_dict': True}
                pred, _ = model(inputs_dict)
                pred = np.argmax(F.softmax(pred.cpu(), dim=1).numpy(), axis=1)
                pred_all.extend(pred.tolist())
                label_all.extend(y_val.numpy().tolist())
        f1 = f1_score(label_all, pred_all, average='macro')
        f1_scores.append(f1)
        logger.info('Epoch {} F1-score : {:.4f}'.format(i, f1))

        # save checkpoint, only save weights of feature extractor
        torch.save(model.pretrained_model.state_dict(),
                   os.path.join(config.get('checkpoint_save_path'), f'extractor_checkpoint_{i}'))
        logger.info(f'Save weights of feature extractor at epoch {i}')
    # save f1 scores in f1_scores.json
    with open(os.path.join(config.get('checkpoint_save_path'), 'f1_scores.json'), 'w') as f:
        json.dump({'f1_scores': f1_scores}, f)
    logger.info(f'Save f1_scores.json to {config.get("checkpoint_save_path")}')


if __name__ == '__main__':
    with open(r'config.json', 'r') as file:
        configs = json.load(file)
    if not os.path.exists(configs.get('checkpoint_save_path')):
        os.makedirs(configs.get('checkpoint_save_path'))
    if not os.path.exists(configs.get('feature_vectors_path')):
        os.makedirs(configs.get('feature_vectors_path'))
    device_ = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    configs['device'] = device_
    logger.info(f'Using device: {device_}')
    train_feature_extractor(configs)

