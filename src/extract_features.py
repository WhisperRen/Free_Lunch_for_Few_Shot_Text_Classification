import logging
import os
import json
from collections import defaultdict
from typing import Dict, AnyStr

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, AlbertModel
from tqdm import tqdm

from src.model_and_dataset import MyDataSet, AlbertClassificationModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_features(config: Dict, base_or_novel: AnyStr, train_or_test: AnyStr):
    bert_tokenizer = BertTokenizer.from_pretrained(config.get('pretrained_weights_path'))
    albert_model = AlbertModel.from_pretrained(config.get('pretrained_weights_path'))
    model = AlbertClassificationModel(pretrained_model=albert_model,
                                      hidden_size=albert_model.config.hidden_size,
                                      label_num=config.get(f'label_num_{base_or_novel}'))
    with open(os.path.join(config.get('checkpoint_save_path'), 'f1_scores.json'), 'r') as f:
        f1_scores = json.load(f).get('f1_scores', [])
        if not f1_scores:
            logger.error(f'There is no f1_scores.json, train feature extractor first')
            return

    device = config.get('device')
    model.to(device)
    best_weights = torch.load(os.path.join(config.get('checkpoint_save_path'),
                                           f'extractor_checkpoint_{np.argmax(f1_scores)}'), map_location=device)
    model.pretrained_model.load_state_dict(best_weights)
    dataset = MyDataSet(tokenizer=bert_tokenizer, config=config, base_or_novel=base_or_novel, mode=train_or_test)
    data_loader = DataLoader(dataset, batch_size=config.get(f'batch_size_{train_or_test}'),
                             shuffle=False, num_workers=0)
    logger.info(f'################### {base_or_novel} {train_or_test} feature extraction start #####################')
    model.eval()
    max_text_length = config.get('max_text_length')
    feature_dict = defaultdict(list)
    with torch.no_grad():
        for x, y in tqdm(data_loader):
            inputs_dict = {'input_ids': x.get('input_ids').squeeze().reshape(-1, max_text_length).to(device),
                           'token_type_ids': x.get('token_type_ids').squeeze().reshape(-1, max_text_length).to(device),
                           'attention_mask': x.get('attention_mask').squeeze().reshape(-1, max_text_length).to(device),
                           'return_dict': True}
            _, features = model(inputs_dict)
            for label, feature in zip(y.cpu().numpy(), features.cpu().numpy()):
                feature_dict[label].append(feature)
    torch.save(feature_dict,
               os.path.join(config.get('feature_vectors_path'), f'{base_or_novel}_{train_or_test}_features.pkl'))
    logger.info(f'Save {base_or_novel} {train_or_test} features to {config.get("feature_vectors_path")}')


if __name__ == '__main__':
    with open(r'config.json', 'r') as file:
        configs = json.load(file)
    if not os.path.exists(configs.get('feature_vectors_path')):
        os.makedirs(configs.get('feature_vectors_path'))
    device_ = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    configs['device'] = device_
    logger.info(f'Using device {device_}')
    extract_features(configs, base_or_novel='base', train_or_test='train')
    extract_features(configs, base_or_novel='novel', train_or_test='train')
    extract_features(configs, base_or_novel='novel', train_or_test='test')
