import logging
import os
import json

import torch

from src.train_feature_extractor import train_feature_extractor
from src.extract_features import extract_features
from src.evaluate_distribution_calibration import setup_seed, evaluate_distribution_calibration


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


if __name__ == '__main__':
    setup_seed()
    with open(r'config.json', 'r') as f:
        config = json.load(f)

    if not os.path.exists(config.get('checkpoint_save_path')):
        os.makedirs(config.get('checkpoint_save_path'))
    if not os.path.exists(config.get('feature_vectors_path')):
        os.makedirs(config.get('feature_vectors_path'))

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    config['device'] = device
    logger.info(f'Using device {device}')

    # fine-tune extractor using base set data
    train_feature_extractor(config)

    # extract features
    extract_features(config, base_or_novel='base', train_or_test='train')
    extract_features(config, base_or_novel='novel', train_or_test='train')
    extract_features(config, base_or_novel='novel', train_or_test='test')

    # evaluate distribution calibration
    evaluate_distribution_calibration(config)
