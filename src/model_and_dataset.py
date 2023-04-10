import numpy as np
import pandas as pd
import torch.nn as nn
from torch.utils.data import Dataset
from joblib import Parallel, delayed


class MyDataSet(Dataset):
    def __init__(self, tokenizer, config, base_or_novel, mode='train'):
        super(MyDataSet, self).__init__()
        if base_or_novel not in ['base', 'novel']:
            raise ValueError('base_or_novel must be "base" or "novel"')
        if mode not in ['train', 'val', 'test']:
            raise ValueError('mode must be "train" or "val" or "test"')
        self.tokenizer = tokenizer
        self.config = config
        data_path = self.config.get('data_path_base').get(mode) if base_or_novel == 'base' else self.config.get(
            'data_path_novel').get(mode)
        data_df = pd.read_csv(data_path)

        with Parallel(n_jobs=config.get('n_jobs')) as parallel:
            data_list = parallel(delayed(self.tokenize_func)(df) for _, df in data_df.groupby('y'))
        self.x = np.concatenate([item[0] for item in data_list])
        self.y = np.concatenate([item[1] for item in data_list])

    def __len__(self):
        return min(len(self.x), len(self.y))

    def __getitem__(self, index):
        if index >= self.__len__():
            raise IndexError
        return self.x[index], self.y[index]

    def tokenize_func(self, df):
        return df['text'].apply(lambda t: self.tokenizer.encode_plus(t,
                                                                     add_special_tokens=True,
                                                                     max_length=self.config.get('max_text_length'),
                                                                     padding='max_length',
                                                                     return_tensors='pt').data).values, df['y'].values


class AlbertClassificationModel(nn.Module):
    def __init__(self, pretrained_model, hidden_size, label_num):
        super(AlbertClassificationModel, self).__init__()
        self.pretrained_model = pretrained_model
        self.post_layer = nn.Linear(hidden_size, label_num, bias=False)

    def forward(self, inputs):
        pooled_features = nn.ReLU()(self.pretrained_model(**inputs).pooler_output)
        logit = self.post_layer(pooled_features)
        return logit, pooled_features


class ClassificationModelDC(nn.Module):
    def __init__(self, hidden_size, label_num):
        super(ClassificationModelDC, self).__init__()
        self.post_layer = nn.Linear(hidden_size, label_num, bias=False)

    def forward(self, inputs):
        return self.post_layer(inputs)

