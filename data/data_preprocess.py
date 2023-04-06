import json

import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_SEED = 99


def split_dataset(df, val_ratio=0.05, test_ratio=0.05):
    train_set, val_test_set = train_test_split(df,
                                               stratify=df['label'],
                                               test_size=val_ratio + test_ratio,
                                               random_state=RANDOM_SEED)
    val_set, test_set = train_test_split(val_test_set,
                                         stratify=val_test_set['label'],
                                         test_size=test_ratio / (val_ratio + test_ratio),
                                         random_state=RANDOM_SEED)
    return train_set, val_set, test_set


if __name__ == '__main__':
    with open(r'../src/config.json', 'r', encoding='utf-8') as f:
        config = json.load(f)

    split_ratio = {'val_ratio': 0.05, 'test_ratio': 0.1}

    data = pd.read_csv(r'./data.txt', sep='\t', header=None, names=['text', 'label'])

    # make label map
    class_map = config.get('class_map')
    data['y'] = data['label'].apply(lambda l: class_map.get(l, 0))

    # show duplication data and data number in different class
    print(data.groupby('y').nunique())

    # split data to base classes and novel classes
    base_class_tag = [i for i in range(config.get('label_num_base'))]
    novel_class_tag = [i for i in range(config.get('label_num_base'), len(class_map))]
    base_class_data = data[data['y'].isin(base_class_tag)]
    novel_class_data = data[data['y'].isin(novel_class_tag)]
    novel_class_new_map = {old_tag: old_tag - config.get('label_num_base')
                           for old_tag in range(config.get('label_num_base'), len(class_map))}
    novel_class_data['y'] = novel_class_data['y'].apply(lambda y: novel_class_new_map.get(y))

    # split base dataset
    train_base, val_base, test_base = split_dataset(base_class_data, **split_ratio)
    train_base.to_csv(r'./train_data_base.csv', index=None)
    val_base.to_csv(r'./val_data_base.csv', index=None)
    test_base.to_csv(r'./test_data_base.csv', index=None)

    # split novel dataset
    train_novel, val_novel, test_novel = split_dataset(novel_class_data, **split_ratio)
    train_data = pd.DataFrame()
    for _, d in train_novel.groupby('y'):
        train_data = pd.concat([train_data, d.head(config.get('n_shot'))])
    train_data.to_csv(r'./train_data_novel.csv', index=None)
    val_novel.to_csv(r'./val_data_novel.csv', index=None)
    test_novel.to_csv(r'./test_data_novel.csv', index=None)
