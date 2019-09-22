import pickle
import pandas as pd
import os
# PATHS
TRAIN_DATASET_FNAME = 'train_dataset.csv'
TEST_DATASET_FNAME = 'test_dataset.csv'
VOCAB_SIZE_FNAME = 'vocab_size'

WORD2INDEX_FNAME = 'word2index.pickle'
INDEX2WORD_FNAME = 'index2word.pickle'
INDEX2COUNT_FNAME = 'index2count.pickle'

MAX_SENTENCE_LENGTH = 64

# TOKENS
PAD_STR = '<pad>'
UNK_STR = '<unk>'
BOS_STR = '<s>'
EOS_STR = '</s>'
PAD = 0
UNK = 1
BOS = 2
EOS = 3


def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def save_pickle(path, obj):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def load_dictionaries(conf):
    word2index = load_pickle(os.path.join(
        conf['dataset_path'],
        WORD2INDEX_FNAME
    ))
    index2word = load_pickle(os.path.join(
        conf['dataset_path'],
        INDEX2WORD_FNAME
    ))

    return word2index, index2word


def load_datasets(conf):
    train_dataset = pd.read_csv(
        os.path.join(conf['dataset_path'], TRAIN_DATASET_FNAME)
    ).values
    test_dataset = pd.read_csv(
        os.path.join(conf['dataset_path'], TEST_DATASET_FNAME)
    ).values
    return train_dataset, test_dataset
