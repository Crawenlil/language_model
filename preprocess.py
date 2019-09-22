from collections import Counter
import os
import pandas as pd
import numpy as np
import argparse
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import gc
import re

from src.utils import (
    TRAIN_DATASET_FNAME,
    TEST_DATASET_FNAME,
    VOCAB_SIZE_FNAME,
    WORD2INDEX_FNAME,
    INDEX2WORD_FNAME,
    INDEX2COUNT_FNAME,
    MAX_SENTENCE_LENGTH,
    PAD_STR, PAD,
    UNK_STR, UNK,
    BOS_STR, BOS,
    EOS_STR, EOS,
    save_pickle
)


def parse_arguments():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus-path", required=True,
                    help="Path to corpus file")
    ap.add_argument("--output-directory", required=True,
                    help="path to output file where result will be stored")
    ap.add_argument("--min-count", required=False, default=0, type=int,
                    help="Minimum word count")
    return vars(ap.parse_args())


def index_words(corpus_path, output_directory, min_count):
    regex = re.compile(r'<s>\s?|\s?</s>|\r\n|\n', re.MULTILINE)
    word2index = {PAD_STR: PAD, BOS_STR: BOS,
                  EOS_STR: EOS, UNK_STR: UNK}
    index2word = {PAD: PAD_STR, BOS: BOS_STR,
                  EOS: EOS_STR, UNK: UNK_STR}
    index2count = Counter()
    word2count = Counter()
    data = []
    with open(corpus_path, 'r') as f:
        total = sum(1 for _ in f)
    with open(corpus_path, 'r') as f:
        for sentence in tqdm(f, total=total, desc='Reading corpus file'):
            sentence = re.sub(regex, '', sentence)
            words = sentence.split()
            for word in words:
                word2count[word] += 1
            data.append(words)
    unk_cnt = 0
    for word, count in tqdm(word2count.most_common(),
                            desc='Fitering words using min_count'):
        if count >= min_count:
            ind = len(word2index)
            word2index[word] = ind
            index2word[ind] = word
            index2count[ind] = count
        else:
            unk_cnt += 1
    index2count[PAD] = 0
    index2count[UNK] = unk_cnt
    index2count[BOS] = 1  # Laplace
    index2count[EOS] = 1  # Laplace

    del word2count

    with open(os.path.join(output_directory, VOCAB_SIZE_FNAME), 'w') as f:
        f.write(len(word2index))

    print("Saving word2index...")
    save_pickle(os.path.join(output_directory, WORD2INDEX_FNAME), word2index)

    print("Saving index2word...")
    save_pickle(os.path.join(output_directory, INDEX2WORD_FNAME), index2word)
    del index2word

    print("Saving index2count...")
    save_pickle(os.path.join(output_directory, INDEX2COUNT_FNAME), index2count)
    del index2count

    def pad(l, pad_token, length):
        return l + [pad_token] * (length - len(l))

    dataset = []
    for sentence in tqdm(data, desc="Creating dataset"):
        seq = [word2index[w] if w in word2index else UNK for w in sentence]
        seq = [BOS] + seq + [EOS]
        if len(seq) < MAX_SENTENCE_LENGTH:
            dataset.append(pad(seq, PAD, MAX_SENTENCE_LENGTH))

    print("Freeing memory...")
    del data
    del word2index
    gc.collect()

    if not os.path.exists(output_directory):
        print("{} doesn't exists, creating".format(output_directory))
        os.mkdir(output_directory)

    print("Creating pandas dataframe with dataset")
    df = pd.DataFrame(dataset, dtype=np.int32)
    df = df.sample(frac=1).reset_index(drop=True)
    train, test = train_test_split(df, test_size=0.2)
    train = train.reset_index(drop=True)
    test = train.reset_index(drop=True)

    print("Saving train dataset")
    train.to_csv(
        os.path.join(output_directory, TRAIN_DATASET_FNAME),
        index=False,
        header=False
    )

    print("Saving test dataset")
    test.to_csv(
        os.path.join(output_directory, TEST_DATASET_FNAME),
        index=False,
        header=False
    )


def main():
    args = parse_arguments()
    corpus_path = args['corpus_path']
    output_directory = args['output_directory']
    min_count = args['min_count']

    index_words(corpus_path, output_directory, min_count)


if __name__ == "__main__":
    main()
