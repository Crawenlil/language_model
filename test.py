from src.models import get_language_model
from src.utils import (
    load_dictionaries
)

import torch
import torch.nn.functional as F
import yaml
import argparse
import os


def parse_arguments():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True,
                    type=str, help="Path to config file in yaml format")
    return vars(ap.parse_args())


def test(model):
    model.eval()

    while True:
        try:
            line = input('>> ')
            line = '<s> ' + line
            sent = line.strip().split()
            X = torch.tensor([model.word2index(w) for w in sent])
            outputs = model(X.view(1, -1))
            outputs = torch.flatten(outputs)
            softmax = F.softmax(outputs, dim=-1)
            y = torch.argsort(softmax, dim=-1, descending=True)
            for i in range(10):
                idx = y[i].item()
                print('{} -> {:.4}'.format(
                    model.index2word(idx),
                    softmax[idx]
                ))
        except EOFError:
            break


def main():
    args = parse_arguments()
    config_path = args['config']

    with open(config_path) as f:
        conf = yaml.safe_load(f)

    w2i, i2w = load_dictionaries(conf)
    model = get_language_model(conf, w2i, i2w)
    model_path = conf['model_path']
    model_path = os.path.join(model_path, model.name)
    model.load_state_dict(torch.load(model_path))

    test(model)


if __name__ == '__main__':
    main()
