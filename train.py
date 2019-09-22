from src.models import get_language_model
from src.datasets import TextDataset
from src.utils import (
    load_datasets,
    load_dictionaries
)

import torch
import math
from tqdm import tqdm
import yaml
import argparse
import os


def parse_arguments():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True,
                    type=str, help="Path to config file in yaml format")
    return vars(ap.parse_args())


def train(train_loader, test_loader, model, conf, device):
    model.train()
    model.to(device)
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(conf['n_epochs']):
        train_loss = 0.0
        test_loss = 0.0
        for sample in tqdm(train_loader):
            X, Y = sample['left_context'].to(device), sample['word'].to(device)

            optimizer.zero_grad()

            outputs = model(X)
            loss = loss_function(outputs, Y)
            loss.backward()
            optimizer.step()

            # print statistics
            train_loss += loss.item()
        with torch.no_grad():
            for sample in test_loader:
                X, Y = X.to(device), Y.to(device)
                outputs = model(X)
                loss = loss_function(outputs, Y)
                test_loss += loss.item()
        print('epoch {:3d}, train_loss: {:.6f} test_loss: {:.6f}, '
              'perplexity: {:.6f}'.format(
                  epoch + 1,
                  train_loss / len(train_loader),
                  test_loss / len(test_loader),
                  math.exp(test_loss / len(test_loader))
              ))


def main():
    args = parse_arguments()
    config_path = args['config']

    with open(config_path) as f:
        conf = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    word2index, index2word = load_dictionaries(conf)
    train_dataset, test_dataset = load_datasets(conf)
    train_dataset = TextDataset(
        dataset=train_dataset,
        left_context_len=conf['left_context_len'],
    )
    test_dataset = TextDataset(
        dataset=test_dataset,
        left_context_len=conf['left_context_len'],
    )
    w2i, i2w = load_dictionaries(conf)
    model = get_language_model(conf, w2i, i2w)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=conf['batch_size'],
        shuffle=True,
        num_workers=2
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=conf['batch_size'],
        shuffle=True,
        num_workers=2
    )
    train(train_loader, test_loader, model, conf, device)
    model_path = conf['model_path']
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    model_path = os.path.join(model_path, model.name)
    torch.save(model.state_dict(), model_path)
    print("Model has been saved to {}".format(model_path))


if __name__ == '__main__':
    main()
