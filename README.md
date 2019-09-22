# Language models


## Getting Started

### Data format
Corpus file should contain sentences, each in separate line.

### Preprocess
Before training model datasets should be prepared based on corpus file. Running 
```bash
python preprocess.py --corpus-path=data/wiki100k/corpus.txt --output-directory=data/wiki100k/
```
will create word2index, index2word, index2count, trainset and testset.

### Model configuration 
Model configurations are stored in ```configs``` directory.

### Training model
```bash
python train.py --config=configs/wiki100k.yaml
```

### Testing model
```bash
python test.py --config=configs/wiki100k.yaml
```


