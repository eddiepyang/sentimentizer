# Introduction

[![PyPI Latest Release](https://img.shields.io/pypi/v/torch-sentiment.svg)](https://pypi.org/project/torch-sentiment/)
![GitHub CI](https://github.com/eddiepyang/torch-sentiment/actions/workflows/ci.yaml/badge.svg)
  
Beta release, api subject to change. Install with:  

```
pip install torch-sentiment
```  
  
This repo contains Neural Nets written with the pytorch framework for sentiment analysis.  
A LSTM based torch model can be found in the rnn folder. In spite of large language models (GPT3.5 as of 2023) 
dominating the conversation, small models can be pretty effective and are nice to learn from. This model focuses on sentiment analysis and was trained on 
a single gpu in minutes and requires less than 1GB of memory.

  
## Usage
```
# where 0 is very negative and 1 is very positive
from torch_sentiment.rnn.tokenizer import get_trained_tokenizer
from torch_sentiment.rnn.model import get_trained_model

model = get_trained_model(64)
tokenizer = get_trained_tokenizer()
review_text = "greatest pie ever, best in town!"
positive_ids = tokenizer.tokenize_text(review_text)
model.predict(positive_ids)
  
>>> tensor(0.9701)
```

## Install for development with miniconda:  
```
conda create -n {env}  
conda install pip  
pip install -e .  
```

## Retrain model
To rerun the model:
* get the yelp [dataset](https://www.yelp.com/dataset), 
* get the glove 6B 100D [dataset](https://nlp.stanford.edu/projects/glove/)
* place both files in the package data directory 
* run with the rnn training entry point:

```
train
```

