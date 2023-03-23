# Introduction

[![PyPI Latest Release](https://img.shields.io/pypi/v/sentimentizer.svg)](https://pypi.org/project/sentimentizer/)
![GitHub CI](https://github.com/eddiepyang/sentimentizer/actions/workflows/ci.yaml/badge.svg)
  
Beta release, api subject to change. Install with:  

```
pip install sentimentizer
```  
  
This repo contains Neural Nets written with the pytorch framework for sentiment analysis.  
A LSTM based torch model can be found in the rnn folder. In spite of large language models (GPT3.5 as of 2023) 
dominating the conversation, small models can be pretty effective and are nice to learn from. This model focuses on sentiment analysis and was trained on 
a single gpu in minutes and requires less than 1GB of memory.

  
## Usage
```
# where 0 is very negative and 1 is very positive
from sentimentizer.tokenizer import get_trained_tokenizer
from sentimentizer.models.rnn import get_trained_model

model = get_trained_model(64, 'cpu')
tokenizer = get_trained_tokenizer()
review_text = "greatest pie ever, best in town!"
positive_ids = tokenizer.tokenize_text(review_text)
model.predict(positive_ids)
  
>> tensor(0.9701)
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
* run the training script in workflows

