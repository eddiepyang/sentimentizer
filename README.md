# Introduction

Beta release at [pypi](https://pypi.org/project/torch-sentiment/);
api subject to change. Install with:  

```
pip install torch-sentiment
```  
  
This repo contains Neural Nets written with the pytorch framework for sentiment analysis.  
A LSTM based torch model can be found in the rnn folder. In spite of the state of the art  
large language models (like GPT3.5 as of 2023), smaller models are still pretty efficient  
and useful to learn from. They work pretty well for some tasks like sentiment analysis. 
This model was trained on a single gpu and required less than 1GB of memory.

  
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
To update the model get the yelp [dataset](https://www.yelp.com/dataset), then run with the rnn training entry point:

```
train
```

