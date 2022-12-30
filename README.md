# Introduction

beta release at https://pypi.org/project/torch-sentiment/, install with:  
```pip install torch-sentiment```

This repo contains Neural Nets written in the pytorch framework for sentiment analysis.  
A LSTM based torch model can be found in the rnn folder. In spite of the state of the art large language models (like GPT3.5 as of 2023), smaller models are still pretty efficient and useful to learn from. They work pretty well for some tasks like sentiment analysis. This model was trained on a local machine with a single GPU. 

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

## Install dev
```
conda create -n {env}  
conda install pip  
pip install -e .  
```

## Example
To setup environments and run the rnn example see instructions below:  

```
get the yelp dataset from https://www.yelp.com/dataset
run
```

