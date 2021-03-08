# Introduction
This repo contains notebook implementations of RNN models with pytorch and keras for sentiment analysis on labeled reviews from Yelp.  
  
A package for a LSTM based torch model can be found in the rnn folder.

To setup environments and run see instructions below:  

*conda create -n yelp  
conda install pip  
pip install -e .  
cd yelp_nlp/rnn  
python train.py*

# Proposed AWS implementation

Data pipe -> Model pipe -> Serving instance

1. Data pipe  
Yelp api -> data -> data processing -> storage (s3)

2. Model pipe  
storage -> data loader -> training instance -> saved model and image (ECR) 

3. Serving instance   
load docker image -> serving instance (elastic beanstalk or sagemaker inference)

# Notebook view
The torch implmentation can be viewed by the jupyter viewer below: 

[view_notebook](https://nbviewer.jupyter.org/github/eddiepyang/yelp_nlp/blob/master/notebook/torch-sentiment-refactor.ipynb)
