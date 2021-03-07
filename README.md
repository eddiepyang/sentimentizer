# Introduction
This repo contains implementations of RNN models with pytorch and keras for sentiment analysis on labeled data from Yelp. Run setup as described to setup environment:  

*conda create env -n yelp_nlp  
conda install pip  
pip install -e .*

# Proposed AWS implementation

Data pipe -> Model pipe -> Serving instance

1. Data pipe
Yelp api -> data -> data processing -> storage (s3)

2. Model pipe
storage -> data loader -> training instance -> saved model and image (ECR) 

3. Serving instance 
Load docker image -> serving instance (elastic beanstalk or sagemaker inference)

# Notebook view
The torch implmentation can be viewed by the jupyter viewer below: 

[view_notebook](https://nbviewer.jupyter.org/github/eddiepyang/yelp_nlp/blob/master/notebook/torch-sentiment.ipynb)
