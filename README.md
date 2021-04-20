# Introduction
This repo contains notebook implementations of RNN models with pytorch and keras for sentiment analysis on labeled reviews from Yelp.  
  
A package for a LSTM based torch model can be found in the rnn folder.

To setup environments and run see instructions below:  

*conda create -n yelp  
conda install pip  
pip install -e .  
cd yelp_nlp/rnn  
python train.py*

# Proposed implementation

Ingestion pipe -> Data pipe -> Model pipe -> Serving instance

1. Ingestion pipe  
Datastream -> ingestion engine -> data store -> bulk download api

3. Data pipe  
Yelp api -> data -> data processing -> storage 

3. Model pipe  
storage -> data loader -> training instance -> docker image

4. Serving instance   
load docker image -> serving instance 
