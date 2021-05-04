# Introduction
This repo contains Neural Nets written in the pytorch framework for sentiment analysis on labeled reviews from Yelp.  
A package for a LSTM based torch model can be found in the rnn folder. To setup environments and run the rnn example see instructions below:  

*conda create -n yelp  
conda install pip  
pip install -e .  
cd yelp_nlp/rnn  
python train.py*

# Proposed implementation

Ingestion pipe -> Data pipe -> Model pipe -> Serving instance

1. Ingestion pipe  
datastreams -> ingestion engine -> data store -> interactive api

3. Data pipe  
interactive api -> data -> data processing -> storage 

3. Model pipe  
storage -> data loader -> training instance -> docker image

4. Serving instance   
load docker image -> serving instance 
