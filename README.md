# DeepCoNN
This is a PyTorch implementation of DeepCoNN from the paper:  
  
_Lei Zheng, Vahid Noroozi, and Philip S Yu. 2017. Joint deep modeling of users and items using reviews for recommendation. In WSDM. ACM, 425-434._


## Before Running Code

#### Get Data
Download and unzip "Digital Music" data set from :  
http://jmcauley.ucsd.edu/data/amazon/  
Then put it under the path `data`

#### Get Pre-trained Word Embedding Model
We use GoogleNews-vectors-negative300.bin as pre-trained word embedding model.
You could find it at:  
https://code.google.com/archive/p/word2vec/  
Then put it under the path `data`

#### Environments
```
pandas~=1.0.3
numpy~=1.18.1
gensim~=3.8.0
pytorch~=1.3.1
nltk~=3.4.5
scikit-learn~=0.22.1
```

## Train & Eval Model

#### Data Pre-processing
```
python -m utils.data_reader
```

#### Train Model
```
python train.py
```
You will find trained model file in `model/checkpoints`

#### Eval Model  
Replace the model path in `eval.py` at first.
```
python eval.py
```
