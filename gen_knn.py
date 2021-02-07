#!/usr/bin/env python
# coding: utf-8

# In[1]:


import time,re
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from tqdm import tqdm
from IPython.display import display

import torch
import torch.optim as optim

from torchtext import data
from torchtext.data import Dataset, Example
from torchtext.data import Field
from torchtext.data import BucketIterator
from torchtext.data.batch import Batch

from sklearn.manifold import TSNE
import plotly.express as px

from lstm import *
from utils import *

#get_ipython().run_line_magic('matplotlib', 'inline')

device = torch.device('cuda:0')

BATCH_SIZE = 32
INPUTS_LEN = 64
MAX_EARLY_STOP = 3

df = pd.read_csv('merge-G.csv',sep = '\t')

def proc_text(x):
    txt = ". ".join([x[0]] +  x[1].split('.')[:3])
    txt = txt.lower()
    txt = txt.replace("\n","")
    txt = re.sub(r'\([^)]*\)', '', txt)
    # removing ALL specials, sounds a bit extreme for now...
    #txt = re.sub('[^A-Za-z0-9]+', '', txt) 
    return txt

df["label"] = pd.factorize(df['handle'])[0]

# create dictionary to retrieve handles later
id2handle = {k: v for k,v in df.drop_duplicates(subset=["label"])[['label','handle']].values}
handle2id = {v: k for k,v in id2handle.items()}

# Split test-train-val using leave-one-out
df_train = df.copy()
df_test  = df_train.groupby('label').head(0)
df_train = df_train.drop(df_test.index)
df_val   = df_train.groupby('label').head(0)
df_train = df_train.drop(df_val.index)


text_field = Field(
    sequential=True,
    tokenize='spacy', 
    fix_length=INPUTS_LEN,
    lower=True,
    use_vocab=True,
    include_lengths=True, 
    batch_first=True
)
label_field = Field(sequential=False, use_vocab=False)

# we preprocess on train so that tokens only in test and val
# will be labelled as "unknown"
preprocessed_text = df_train['text'].apply(
    lambda x: text_field.preprocess(x)
)


# In[ ]:


data_fields = [
    ('text', text_field),
    ('label', label_field), 
]

trainds = DataFrameDataset(df_train,data_fields)
testds  = DataFrameDataset(df_test,data_fields)
valds   = DataFrameDataset(df_val,data_fields)

traindl, testdl, valdl = data.BucketIterator.splits(datasets=(trainds, testds, valds), # specify train and validation Tabulardataset
                                                    batch_sizes=[BATCH_SIZE]*3,  # batch size of train and validation
                                                    sort_key=lambda x: len(x.text), # on what attribute the text should be sorted
                                                    device=device, # -1 mean cpu and 0 or None mean gpu
                                                    sort_within_batch=True, 
                                                    repeat=False,
                                                    shuffle=True)


# build the vocabulary using train and validation dataset and assign the vectors
text_field.build_vocab(preprocessed_text, max_size=100000, vectors='fasttext.simple.300d')
# build vocab for labels
label_field.build_vocab(trainds)


model = torch.load("model_1m.pt")
# checked_names = ['teralytics','NetGuardians SA','Safe Swiss Cloud AG','United Security Providers AG',                  'Exeon Analytics AG','Agam Security SA','Cyber Resilience','Hacknowledge SA','Tebicom SA',                  'Quantum Integrity SA','OS Objectif Sécurité SA','InterHyve SARL','SIETEC Zaugg',                  'Sisa Green Systems SA','HACKNET SA','Bavitech Systems Sàrl','Backup ONE AG','Acronis AG',                  'Golden Frog GmbH']
com_com = {}
for com in tqdm(id2handle.values()):
    try:
        com_com[com] = []
        top_vals = (model.m_embeddings.weight[handle2id[com],:] * model.m_embeddings.weight).sum(1)
        top_ids = top_vals.argsort().tolist()

        for i in top_ids[::-1][:40]:
            com_com[com].append(id2handle[i])
            #print(top_vals[i].item(),id2handle[i])
    except Exception as e:
        print(e)
        continue

# com_tech:
set_tech = list(set(df['text'].tolist()))
vecs = []
lens = []
for i in set_tech:
    indice = text_field.vocab.lookup_indices(i.split())
    lens.append(len(indice))
    indice.extend([1]*(64 - len(indice)))
    vecs.append(indice)
lens, vecs = zip(*sorted(zip(lens, vecs), reverse = True))
vecs = torch.LongTensor(vecs).cuda()
lens = torch.LongTensor(lens).cuda()
feature_emb = model.featurize((vecs,lens))

com_tech = {}
for com in tqdm(id2handle.values()):
    try:
        com_tech[com] = []
        top_vals = (model.m_embeddings.weight[handle2id[com],:] * feature_emb).sum(1)
        top_ids = top_vals.argsort().tolist()

        for i in top_ids[::-1][:40]:
            com_tech[com].append(set_tech[i])
            #print(top_vals[i].item(),id2handle[i])
    except Exception as e:
        print(e)
        continue

import json
with open('top40_com.json','w',encoding = 'utf-8') as f:
    json.dump(com_com,f)

with open('top40_tech.json','w',encoding = 'utf-8') as f:
    json.dump(com_tech,f)

