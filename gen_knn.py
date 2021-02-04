#!/usr/bin/env python
# coding: utf-8

# In[1]:


import time,re
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from tqdm.notebook import tqdm
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

# adatpted from:
# https://medium.com/@sonicboom8/sentiment-analysis-torchtext-55fb57b1fab8


# In[2]:


NUM_EPOCHS = 400
LEARNING_RATE = 1e-3
BATCH_SIZE = 32
INPUTS_LEN = 64
MAX_EARLY_STOP = 3


# In[3]:


# cols = ['handle','language','title','text','favorite','reply','quote','retweet']
# df = pd.read_csv("/mnt/localdata/rappaz/mediaobs/articles_en_1m.csv", header=None, names=cols)


# In[4]:


df = pd.read_csv('merge_full.csv',sep = '\t')


# In[5]:


# filter sources with really low article counts
#ssz = df.groupby('handle').size()
#low_cnt_sources = ssz[ssz<80].index.tolist()
#df = df[~df.handle.isin(low_cnt_sources)]


# In[5]:


def proc_text(x):
    txt = ". ".join([x[0]] +  x[1].split('.')[:3])
    txt = txt.lower()
    txt = txt.replace("\n","")
    txt = re.sub(r'\([^)]*\)', '', txt)
    # removing ALL specials, sounds a bit extreme for now...
    #txt = re.sub('[^A-Za-z0-9]+', '', txt) 
    return txt

# Take only 3 sentences + the title and tokenize
# Re-use the same column, which is probably a bad practice 
# df["text"] = df[['text']].apply(proc_text, axis=1)

# convert handles to ids
df["label"] = pd.factorize(df['handle'])[0]

# create dictionary to retrieve handles later
id2handle = {k: v for k,v in df.drop_duplicates(subset=["label"])[['label','handle']].values}
handle2id = {v: k for k,v in id2handle.items()}

# Split test-train-val using leave-one-out

model = torch.load("model_1m.pt")
checked_names = ['teralytics','NetGuardians SA','Safe Swiss Cloud AG','United Security Providers AG',                  'Exeon Analytics AG','Agam Security SA','Cyber Resilience','Hacknowledge SA','Tebicom SA',                  'Quantum Integrity SA','OS Objectif Sécurité SA','InterHyve SARL','SIETEC Zaugg',                  'Sisa Green Systems SA','HACKNET SA','Bavitech Systems Sàrl','Backup ONE AG','Acronis AG',                  'Golden Frog GmbH']
d = {}
for com in id2handle.values():
    print('\n\n',com,'\n')
    try:
        d[com] = []
        top_vals = (model.m_embeddings.weight[handle2id[com],:] * model.m_embeddings.weight).sum(1)
        top_ids = top_vals.argsort().tolist()

        for i in top_ids[::-1][:40]:
            d[com].append(id2handle[i])
            #print(top_vals[i].item(),id2handle[i])
    except Exception as e:
        print(e)
        continue
import json
with open('top40_com.json','w',encoding = 'utf-8') as f:
    json.dump(d,f)
