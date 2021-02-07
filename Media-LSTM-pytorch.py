import time,re
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from tqdm.notebook import tqdm

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


device = torch.device('cuda:0')


NUM_EPOCHS = 400
LEARNING_RATE = 1e-3
BATCH_SIZE = 32
INPUTS_LEN = 64
MAX_EARLY_STOP = 3


df = pd.read_csv('merge-G.csv',sep = '\t')
# df["weight"] = pd.to_numeric(df["weight"])
df = df[:100]


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
df_train = df.copy()
df_test  = df_train.groupby('label').head(0)
df_train = df_train.drop(df_test.index)
df_val   = df_train.groupby('label').head(0)
df_train = df_train.drop(df_val.index)


# In[6]:


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
weight_field = Field(sequential=False, use_vocab=False)
# we preprocess on train so that tokens only in test and val
# will be labelled as "unknown"
preprocessed_text = df_train['text'].apply(
    lambda x: text_field.preprocess(x)
)


# In[ ]:


data_fields = [
    ('text', text_field),
    ('label', label_field), 
    ('weight', weight_field)
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


# In[8]:

import pdb
#pdb.set_trace()
# build the vocabulary using train and validation dataset and assign the vectors
text_field.build_vocab(preprocessed_text, max_size=100000, vectors='fasttext.simple.300d')
# build vocab for labels
label_field.build_vocab(trainds)
weight_field.build_vocab(trainds)
# get the vocab instance
vocab = text_field.vocab

print("unknown token: ", vocab.stoi['<unk>'])
print("padding token: ", vocab.stoi['<pad>'])


# In[9]:


model = Weighted_MediaLSTM(len(id2handle), 
                  vocab, 
                  att=True, 
                  padding_idx=vocab.stoi['<pad>'],
                  subword=False,
                  num_layers=2).to(device)


# In[10]:


def neg_sampling(pos_label,ds):
    tot_ex = len(trainds.examples)
    rand_idx = np.random.randint(tot_ex)
    while pos_label==ds.examples[rand_idx].label:
        rand_idx = np.random.randint(tot_ex)
    return ds.examples[rand_idx]

def neg_sampling_batch(pos_batch,ds,device):
    neg_examples = [neg_sampling(p.item(),ds) for p in pos_batch.label]
    neg_examples = sorted(neg_examples, key=lambda x: len(x.text), reverse=True)
    return Batch(neg_examples,ds,device)

def compute_recalls(ds,model,vocab):
    scores_5 = [];scores_10 = [];scores_20 = []
    for ex in ds:
        inputs  = [vocab.stoi[w] for w in ex.text][:INPUTS_LEN]
        inputs  = torch.LongTensor(inputs).unsqueeze(0).to(device)
        lengths = torch.LongTensor([min(len(ex.text),INPUTS_LEN)]).to(device)

        x = model.featurize((inputs,lengths))

        sc = (x * model.m_embeddings.weight).sum(1)

        _, indices_5 = sc.topk(5)
        scores_5 += [int(ex.label in indices_5)]
        _, indices_10 = sc.topk(10)
        scores_10 += [int(ex.label in indices_10)]
        _, indices_20 = sc.topk(20)
        scores_20 += [int(ex.label in indices_20)]
    
    return np.mean(scores_5),np.mean(scores_10),np.mean(scores_20)

optimizer = optim.Adam(model.parameters(), LEARNING_RATE)

best_r = 0.0
early_stop = 0
opti_epochs = 0
for epoch in range(NUM_EPOCHS):
    losses = []
    tic = time.time()
    flag = 0
    for batch_pos in tqdm(traindl):
        optimizer.zero_grad()
        
        batch_neg = neg_sampling_batch(batch_pos,trainds,device)
        
        out_pos = model(batch_pos.text,batch_pos.label, batch_pos.weight)
        out_neg = model(batch_neg.text,batch_pos.label, batch_pos.weight) # pos handle, neg text examples
        
        loss = -(out_pos - out_neg).sigmoid().log().sum()
        if torch.isnan(loss).item():
            flag = 1
            break
        loss.backward()
        #print(loss)
        optimizer.step()
        losses.append(loss.item())
    if flag == 1:
        break
    r5,r10,r20 = compute_recalls(valds,model,vocab)
    vals = (epoch,r5,r10,r20,np.mean(losses),time.time()-tic)
    print("[%d] recall@[5/10/20]:  %.4f %.4f %.4f loss: %.4f time[s]: %.1f" % vals)
    
    if True:#r10>=best_r:
        torch.save(model,"model_1m.pt")
        best_r = r10
        opti_epochs = epoch
#         early_stop  = 0 # reset counter
#     else:
#         early_stop += 1
#         if early_stop >= MAX_EARLY_STOP:
            
#             break

model = torch.load("model_1m.pt")
r5,r10,r20 = compute_recalls(testds,model,vocab)
print("final recall@[5/10/20]:  %.4f %.4f %.4f" % (r5,r10,r20))
print("in %d epochs" % opti_epochs)


# # Analysis

# In[50]:


# find clostest source
checked_names = ['teralytics','NetGuardians SA','Safe Swiss Cloud AG','United Security Providers AG',                  'Exeon Analytics AG','Agam Security SA','Cyber Resilience','Hacknowledge SA','Tebicom SA',                  'Quantum Integrity SA','OS Objectif Sécurité SA','InterHyve SARL','SIETEC Zaugg',                  'Sisa Green Systems SA','HACKNET SA','Bavitech Systems Sàrl','Backup ONE AG','Acronis AG',                  'Golden Frog GmbH']
for com in checked_names:
    print('\n\n',com,'\n')
    try:
        top_vals = (model.m_embeddings.weight[handle2id[com],:] * model.m_embeddings.weight).sum(1)
        top_ids = top_vals.argsort().tolist()

        for i in top_ids[::-1][:5]:
            print(top_vals[i].item(),id2handle[i])
    except Exception as e:
        print(e)
        continue
