import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as pad
from torch.nn.utils.rnn import PackedSequence
from torch.autograd import Variable


class TextModel(nn.Module):

    def __init__(self, num_features, num_classes,
                 num_embeddings, embedding_dim, embedding_dropout=0,
                 padding_idx=None, max_norm=None, norm_type=2, scale_grad_by_freq=False,
                 subword=False, fine_tuning=False):

        super(TextModel, self).__init__()

        if subword:
            self.embeddings = nn.EmbeddingBag(num_embeddings, embedding_dim,
                                              max_norm=max_norm, norm_type=norm_type,
                                              scale_grad_by_freq=scale_grad_by_freq, mode='mean')
        else:
            self.embeddings = nn.Embedding(num_embeddings, embedding_dim,
                                           max_norm=max_norm, norm_type=norm_type,
                                           scale_grad_by_freq=scale_grad_by_freq)

        # enables embeddings fine-tuning
        self.embeddings.weight.requires_grad = fine_tuning

        self.embedding_dim = embedding_dim
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq=scale_grad_by_freq
        self.embedding_dropout = embedding_dropout
        self.subword = subword
        self.num_features = num_features
        self.num_classes = num_classes
        self.fc = nn.Linear(num_features, num_classes)

    def embedding_weights(self):
        return self.embeddings.weight.data

    def classify(self, x):
        return self.fc(x)

    def word_vectors(self, inputs):
        if self.subword:
            x = [self.embeddings.forward(i, o).unsqueeze(0) for i, o in inputs]
            nelt = [t.nelement() for t in x]
            if not all(z == x[0].nelement() for z in nelt):
                import pdb
                pdb.set_trace()
            x = torch.cat(x, 0)
        else:
            x = self.embeddings(inputs)
        x = F.dropout(x, p=self.embedding_dropout, training=self.training)
        return x

    def forward(self, x):
        x = self.featurize(x)
        logit = self.classify(x)
        return logit
    
    
class SelfAttention(nn.Module):
    def __init__(self, hidden_size, batch_first=False):
        super(SelfAttention, self).__init__()

        self.hidden_size = hidden_size
        self.batch_first = batch_first

        self.att_weights = nn.Parameter(torch.Tensor(1, hidden_size),
                                        requires_grad=True)

        nn.init.xavier_uniform_(self.att_weights.data)

    def get_mask(self):
        pass

    def forward(self, inputs):

        if isinstance(inputs, PackedSequence):
            # unpack output
            inputs, lengths = pad(inputs, batch_first=self.batch_first)
        if self.batch_first:
            batch_size, max_len = inputs.size()[:2]
        else:
            max_len, batch_size = inputs.size()[:2]
            inputs = inputs.permute(1, 0, 2)

        # att = torch.mul(inputs, self.att_weights.expand_as(inputs))
        # att = att.sum(-1)
        weights = torch.bmm(inputs,
                            self.att_weights  # (1, hidden_size)
                            .permute(1, 0)  # (hidden_size, 1)
                            .unsqueeze(0)  # (1, hidden_size, 1)
                            # (batch_size, hidden_size, 1)
                            .repeat(batch_size, 1, 1)
                            )

        attentions = F.softmax(F.relu(weights.squeeze()))

        # apply weights
#         import pdb
#         pdb.set_trace()
        attentions = F.softmax(F.relu(weights))

        # apply weights
        weighted = torch.mul(
            inputs, attentions.expand_as(inputs))

        # get the final fixed vector representations of the sentences
        representations = weighted.sum(1).squeeze()
        
# #         inputs = inputs.squeeze()
#         temp = len(inputs.shape) - len(attentions.shape)
#         att = attentions.clone()
#         for i in range(temp):
#             att = att.unsqueeze(-1)
            
            
#         try:
#             weighted = torch.mul(
#                 inputs, att.expand_as(inputs))
#         except Exception as e:
#             print(e)
#             return inputs.sum(1).squeeze(), attentions

#         # get the final fixed vector representations of the sentences
#         representations = weighted.sum(1).squeeze()

        return representations, attentions
    
class MediaLSTM(TextModel):

    def __init__(self,
                 num_classes, vocab, att=False, we_dropout=0, dropout=0,
                 we_dim=50, num_features=100, bidirectional=False, num_layers=1,
                 max_norm=None, norm_type=2, scale_grad_by_freq=True, fine_tuning=False, **kwargs):

        self.att = att
        D = we_dim
        C = num_classes
        Co = num_features
        hidden_dim = Co // 2 if bidirectional else Co
        H = Co

        super(MediaLSTM, self).__init__(H, C, len(vocab), we_dim,
                                       embedding_dropout=we_dropout,
                                       padding_idx=0,
                                       max_norm=max_norm, norm_type=norm_type,
                                       scale_grad_by_freq=scale_grad_by_freq,
                                       subword=False, fine_tuning=fine_tuning)

        self.lstm = nn.LSTM(
            D, hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional
        )
        if self.att:
            self.att_layer = SelfAttention(Co, batch_first=True)
            
        # Media embedding
        self.m_embeddings = nn.Embedding(num_classes, self.num_features,
                                         max_norm=self.max_norm, norm_type=self.norm_type,
                                         scale_grad_by_freq=self.scale_grad_by_freq)

    def forward(self, ex, media):
        embs = self.m_embeddings(media)
        x = self.featurize(ex)
        return (x * embs).sum(dim=1)
        
    def featurize(self, x):
        import pdb
        #pdb.set_trace()
        wrd_ix = x[0]
        lengths = x[1]
        x = self.word_vectors(wrd_ix)
        x = pack(x, lengths.tolist(), batch_first=True)
        lstm_out, (hidden_state, cell_state) = self.lstm(x)
        if self.att:
            x, self.attentions = self.att_layer(lstm_out)
        else:
            output, _ = pad(lstm_out, batch_first=True)
            # get the last time step for each sequence
            idx = (lengths - 1).view(-1, 1).expand(output.size(0),
                                                   output.size(2)).unsqueeze(1)
            x = output.gather(1, Variable(idx)).squeeze(1)
        return x
