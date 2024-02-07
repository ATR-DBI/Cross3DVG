import numpy as np
import torch
from torch import nn

class ScaledDotProductAttention(nn.Module):
    '''
    Scaled dot-product attention
    '''

    def __init__(self, d_model, d_k, d_v, h):
        '''
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        '''
        super(ScaledDotProductAttention, self).__init__()
        self.fc_q = nn.Linear(d_model, h * d_k)
        self.fc_k = nn.Linear(d_model, h * d_k)
        self.fc_v = nn.Linear(d_model, h * d_v)
        self.fc_o = nn.Linear(h * d_v, d_model)

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h

        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.fc_q.weight)
        nn.init.xavier_uniform_(self.fc_k.weight)
        nn.init.xavier_uniform_(self.fc_v.weight)
        nn.init.xavier_uniform_(self.fc_o.weight)
        nn.init.constant_(self.fc_q.bias, 0)
        nn.init.constant_(self.fc_k.bias, 0)
        nn.init.constant_(self.fc_v.bias, 0)
        nn.init.constant_(self.fc_o.bias, 0)

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None, way='mul'):
        '''
        Computes
        :param queries: Queries (b_s, nq, d_model)
        :param keys: Keys (b_s, nk, d_model)
        :param values: Values (b_s, nk, d_model)
        :param attention_mask: Mask over attention values (b_s, h, nq, nk). True indicates masking.
        :param attention_weights: Multiplicative weights for attention values (b_s, h, nq, nk).
        :return:
        '''
        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]
        #print(queries.shape, keys.shape, values.shape)
        q = self.fc_q(queries)
        #print(q.shape, b_s, nq, self.h, self.d_k)
        q = q.view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        k = self.fc_k(keys).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk)
        v = self.fc_v(values).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

        att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)
        if attention_weights is not None:
            if way == 'mul':
                att = att * attention_weights
            elif way == 'add':
                # print(att.shape, attention_weights.shape, '<< att shape; add')
                att = att + attention_weights
            else:
                raise NotImplementedError(way)
        if attention_mask is not None:
            att = att.masked_fill(attention_mask, -1e9) # -np.inf)
        att = torch.softmax(att, -1)
        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out = self.fc_o(out)  # (b_s, nq, d_model)
        return out

    def forward_faster(self, queries, keys, values, attention_pos, attention_weights, way='mul'):
        '''
        Computes
        :param queries: Queries (b_s, nq, d_model)
        :param keys: Keys (b_s, nk, d_model)
        :param values: Values (b_s, nk, d_model)
        :param attention_pos: Mask over attention values (b_s, nq, pk). True indicates masking indices. pk << nk_real
        :param attention_weights: Multiplicative weights for attention values (b_s, h, nq, pk).
        :return:
        '''
        b_s, nq, d_model = queries.shape
        nk = keys.shape[1]
        pk = attention_pos.shape[2]
        # print("attention_pos0", attention_pos.shape) #4 256 20
        #print(queries.shape, keys.shape, values.shape)
        #print(q.shape, b_s, nq, self.h, self.d_k)
        q = self.fc_q(queries).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)[: ,:, :, None, :]  # (b_s, h, nq, 1, d_k)

        attention_pos = attention_pos.view(b_s, nq*pk)

        k = self.fc_k(keys)
        v = self.fc_v(values)
        i_ind = torch.arange(b_s)[:, None].type_as(attention_pos) * nq
        attention_pos = attention_pos + i_ind.long()  # faster...
        # return queries
        # print("attention_pos", attention_pos.shape) #4, 5120
        # print("k", k.view(b_s*nq, -1).shape) #1024, 128
        # print("b_s, nq, pk, self.h, self.d_k", b_s, nq, pk, self.h, self.d_k) #4 256 20 4 32
        # print("k[attention_pos]", k.view(b_s*nq, -1)[attention_pos].shape)
        k = k.view(b_s*nq, -1)[attention_pos].view(b_s, nq, pk, self.h, self.d_k)  # (b_s, nq, pk, h, dk)
        v = v.view(b_s*nq, -1)[attention_pos].view(b_s, nq, pk, self.h, self.d_v)  # (b_s, nq, pk, h, dv)
        # return queries  # 26ms
        # i_ind = torch.arange(b_s)[:, None].type_as(attention_pos).repeat(1, nq*pk)
        # k = k[i_ind, attention_pos].view(b_s, nq, pk, self.h, self.d_k)  # (b_s, nq, pk, h, dk)
        # v = v[i_ind, attention_pos].view(b_s, nq, pk, self.h, self.d_v)  # (b_s, nq, pk, h, dv)

        k = k.permute(0, 3, 1, 4, 2)  # (b_s, h, nq, d_k, pk)
        v = v.permute(0, 3, 1, 2, 4)  # (b_s, h, nq, pk, d_v)

        att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, 1, p_k)
        if attention_weights is not None:
            attention_weights = attention_weights[:, :, :, None, :]
            if way == 'mul':
                att = att * attention_weights
            elif way == 'add':
                # print(att.shape, attention_weights.shape, '<< att shape; add')
                att = att + attention_weights
            else:
                raise NotImplementedError(way)
        att = torch.softmax(att, -1)  # softmax;
        out = torch.matmul(att, v)  # (b_s, h, nq, 1, d_v)
        out = out.permute(0, 2, 1, 4, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out = self.fc_o(out)  # (b_s, nq, d_model)
        return out

class MultiHeadAttention(nn.Module):
    '''
    Multi-head attention layer with Dropout and Layer Normalization.
    '''

    def __init__(self, d_model, d_k, d_v, h, dropout=.1, identity_map_reordering=False, can_be_stateful=False,
                 attention_module=None, attention_module_kwargs=None):
        super(MultiHeadAttention, self).__init__()
        self.identity_map_reordering = identity_map_reordering
        if attention_module is not None:
            if attention_module_kwargs is not None:
                self.attention = attention_module(d_model=d_model, d_k=d_k, d_v=d_v, h=h, **attention_module_kwargs)
            else:
                self.attention = attention_module(d_model=d_model, d_k=d_k, d_v=d_v, h=h, m = 20)
        else:
            self.attention = ScaledDotProductAttention(d_model=d_model, d_k=d_k, d_v=d_v, h=h)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model)

        self.can_be_stateful = can_be_stateful
        if self.can_be_stateful:
            self.register_state('running_keys', torch.zeros((0, d_model)))
            self.register_state('running_values', torch.zeros((0, d_model)))

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None, way='mul'):
        if self.can_be_stateful and self._is_stateful:
            self.running_keys = torch.cat([self.running_keys, keys], 1)
            keys = self.running_keys

            self.running_values = torch.cat([self.running_values, values], 1)
            values = self.running_values

        if self.identity_map_reordering:
            q_norm = self.layer_norm(queries)
            k_norm = self.layer_norm(keys)
            v_norm = self.layer_norm(values)
            out = self.attention(q_norm, k_norm, v_norm, attention_mask, attention_weights, way)
            out = queries + self.dropout(torch.relu(out))
        else:
            out = self.attention(queries, keys, values, attention_mask, attention_weights, way)
            out = self.dropout(out)
            out = self.layer_norm(queries + out)
        return out

    def forward_faster(self, queries, keys, values, attention_pos, attention_weights, way):
        if self.can_be_stateful and self._is_stateful:
            self.running_keys = torch.cat([self.running_keys, keys], 1)
            keys = self.running_keys

            self.running_values = torch.cat([self.running_values, values], 1)
            values = self.running_values

        if self.identity_map_reordering:
            q_norm = self.layer_norm(queries)
            k_norm = self.layer_norm(keys)
            v_norm = self.layer_norm(values)
            out = self.attention.forward_faster(q_norm, k_norm, v_norm, attention_pos, attention_weights, way)
            out = queries + self.dropout(torch.relu(out))
        else:
            out = self.attention.forward_faster(queries, keys, values, attention_pos, attention_weights, way)
            out = self.dropout(out)
            out = self.layer_norm(queries + out)
        return out


import math

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class NewGELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

class FeedFowardNetwork(nn.Module):
    def __init__(self, d_model, pdrop=0.1):
        super(FeedFowardNetwork, self).__init__()    
        self.mlp = nn.ModuleDict(dict(
            c_fc    = nn.Linear(d_model, 4 * d_model),
            c_proj  = nn.Linear(4 * d_model, d_model),
            act     = NewGELU(),
            dropout = nn.Dropout(pdrop),
        ))    
        m = self.mlp
        self.mlpf = lambda x: m.dropout(m.c_proj(m.act(m.c_fc(x)))) # MLP forward        
        
    def forward(self, x):
        x = self.mlpf(x)
        return x
    
class Block(nn.Module):
    def __init__(self, d_model, d_k, d_v, h, dropout=.1, identity_map_reordering=False, can_be_stateful=False,
                attention_module=None, attention_module_kwargs=None):
        super(Block, self).__init__()
        
        self.self_attn = MultiHeadAttention(d_model=d_model, d_k=d_k, d_v=d_v, h=h)
        self.cross_attn = MultiHeadAttention(d_model=d_model, d_k=d_k, d_v=d_v, h=h)
        self.ffn = FeedFowardNetwork(d_model, dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None, way='mul'):
        feature = self.self_attn(queries, queries, queries, attention_weights=attention_weights, way=way)
        feature = self.cross_attn(feature, keys, values, attention_mask)
        feature = feature + self.ffn(feature)
        feature = self.layer_norm(feature)
        return feature