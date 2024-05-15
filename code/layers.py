import numpy as np
import torch
import torch.nn.functional as F
import math

class FeaturesEmbedding(torch.nn.Module):
    def __init__(self, vocabulary_size, embed_dim):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocabulary_size, embed_dim)
#         torch.nn.init.xavier_uniform_(self.embedding.weight.data)
        # maxval = np.sqrt(6. / np.sum(embed_dim))
        # minval = -maxval
        # torch.nn.init.uniform_(self.embedding.weight, minval, maxval)
    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        return self.embedding(x)


class Linear(torch.nn.Module):
    def __init__(self, vocabulary_size, output_dim=1):
        super().__init__()
        self.fc = torch.nn.Embedding(vocabulary_size, output_dim)
        self.bias = torch.nn.Parameter(torch.zeros((output_dim,)))

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        return torch.sum(self.fc(x), dim=1) + self.bias


class FactorizationLayer(torch.nn.Module):
    def __init__(self, reduce_sum=True):
        super().__init__()
        self.reduce_sum = reduce_sum

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
        """
        square_of_sum = torch.sum(x, dim=1) ** 2
        sum_of_square = torch.sum(x ** 2, dim=1)
        ix = square_of_sum - sum_of_square
        if self.reduce_sum:
            ix = torch.sum(ix, dim=1, keepdim=True)
        return 0.5 * ix #[B, 1]

class FieldAwareFactorizationLayer(torch.nn.Module):

    def __init__(self, vocabulary_size, num_fields, embed_dim):
        super().__init__()
        self.num_fields = num_fields   
        self.embeddings = torch.nn.ModuleList([
            torch.nn.Embedding(vocabulary_size, embed_dim) for _ in range(self.num_fields)
        ]) 
        for embedding in self.embeddings:
            torch.nn.init.xavier_uniform_(embedding.weight.data)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        xs = [self.embeddings[i](x) for i in range(self.num_fields)]  
        ix = list()
        for i in range(self.num_fields - 1):
            for j in range(i + 1, self.num_fields):
                ix.append(xs[j][:, i] * xs[i][:, j])
        ix = torch.stack(ix, dim=1)
        return torch.sum(torch.sum(ix,dim=1),dim=1, keepdim=True)

class MultiLayerPerceptron(torch.nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout, output_layer=True, act = 'relu'):
        super().__init__()
        layers = list()
        for hidden_dim in hidden_dims:
            layers.append(torch.nn.Dropout(p=dropout))
            layers.append(torch.nn.Linear(input_dim, hidden_dim))
            layers.append(torch.nn.BatchNorm1d(hidden_dim))
            if act == 'relu':
                layers.append(torch.nn.ReLU())
            elif act == 'tanh':
                layers.append(torch.nn.Tanh())
            input_dim = hidden_dim
        if output_layer:
            layers.append(torch.nn.Linear(input_dim, 1))
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, input_dim)``
        """
        return self.mlp(x)


class InnerProductNetwork(torch.nn.Module):
    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
        """
        num_fields = x.shape[1]
        row, col = list(), list()
        for i in range(num_fields - 1):
            for j in range(i + 1, num_fields):
                row.append(i), col.append(j)
        # return size: (batch_size, num_fields*(num_fields-1)/2)
        return torch.sum(x[:, row] * x[:, col], dim=2)

class CrossNet(torch.nn.Module):
    def __init__(self, input_dim, num_layers):
        super(CrossNet, self).__init__()
        self.num_layers = num_layers
        self.cross_net = torch.nn.ModuleList(CrossInteractionLayer(input_dim)
                                       for _ in range(self.num_layers))

    def forward(self, X_0):
        X_i = X_0 # b x dim
        for i in range(self.num_layers):
            X_i = X_i + self.cross_net[i](X_0, X_i)
        return X_i


class CrossInteractionLayer(torch.nn.Module):
    def __init__(self, input_dim):
        super(CrossInteractionLayer, self).__init__()
        self.weight = torch.nn.Linear(input_dim, 1, bias=False)
        self.bias = torch.nn.Parameter(torch.zeros(input_dim))

    def forward(self, X_0, X_i):
        interaction_out = self.weight(X_i) * X_0 + self.bias
        return interaction_out


class HistAtt(torch.nn.Module):
    def __init__(self, q_dim):
        super().__init__()
        self.null_attention = -2 ** 10
        input_dim = 4 * q_dim # [q, k, q * k, q - k]
        layers = list()
        for hidden_dim in [200, 80]:
            layers.append(torch.nn.Linear(input_dim, hidden_dim))
            layers.append(torch.nn.Dropout(p=0.2))
            layers.append(torch.nn.ReLU())
            input_dim = hidden_dim
        layers.append(torch.nn.Linear(input_dim, 1))
        self.atten_net = torch.nn.Sequential(*layers)
    
    def forward(self, x_item, user_hist, hist_len):
        _, len, dim = user_hist.shape # batch_size , padded_length, item_num_field*embed_dim
        # x_item_tile = torch.tile(x_item.reshape([-1, 1, dim]), [1, len, 1])
        x_item_tile = x_item.reshape([-1, 1, dim]).repeat((1, len, 1))
        attention_inp = torch.cat((x_item_tile, user_hist, x_item_tile * user_hist, x_item_tile - user_hist), dim=2)
        score = self.atten_net(attention_inp)
        mask = torch.arange(user_hist.shape[1])[None, :].to(hist_len.device) < hist_len[:, None]
        score[~mask] = self.null_attention

        atten_score = torch.nn.Softmax(dim = 1)(score)
        user_hist_rep = torch.sum(user_hist * atten_score, dim=1)

        return user_hist_rep, score.squeeze()

class HistAtt_S(torch.nn.Module):
    def __init__(self, q_dim):
        super().__init__()
        self.null_attention = -2 ** 22
        input_dim = q_dim
        layers = list()
        for hidden_dim in [200, 80]:
            layers.append(torch.nn.Linear(input_dim, hidden_dim))
            layers.append(torch.nn.Dropout(p=0.2))
            layers.append(torch.nn.ReLU())
            input_dim = hidden_dim
        layers.append(torch.nn.Linear(input_dim, 1))
        self.atten_net = torch.nn.Sequential(*layers)
    
    def forward(self, x_item, user_hist, hist_len):
        _, len, dim = user_hist.shape
        # x_item_tile = torch.tile(x_item.reshape([-1, 1, dim]), [1, len, 1])
        x_item_tile = x_item.reshape([-1, 1, dim]).repeat((1, len, 1))
#       attention_inp = torch.cat((x_item_tile, user_hist, x_item_tile * user_hist, x_item_tile - user_hist), dim=2)
        score = torch.sum(x_item_tile * user_hist, dim=2, keepdim=True)
        mask = torch.arange(user_hist.shape[1])[None, :].to(hist_len.device) < hist_len[:, None]
        score[~mask] = self.null_attention

        atten_score = torch.nn.Softmax(dim = 1)(score)
        user_hist_rep = torch.sum(user_hist * atten_score, dim=1)

        return user_hist_rep, score.squeeze()

class CoAtt(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.null_attention = -2 ** 22
        layers = list()
        input_dim = 3*dim
        for hidden_dim in [200, 80]:
            layers.append(torch.nn.Linear(input_dim, hidden_dim))
            layers.append(torch.nn.Dropout(p=0.2))
            layers.append(torch.nn.ReLU())
            input_dim = hidden_dim
        layers.append(torch.nn.Linear(input_dim, 1))
        self.atten_net = torch.nn.Sequential(*layers)
        self.fc1 = torch.nn.Linear(2*dim,dim)
        
    def forward(self, item_emb,x_session,session_len, user_hist, hist_len):

        batch_size,s_len,dim = x_session.shape
        h_len = user_hist.shape[1]

        # user_hist_tile = torch.tile(user_hist.reshape([-1,1,h_len,dim]),[1,s_len,1,1])
        # x_session_tile = torch.tile(x_session.reshape([-1,s_len,1,dim]),[1,1,h_len,1])
        # item_emb_tile = torch.tile(item_emb.reshape([-1,1,1,dim]),[1,s_len,h_len,1])
        user_hist_tile = user_hist.reshape([-1,1,h_len,dim]).repeat((1,s_len,1,1))
        x_session_tile = x_session.reshape([-1,s_len,1,dim]).repeat((1,1,h_len,1))
        item_emb_tile = item_emb.reshape([-1,1,1,dim]).repeat((1,s_len,h_len,1))

        query = self.fc1(torch.cat([item_emb_tile,x_session_tile],dim=3))

        att_score = torch.sum(user_hist_tile*query,dim=3)

        # inp = torch.cat([x_session_tile,user_hist_tile,item_emb_tile],dim=3)

        # att_score = self.atten_net(inp).squeeze()

        mask_session = torch.arange(x_session.shape[1])[None, :].to(hist_len.device) < session_len[:, None]
        # mask_session = torch.tile(mask_session.reshape(-1,s_len,1),[1,1,h_len])
        mask_session = mask_session.reshape(-1,s_len,1).repeat((1,1,h_len))
        mask_hist =torch.arange(user_hist.shape[1])[None, :].to(hist_len.device) < hist_len[:, None]
        # mask_hist = torch.tile(mask_hist.reshape(-1,1,h_len),[1,s_len,1])
        mask_hist = mask_hist.reshape(-1,1,h_len).repeat((1,s_len,1))

        att_score[~mask_session] = self.null_attention
        att_score[~mask_hist] = self.null_attention

        #先softmax再mean
        # att_score = (torch.nn.Softmax(dim=1)(att_score.reshape(-1,s_len*h_len))).reshape(-1,s_len,h_len)
        # att_score = torch.sum(att_score,dim=1)
        
        #先max再softmax
        score = torch.max(att_score.squeeze(),dim=1)[0]
        att_score = torch.nn.Softmax(dim=1)(score)

        user_hist_rep = torch.sum((user_hist * (att_score.unsqueeze(dim=2))), dim=1)
        
        return user_hist_rep, score

def sequence_mask(X, valid_len, value=0): 
    maxlen = X.size(1)
    mask = torch.arange((maxlen), dtype=torch.float32,device=X.device)[None, :] < valid_len[:, None]
    X[~mask] = value
    return X

# X.shape (batch_size,1(q_num),h_len(kv_num)) valid_lens (batch_size,)
def masked_softmax(X, valid_lens, mask_ls=None):
    if valid_lens is None:
        return torch.nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        X = sequence_mask(X.reshape(-1, shape[-1]), valid_lens,value= -2 ** 22)
        # valid_lens (batch_size,)
        # X (batch_size*1,h_len)
        if mask_ls is not None:
            X[mask_ls] = -2 ** 22
        return torch.nn.functional.softmax(X.reshape(shape), dim=-1)


class DotProductAttention(torch.nn.Module): 
    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens=None, mask_ls=None, return_unsoftmax=False):
        d = queries.shape[-1]
        scores = torch.bmm(queries, keys.transpose(1,2)) / math.sqrt(d) 
        self.attention_weights = masked_softmax(scores, valid_lens, mask_ls) 
        if return_unsoftmax:
            return torch.bmm(self.dropout(self.attention_weights), values), scores
        return torch.bmm(self.dropout(self.attention_weights), values), self.attention_weights

def transpose_qkv(X, num_heads): 
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)
    X = X.permute(0, 2, 1, 3)
    return X.reshape(-1, X.shape[2], X.shape[3])

def transpose_output(X, num_heads):
    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2]) 
    X = X.permute(0, 2, 1, 3)
    return X.reshape(X.shape[0], X.shape[1], -1)

class MultiheadAtt(torch.nn.Module):
    def __init__(self,dim,dropout=0.1,num_heads=4,bias=False,**kargs):
        super().__init__()
        self.num_heads=num_heads
        self.dim=dim
        self.attention = DotProductAttention(dropout)
        self.W_q=torch.nn.Linear(dim, dim, bias=bias)
        self.W_k=torch.nn.Linear(dim, dim, bias=bias)
        self.W_v=torch.nn.Linear(dim, dim, bias=bias)
        self.W_o=torch.nn.Linear(dim, dim, bias=bias)

    # 当前session表示，长度为1，(batch_size,1,dim)，历史session表示，session长度(batch_size,h_len,dim)
    def forward(self, x_session, hist_session, hist_session_len,mask_ls=None,return_unsoftmax=False):
        # h_len = hist_session.shape[1]
        x_session = x_session.unsqueeze(1)
        queries = transpose_qkv(self.W_q(x_session), self.num_heads)
        keys = transpose_qkv(self.W_k(hist_session), self.num_heads)
        values = transpose_qkv(self.W_v(hist_session), self.num_heads)
        if hist_session_len is not None:
            hist_session_len = torch.repeat_interleave(hist_session_len,repeats=self.num_heads,dim=0)
        # 注意针对多头处理mask,待确认
        if mask_ls is not None:
            mask_ls = mask_ls.repeat(self.num_heads, 1)
            # print(mask_ls.shape)
        output,score = self.attention(queries, keys, values, hist_session_len, mask_ls, return_unsoftmax)
        output_concat = transpose_output(output, self.num_heads) 
        # print("score.shape: ",score.shape)
        return self.W_o(output_concat), score


