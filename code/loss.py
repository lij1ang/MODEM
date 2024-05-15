import torch
import torch.nn as nn

class CompareLoss(nn.Module):
    def __init__(self, margin = 0.1):
        super(CompareLoss,self).__init__()
        self.margin = margin

    def forward(self, score_ll, score_ls, score_ss, score_sl, valid_lens):
        # score(b,1,max_session_num) ==> (b,max_session_num)
        score_ll, score_ls, score_ss, score_sl = score_ll.squeeze(), score_ls.squeeze(), score_ss.squeeze(), score_sl.squeeze()
        # print("\nloss_input: ",score_ll.shape,score_ls.shape,score_ss.shape,score_sl.shape,valid_lens.shape)
        
        # loss1-4 (b,max_session_num)
        loss1 = torch.exp(score_ll*score_ls) * torch.clamp(self.margin - torch.abs(score_ll-score_ls),min=0.0)
        loss2 = torch.exp(score_sl*score_ss) * torch.clamp(self.margin - torch.abs(score_sl-score_ss),min=0.0)
        loss3 = torch.exp(score_ls*score_ss) * torch.clamp(self.margin - torch.abs(score_ls-score_ss),min=0.0)
        loss4 = torch.exp(score_ll*score_sl) * torch.clamp(self.margin - torch.abs(score_ll-score_sl),min=0.0)

        # 根据len计算avg，len为（b,）
        # (b,max_session_num) loss进行mask
        loss = loss1 + loss2 +loss3 +loss4 
        batch_size, max_length = loss.shape
        # print("\nloss1-4.shape: ",loss1.shape,loss2.shape,loss3.shape,loss4.shape)
        mask = torch.arange(max_length,device= valid_lens.device).expand(batch_size, max_length) < valid_lens.unsqueeze(1)
        loss = loss * mask.float()
        # (b,)
        loss = torch.sum(loss,dim=1) / torch.clamp(valid_lens,min=1)
        # return a mean value
        return loss.mean()
    
class InfoNCE(torch.nn.Module):
     
    def __init__(self, tau = 1):
        super(InfoNCE,self).__init__()
        self.tau = tau
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, scores):
        # (b,num_neg+1),[:,0]位置为positive
        scores = scores / self.tau
        scores = self.softmax(scores)
        loss = -torch.log(scores[:,0])
        return loss.mean()

class BPRLoss(nn.Module):
    """ BPRLoss, based on Bayesian Personalized Ranking
    Args:
        - gamma(float): Small value to avoid division by zero
    Shape:
        - Pos_score: (N)
        - Neg_score: (N), same shape as the Pos_score
        - Output: scalar.
    Examples::
        >>> loss = BPRLoss()
        >>> pos_score = torch.randn(3, requires_grad=True)
        >>> neg_score = torch.randn(3, requires_grad=True)
        >>> output = loss(pos_score, neg_score)
        >>> output.backward()
    """

    def __init__(self, gamma=1e-10):
        super(BPRLoss, self).__init__()
        self.gamma = gamma

    def forward(self, pos_score, neg_score):
        loss = -torch.log(self.gamma + torch.sigmoid(pos_score - neg_score)).mean()
        return loss

class RegLoss(nn.Module):
    """ RegLoss, L2 regularization on model parameters
    """

    def __init__(self):
        super(RegLoss, self).__init__()

    def forward(self, parameters):
        reg_loss = None
        for W in parameters:
            if reg_loss is None:
                reg_loss = W.norm(2)
            else:
                reg_loss = reg_loss + W.norm(2)
        return reg_loss

class TopKLoss(nn.Module):
    def __init__(self, gamma=1e-10):
        super(TopKLoss, self).__init__()
        self.gamma = gamma
    
    def forward(self, pred_s, pred_t, k, list_len):
        pred_s = pred_s.reshape([-1, list_len])
        pred_t = pred_t.reshape([-1, list_len])
        
        idx = torch.argsort(pred_t, dim=1, descending=True)
        sorted_pred_s = torch.gather(pred_s, dim=1, index=idx)
        topk_scores = torch.mean(sorted_pred_s[:,:k], dim=1)
        no_topk_scores = torch.mean(sorted_pred_s[:,k:], dim=1)
        loss = -torch.log(self.gamma + torch.sigmoid(topk_scores - no_topk_scores)).mean()
        return loss

class SoftmaxLoss(nn.Module):
    def __init__(self):
        super(SoftmaxLoss, self).__init__()
        self.bce_loss_func = torch.nn.BCELoss()
        self.softmax_func = torch.nn.Softmax(dim=1)
    
    def forward(self, pred, label, batch_neg_size):
        pred = torch.transpose(pred.reshape((batch_neg_size + 1, -1)), 0, 1)
        label = torch.transpose(label.reshape((batch_neg_size + 1, -1)), 0, 1)
        pred_softmax = self.softmax_func(pred)
        loss = self.bce_loss_func(pred_softmax, label)
        return loss

class ICCLoss(nn.Module):
    def __init__(self):
        super(ICCLoss, self).__init__()
        self.bce_loss_func = torch.nn.BCELoss()
    
    def forward(self, model_preds, label):
        indicators = None
        for i in range(model_preds.shape[1]):
            k = torch.sigmoid(torch.tensor([i])).to(model_preds.device)
            indicator = torch.sigmoid(model_preds[:,i] - k).unsqueeze(1)
            indicators = indicator if indicators == None else torch.cat((indicators, indicator), dim=1)
        weights = torch.cumprod(torch.cat((torch.ones((indicators.shape[0],1)).to(indicators.device), indicators), dim=1), dim=1)[:,:-1]
        weights = (1 - weights) * weights
        score = torch.sum(model_preds * weights, dim=1)
        loss = self.bce_loss_func(score, label)
        return loss