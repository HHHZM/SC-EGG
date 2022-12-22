import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.autograd as autograd
from models.utils import WeightedL1


class MinMaxScaler_Torch(nn.Module):
    def __init__(self, scale_min, scale_max):
        super(MinMaxScaler_Torch, self).__init__()  
        self.scale_min = nn.Parameter(scale_min, requires_grad=False)
        self.scale_max = nn.Parameter(scale_max, requires_grad=False)
    def forward(self, input):
        return (input - self.scale_min) / (self.scale_max - self.scale_min)


class DAZLE(nn.Module):  
    def __init__(self, dim_f, dim_v, init_w2v_att, att, normalize_att,
                 seenclass, unseenclass, lambda_1, lambda_2, lambda_3, device,
                 trainable_w2v=False, normalize_V=False, normalize_F=False, is_conservative=False,
                 prob_prune=0, desired_mass=-1, uniform_att_1=False, uniform_att_2=False, 
                 is_conv=False, is_bias=False, bias=1, non_linear_act=False,
                 loss_type='CE', non_linear_emb=False, is_sigmoid=False, margin=1,
                 att_compose_type='avg', att_compose_norm=False, #Scaler=None,
                 lambda_localCE=1., lambda_globalCE=1.):
        super(DAZLE, self).__init__()  
        self.dim_f = dim_f  
        self.dim_v = dim_v  
        self.dim_att = att.shape[1]  
        self.nclass = att.shape[0]  
        self.hidden = self.dim_att//2
        self.init_w2v_att = init_w2v_att
        self.non_linear_act = non_linear_act
        self.loss_type = loss_type
        self.device = device

        self.att_compose_norm = att_compose_norm
        self.att_compose_type = att_compose_type
        self.lambda_localCE = lambda_localCE
        self.lambda_globalCE = lambda_globalCE
        self.Scaler = MinMaxScaler_Torch(torch.zeros(dim_f), torch.ones(dim_f))

        if self.att_compose_type == 'mlp':
            self.mlp_compose = nn.Sequential(nn.Linear(self.dim_att, self.hidden),
                                            nn.LeakyReLU(),
                                            #  nn.ReLU(),
                                            nn.Linear(self.hidden, 1),
                                            #  nn.LeakyReLU()
                                            #  nn.Tanh()
                                            )
        self.compose_cls = nn.Sequential(
            nn.Linear(self.dim_f, self.nclass, bias=True),
            # nn.Linear(self.dim_f, self.nclass),
            # nn.LeakyReLU(),
            # nn.Linear(self.nclass, self.nclass)
        )
        
        if is_conv:
            r_dim = dim_f//2
            self.conv1 = nn.Conv2d(dim_f, r_dim, 2) #[2x2] kernel with same input and output dims
            print('***Reduce dim {} -> {}***'.format(self.dim_f,r_dim))
            self.dim_f = r_dim
            self.conv1_bn = nn.BatchNorm2d(self.dim_f)
            
        if init_w2v_att is None:  
            self.V = nn.Parameter(nn.init.normal_(
                torch.empty(self.dim_att,self.dim_v)).to(device))  
        else:
            self.init_w2v_att = F.normalize(torch.tensor(init_w2v_att))
            if trainable_w2v:
                self.V = nn.Parameter(self.init_w2v_att.clone().to(device))
            else:
                self.V = self.init_w2v_att.clone().to(device)
        
        self.att = F.normalize(torch.tensor(att)).to(device)
        
        self.W_1 = nn.Parameter(nn.init.normal_(torch.empty(self.dim_v,self.dim_f)).to(device))
        self.W_2 = nn.Parameter(nn.init.zeros_(torch.empty(self.dim_v,self.dim_f)).to(device))
        self.W_3 = nn.Parameter(nn.init.zeros_(torch.empty(self.dim_v,self.dim_f)).to(device))
        
        ## Compute the similarity between classes  
        self.P = torch.mm(self.att,torch.transpose(self.att,1,0))  
        assert self.P.size(1)==self.P.size(0) and self.P.size(0)==self.nclass  
        self.weight_ce = torch.eye(self.nclass).float().to(device)

        self.normalize_V = normalize_V  
        self.normalize_F = normalize_F   
        self.is_conservative = is_conservative  
        self.is_conv = is_conv
        self.is_bias = is_bias
        
        self.seenclass = seenclass  
        self.unseenclass = unseenclass  
        self.normalize_att = normalize_att   
        
        if is_bias:
            self.bias = torch.tensor(bias).to(device)
            mask_bias = np.ones((1,self.nclass))
            mask_bias[:,self.seenclass.cpu().numpy()] *= -1
            self.mask_bias = torch.tensor(mask_bias).float().to(device)
        
        margin_CE = np.ones((1,self.nclass))
        margin_CE[:,self.seenclass.cpu().numpy()] = margin 
        margin_CE[:,self.unseenclass.cpu().numpy()] = - margin 
        self.margin_CE = torch.tensor(margin_CE).float().to(device)
        
        if desired_mass == -1:  
            self.desired_mass = self.unseenclass.size(0)/self.nclass
        else:  
            self.desired_mass = desired_mass
        self.prob_prune = torch.tensor(prob_prune).to(device)
         
        self.lambda_1 = lambda_1  
        self.lambda_2 = lambda_2  
        self.lambda_3 = lambda_3  
        self.loss_att_func = nn.BCEWithLogitsLoss()
        self.log_softmax_func = nn.LogSoftmax(dim=1)  
        self.uniform_att_1 = uniform_att_1
        self.uniform_att_2 = uniform_att_2
        
        self.non_linear_emb = non_linear_emb
        
        if self.non_linear_emb:
            print('non_linear embedding')
            self.emb_func = torch.nn.Sequential(
                torch.nn.Linear(self.dim_att, self.dim_att//2),
                torch.nn.ReLU(),
                torch.nn.Linear(self.dim_att//2, 1))
        
        self.is_sigmoid = is_sigmoid           
          
    def compute_V(self):
        if self.normalize_V:  
            V_n = F.normalize(self.V)
        else:  
            V_n = self.V  
        return V_n
    
    def compute_aug_cross_entropy(self,in_package,is_conservative = None, override_bias = False):  
        batch_label = in_package['batch_label'] 
        
        if override_bias:
            is_bias = False
        else:
            is_bias = self.is_bias
        
        if is_conservative is None:
            is_conservative = self.is_conservative
        
        if len(in_package['batch_label'].size()) == 1:
            batch_label = self.weight_ce[batch_label]  
        
        S_pp = in_package['S_pp']  
        
        Labels = batch_label
            
        if is_bias:
            S_pp = S_pp - self.vec_bias
        
        if not is_conservative:  
            S_pp = S_pp[:,self.seenclass]  
            Labels = Labels[:,self.seenclass]  
            assert S_pp.size(1) == len(self.seenclass)  
        
        Prob = self.log_softmax_func(S_pp)  
          
        loss = -torch.einsum('bk,bk->b',Prob,Labels)  
        loss = torch.mean(loss)  
        return loss      
    
    def compute_local_global_reg(self, in_package):
        local_cls_embed = in_package['local_cls_embed']
        global_cls_embed = in_package['global_cls_embed']
        loss = WeightedL1(global_cls_embed, local_cls_embed)
        return loss

    def compute_loss_CLS(self,in_package):
        
        if len(in_package['batch_label'].size()) == 1:
            in_package['batch_label'] = self.weight_ce[in_package['batch_label']]  
        
        local_package = {'S_pp': in_package['local_cls_embed'],
                         'batch_label': in_package['batch_label']}
        global_package = {'S_pp': in_package['global_cls_embed'],
                         'batch_label': in_package['batch_label']}
        loss_CE_local = self.compute_aug_cross_entropy(local_package)
        loss_CE_global = self.compute_aug_cross_entropy(global_package)

        # lg_reg_package = {'local_cls_embed': in_package['local_cls_embed'],
        #                   'global_cls_embed': in_package['global_cls_embed']}
        # loss_lg_reg = self.compute_local_global_reg(lg_reg_package)
        
        ## total loss  
        loss = self.lambda_localCE * loss_CE_local
        loss += self.lambda_globalCE * loss_CE_global
        # loss += self.lambda_lg_reg * loss_lg_reg
          
        out_package = {'loss': loss,
                       'loss_CE_local': loss_CE_local,
                       'loss_CE_global': loss_CE_global}
          
        return out_package  

    def compute_CE_loss_non_conservative(self,in_package):
        
        if len(in_package['batch_label'].size()) == 1:
            in_package['batch_label'] = self.weight_ce[in_package['batch_label']]  
        
        loss_CE = self.compute_aug_cross_entropy(in_package, is_conservative=False)
        return {'loss_CE':loss_CE}

    def compute_CE_loss_conservative(self,in_package):
        
        if len(in_package['batch_label'].size()) == 1:
            in_package['batch_label'] = self.weight_ce[in_package['batch_label']]  
        
        loss_CE = self.compute_aug_cross_entropy(in_package, is_conservative=True)
        return {'loss_CE':loss_CE}
    
    def extract_attention(self,Fs):
        if self.is_conv:
            Fs = self.conv1(Fs)
            Fs = self.conv1_bn(Fs)
            Fs = F.relu(Fs)
        
        shape = Fs.shape
        Fs = Fs.reshape(shape[0],shape[1],shape[2]*shape[3])
        
        V_n = self.compute_V()
          
        if self.normalize_F and not self.is_conv:  
            Fs = F.normalize(Fs,dim = 1)
        
        A = torch.einsum('iv,vf,bfr->bir',V_n,self.W_2,Fs)   
        A = F.softmax(A,dim = -1)
        Hs = torch.einsum('bir,bfr->bif',A,Fs)
        
        package = {'A':A,'Hs':Hs}
        #What the attribute does not appear in the image
        return package        #bif

    def extract_attributes_compose_features(self, Hs, compose_type=None, compose_norm=None):
        if compose_type == None:
            compose_type = self.att_compose_type
        if compose_type == 'avg':
            gHs = F.adaptive_avg_pool1d(Hs.permute(0, 2, 1), 1).squeeze()
        elif compose_type == 'mlp':
            gHs = self.mlp_compose(Hs.permute(0, 2, 1)).squeeze()
        if compose_norm == None:
            compose_norm = self.att_compose_norm
        if compose_norm:
            gHs = F.normalize(gHs, dim=1)
        # if self.Scaler != None:
        gHs = self.Scaler(gHs)
        return gHs

    def extract_pooling_attributes_features(self, Fs, compose_type=None):
        f_local = self.extract_attention(Fs)['Hs']
        f_global = self.extract_attributes_compose_features(f_local, compose_type)
        return f_global
    
    def compute_attribute_embed(self,Hs):
        B = Hs.size(0)  
        V_n = self.compute_V()
        S_p = torch.einsum('iv,vf,bif->bi',V_n,self.W_1,Hs) 
        
        ## Attribute attention
        A_p = torch.einsum('iv,vf,bif->bi',V_n,self.W_3,Hs)
        A_p = torch.sigmoid(A_p) 
        ##  
        
        A_b_p = self.att.new_full((B,self.dim_att),fill_value = 1)  
        
        if self.uniform_att_2:
            S_pp = torch.einsum('ki,bi,bi->bik',self.att,A_b_p,S_p)
        else:
            S_pp = torch.einsum('ki,bi,bi->bik',self.att,A_p,S_p)
        
        if self.non_linear_emb:
            S_pp = torch.transpose(S_pp,2,1)    #[bki] <== [bik]
            S_pp = self.emb_func(S_pp)          #[bk1] <== [bki]
            S_pp = S_pp[:,:,0]                  #[bk] <== [bk1]
        else:
            S_pp = torch.sum(S_pp,axis=1)        #[bk] <== [bik]
        
        if self.is_bias:
            self.vec_bias = self.mask_bias*self.bias
            S_pp = S_pp + self.vec_bias
        
        package = {'S_pp':S_pp,'A_p':A_p, 'S_p':S_p}  
        
        return package  

    def compute_Fs2Sp(self,Fs):
        Hs = self.extract_attention(Fs)['Hs']
        V_n = self.compute_V()
        S_p = torch.einsum('iv,vf,bif->bi',V_n,self.W_1,Hs)
        return S_p

    def compute_Sp2Spp(self,S_p):
        
        A_b_p = self.att.new_full((S_p.size(0),self.dim_att),fill_value = 1)  
        
        if self.uniform_att_2:
            S_pp = torch.einsum('ki,bi,bi->bik',self.att,A_b_p,S_p)
        else:
            raise Exception
        
        if self.non_linear_emb:
            S_pp = torch.transpose(S_pp,2,1)    #[bki] <== [bik]
            S_pp = self.emb_func(S_pp)          #[bk1] <== [bki]
            S_pp = S_pp[:,:,0]                  #[bk] <== [bk1]
        else:
            S_pp = torch.sum(S_pp,axis=1)        #[bk] <== [bik]
        
        if self.is_bias:
            self.vec_bias = self.mask_bias*self.bias
            S_pp = S_pp + self.vec_bias
        
        package = {'S_pp':S_pp}
        
        return package  

    def forward(self,Fs):
        package_1 = self.extract_attention(Fs)
        Hs = package_1['Hs']
        package_2 = self.compute_attribute_embed(Hs)
        local_cls_embed = package_2['S_pp']
        S_p = package_2['S_p']

        gHs = self.extract_attributes_compose_features(Hs)
        global_cls_embed = self.compose_cls(gHs)

        # S_pp = local_cls_embed + global_cls_embed 
        # S_pp /= 2
        S_pp = local_cls_embed
        # S_pp = global_cls_embed
        
        package_out = {'S_pp': S_pp,
                       'S_p': S_p,
                       'local_cls_embed': local_cls_embed,
                       'global_cls_embed': global_cls_embed}
        
        return package_out


