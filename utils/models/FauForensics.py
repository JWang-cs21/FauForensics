import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from torch import nn

from functools import partial
from einops import rearrange, repeat

try:
    from base import *
    from backbone.csn import csn_temporal_no_head
    from backbone.MEFarg.ANFL import MEFARG
    from backbone.WhisperAEncoder import Whisper, ModelDimensions
except:
    from .base import *
    from .backbone.csn import csn_temporal_no_head
    from .backbone.MEFarg.ANFL import MEFARG
    from .backbone.WhisperAEncoder import Whisper, ModelDimensions

class FauForensics(nn.Module):
    def __init__(self, cfg, **kwargs):
        super().__init__()
        feat_dim = 512
        frame_num = 25

        ## AU encoder
        self.FAUEncoder_cnn = MEFARG(num_classes=12, backbone='resnet50', metric="dots")
        for p in self.parameters():
            p.requires_grad=False
        self.au_pool_norm = LayerNorm(feat_dim)

        ## audio encoder
        AEncoderDims = {'n_mels': 80, 'n_vocab': 51865, 'n_audio_ctx': 50, 'n_audio_state': 512, 'n_audio_head': 8, 
                        'n_audio_layer': 6, 'n_text_ctx': 448, 'n_text_state': 512, 'n_text_head': 8, 'n_text_layer': 6}
        dims = ModelDimensions(**AEncoderDims)
        self.AEncoder = Whisper(dims)    #.cuda()

        ## video encoder
        self.VEncoder_cnn = csn_temporal_no_head()    #.cuda() 
        self.v_pool = nn.Linear(2048, feat_dim, bias=False)
        self.v_pool_norm = LayerNorm(feat_dim)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        ## query-shared self-attention:
        q_num_queries = frame_num
        self.queries = nn.Parameter(torch.randn(q_num_queries, feat_dim)) 
        self.aud_attn_pool = CrossAttention(dim=feat_dim, context_dim=feat_dim, norm_context=True)
        self.aud_attn_pool_norm = LayerNorm(feat_dim)
        self.v_attn_pool = CrossAttention(dim=feat_dim, context_dim=feat_dim, norm_context=True)
        self.v_attn_pool_norm = LayerNorm(feat_dim)

        # MLP:
        self.num_cls     = cfg['num_classes']
        self.is_mul_loss = cfg['is_mul_loss']
        self.is_get_feat = cfg['is_get_feat']
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.logit_scale_v = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.logit_scale_a = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.mlp_joint = nn.Linear(frame_num**2, self.num_cls, bias=False)
        self.mlp_v = nn.Linear(frame_num**2, 2, bias=False)
        self.mlp_a = nn.Linear(frame_num**2, 2, bias=False)

    def get_fau_feat(self, video):
        video = video.permute(0,2,1,3,4)
        bs, frames, c, w, h = video.size()
        video = video.contiguous().view(-1, c, h, w)        
        feat_au = self.FAUEncoder_cnn(video)                
        feat_au = feat_au.mean(-1)
        feat_au = feat_au.view(bs, frames, -1)             
        return self.au_pool_norm(feat_au)
        
    def get_v_feat(self, video):
        feat_v = self.VEncoder_cnn(video)               
        feat_v = self.v_pool(feat_v.permute(0,2,1))     
        return self.v_pool_norm(feat_v)                

    def v_att_p(self, au_tokens):
        au_queries = repeat(self.queries, 'n d -> b n d', b=au_tokens.shape[0])
        au_queries = self.v_attn_pool(au_queries, au_tokens)
        au_queries = self.v_attn_pool_norm(au_queries)
        return au_queries  

    def a_att_p(self, audio_tokens):
        audio_queries = repeat(self.queries, 'n d -> b n d', b=audio_tokens.shape[0])
        audio_queries = self.aud_attn_pool(audio_queries, audio_tokens)
        audio_queries = self.aud_attn_pool_norm(audio_queries)
        return audio_queries   

    def get_unimodal_ap(self, feat, logit_scale):
        feat = feat/feat.norm(dim=-1, keepdim=True)
        feat = logit_scale * torch.matmul( feat, feat.permute(0,2,1) )
        feat = torch.nn.functional.normalize(feat)
        return feat.view(feat.size(0), -1)

    def get_TAPooler(self, visual_features, audio_features):
        v_feat_intra = self.get_unimodal_ap(visual_features, self.logit_scale_v.exp())
        a_feat_intra = self.get_unimodal_ap(audio_features, self.logit_scale_a.exp())
        visual_features = visual_features/visual_features.norm(dim=-1, keepdim=True)
        audio_features = audio_features/audio_features.norm(dim=-1, keepdim=True)
        logit_scale = self.logit_scale.exp()
        feat_inter = logit_scale * torch.matmul( visual_features, audio_features.permute(0,2,1) )
        feat_inter = torch.nn.functional.normalize(feat_inter)
        feat_inter = feat_inter.view(feat_inter.size(0), -1)
        return feat_inter, a_feat_intra, v_feat_intra

    def classifier_v6(self, imgs, audio, mask_ratio):
        feat_a = self.AEncoder.embed_audio(audio.squeeze(1).permute(0,2,1))     
        feat_vid = self.get_v_feat(imgs)                                        
        feat_au  = self.get_fau_feat(imgs)                                      
        feat_v = feat_au+feat_vid
        feat_v_p = self.v_att_p(feat_v)                                         
        feat_a_p = self.a_att_p(feat_a)                                         
        feat_inter, a_feat_intra, v_feat_intra =  self.get_TAPooler(feat_v_p, feat_a_p)
        pred_a = self.mlp_a(a_feat_intra)
        pred_v = self.mlp_v(v_feat_intra)
        pred_joint = self.mlp_joint(feat_inter)                                 
        if self.is_get_feat == 0:
            return { 'audio':pred_a, 'visual':pred_v, 'total':pred_joint }
        else:
            return { 'feat_inter':feat_inter, 'v_feat_intra':v_feat_intra, 'a_feat_intra':a_feat_intra , 'pred_total': pred_joint}

    def forward(self, imgs, audio, label=None, split='train', mask_ratio=0.5):
        preds = self.classifier_v6(imgs, audio, mask_ratio=0)
        if self.is_get_feat == 0:
            return preds['total'], self.get_loss(preds, label)    ## changed for unimodal detection
        else:
            return preds, torch.rand(1).to(self.device)

    def get_loss(self, prediction, target):
        if self.is_mul_loss == 0:
            return self.get_celoss(prediction['total'], target.cuda())
        else:
            loss_a = self.get_celoss(prediction['audio'], target['label_a'].to(self.device))
            loss_v = self.get_celoss(prediction['visual'], target['label_v'].to(self.device))
            loss_av = self.get_celoss(prediction['total'], target['label'].to(self.device))
            loss_total = loss_a*0.1 + loss_v*0.1 + loss_av*0.8
            return loss_total

    def get_celoss(self, prediction, target):
        if self.num_cls == 2 or self.num_cls == 4:  return F.cross_entropy(prediction, target)
        elif self.num_cls == 1:                     return F.binary_cross_entropy_with_logits(prediction.squeeze(-1), target.float())
        else:                                       print('##please check num_classes!'); exit()

    '''return all layers'''
    def alllayers(self):
        return {
            'audio': self.AEncoder,
            'visual': self.VEncoder_cnn,
            'fau': self.FAUEncoder_cnn,
            'v_pool': self.v_pool,
            'queries': self.queries,
            'v_attn_pool': self.v_attn_pool,
            'aud_attn_pool': self.aud_attn_pool,
            'mlp_joint': self.mlp_joint,
            'mlp_a': self.mlp_a,
            'mlp_v': self.mlp_v,
            'logit_scale': self.logit_scale,
            'logit_scale_v': self.logit_scale_v,
            'logit_scale_a': self.logit_scale_a,
        }       











