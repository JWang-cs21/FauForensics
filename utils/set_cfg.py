from .augmentation import *
# from .aug_gray import *
from torchvision import transforms as tfs

import os

def set_cfg(cfg, args, logger_handle):
 
## set MODEL_CFG
    cfg.MODEL_CFG['type'] = args.net_type    # 'nn'
    cfg.MODEL_CFG['num_classes'] = args.num_cls
    # cfg.MODEL_CFG['is_frozen'] = args.is_frozen
    cfg.MODEL_CFG['imgSize'] = args.imgSize
    cfg.MODEL_CFG['is_step2'] = args.is_step2
    if args.data_type == 'a':
        cfg.MODEL_CFG['in_chans'] = 1
    elif args.data_type == 'v' or  args.data_type == 'f':
        cfg.MODEL_CFG['in_chans'] = 3
    cfg.MODEL_CFG['data_type'] = args.data_type
    cfg.MODEL_CFG['is_mul_loss'] = args.is_mul_loss
    cfg.MODEL_CFG['is_get_feat'] = args.is_get_feat


## set OPTIMIZER_CFG
    cfg.OPTIMIZER_CFG['type'] = args.opt_type
    cfg.OPTIMIZER_CFG['policy']['type'] = args.LR_policy
    cfg.OPTIMIZER_CFG['max_epochs'] = args.max_epochs
    cfg.OPTIMIZER_CFG[cfg.OPTIMIZER_CFG['type']]['learning_rate'] = args.LR
    cfg.OPTIMIZER_CFG['LR_decay'] = args.LR_decay

    if args.is_multi_lr == 1:
        m_LR_decay = args.m_LR_decay   #0.01
        cfg.OPTIMIZER_CFG['params_rules'] = {'visual':m_LR_decay, 'audio':m_LR_decay, 'others':1} #, 'logit_scale':0.01
        logger_handle.warning('##please check the params_rules!!!') 

    if args.is_frozen==1 or args.is_frozen==2:
        cfg.OPTIMIZER_CFG['filter_params'] = True
    elif args.is_frozen==0:
        cfg.OPTIMIZER_CFG['filter_params'] = False
    else:
        logger_handle.info('## bug in Label is_frozen: %s' % (args.is_frozen))
    return cfg