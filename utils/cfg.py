'''base config'''
# config for optimizer
OPTIMIZER_CFG = {
    'type': 'sgd',
    'sgd': {
        'learning_rate': 0.0001,
        'momentum': 0.9,
        'weight_decay': 5e-4,
    },
    'adam': {
        'learning_rate': 0.0002, 
        'weight_decay': 1e-4,
    },
    'adamw': {
        'learning_rate': 0.0002, 
        'weight_decay': 5e-4,
    },
    'adamp': {
        'learning_rate': 0.0002, 
        'weight_decay': 5e-4,
    },
    'max_epochs': 200,
    'params_rules': {},
    'filter_params': False,    
    'policy': {
        'type': 'poly',
        'opts': {'power': 0.9, 'max_iters': None, 'num_iters': None, 'num_epochs': None}
    },
    'adjust_period': ['iteration', 'epoch'][0],
    'warmup_epochs': 10,
    'LR_decay':0.01,
}

# config for model
MODEL_CFG = {
    'type': 'net_name',
    'num_classes': -1,
    'benchmark': True,
    'align_corners': False,
    'is_multi_gpus': True,
    'distributed': {'is_on': True, 'backend': 'nccl'},
    'norm_cfg': {'type': 'syncbatchnorm', 'opts': {}},
    'act_cfg': {'type': 'relu', 'opts': {'inplace': True}},
    'imgSize':224,
    'in_chans':3,
    'data_type':'a',
}

