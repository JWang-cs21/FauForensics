from .models import *

'''build model'''
def BuildModel(cfg, mode, **kwargs):
    supported_models = {
        'FauForensics': FauForensics,
    }
    model_type = cfg['type']
    assert model_type in supported_models, 'unsupport model_type %s...' % model_type
    return supported_models[model_type](cfg, mode=mode)


from .optimizers import SGDBuilder, AdamBuilder, AdamWBuilder, AdamPBuilder
'''build optimizer'''
def BuildOptimizer(model, cfg, **kwargs):
    supported_optimizers = {
        'sgd': SGDBuilder,
        'adam': AdamBuilder,
        'adamw': AdamWBuilder,
        'adamp': AdamPBuilder
    }
    assert cfg['type'] in supported_optimizers, 'unsupport optimizer type %s...' % cfg['type']
    selected_optim_cfg = {
        'params_rules': cfg.get('params_rules', {}),
        'filter_params': cfg.get('filter_params', False)
    }
    selected_optim_cfg.update(cfg[cfg['type']])
    return supported_optimizers[cfg['type']](model, selected_optim_cfg, **kwargs)

