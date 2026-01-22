
import torch.nn as nn
import torch.optim as optim


'''sgd builder'''
def SGDBuilder(model, cfg, **kwargs):
    params_rules, filter_params = cfg.get('params_rules', {}), cfg.get('filter_params', False)
    # print('##sgd ##filter_params:',filter_params)
    if not params_rules:
        optimizer = optim.SGD(model.parameters() if not filter_params else filter(lambda p: p.requires_grad, model.parameters()), 
                              lr=cfg['learning_rate'], 
                              momentum=cfg['momentum'], 
                              weight_decay=cfg['weight_decay'],
                              dampening=cfg.get('dampening', 0),
                              nesterov=cfg.get('nesterov', False))
    else:
        params, all_layers = [], model.alllayers()
        assert 'others' not in all_layers, 'potential bug in model.alllayers...'
        for key, value in params_rules.items():
            if key == 'others': continue
            params.append({
                'params': all_layers[key].parameters() if not filter_params else filter(lambda p: p.requires_grad, all_layers[key].parameters()), 
                'lr': cfg['learning_rate'] * value, 
                'name': key
            })
        others = []
        for key, layer in all_layers.items():
            if key not in params_rules: others.append(layer)
        others = nn.Sequential(*others)
        params.append({
            'params': others.parameters() if not filter_params else filter(lambda p: p.requires_grad, others.parameters()), 
            'lr': cfg['learning_rate'] * params_rules['others'], 
            'name': 'others'
        })
        optimizer = optim.SGD(params, 
                              lr=cfg['learning_rate'], 
                              momentum=cfg['momentum'], 
                              weight_decay=cfg['weight_decay'],
                              dampening=cfg.get('dampening', 0),
                              nesterov=cfg.get('nesterov', False))
    return optimizer


'''adam builder'''
def AdamBuilder(model, cfg, **kwargs):

    params_rules, filter_params = cfg.get('params_rules', {}), cfg.get('filter_params', False)
    # print('##adam ##filter_params:',filter_params)
    if not params_rules:
        optimizer = optim.Adam(model.parameters() if not filter_params else filter(lambda p: p.requires_grad, model.parameters()),
                               lr=cfg['learning_rate'],
                               weight_decay=cfg['weight_decay'],
                               betas=cfg.get('betas', (0.9, 0.999)),
                               eps=cfg.get('eps', 1e-03),
                            #    amsgrad=cfg.get('amsgrad', False)
                               )
    else:
        params, all_layers = [], model.alllayers()
        assert 'others' not in all_layers, 'potential bug in model.alllayers...'
        for key, value in params_rules.items():
            if key == 'others': continue
            params.append({
                'params': all_layers[key].parameters() if not filter_params else filter(lambda p: p.requires_grad, all_layers[key].parameters()), 
                'lr': cfg['learning_rate'] * value, 
                'name': key
            })
        others = []
        for key, layer in all_layers.items():
            if key not in params_rules: others.append(layer)
        others = nn.Sequential(*others)
        params.append({
            'params': others.parameters() if not filter_params else filter(lambda p: p.requires_grad, others.parameters()), 
            'lr': cfg['learning_rate'] * params_rules['others'], 
            'name': 'others'
        })
        optimizer = optim.Adam(params,
                               lr=cfg['learning_rate'],
                               weight_decay=cfg['weight_decay'],
                               betas=cfg.get('betas', (0.9, 0.999)),
                               eps=cfg.get('eps', 1e-08),
                               amsgrad=cfg.get('amsgrad', False))
    return optimizer


'''adam builder'''
def AdamWBuilder(model, cfg, **kwargs):

    params_rules, filter_params = cfg.get('params_rules', {}), cfg.get('filter_params', False)
    if not params_rules:
        print('##adamW 0##filter_params:',filter_params)
        optimizer = optim.AdamW(model.parameters() if not filter_params else filter(lambda p: p.requires_grad, model.parameters()),
                               lr=cfg['learning_rate'],
                               weight_decay=cfg['weight_decay'],
                               betas=cfg.get('betas', (0.9, 0.999)),
                               eps=cfg.get('eps', 1e-08),
                            #    amsgrad=cfg.get('amsgrad', False)
                               )
    else:
        print('##adamW 1##filter_params:',filter_params)
        try:
            params, all_layers = [], model.alllayers()
        except:
            params, all_layers = [], model.module.alllayers()
            
        assert 'others' not in all_layers, 'potential bug in model.alllayers...'
        for key, value in params_rules.items():
            if key == 'others': continue
            params.append({
                'params': all_layers[key].parameters() if not filter_params else filter(lambda p: p.requires_grad, all_layers[key].parameters()), 
                'lr': cfg['learning_rate'] * value, 
                'name': key
            })
        others = []
        for key, layer in all_layers.items():
            if key not in params_rules: 
                others.append(layer)
                # print('##key:',key)
        others = nn.Sequential(*others)
        params.append({
            'params': others.parameters() if not filter_params else filter(lambda p: p.requires_grad, others.parameters()), 
            'lr': cfg['learning_rate'] * params_rules['others'], 
            'name': 'others'
        })
        optimizer = optim.AdamW(params,
                               lr=cfg['learning_rate'],
                               weight_decay=cfg['weight_decay'],
                               betas=cfg.get('betas', (0.9, 0.999)),
                               eps=cfg.get('eps', 1e-08),
                               amsgrad=cfg.get('amsgrad', False))
    return optimizer



'''adam builder  '''
def AdamPBuilder(model, cfg, **kwargs):
    from adamp import AdamP
    params_rules, filter_params = cfg.get('params_rules', {}), cfg.get('filter_params', False)
    if not params_rules:
        optimizer = AdamP(model.parameters() if not filter_params else filter(lambda p: p.requires_grad, model.parameters()), 
                    lr=cfg['learning_rate'], betas=(0.9, 0.999), weight_decay=1e-2)

        # print('##adamP 0##filter_params:',filter_params)
    else:
        # print('##adamP 1##filter_params:',filter_params)
        try:
            params, all_layers = [], model.alllayers()
        except:
            params, all_layers = [], model.module.alllayers()
            
        assert 'others' not in all_layers, 'potential bug in model.alllayers...'
        for key, value in params_rules.items():
            if key == 'others': continue
            params.append({
                'params': all_layers[key].parameters() if not filter_params else filter(lambda p: p.requires_grad, all_layers[key].parameters()), 
                'lr': cfg['learning_rate'] * value, 
                'name': key
            })
        others = []
        for key, layer in all_layers.items():
            if key not in params_rules: 
                others.append(layer)
        others = nn.Sequential(*others)
        params.append({
            'params': others.parameters() if not filter_params else filter(lambda p: p.requires_grad, others.parameters()), 
            'lr': cfg['learning_rate'] * params_rules['others'], 
            'name': 'others'
        })
        optimizer = AdamP(params,  lr=cfg['learning_rate'], betas=(0.9, 0.999), weight_decay=1e-2)

    return optimizer

'''adjust learning rate'''
def adjustLearningRate(optimizer, optimizer_cfg=None, batch_idx=None, data_nums=None):          # policy_cfg['opts']['num_epochs']
    # parse and check the config for optimizer
    policy_cfg, selected_optim_cfg = optimizer_cfg['policy'], optimizer_cfg[optimizer_cfg['type']]
    if ('params_rules' in optimizer_cfg) and (optimizer_cfg['params_rules']):
        assert len(optimizer.param_groups) == len(optimizer_cfg['params_rules'])
    # adjust the learning rate according the policy
    if policy_cfg['type'] == 'poly':
        base_lr = selected_optim_cfg['learning_rate']
        # min_lr = selected_optim_cfg.get('min_lr', base_lr * 0.1)
        min_lr = selected_optim_cfg.get('min_lr', base_lr * optimizer_cfg['LR_decay'])
        max_iters, power = policy_cfg['opts']['max_iters'], policy_cfg['opts']['power']
        num_iters = min(policy_cfg['opts']['num_iters'], max_iters) 
        coeff = (1 - num_iters / max_iters) ** power
        target_lr = coeff * (base_lr - min_lr) + min_lr 
        for param_group in optimizer.param_groups:
            if ('params_rules' in optimizer_cfg) and (optimizer_cfg['params_rules']):
                param_group['lr'] = target_lr * optimizer_cfg['params_rules'][param_group['name']]
            else:
                param_group['lr'] = target_lr
    elif policy_cfg['type'] == 'stair':
        target_lr = selected_optim_cfg['learning_rate']*0.8**(int((policy_cfg['opts']['num_epochs']-10)/3))
        # if policy_cfg['opts']['num_epochs'] % 3 == 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] = target_lr        #param_group['lr'] * 0.8
        # target_lr = param_group['lr']
    elif policy_cfg['type'] == 'warmup_linear':
        warmup_epochs = 10
        epoch = policy_cfg['opts']['num_epochs']
        # batch_idx = policy_cfg['opts']['num_iters']
        base_lr = selected_optim_cfg['learning_rate']
        if epoch < warmup_epochs:
            epoch += float(batch_idx + 1) / data_nums
            lr_adj = 1. * (epoch / warmup_epochs)
        else:
            if epoch < 30 + warmup_epochs:
                lr_adj = 1.
            elif epoch < 60 + warmup_epochs:
                lr_adj = 1e-1
            elif epoch < 90 + warmup_epochs:
                lr_adj = 1e-2
            else:
                lr_adj = 1e-3
        for param_group in optimizer.param_groups:
            param_group['lr'] = base_lr * lr_adj
        return base_lr * lr_adj
    elif policy_cfg['type'] == 'warmup_poly':
        warmup_epochs = 3
        epoch = policy_cfg['opts']['num_epochs']
        # batch_idx = policy_cfg['opts']['num_iters']
        base_lr = selected_optim_cfg['learning_rate']
        if epoch < warmup_epochs:
            epoch += float(batch_idx + 1) / data_nums
            lr_adj = 1. * (epoch / warmup_epochs)
            for param_group in optimizer.param_groups:
                param_group['lr'] = base_lr * lr_adj
            return base_lr * lr_adj
        else:
        # base_lr = selected_optim_cfg['learning_rate']
        # min_lr = selected_optim_cfg.get('min_lr', base_lr * 0.1)
            min_lr = selected_optim_cfg.get('min_lr', base_lr * optimizer_cfg['LR_decay'])
            num_iters = policy_cfg['opts']['num_iters'] - warmup_epochs*data_nums
            max_iters = policy_cfg['opts']['max_iters']
            power = policy_cfg['opts']['power']
            coeff = (1 - num_iters / max_iters) ** power
            target_lr = coeff * (base_lr - min_lr) + min_lr
            for param_group in optimizer.param_groups:
                if ('params_rules' in optimizer_cfg) and (optimizer_cfg['params_rules']):
                    param_group['lr'] = target_lr * optimizer_cfg['params_rules'][param_group['name']]
                else:
                    param_group['lr'] = target_lr
            return target_lr
    elif policy_cfg['type'] == 'warmup_cosine':
        warmup_epochs = optimizer_cfg['warmup_epochs']
        epoch = policy_cfg['opts']['num_epochs']
        base_lr = selected_optim_cfg['learning_rate']
        if epoch < warmup_epochs:
            epoch += float(batch_idx + 1) / data_nums
            lr_adj = 1. * (epoch / warmup_epochs)
        else:
            run_epochs = epoch - warmup_epochs
            total_epochs = args.epochs - warmup_epochs
            T_cur = float(run_epochs * data_nums) + batch_idx
            T_total = float(total_epochs * data_nums)
            lr_adj = 0.5 * (1 + math.cos(math.pi * T_cur / T_total))
        target_lr = base_lr * lr_adj
        for param_group in optimizer.param_groups:
            param_group['lr'] = target_lr
    else:
        raise ValueError('Unsupport policy %s...' % policy)
    return target_lr










