import torch

def load_pretrained_weights(start_epoch, pretrained_dict_a=None, pretrained_dict_v=None, model=None, args=None, logger=None):
    # return model
    PATH_root='/home/ubuntu/code/mm_detectors/PSForensics/'
    # PATH_root='/path_to_pretrained_weights/'

    path_fau = PATH_root+'checkpoints/pretrain/au_112.pth'
    path_fau = PATH_root+'checkpoints/pretrain/au_112_12cls2.pth'
    state_dict_au = torch.load(path_fau, map_location='cpu')['state_dict']
    state_dict_v = torch.load(PATH_root+'checkpoints/pretrain/CSN_32x2_R101.pyth', map_location='cpu')['model_state']  ##keys: dict_keys(['epoch', 'model_state', 'optimizer_state', 'cfg'])
    state_dict_a = torch.load(open(PATH_root+'checkpoints/pretrain/whisper_base.pt', "rb"), map_location="cpu")["model_state_dict"]
    del state_dict_a['encoder.positional_embedding']

    new_state_dict = {}
    for key, value in state_dict_v.items():  #.state_dict().items()
        new_state_dict['VEncoder_cnn.'+key] = value

    for key, value in state_dict_au.items():  #.state_dict().items()
        new_state_dict[key.replace('module.','FAUEncoder_cnn.')] = value

    for key, value in state_dict_a.items():  #.state_dict().items()
        new_state_dict['AEncoder.'+key] = value
    model.load_state_dict(new_state_dict, strict=False)
    logger.info('##successfully load weights for csn, whisper, and aunet')
    return model


