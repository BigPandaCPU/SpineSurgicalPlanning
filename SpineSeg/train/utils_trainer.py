import glob, os
import torch
import torch.nn as nn 
import torch.nn.init as init

def init_weight(net, init_type='normal', init_gain=0.02):

    """
    Available initialzation methods: normal | xavier | kaiming | orthogonal

    """
    print('initialize network with method <{}>'.format(init_type))

    for m in net.modules():
        if isinstance(m, nn.Conv3d):
            if init_type == 'normal':
                init.normal_(m.weight, mean=0.0, std=1.0)
            elif init_type == 'uniform':
                init.uniform_(m.weight, a=0.0, b=1.2)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight, gain=init_gain)
            elif init_type == 'ones':
                init.ones_(m.weight)
            else:
                raise NotImplementedError('initialzation method {} NOT implemented.'.format(init_type))
            if m.bias is not None:
                init.constant_(m.bias, 0.0)
        if isinstance(m, nn.BatchNorm3d):
            init.normal_(m.weight, mean=0.0, std=1.0)
            init.constant_(m.bias, 0.0)


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[0]):

    assert torch.cuda.is_available()
    if len(gpu_ids) == 1:
        net = net.cuda()
    else:
        print('multi gpus processing.')
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)
        print('gpu ids: ', gpu_ids)
    init_weight(net, init_type=init_type, init_gain=init_gain)

    return net

def findLastCheckpoint(save_dir):   # by looking at the index from the model name
    import re 
    model_list = sorted(glob.glob(os.path.join(save_dir, 'generator_*.pth')))
    if model_list:
        last_model = model_list[-1]
        initial_epoch = [int(s) for s in re.findall(r'\d+', last_model)]
        initial_epoch = initial_epoch[-1]
    else:
        initial_epoch = 0
    return initial_epoch 


def dict_mapping(state_dict):

    from collections import OrderedDict
    new_state_dict = OrderedDict()

    for k, v in state_dict.items():
        if 'module' not in k:
            k = 'module.'+k
        else:
            k = k.replace('features.module.', 'module.features.')
        new_state_dict[k]=v

    return new_state_dict

def log2file(log_dir, epoch, model_name, text):
    log_dir = log_dir
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    log_file = model_name+'.txt'
    log_path = os.path.join(log_dir, log_file)
    if epoch == 0:
        with open(log_path, 'w+') as logg:
            logg.write(model_name)
            logg.write('\n')
            logg.write(text)
            logg.write('\n')
    else:
        with open(log_path, 'a') as logg:
            logg.write(text)
            logg.write('\n')
            logg.close()
