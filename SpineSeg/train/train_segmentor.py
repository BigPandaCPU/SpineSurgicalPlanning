import argparse, os, time, glob
import numpy as np 
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from utils_trainer import *
from data_loader import *


parser = argparse.ArgumentParser('spine or vertebra segmentor train script')

# parser.add_argument('--gpu_device', type=int, default=[0, 1], nargs='+')
parser.add_argument('--task', type=str, help='options: spine | vertebra')
parser.add_argument('--save_dir', type=str, help='folder to save trained models')
parser.add_argument('--model_name', type=str)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--lambda_l2', type=float, default=20)
parser.add_argument('--dataset_dir', type=str, help='path to the generated 3D cubes')
parser.add_argument('--train_ID_list', nargs='+', help='a list of scan IDs, eg. verse008, GL003, ...')
parser.add_argument('--batch_size', type=int, default=10)
parser.add_argument('--n_epoch', type=int, default=150)
parser.add_argument('--log_dir', type=str, help='folder to save logs')
args = parser.parse_args()


# use all available gpus
gpu_device = np.arange(torch.cuda.device_count()).tolist()

# if len(args.gpu_device) == 1:
# torch.backends.cudnn.enabled = False 

# create result folder
save_dir = os.path.join(args.save_dir, args.model_name)
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

print('model save_dir: ', save_dir)

# initialize the generator 
from .segmentor import Unet3D_attention

if args.task == 'vertebra':
    generator = Unet3D_attention(in_channels=2, out_channels=1, 
                                 activation2='sigmoid', feature_maps=[16, 32, 64, 128, 256])
elif args.task == 'spine':
    generator = Unet3D_attention(in_channels=1, out_channels=1, 
                                 activation2='sigmoid', feature_maps=[16, 32, 64, 128, 256])    
else: 
    raise NotImplementedError('Unknwn task {}, task options: spine | vertebra'.format(args.task))

# initialize weights and gpu devices
generator = init_net(generator, gpu_ids=gpu_device)


# load pretrained model
initial_epoch = findLastCheckpoint(save_dir)
if initial_epoch > 0:
    print('resuming from epoch {}'.format(initial_epoch))

    state_dict = torch.load(os.path.join(save_dir, 'generator_{:05d}.pth'.format(initial_epoch)))
    if len(gpu_device) > 1:
        state_dict = dict_mapping(state_dict)
    generator.load_state_dict(state_dict)

    initial_epoch += 1

# Optimizers
optimizer = torch.optim.Adam(generator.parameters(), lr=args.lr)


# load dataset
if args.task == 'spine':
    dataset = segmentation_cubes_loader(args.dataset_dir, args.train_ID_list, False)
elif args.task == 'vertebra':
    dataset = segmentation_cubes_loader(args.dataset_dir, args.train_ID_list, True)

dataloader = DataLoader(dataset=dataset, num_workers=10, drop_last=True, batch_size=args.batch_size, shuffle=True)
print('dataset: ', len(dataset))


# loss
l2_loss = torch.nn.MSELoss()
from losses import dice_loss


# ---------
# train
# ---------

for epoch in range(initial_epoch, args.n_epoch):

    strat_time = time.time()
    epoch_loss = 0

    for i, batch in enumerate(dataloader):

        # model inputs
        img = batch[0].cuda()
        real_msk = batch[1].cuda().requires_grad_(False)

        optimizer.zero_grad()

        pred = generator(img)

        # l2 loss
        loss_l2 = l2_loss(input=pred, target=real_msk)

        # dice loss
        loss_dice = dice_loss(input=pred, target=real_msk)

        # total loss
        loss = loss_dice + args.lambda_l2 * loss_l2

        epoch_loss += loss.item()

        loss.backward()
        optimizer.step()

        print('{}, {}/{}, loss_l2 = {}, loss_dice = {}, loss = {}'.format(epoch+1, i, len(dataset)//args.batch_size, loss_l2, loss_dice, loss))


    elapsed_time = time.time() - strat_time

    epoch_info = '%4d/%4d  loss = %2.4f, time= %4.2f s' % (epoch+1, args.n_epoch, epoch_loss, elapsed_time)
    print(epoch_info_G, '\n')

    # log the loss to file
    log2file(args.log_dir, epoch, args.model_name, epoch_info)

    # save the model dictionary
    if len(gpu_device) == 1:
        torch.save(generator.state_dict(), os.path.join(save_dir, 'generator_{:05d}.pth'.format(epoch)))
    else:
        torch.save(generator.module.state_dict(), os.path.join(save_dir, 'generator_{:05d}.pth'.format(epoch)))
















