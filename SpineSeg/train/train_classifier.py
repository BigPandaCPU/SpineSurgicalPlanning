

import argparse, os, torch, time
from .utils import mkpath
from .classifier import encoder3d
from utils.trainer import init_net,log2file, findLastCheckpoint, dict_mapping
import torch.nn.functional as F
from torch.utils.data import DataLoader
from class_weight import get_class_weight


parser = argparse.ArgumentParser()

parser.add_argument('--save_dir', type=str, help='folder to save the trained models')
parser.add_argument('--model_name', type=str)
parser.add_argument('--classify_level', type=str, help='group | cervical | thoracic | lumbar')
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--dataset_dir', type=str, help='path to the generated verse20 training set')
parser.add_argument('--train_ID_list', nargs='+', help='a list of scan IDs, eg. verse008, GL003, ...')
parser.add_argument('--batch_size', type=int, default=5)
parser.add_argument('--n_epoch', type=int, default=150)
parser.add_argument('--log_dir', type=str, help='folder to save the logs')

args = parser.parse_args()

gpu_device = np.arange(torch.cuda.device_count()).tolist()
# torch.backends.cudnn.enabled = False 

### create result folder
save_dir = os.path.join(args.save_dir, args.model_name)
mkpath(save_dir)
print('models will be saved to ', save_dir)


### load model & calss weights & datasets
if args.classify_level == 'group':
    model = encoder3d(n_classes=3)
    class_weights = get_class_weight('group')
    dataset = group_cubes_loader(train_ids=args.train_ID_list, data_dir=args.dataset_dir)

elif args.classify_level == 'cervical':
    model = encoder3d(n_classes=7)
    class_weights = get_class_weight('cervical')
    dataset = cervical_cubes_loader(train_ids=args.train_ID_list, data_dir=args.dataset_dir)

elif args.classify_level == 'thoracic':
    model = encoder3d(n_classes=12)
    class_weights = get_class_weight('thoracic')
    dataset = thoracic_cubes_loader(train_ids=args.train_ID_list, data_dir=args.dataset_dir)

elif args.classify_level == 'lumbar':
    model = encoder3d(n_classes=5)
    class_weights = get_class_weight('lumbar')
    dataset = lumbar_cubes_loader(train_ids=args.train_ID_list, data_dir=args.dataset_dir)

else:
    raise NotImplementedError('Unknown classify level {}'.format(args.classify_level))

model = init_net(model, gpu_ids=gpu_device)

### load pretrained model
initial_epoch = findLastCheckpoint(save_dir)
if initial_epoch > 0:
    print('resuming from epoch {}'.format(initial_epoch))

    state_dict = torch.load(os.path.join(save_dir, 'model_{:05d}.pth'.format(initial_epoch)))
    if len(gpu_device) > 1:
        state_dict = dict_mapping(state_dict)
    model.load_state_dict(state_dict)

    initial_epoch += 1


### optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)


### load dataset
dataloader = DataLoader(dataset=dataset, num_workers=16, drop_last=True, batch_size=args.batch_size, shuffle=True) # num_workers=num_cpu-4,
print('dataset: ', len(dataset))


### class weights
class_weights = torch.from_numpy(class_weights).to(torch.float).cuda()

##-----------
##  train
##-----------

for epoch in range(initial_epoch, args.n_epoch):

    strat_time = time.time()

    epoch_loss = 0

    for i, batch in enumerate(dataloader):
        optimizer.zero_grad()

        # model inputs
        img = batch['image'].cuda()
        label = batch['label'].cuda()

        # predict
        pred = model(img)

        # CE loss
        loss = F.cross_entropy(input=pred, target=label, weight=class_weights)

        epoch_loss += loss.item()

        loss.backward()

        optimizer.step()

        print('{}, {}/{}, loss = {}'.format(epoch+1, i, len(dataset)//args.batch_size, loss))


    elapsed_time = time.time() - strat_time
    epoch_info = '%4d/%4d  loss = %2.4f, time= %4.2f s' % (epoch+1, args.n_epoch, epoch_loss, elapsed_time)
    print(epoch_info, '\n')

    # log the loss to file
    log2file(args.log_dir, epoch, args.model_name, epoch_info)

    # save the model dictionary
    if len(gpu_device) == 1:
        torch.save(model.state_dict(), os.path.join(save_dir, 'model_{:05d}.pth'.format(epoch)))
    else:
        torch.save(model.module.state_dict(), os.path.join(save_dir, 'model_{:05d}.pth'.format(epoch)))








