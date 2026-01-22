
#!/usr/bin/python
# -*- coding: utf-8 -*-


import os
import argparse
import copy
import time
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
import pynvml
import warnings
warnings.filterwarnings("ignore")

import argparse
parser = argparse.ArgumentParser(description='a general framework by Miracle Team.')
# parser.add_argument('--loginterval',    type=int,   default=2000,   help='')
parser.add_argument('--PATH_root',       type=str,   default='',      help='Path for dataset')
parser.add_argument('--Path_checkpoints',type=str,   default='./checkpoints/', help='Checkpoints you want to resume from')
parser.add_argument('--PATH_save',     type=str,   default='0807',  help='')
parser.add_argument('--train_file',      type=str,   default='train', help='')
parser.add_argument('--labels_folder',   type=str,   default='labels_end', help='')
parser.add_argument('--test_file',       type=str,   default='default', help='')
parser.add_argument('--val_file',        type=str,   default='default', help='')
parser.add_argument('--checkpointspath', type=str,   default='',      help='')
parser.add_argument('--database',        type=str,   default='vox',   help='')
parser.add_argument('--dataset_cfg',     type=str,   default='cfg/dataset.yaml',      help='path for dataset.yaml')
parser.add_argument('--opt_type',        type=str,   default='adam',  help='optimizer type.')
parser.add_argument('--LR',              type=float, default=0.001,   help='')
parser.add_argument('--LR_policy',       type=str,   default='poly',  choices=['poly', 'stair', 'warmup_linear', 'warmup_cosine', 'warmup_poly'], help='')
parser.add_argument('--m_LR_decay',      type=float, default=0.001,   help='')
parser.add_argument('--LR_decay',        type=float, default=0.01,   help='')
parser.add_argument('--max_epochs',      type=int,   default=500,     help='epoch.')
parser.add_argument('--batch_size',      type=int,   default=32,      help='')
parser.add_argument('--net_type',        type=str,   default='nn',    help='model type.')
parser.add_argument('--data_type',       type=str,   default='va',    choices=['a', 'v', 'rf', 'mm'],  help='data type: visual or audio or audio-visual.')
parser.add_argument('--seed',            type=int,   default=0,       help='')
parser.add_argument('--imgSize',         type=int,   default=0,       help='')
parser.add_argument('--cropSize',        type=int,   default=0,       help='')
parser.add_argument('--num_cls',         type=int,   default=2,      help='')
parser.add_argument('--num_workers',     type=int,   default=16,      help='')
parser.add_argument('--iters_accumulate',type=int,   default=2,      help='')
parser.add_argument('--gpus',            type=int,   default=1,      help='')
parser.add_argument('--use_tensorboard', type=int,   default=1,       help='')
parser.add_argument('--is_tqdm',         type=int,   default=1,       help='')
parser.add_argument('--is_frozen',       type=int,   default=0,       help='')
parser.add_argument('--is_DB',           type=str,   default='',      help='')
parser.add_argument('--is_eval_test',    type=int,   default=1,       help='')
parser.add_argument('--is_eval_val',     type=int,   default=1,       help='')
parser.add_argument('--is_test',         type=int,   default=0,       help='')
parser.add_argument('--is_adjustLR',     type=int,   default=0,       help='')
parser.add_argument('--is_multi_lr',     type=int,   default=0,       help='')
parser.add_argument('--is_step2',        type=int,   default=0,       choices=[0,1,2],  help='.')
parser.add_argument('--is_checkdata',    type=int,   default=0,       choices=[0,1],  help='.')
parser.add_argument('--is_save',         type=int,   default=1,       help='')
parser.add_argument('--is_random_clip',  type=int,   default=1,       help='')
parser.add_argument('--is_mul_loss',     type=int,   default=0,       help='')
parser.add_argument('--is_label_dict',   type=int,   default=0,       help='')
parser.add_argument('--is_lr_scheduler', type=int,   default=0,       help='')
parser.add_argument('--is_amp',          type=int,   default=1,       help='')
parser.add_argument('--is_out_by_softmax',type=int,   default=0,       help='')
parser.add_argument('--is_get_feat',     type=int,   default=0,       help='')
args = parser.parse_args()

best_acc = {'train':0, 'test':0,'val':0}
best_auc = {'train':0, 'test':0,'val':0}

## check path ##
def checkPath(_PATH_):
    if not os.path.exists(_PATH_):
        os.makedirs(_PATH_)
checkPath(os.path.join(args.Path_checkpoints, args.PATH_save))

if args.use_tensorboard == 1:
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(os.path.join(args.Path_checkpoints, args.PATH_save, 'tensorboard'))

## tools
from tqdm import tqdm
if args.is_tqdm == 1:    is_tqdm = False
else:                    is_tqdm = True

if args.is_amp == 1:     enabled_amp = True
else:                    enabled_amp = False

## father class
class helper():
    def __init__(self, split='train'):
        self.loss = nn.CrossEntropyLoss()
        self.use_cuda = torch.cuda.is_available()
        self.FloatTensor = torch.cuda.FloatTensor if self.use_cuda else torch.FloatTensor
        self.LongTensor = torch.cuda.LongTensor if self.use_cuda else torch.LongTensor
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        self.optimizer_cfg = cfg.OPTIMIZER_CFG

    def get_label(self, samples, num_cls, is_label_dict=1):
        if num_cls <= 2 :
            return {'label_v': samples['label_v'].to(self.device), 'label_a': samples['label_a'].to(self.device), 'label': samples['label'].to(self.device), 'path': samples['path'] }
        else:
            return {'label_v': samples['label_v'].to(self.device), 'label_a': samples['label_a'].to(self.device), 'label': (samples['label_a'].to(self.device)*2 + samples['label_v'].to(self.device)*1).to(self.device), 'path': samples['path']  }

    def get_gtest_sets(self, train_database):
        if train_database == 'FakeAV':           gtest_sets = ['LipSyncTimit']  # ,'PolyGlot','LAV-DF'
        elif train_database == 'LAV-DF':         gtest_sets = ['LipSyncTimit','PolyGlot','FakeAV']  
        elif train_database == 'PolyGlot':       gtest_sets = []
        elif train_database == 'LipSyncTimit':   gtest_sets = []
        else:                                    logger_handle.warning('##check test database in main.py'); exit()
        return gtest_sets

    def get_perf(self, pred_score_list, gt_label_list, pred_label_list, num_cls, video_index_list=None, split='train'):
        gt_label = np.concatenate(tuple([_ for _ in gt_label_list]), axis=0)        
        pred_label = np.concatenate(tuple([_ for _ in pred_label_list]), axis=0)
        acc_score = accuracy_score(gt_label, pred_label)

        if num_cls>2:
            pred_score = np.concatenate(pred_score_list, axis=0)   
            gt_label = F.one_hot(torch.tensor(gt_label), num_classes=num_cls).numpy()
            auc_score = roc_auc_score(gt_label, pred_score)
        else:
            pred_score = np.concatenate(tuple([_ for _ in pred_score_list]), axis=0)    
            auc_score = roc_auc_score(gt_label, pred_score)

        return acc_score, auc_score
    def get_feat(self, inter_feats, a_feats, v_feats, gt_label_list, pred_scores=None, path_list=None):
        inter_feats_np = np.concatenate(inter_feats, axis=0)
        a_feats_np = np.concatenate(a_feats, axis=0)
        v_feats_np = np.concatenate(v_feats, axis=0)
        gt_label_np = np.concatenate(gt_label_list, axis=0)
        pred_scores = np.concatenate(pred_scores, axis=0)

        if 'val' in args.checkpointspath:  split ='val'
        if 'test' in args.checkpointspath:  split ='test'

        checkPath(os.path.join(args.Path_checkpoints, args.PATH_save, 'feat'))
        np.save(os.path.join(args.Path_checkpoints, args.PATH_save, 'feat', 'gt_label_%s.npy'%split), gt_label_np)
        np.save(os.path.join(args.Path_checkpoints, args.PATH_save, 'feat', 'inter_feats_%s.npy'%split), inter_feats_np)
        np.save(os.path.join(args.Path_checkpoints, args.PATH_save, 'feat', 'a_feats_%s.npy'%split), a_feats_np)
        np.save(os.path.join(args.Path_checkpoints, args.PATH_save, 'feat', 'v_feats_%s.npy'%split), v_feats_np)
        np.save(os.path.join(args.Path_checkpoints, args.PATH_save, 'feat', 'pred_scores_%s.npy'%split), pred_scores)
        if path_list:
            file_path = open(os.path.join(args.Path_checkpoints, args.PATH_save, 'feat', 'path_%s.txt'%split), 'a+')
            for path_ in path_list:
                file_path.write(path_+ '\n')
            file_path.close()


    def get_score(self, preds, output_dim=2, cls_thr=0.5 ):
        if output_dim==2:
            pred_score = F.softmax(preds,dim=1)[:, 1].data.cpu().numpy()  if args.is_out_by_softmax==0 else preds[:, 1].data.cpu().numpy()
            pred_label = preds.data.cpu().max(1)[1].numpy()
        elif output_dim==1:
            pred_score = torch.sigmoid(preds).data.cpu().numpy()
            pred_label = (torch.sigmoid(preds)>cls_thr).data.cpu().numpy()
        elif output_dim==4:
            pred_score = F.softmax(preds,dim=-1).data.cpu().numpy()  if args.is_out_by_softmax==0 else preds.data.cpu().numpy()
            pred_label = preds.data.cpu().max(1)[1].numpy()
        else:
            logger_handle.warning('##check output_dim:\n   %s'%(output_dim))
        return pred_score, pred_label



class Trainer(helper):
    def __init__(self, train_dataset, cfg):
        super(Trainer, self).__init__()
        self.split = 'train'
        if not self.use_cuda:    logger_handle.warning('Cuda is not available, only cpu is used to train the model...')

        self.train_loader, num_videos = train_dataset
        logger_handle.info('Dataset used: %s. Number of videos: %s. File: %s' % ( args.database, num_videos, args.train_file))    # len(train_dataset), 
        self.MODEL_CFG = cfg.MODEL_CFG

    def checkGrad(self, model):
        name_check = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                block_name = name.split('.')[0]
                if block_name not in name_check:
                    logger_handle.warning('##requires_grad False: %s branch' % (block_name))
                    name_check.append(block_name)

    def check_data(self, ):
        print('##start check data')
        for batch_idx, samples in enumerate(tqdm(self.train_loader,disable=is_tqdm)):
            input_v, input_a, label = samples['video_aug'], samples['audio_aug'], samples['label']
            input_v.shape
            input_a.shape
            label.shape
        print('## check done.')

    def start(self, model, clientValer, clientTester, start_epoch=0):
        global best_auc
        global best_acc

        end_epoch = self.optimizer_cfg['max_epochs']
        optimizer = BuildOptimizer(model, copy.deepcopy(cfg.OPTIMIZER_CFG))
        if args.is_lr_scheduler == 1:
            scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=3, verbose=True, min_lr=1e-8)

        if self.use_cuda: model.cuda()

        self.checkGrad(model)

        learning_rate = self.optimizer_cfg[self.optimizer_cfg['type']]['learning_rate']
        num_iters, max_iters, data_nums = start_epoch * len(self.train_loader), end_epoch * len(self.train_loader), len(self.train_loader)
        eval_count = 0
        eval_interval = int(len(self.train_loader))
        iters_to_accumulate = args.iters_accumulate

        scaler = GradScaler(enabled=enabled_amp)   # torch.cuda.amp.GradScaler()

        if args.is_checkdata == 1:
            self.check_data()

        for epoch in range(start_epoch, end_epoch):
            model.train()
            loss_epoch, loss_stage = 0, 0
            pred_score_list, gt_label_list, pred_label_list = [], [], []
            time0 = time.time()
            for batch_idx, samples in enumerate(tqdm(self.train_loader, disable=is_tqdm)):
                input_v, input_a = samples['video_aug'].type(self.FloatTensor), samples['audio_aug'].type(self.FloatTensor)
                label = self.get_label(samples, args.num_cls, is_label_dict=args.is_label_dict)
                if args.is_adjustLR==1 and args.is_lr_scheduler == 0:
                    self.optimizer_cfg['policy']['opts'].update({'num_iters': num_iters, 'max_iters': max_iters, 'num_epochs': epoch })
                    learning_rate = adjustLearningRate(optimizer, self.optimizer_cfg, batch_idx, data_nums)
                if args.net_type == 'PVASS':
                    input_v = [samples['video'].type(self.FloatTensor), input_v]
                if args.is_step2 >= 1 :
                    with autocast():
                        preds, loss = model(input_v, input_a, label, 'train')            
                        if num_gpu > 1:    
                            loss = loss.mean()
                        loss = loss/iters_to_accumulate
                    pred_score, pred_label = self.get_score(preds, output_dim=args.num_cls)
                    pred_score_list.append(pred_score)
                    pred_label_list.append(pred_label)
                    gt_label_list.append(label['label'].cpu().numpy()) if isinstance(label, dict) else gt_label_list.append(label.cpu().numpy())

                else:
                    with autocast():
                        _, loss = model(input_v, input_a, label, 'train')           
                        if num_gpu > 1:    loss = loss.mean()
                        loss = loss/iters_to_accumulate
                scaler.scale(loss).backward()       # loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if (batch_idx+1)%iters_to_accumulate == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                loss_epoch += loss.item()
                num_iters += 1

                if (batch_idx+1)%eval_interval == 0 and args.is_eval_val == 1:
                    eval_count+=1
                    if args.is_step2 == 0:     self.save_model('train', model, epoch=epoch)
                    else:
                            G_test_sets = self.get_gtest_sets(args.database)
                            for G_test_set in G_test_sets:
                                _, acc_score, auc_score = clientTester.start(model, gtest_set=G_test_set)
                                self.save_model('test', model, epoch, acc_score, auc_score)

            time1 = time.time()
            if args.is_step2 >= 1 :
                acc_score, auc_score = self.get_perf(pred_score_list, gt_label_list, pred_label_list, num_cls=args.num_cls)
                logger_handle.info('[train][{}/{}] [{:0.1f}s]  LR:{:0.8f} || loss:{:0.6f}  ACC:{:0.4f} AUC:{:0.4f}'.format( \
                                    epoch, end_epoch, time1-time0, learning_rate, loss_epoch/len(self.train_loader), acc_score, auc_score))
            else:
                logger_handle.info('[train][{}/{}] [{:0.1f}s]  LR:{:0.8f} || loss:{:0.6f} '.format(epoch, \
                                    end_epoch, time1-time0, learning_rate, loss_epoch/len(self.train_loader)))

            if args.is_eval_val == 1:
                _, acc_score, auc_score = clientValer.start(model)
                self.save_model('val', model, epoch, acc_score, auc_score)

            if args.is_eval_test == 1:
                G_test_sets = self.get_gtest_sets(args.database)
                for G_test_set in G_test_sets:
                    _, acc_score, auc_score = clientTester.start(model, gtest_set=G_test_set)
                    self.save_model('test', model, epoch, acc_score, auc_score)
                logger_handle.info('')

            self.save_model('End', model, epoch)

            if args.is_lr_scheduler == 1:  
                scheduler.step(loss)       
                learning_rate = scheduler.optimizer.param_groups[-1]['lr']

    def save_model(self, split, model, epoch=0, acc_score=0.0, auc_score=0.0):
        global best_auc
        global best_acc
        if args.use_tensorboard == 1:
            writer.add_scalar('%s_acc_score'%(split), acc_score, epoch)
            writer.add_scalar('%s_auc_score'%(split), auc_score, epoch)
        testLabel = False

        if split == 'End':
            if args.is_save == 1 :
                if num_gpu > 1:    net_ = model.module.state_dict()
                else:              net_ = model.state_dict()
                state_dict = {'model': net_, 'epoch': epoch, 
                            #   'val_auc_best': best_auc['val'], 'val_acc_best': best_acc['val'], 
                            'auc': best_auc['test'], 'acc': best_acc['test']}
                torch.save(state_dict, os.path.join(args.Path_checkpoints, args.PATH_save, 'net_end.pth'))
            return testLabel

        if args.is_step2 >= 1 :
            if auc_score > best_auc[split]:   
                best_auc[split] = auc_score
                if split == 'val':   testLabel = True
                if ( split == 'val' or split == 'test' ) and args.is_save == 1:
                    state_dict = {  'epoch': epoch,  'auc': auc_score, 'acc': acc_score,
                                    'model': model.module.state_dict() if num_gpu>1 else model.state_dict(),
                    }
                    torch.save(state_dict, os.path.join(args.Path_checkpoints, args.PATH_save, 'net_best_'+split+'_auc.pth'))
                logger_handle.info('**   %s  auc best     **'%(split))

            if acc_score > best_acc[split]:    
                best_acc[split] = acc_score
                if split == 'val':   testLabel = True
                if ( split == 'val' or split == 'test' ) and args.is_save == 1:
                    state_dict = {  'epoch': epoch,  'auc': auc_score, 'acc': acc_score,
                                    'model': model.module.state_dict() if num_gpu>1 else model.state_dict(),
                    }
                    torch.save(state_dict, os.path.join(args.Path_checkpoints, args.PATH_save, 'net_best_'+split+'_acc.pth'))
                logger_handle.info('**   %s  acc best     **'%(split))
        else:
            if epoch % 10 == 0 and args.is_save == 1:
                if num_gpu > 1: net_ = model.module.state_dict()
                else:           net_ = model.state_dict()
                state_dict = { 'epoch': epoch,  'model': net_, }
                torch.save(state_dict, os.path.join(args.Path_checkpoints, args.PATH_save, 'net_'+str(epoch)+'.pth'))
        return testLabel


class Valer(helper):
    def __init__(self, cfg):
        super(Valer, self).__init__()
        datasets = VADataLoader(args)  
        self.val_loader, num_videos = datasets.val_dataloader()
        logger_handle.info('Dataset used: val. Number of videos: %s. File: %s' % (num_videos, args.val_file)) # len(self.val_loader)

    def start(self, model):
        if self.use_cuda: model.cuda()
        model.eval()
        with torch.no_grad():
            loss_epoch = 0
            pred_score_list, gt_label_list, pred_label_list  = [], [], []
            inter_feats, a_feats, v_feats, pred_scores, path_list = [], [], [], [], []
            time0 = time.time()

            for batch_idx, samples in enumerate(tqdm(self.val_loader, disable=is_tqdm)):
                input_v, input_a = samples['video'].type(self.FloatTensor), samples['audio'].type(self.FloatTensor)
                label = self.get_label(samples, args.num_cls, is_label_dict=args.is_label_dict)
                if args.net_type == 'PVASS':
                    input_v = [input_v, samples['video_aug'].type(self.FloatTensor)]
                with autocast(enabled=enabled_amp):
                    preds, loss = model(input_v, input_a, label, 'val')  ##preds: torch.Size([256, 2])       ##label: torch.Size([256])
                if num_gpu > 1:  loss = loss.mean()

                if args.is_get_feat == 0:
                    if args.is_step2 >= 1:
                        pred_score, pred_label = self.get_score(preds, output_dim=args.num_cls )
                        pred_score_list.append(pred_score)
                        pred_label_list.append(pred_label)
                        gt_label_list.append(label['label'].cpu().numpy()) if isinstance(label, dict) else gt_label_list.append(label.cpu().numpy())      ##
                else:
                    inter_feats.append(preds['feat_inter'].cpu().numpy())
                    a_feats.append(preds['a_feat_intra'].cpu().numpy())
                    v_feats.append(preds['v_feat_intra'].cpu().numpy())
                    path_list.extend(label['path'])
                    if 'pred_total' in preds.keys(): pred_scores.append(preds['pred_total'].cpu().numpy())
                    gt_label_list.append(label['label'].cpu().numpy()) if isinstance(label, dict) else gt_label_list.append(label.cpu().numpy())

                loss_epoch += loss.item()

        if args.is_get_feat == 1:
            self.get_feat(inter_feats, a_feats, v_feats, gt_label_list, pred_scores, path_list); return 

        time1 = time.time()
        if args.is_step2 >= 1:
            acc_score, auc_score = self.get_perf(pred_score_list, gt_label_list, pred_label_list, num_cls=args.num_cls, split='val')
            logger_handle.info('[Val  ] [{:0.1f}s] || LOSS:{:0.6f}  ACC:{:0.4f} AUC:{:0.4f}  '.format(time1-time0, \
                            loss_epoch/len(self.val_loader), acc_score, auc_score))
            return loss_epoch/len(self.val_loader), acc_score, auc_score
        else:
            logger_handle.info('[Val  ] [{:0.1f}s] || LOSS:{:0.6f}  '.format(time1-time0, \
                            loss_epoch/len(self.val_loader)))
            return loss_epoch/len(self.val_loader),


class Tester(helper):
    def __init__(self, cfg):
        super(Tester, self).__init__()

    def start(self, model, gtest_set):
        if self.use_cuda: model.cuda()

        datasets = VADataLoader(args)  
        test_loader, num_videos = datasets.test_dataloader(gtest_set)
        logger_handle.info('Dataset used: %s. Number of videos: %s, File: %s' % (gtest_set, num_videos, args.test_file))  # len(test_loader)

        model.eval()
        with torch.no_grad():
            loss_epoch = 0
            pred_score_list, gt_label_list, pred_label_list = [], [], []
            time0 = time.time()
            video_index_list = []
            inter_feats, a_feats, v_feats, pred_scores, path_list = [], [], [], [], []

            for batch_idx, samples in enumerate(tqdm(test_loader,disable=is_tqdm)):
                input_v, input_a = samples['video'].type(self.FloatTensor), samples['audio'].type(self.FloatTensor)
                label = self.get_label(samples, args.num_cls, is_label_dict=args.is_label_dict)
                video_index = samples['video_index']
                if args.net_type == 'PVASS':
                    input_v = [input_v, samples['video_aug'].type(self.FloatTensor)]
                with autocast():
                    preds, loss = model(input_v, input_a, label, 'test')
                if num_gpu > 1:
                    loss = loss.mean()

                if args.is_get_feat == 0:
                    if args.is_step2 >= 1:
                        pred_score, pred_label = self.get_score(preds, output_dim=args.num_cls )
                        pred_score_list.append(pred_score)
                        pred_label_list.append(pred_label)
                        gt_label_list.append(label['label'].cpu().numpy()) if isinstance(label, dict) else gt_label_list.append(label.cpu().numpy())        ## changed for bimodal detection
                        video_index_list.append(video_index.numpy())
                    else:
                        pred_score, pred_label = self.get_score(preds, output_dim=args.num_cls )
                        gt_label_list.append(label['label'].cpu().numpy()) if isinstance(label, dict) else gt_label_list.append(label.cpu().numpy())
                else:
                    inter_feats.append(preds['feat_inter'].cpu().numpy())
                    a_feats.append(preds['a_feat_intra'].cpu().numpy())
                    v_feats.append(preds['v_feat_intra'].cpu().numpy())
                    path_list.extend(label['path'])
                    if 'pred_total' in preds.keys(): pred_scores.append(preds['pred_total'].cpu().numpy())
                    gt_label_list.append(label['label'].cpu().numpy()) if isinstance(label, dict) else gt_label_list.append(label.cpu().numpy())

                loss_epoch += loss.item()
        if args.is_get_feat == 1:
            self.get_feat(inter_feats, a_feats, v_feats, gt_label_list, pred_scores, path_list); return 

        acc_score, auc_score = self.get_perf(pred_score_list, gt_label_list, pred_label_list, num_cls=args.num_cls, video_index_list=video_index_list, split='test')
        time1 = time.time()
        logger_handle.info('[Test ] [{:0.1f}s] || LOSS:{:0.6f}  ACC:{:0.4f} AUC:{:0.4f}'.format(time1-time0, \
                            loss_epoch/len(test_loader), acc_score, auc_score))

        return loss_epoch/len(test_loader), acc_score, auc_score



def main():
    global cfg
    cfg = set_cfg(cfg, args, logger_handle)
    pynvml.nvmlInit()
    for idx in range(num_gpu):
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # len()
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        GPU_total = meminfo.total/1024/1024/1024
        GPU_used = meminfo.used/1024/1024/1024
        GPU_free = meminfo.free/1024/1024/1024
        logger_handle.info('##GPU-%s info## total:%0.2fGB, used:%0.2fGB, free:%0.2fGB ' % (idx, GPU_total, GPU_used, GPU_free))

    if args.is_test == 0:
        model = BuildModel(cfg=copy.deepcopy(cfg.MODEL_CFG), mode='TRAIN')
        start_epoch = 0
        if args.checkpointspath and os.path.exists(args.checkpointspath):
            checkpoints =  torch.load(args.checkpointspath)
            try:
                model.load_state_dict(checkpoints['model'])
            except:
                model.load_state_dict(checkpoints['model'], strict=False)
                logger_handle.warning('##load via strict=False')
            if args.is_step2 == 0 or args.is_step2 == 2:
                start_epoch  = checkpoints['epoch']
                best_auc['test'], best_acc['test'] = checkpoints['auc'], checkpoints['acc']
            else:
                start_epoch, best_auc['test'], best_acc['test'] = checkpoints['epoch']+1, checkpoints['auc'], checkpoints['acc']
            logger_handle.info('##load %s successfully in %s epoch' % (args.checkpointspath, start_epoch))
        else:
            if args.is_step2 == 0:
                # model = load_pretrained_dict_v0p2(start_epoch, model=model, args=args, logger=logger_handle)
                logger_handle.info('##check weight transfer..(001)'); exit()
            else:
                if "FauForensics" in args.net_type:
                    model = load_pretrained_weights(start_epoch, model=model, args=args, logger=logger_handle) ## imp

        if num_gpu > 1:      model = nn.DataParallel(model)
        datasets = VADataLoader(args)  # 初始化
        train_dataset = datasets.train_dataloader()
        trainer = Trainer(train_dataset, cfg)
        valer = Valer(cfg)
        tester = Tester(cfg)
        trainer.start(model, valer, tester, start_epoch)

    elif args.is_test == 1:
        logger_handle.info('## eval val set')
        model = BuildModel(cfg=copy.deepcopy(cfg.MODEL_CFG), mode='val')
        if args.checkpointspath and os.path.exists(args.checkpointspath):
            checkpoints =  torch.load(args.checkpointspath)
            model.load_state_dict(checkpoints['model'])
            logger_handle.info('##load %s successfully' % (args.checkpointspath))
        else:
            logger_handle.warning('##not load pretrained model from %s' % (args.checkpointspath)); exit()
        valer = Valer(cfg)
        valer.start(model)
    else:
        model = BuildModel(cfg=copy.deepcopy(cfg.MODEL_CFG), mode='test')
        if args.checkpointspath and os.path.exists(args.checkpointspath):
            model.load_state_dict(torch.load(args.checkpointspath)['model'])
            logger_handle.info('##load %s successfully' % (args.checkpointspath))
        else:
            logger_handle.warning('##not load pretrained model from %s' % (args.checkpointspath)); exit()
        tester = Tester(cfg)
        gtest_sets = tester.get_gtest_sets(args.database)
        logger_handle.info('## eval test set: ' + ' | '.join(_ for _ in gtest_sets))
        for gtest_set in gtest_sets:
            tester.start(model, gtest_set)  # model
            print()

if __name__ == '__main__':

    from logger import Logger
    logger_handle = Logger(os.path.join(args.Path_checkpoints, args.PATH_save, 'train.log'))
    num_gpu = torch.cuda.device_count()
    args.gpus = num_gpu
    from utils import MMLoader as VADataLoader
    from utils import *
    import utils.cfg as cfg

    ## random seed setting:
    if args.is_test >= 1:       seed = int( input("please enter random seed:") ) 
    else:                       seed = np.random.randint(0,12000) if args.seed == 0  else args.seed
    fix_seed(seed)

    logger_handle.info('##starting... ')
    logger_handle.info('##MMLoader for loading data ')
    logger_handle.info('##random seed: %s'%(seed))

    main()


    ## training end:
    if args.use_tensorboard == 1:
        writer.close()
    logger_handle.info('##training done...')
