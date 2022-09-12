import numpy as np
from tqdm import tqdm
from glob import glob
import os
import pickle

import shutil

import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.backends import cudnn
import numpy as np

from datetime import datetime

from agents.base import BaseAgent
from graphs.losses.mlsoft import MultiLabelSoftmax, HingeCalibratedRanking
from graphs.losses.asl import AsymmetricLossOptimized
from graphs.models.multi_scale_parallel import MultiScaleParallelNet
import graphs.models.resnet as resnet

from datasets.multi_scale_loader import MultiScaleLIDCLoader

from utils import AverageMeter
from utils.indices import multi_label_metrics
from utils.pytorchtools import EarlyStopping

cudnn.benchmark = True
np.set_printoptions(precision=4, suppress=True)

# begin
class MLSLNetLIDCAgent(BaseAgent):
    def __init__(self, config):
        super().__init__(config)
        if self.config.mode == 'cross-validation':
            self.data_loader = MultiScaleLIDCLoader(self.config)
        else:
            raise NotImplementedError
        self.config.pretrain = os.path.join(self.config.pretrain, '%s%s.pth' % (self.config.backbone, self.config.model_depth))

        if self.config.base_loss == 'ml_softmax':
            self.loss = MultiLabelSoftmax(gamma_pos=self.config.mlsl_gamma_pos, gamma_neg=self.config.mlsl_gamma_neg)
        elif self.config.base_loss == 'asl':
            self.loss = AsymmetricLossOptimized(gamma_neg=self.config.asl_gamma_neg, gamma_pos=self.config.asl_gamma_pos, clip=self.config.asl_clip)
        elif self.config.base_loss == 'calibrated_hinge':
            self.loss = HingeCalibratedRanking()
        else:
            raise NotImplementedError

        self.current_fold = 0
        self.best_model = self.config.best_model

        self.is_cuda = torch.cuda.is_available()
        self.cuda = self.is_cuda & self.config.cuda
        if self.cuda:
            self.device = torch.device("cuda")
            torch.cuda.manual_seed_all(self.config.seed)
            torch.cuda.set_device(self.config.gpu_device)
            self.logger.info("Operation will be on **********GPU-CUDA*********")
        else:
            self.device = torch.device("cpu")
            torch.manual_seed(self.config.seed)
            self.logger.info("Operation will be on *****CPU****** ")
        self._init_one_fold()
        self.loss = self.loss.to(self.device)
        self.load_checkpoint(self.config.checkpoint_file)
    
    def _init_one_fold(self):
        self.current_epoch = 0
        self.best_metric = 0
        self.best_epoch = 0

        if self.config.multi_scale_method == 'ensemble_network': # train and test on multiple scale images
            self.model = MultiScaleParallelNet(self.config)
        elif self.config.multi_scale_method == 'single_network':
            # using 3D ResNet by default 
            Model = getattr(resnet, self.config.backbone + str(self.config.model_depth))
            self.model = Model(num_classes=self.config.num_classes)
            if self.config.finetune:
                    self.model.load_pretrain(self.config.pretrain)
        else:
            raise NotImplementedError
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.config.learning_rate, momentum=0.9, weight_decay=self.config.weight_decay)
        self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', factor=0.1, patience=16)
        self.early_stop = EarlyStopping(patience=40, delta=0., trace_func=self.logger.info)
        self.model = self.model.to(self.device)

    def save_checkpoint(self, filename='ckpts.pth.tar', is_best=False):
        if not is_best:
            state = {
                'fold': self.current_fold, 
                'epoch': self.current_epoch, 
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(), 
            }
        else:
            state = {'state_dict': self.model.state_dict()}
        torch.save(state, self.config.checkpoint_dir + filename)

    def load_checkpoint(self, filename):
        filename = os.path.join(self.config.checkpoint_dir, filename)
        try:
            self.logger.info("Loading checkpoint '{}'".format(filename))
            ckpt = torch.load(filename)
            is_best = False if 'optimizer' in ckpt else True
            if not is_best:
                self.current_fold = ckpt['fold']
                self.current_epoch = ckpt['epoch']
                self.model.load_state_dict(ckpt['state_dict'])
                self.optimizer.load_state_dict(ckpt['optimizer'])
            else:
                self.logger.info("*** Load checkpoint from the best model! ***")
                self.model.load_state_dict(ckpt['state_dict'])
            self.logger.info("Checkpoint loaded successfully from {}\n".format(self.config.checkpoint_dir))
        except OSError as e:
            self.logger.info("No checkpoint exists from '{}'. Skipping...".format(self.config.checkpoint_dir))
            self.logger.info("**** First time to train ****")

    def run(self):
        if self.config.mode == 'test':
            pass
        elif self.config.mode == 'cross-validation':
            # cross valid experiments data split root.
            self.cross_fold = glob(os.path.join(self.config.CV_ROOT, 'fold*'))
            self.cross_fold.sort()
            
            start = datetime.now()
            for i in range(self.current_fold, len(self.cross_fold)):
                self.current_fold = i
                self.train_loader, self.val_loader, self.test_loader = self.data_loader.get_current_loader(self.cross_fold[i])
                self.train()
                self.load_checkpoint(self.best_model)
                self.test()
                self.logger.info('Execute time: {}'.format(datetime.now() - start))
                if i != len(self.cross_fold) -  1:
                    self._init_one_fold()

            results, all_auc = [], []
            predfiles = glob(self.config.out_dir + 'test_preds_f*.pkl')
            for each in predfiles:
                with open(each, 'rb') as f:
                    cur = pickle.load(f)
                prob, targ = cur['probs'], cur['targs']
                ml_res, aucs = multi_label_metrics(prob, targ)
                results.append(ml_res)
                all_auc.append(aucs)
            results = np.asarray(results) * 100
            all_auc = np.asarray(all_auc) * 100 
            out_str = ' ~~~~~~~~~~~~~~ Final results! ~~~~~~~~~~~~~~~\n'
            np.set_printoptions(precision=2, suppress=True)
            out_str +='Multi-label: subacc, hloss, rloss, avgprec, rec, prec, f1, auc\n'
            out_str +='~~~~ MEAN: {}\n~~~~  STD: {}\n'.format(results.mean(0), results.std(0))
            out_str +='AUC results: \n'
            out_str +='~~~~ MEAN: {}\n~~~~  STD: {}\n'.format(all_auc.mean(0), all_auc.std(0))
            self.logger.info(out_str)
        else:
            raise NotImplementedError

    def train(self):
        for epoch in range(self.current_epoch, self.config.max_epoch):
            self.current_epoch = epoch
            start = datetime.now()
            self.train_one_epoch()
            val_loss = self.validate(start)
            self.scheduler.step(val_loss)
            self.save_checkpoint()
            self.early_stop(val_loss)
            if self.early_stop.early_stop:
                self.logger.info('Early stopping!')
                break

    def train_one_epoch(self):
        self.model.train()
        for idx, sample in enumerate(self.train_loader):
            y = sample['label'].to(self.device)
            X = sample['data']
            if X.dim() == 5:
                X = X.to(self.device)
                outs = self.model(X)
            else:
                bs, crops, c, d, h, w = X.size()
                X = torch.autograd.Variable(X.view(-1, c, d, h, w), requires_grad=False).to(self.device)
                outs = self.model(X).view(bs, crops, -1).mean(1)
            loss = self.loss(outs, y)
            if np.isnan(float(loss.item())):
                import pdb; pdb.set_trace()
                raise ValueError('Loss is nan during training...')
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def validate(self, start_time):
        self.model.eval()
        T, P = [], []
        epoch_loss = AverageMeter()
        with torch.no_grad():
            for sample in self.val_loader:
                X = sample['data']
                y = sample['label'].to(self.device)
                if X.dim() == 5:
                    X = X.to(self.device)
                    outs = self.model(X)
                else:
                    bs, crops, c, d, h, w = X.size()
                    X = X.view(-1, c, d, h, w).to(self.device)
                    outs = self.model(X).view(bs, crops, -1).mean(1)
                loss = self.loss(outs, y)
                prob = torch.sigmoid(outs)
                epoch_loss.update(loss.item())
                T.append(y.cpu())
                P.append(prob.cpu())
            T = torch.cat(T, dim=0)
            P = torch.cat(P, dim=0)
            res, auc = multi_label_metrics(P, T)
        mauc = res[-1]
        if mauc < self.best_metric: 
            self.best_metric = mauc
            self.best_epoch = self.current_epoch
            self.save_checkpoint(filename=self.best_model, is_best=True)
            current_file = glob(self.config.out_dir + 'val_preds_f{}_*.pkl'.format(self.current_fold))
            if len(current_file) > 0:
                os.remove(current_file[0])
            newfile = self.config.out_dir + 'val_preds_f{}_0{}.pkl'.format(self.current_fold, int(mauc*1000))
            with open(newfile, 'wb') as f:
                pickle.dump((P, T), f)
        out_lines = 'Fold-{} Epoch-{}: Loss {:.4f} | Multi-label {} | LR {:.6f} | Best {:.4f} (@{}) | Time {} |'.format(
            self.current_fold, str(self.current_epoch).zfill(3), epoch_loss.val, res, self.optimizer.state_dict()['param_groups'][0]['lr'], 
            self.best_metric, str(self.best_epoch).zfill(3), (datetime.now() - start_time).seconds)
        self.logger.info(out_lines)
        return epoch_loss.val


    def test(self):
        tqdm_batch = tqdm(self.test_loader, total=self.data_loader.test_iters, desc=' * Testing * ')
        self.model.eval()
        T, P  = [], []
        with torch.no_grad():
            for sample in tqdm_batch:
                X = sample['data']
                y = sample['label'].to(self.device)
                if X.dim() == 5:
                    X = X.to(self.device)
                    logits = self.model(X)
                else:
                    bs, crops, c, d, h, w = X.size()
                    X = X.view(-1, c, d, h, w).to(self.device)
                    logits = self.model(X).view(bs, crops, -1).mean(1)
                prob = torch.sigmoid(logits)
                T.append(y.cpu())
                P.append(prob.cpu())
            T = torch.cat(T, dim=0)
            P = torch.cat(P, dim=0)
            res, auc = multi_label_metrics(P, T)
        self.logger.info('*************** FINAL RESULTS **************')
        self.logger.info('Multi label: {}'.format(res))
        self.logger.info('AUCs: {}'.format(auc))

        final_file = self.config.out_dir + 'test_preds_f{}.pkl'.format(self.current_fold)
        save_data = {'probs': P, 'targs': T}
        with open(final_file, 'wb') as f:
            pickle.dump(save_data, f)
        tqdm_batch.close()

    def finalize(self):
        pass



