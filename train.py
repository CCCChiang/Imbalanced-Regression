import os
import numpy as np
import pandas as pd
import time
from tqdm import tqdm
import torch
import torch.nn as nn
from scipy.stats import gmean
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import torch.backends.cudnn as cudnn
import seaborn as sns
import matplotlib.pyplot as plt
from dataloader import RBMURA
from utils import DataClean, AverageMeter, ProgressMeter, adjust_learning_rate, save_checkpoint
from model import ModelBuild
from evaluation import weighted_l1_loss, weighted_mse_loss, weighted_focal_mse_loss, weighted_focal_l1_loss, weighted_huber_loss

class Exp:
    def __init__(self, args):
        self.args = args
        

    def train(self):

        if self.args.gpu is not None:
            print(f"Use GPU: {self.args.gpu} for training")

        # Data
        print('=====> Preparing data...')
        clean = DataClean()
        # data = pd.read_csv(self.args.data_dir, encoding='big5', low_memory=False)

        # data = clean.choose_float(data)
        # data = clean.drop_No(data)
        # data = clean.drop_unit(data)
        # data = clean.drop_miss(data, 0.7)
        # data = clean.drop_dup(data)
        # data = clean.drop_na(data)
        # data.to_csv("D:/UserData/AndyCCChiang/Boston_Housing_dataset/data/processdata.csv")
        data = pd.read_csv("D:/UserData/AndyCCChiang/Boston_Housing_dataset/data/processdata.csv", encoding='big5', low_memory=False)
        # data = clean.drop_JND_greater_digit(data, 4.3, "Defect Value")
        y = np.array(data.iloc[:, 0])
        X = np.array(data.iloc[:, 1:])
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=0)
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 0.3, random_state=0)
        train_labels = y_train
        self.training = y_train
        pd.DataFrame(x_train).to_csv("D:/UserData/AndyCCChiang/Boston_Housing_dataset/data/x_train.csv",index=False)
        pd.DataFrame(y_train).to_csv("D:/UserData/AndyCCChiang/Boston_Housing_dataset/data/y_train.csv",index=False)
        pd.DataFrame(x_val).to_csv("D:/UserData/AndyCCChiang/Boston_Housing_dataset/data/x_val.csv",index=False)
        pd.DataFrame(y_val).to_csv("D:/UserData/AndyCCChiang/Boston_Housing_dataset/data/y_val.csv",index=False)
        pd.DataFrame(x_test).to_csv("D:/UserData/AndyCCChiang/Boston_Housing_dataset/data/x_test.csv",index=False)
        pd.DataFrame(y_test).to_csv("D:/UserData/AndyCCChiang/Boston_Housing_dataset/data/y_test.csv",index=False)
        train_dataset = RBMURA(x_train, y_train, split='train', reweight=self.args.reweight, lds=self.args.lds, lds_kernel=self.args.lds_kernel, lds_ks=self.args.lds_ks, lds_sigma=self.args.lds_sigma)
        self.W = train_dataset.weights
        val_dataset = RBMURA(x_val, y_val, split='val')
        test_dataset = RBMURA(x_test, y_test, split='test')
#1251 #537 #448
        train_loader = DataLoader(train_dataset, batch_size=self.args.batch_size[0], shuffle=True,
                                    num_workers=self.args.workers, pin_memory=False, drop_last=False)
        val_loader = DataLoader(val_dataset, batch_size=self.args.batch_size[1], shuffle=False,
                                num_workers=self.args.workers, pin_memory=True, drop_last=False)
        test_loader = DataLoader(test_dataset, batch_size=self.args.batch_size[2], shuffle=False,
                                    num_workers=self.args.workers, pin_memory=True, drop_last=False)
        print(f"Training data size: {len(train_dataset)}")
        print(f"Validation data size: {len(val_dataset)}")
        print(f"Test data size: {len(test_dataset)}")

        config = dict(
            fds=self.args.fds, bucket_num=self.args.bucket_num, bucket_start=self.args.bucket_start,
            start_update=self.args.start_update, start_smooth=self.args.start_smooth,
            kernel=self.args.fds_kernel, ks=self.args.fds_ks, sigma=self.args.fds_sigma, momentum=self.args.fds_mmt)
        
        modelbuild = ModelBuild(self.args.model)
        model = modelbuild.build(x_train.shape[1], **config)
        if self.args.model == "LR":
            for idx, (inputs, targets, weights) in enumerate(train_loader):
                pass
            if weights is None:
                model.fit(inputs.numpy(), targets.numpy())
            else:
                model.fit(inputs.numpy(), targets.numpy(), sample_weight=weights.numpy().ravel())
        
            batch_time = AverageMeter('Time', ':6.3f')
            losses_mse = AverageMeter('Loss (MSE)', ':.3f')
            losses_l1 = AverageMeter('Loss (L1)', ':.3f')
            progress = ProgressMeter(
                len(test_loader),
                [batch_time, losses_mse, losses_l1],
                prefix=f'Test'
            )

            criterion_mse = nn.MSELoss()
            criterion_l1 = nn.L1Loss()
            criterion_gmean = nn.L1Loss(reduction='none')

            losses_all = []
            preds, labels = [], []
            for idx, (pred_inputs, pred_targets, _)  in enumerate(test_loader):
                outputs = model.predict(pred_inputs.numpy())

                preds.extend(outputs)
                labels.extend(pred_targets.numpy())

                loss_mse = criterion_mse(torch.tensor(outputs.reshape(-1, 1)).squeeze(), pred_targets)
                loss_l1 = criterion_l1(torch.tensor(outputs.reshape(-1, 1)).squeeze(), pred_targets)
                loss_all = criterion_gmean(torch.tensor(outputs.reshape(-1, 1)).squeeze(), pred_targets)
                losses_all.extend(loss_all.cpu().numpy())

                losses_mse.update(loss_mse.item(), inputs.size(0))
                losses_l1.update(loss_l1.item(), inputs.size(0))

                if idx % self.args.print_freq == 0:
                    progress.display(idx)
            loss_gmean = gmean(np.hstack(losses_all), axis=None).astype(float)
            print(f" * Overall: MSE {losses_mse.avg:.3f}\tL1 {losses_l1.avg:.3f}\tG-Mean {loss_gmean:.3f}")

            plt.plot(labels, label='True values in {} set'.format('Test'))
            plt.plot(preds, label='Pred. values in {} set'.format('Test'))
            plt.legend()

        else:
            def train(train_loader, model, optimizer, epoch):
                batch_time = AverageMeter('Time', ':6.2f')
                data_time = AverageMeter('Data', ':6.4f')
                losses = AverageMeter(f'Loss ({self.args.loss.upper()})', ':.3f')
                progress = ProgressMeter(
                    len(train_loader),
                    [batch_time, data_time, losses],
                    prefix="Epoch: [{}]".format(epoch)
                )

                model.train()
                end = time.time()
                for idx, (inputs, targets, weights) in enumerate(train_loader):
                    data_time.update(time.time() - end)
                    if not self.args.cpu_only:
                        inputs, targets, weights = \
                            inputs.cuda(non_blocking=True), targets.cuda(non_blocking=True), weights.cuda(non_blocking=True)
                    inputs = inputs.to(torch.float32)
                    outputs, _ = model(inputs, targets, epoch)

                    loss = globals()[f"weighted_{self.args.loss}_loss"](outputs.squeeze(), targets, weights)
                    assert not (np.isnan(loss.item()) or loss.item() > 1e6), f"Loss explosion: {loss.item()}"

                    losses.update(loss.item(), inputs.size(0))

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    batch_time.update(time.time() - end)
                    end = time.time()
                    # if idx % self.args.print_freq == 0:
                    #     progress.display(idx)
                if self.args.fds and epoch >= self.args.start_update:
                    # print(f"Create Epoch [{epoch}] features of all training data...")
                    encodings, labels = [], []
                    with torch.no_grad():
                        for (inputs, targets, _) in train_loader:
                            # inputs = inputs.cuda(non_blocking=True)
                            inputs = inputs.to(torch.float32)
                            outputs, feature = model(inputs, targets, epoch)
                            encodings.extend(feature.data.squeeze().cpu().numpy())
                            labels.extend(targets.data.squeeze().cpu().numpy())

                    # encodings, labels = torch.from_numpy(np.vstack(encodings)).cuda(), torch.from_numpy(np.hstack(labels)).cuda()
                    encodings, labels = torch.from_numpy(np.vstack(encodings)), torch.from_numpy(np.hstack(labels))

                    # 如果訓練模型的時候如果用的是 torch.nn.DataParalle
                    # 會得到一個“額外的” module
                    # model.module.FDS.update_last_epoch_stats(epoch)
                    # model.module.FDS.update_running_stats(encodings, labels, epoch)
                    model.FDS.update_last_epoch_stats(epoch)
                    model.FDS.update_running_stats(encodings, labels, epoch)
                return losses.avg

            def validate(val_loader, model, train_labels=None, prefix='Val', show_fig=False):
                    batch_time = AverageMeter('Time', ':6.3f')
                    losses_mse = AverageMeter('Loss (MSE)', ':.3f')
                    losses_l1 = AverageMeter('Loss (L1)', ':.3f')
                    progress = ProgressMeter(
                        len(val_loader),
                        [batch_time, losses_mse, losses_l1],
                        prefix=f'{prefix}: '
                    )

                    criterion_mse = nn.MSELoss()
                    criterion_l1 = nn.L1Loss()
                    criterion_gmean = nn.L1Loss(reduction='none')

                    model.eval()
                    losses_all = []
                    preds, labels = [], []
                    with torch.no_grad():
                        end = time.time()
                        for idx, (inputs, targets, _) in enumerate(val_loader):
                            if not self.args.cpu_only:
                                inputs, targets = inputs.cuda(non_blocking=True), targets.cuda(non_blocking=True)
                            inputs = inputs.to(torch.float32)
                            outputs, _ = model(inputs)

                            preds.extend(outputs.data.cpu().numpy())
                            labels.extend(targets.data.cpu().numpy())

                            loss_mse = criterion_mse(outputs.squeeze(), targets)
                            loss_l1 = criterion_l1(outputs.squeeze(), targets)
                            loss_all = criterion_gmean(outputs.squeeze(), targets)
                            losses_all.extend(loss_all.cpu().numpy())

                            losses_mse.update(loss_mse.item(), inputs.size(0))
                            losses_l1.update(loss_l1.item(), inputs.size(0))

                            batch_time.update(time.time() - end)
                            end = time.time()
                            # if idx % self.args.print_freq == 0:
                            #     progress.display(idx)

                        loss_gmean = gmean(np.hstack(losses_all), axis=None).astype(float)
                        # print(f" * Overall: MSE {losses_mse.avg:.3f}\tL1 {losses_l1.avg:.3f}\tG-Mean {loss_gmean:.3f}")
                    
                    if show_fig==True:
                        plt.plot(labels, label='True values in {} set'.format(prefix))
                        plt.plot(preds, label='Pred. values in {} set'.format(prefix))
                        plt.legend()
                    return losses_mse.avg, losses_l1.avg, loss_gmean

            if not self.args.cpu_only:
                model = model.cuda()

            # evaluate only
            if self.args.evaluate:
                assert self.args.resume, 'Specify a trained model using [args.resume]'
                checkpoint = torch.load(self.args.resume)
                model.load_state_dict(checkpoint['state_dict'], strict=False)
                print(f"===> Checkpoint '{self.args.resume}' loaded (epoch [{checkpoint['epoch']}]), testing...")
                validate(test_loader, model, train_labels=train_labels, prefix='Test', show_fig=True)
                return

            # Loss and optimizer
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr) if self.args.optimizer == 'adam' else \
                torch.optim.SGD(model.parameters(), lr=self.args.lr, momentum=self.args.momentum, weight_decay=self.args.weight_decay)

            if self.args.resume:
                if os.path.isfile(self.args.resume):
                    print(f"===> Loading checkpoint '{self.args.resume}'")
                    checkpoint = torch.load(self.args.resume) if self.args.gpu is None else \
                        torch.load(self.args.resume, map_location=torch.device(f'cuda:{str(self.args.gpu)}'))
                    self.args.start_epoch = checkpoint['epoch']
                    self.args.best_loss = checkpoint['best_loss']
                    model.load_state_dict(checkpoint['state_dict'])
                    optimizer.load_state_dict(checkpoint['optimizer'])
                    print(f"===> Loaded checkpoint '{self.args.resume}' (Epoch [{checkpoint['epoch']}])")
                else:
                    print(f"===> No checkpoint found at '{self.args.resume}'")

            if not self.args.cpu_only:
                cudnn.benchmark = True

            for epoch in range(self.args.start_epoch, self.args.epoch):
                adjust_learning_rate(optimizer, epoch, self.args)
                train_loss = train(train_loader, model, optimizer, epoch)
                val_loss_mse, val_loss_l1, val_loss_gmean = validate(val_loader, model, train_labels=train_labels, show_fig=False)

                loss_metric = val_loss_mse if self.args.loss == 'mse' else val_loss_l1
                is_best = loss_metric < self.args.best_loss
                self.args.best_loss = min(loss_metric, self.args.best_loss)
                # print(f"Best {'L1' if 'l1' in self.args.loss else 'MSE'} Loss: {self.args.best_loss:.3f}")
                save_checkpoint(self.args, {
                    'epoch': epoch + 1,
                    'model': self.args.model,
                    'best_loss': self.args.best_loss,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }, is_best)
                # print(f"Epoch #{epoch}: Train loss [{train_loss:.4f}]; "
                #     f"Val loss: MSE [{val_loss_mse:.4f}], L1 [{val_loss_l1:.4f}], G-Mean [{val_loss_gmean:.4f}]")

            # test with best checkpoint
            print("=" * 120)
            print("Test best model on testset...")
            checkpoint = torch.load(f"{self.args.store_root}/{self.args.store_name}/ckpt.best.pth.tar")
            model.load_state_dict(checkpoint['state_dict'])
            print(f"Loaded best model, epoch {checkpoint['epoch']}, best val loss {checkpoint['best_loss']:.4f}")
            test_loss_mse, test_loss_l1, test_loss_gmean = validate(test_loader, model, train_labels=train_labels, prefix='Test', show_fig=True)
            print(f"Test loss: MSE [{test_loss_mse:.4f}], L1 [{test_loss_l1:.4f}], G-Mean [{test_loss_gmean:.4f}]\nDone")


