#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 5 11:00:22 2020

@author: satish
"""
import numpy as np
import os
import sys
import math
import argparse
import random
import glob
import json
from datetime import datetime
import torch
import torch.nn as nn
import torchvision
from torchvision import models
import torch.optim as optim
from tqdm import tqdm
import wandb
# from torch.nn.functional import one_hot

from model.network import Merge_LSTM as net_model
from dataloader.dataloader import thermaldataset
from loss_function.utils_loss import loss_cls
# from loss_function.pearson_loss_1D import Pearson_Correlation

print("PyTorch Version:", torch.__version__)
print("Torchvision Version:", torchvision.__version__)

#Result Directory
try:
        os.mkdir('../results/')
except OSError as exc:
        pass

def main():
        #Input arguments/training settings
        parser = argparse.ArgumentParser()
        parser.add_argument('-e','--epochs',type=int,required=False,default=300, help='Number_of_Epochs')
        parser.add_argument('-lr','--learning_rate',type=float,required=False,default=0.0001, \
                                                help='Learning_Rate')
        parser.add_argument('-ba','--batch_size',type=int,required=False, default=1,help='Batch_Size')
        parser.add_argument('-nw','--workers',type=int,required=False, default=0,help='number of workers')
        parser.add_argument('-seed',type=int,required=False, default=5,help='random seed')
        parser.add_argument('-data',type=str,required=False, default='../data/mat_files/', \
                                                help='data path')
        parser.add_argument('-label',type=str,required=False, default='../data/normalized_label_data/', \
                                                help='label path')
        parser.add_argument('-sync',type=str,required=False, default='../data/sync_data/', \
                                                help='ecg&vid sync')
        parser.add_argument('-phase',type=str,required=False, default='train',help='train/test mode')
        parser.add_argument('-split','--train_val_split', type=float, required=False, default=0.95,\
                                                help='train/test mode')
        parser.add_argument('-min_batch', '--frames_in_GPU',type=int,required=False, default=45, \
                                                help='number of frames per batch from the video to go in GPU')

        #Parameters for existing model reload
        parser.add_argument('-resume',type=str,required=False, default='F',help='resume training')
        parser.add_argument('-hyper_param',type=str,required=False, default='F', \
                                                help='existing hyper-parameters')
        parser.add_argument('-cp','--checkpoint_path',type=str,required=False, \
                                                default='../model_checkpoints_r50/', help='resume training')
        parser.add_argument('-use_wandb', default=False, type=bool)

        #parameters
        args   = parser.parse_args()
        epochs = args.epochs
        l_rate = args.learning_rate
        data   = args.data
        label  = args.label
        sync   = args.sync
        phase  = args.phase
        seed   = args.seed
        split  = args.train_val_split
        fps    = args.frames_in_GPU  #numbers of frames per batch
        batch_size = args.batch_size
        workers = args.workers

        if args.use_wandb:
                wandb.init(project="stressnet", entity="satish1901")

                wandb.config = {
                        "learning_rate": l_rate,
                        "epochs" : epochs,
                        "frame_inGPU": fps,
                        "phase": phase
                }

        #Parameters for exisitng model reload
        c_point_dir = args.checkpoint_path
        resume          = args.resume
        hyper_param = args.hyper_param
        cur_epoch       = 0

        #Fix seed
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed) #sets the seed for random number
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        random.seed(seed)
        
        #Initializing Network & LSTM dimension
        frame_rate = 15; in_dim = frame_rate*2048; h_dim = frame_rate*30; num_l = 6
        print("Initializing Network")
        model = net_model(in_dim, h_dim, num_l, frame_rate, fps)

        #Freez parameters and layers training control
        layers           = ('transformer', 'cls_layer')
        # trans_train  = [p for n, p in model.named_parameters() if "transformer" in n or "cls_layer" in n]
        # resnet_train  = [p for n, p in model.named_parameters() if "cnn" in n]

        trans_train  = []
        resnet_train  = []
        mlp_train = []
        # breakpoint()

        for n, p in model.named_parameters():
                try:
                        layer = n.split('.')[0]
                except:
                        pass
                if layer in {'encoder'}:
                        trans_train.append(p)
                elif layer in {'spatial_backbone', 'proj',}:
                        resnet_train.append(p)
                elif layer in {'classifier'}:
                    mlp_train.append(p)


        #Optimizer
        print("Initializing optimizer")
        optimizer = optim.Adam([{"params": resnet_train, "lr": 1e-5},
                                {"params": trans_train, "lr": 1e-4},
                                {"params": mlp_train, "lr": 1e-4}])

        #Network to GPU
        model.cuda()
        
        #Scheduler
        print("Initializing scheduler")
        lambda1   = lambda epoch : 1.0 if epoch<10 else (0.1 if epoch<20 else 0.1)
        lambda2   = lambda epoch : 1.0 if epoch<20 else (0.1 if epoch<30 else 0.1)
        lambda3   = lambda epoch : 1.0 if epoch<20 else (0.1 if epoch<30 else 0.1)
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[lambda1, lambda2, lambda3])
                
        #Dataloader
        print("Initializing dataloader")
        datasets  = {}
        dataset   = thermaldataset(label, data, sync, phase='train')
        '''Indexes for train/val'''
        idxs = list(range(0, len(dataset)))
        random.shuffle(idxs)
        split_idx = int(len(idxs) * split)
        trainIdxs = idxs[:split_idx]; valIdxs = idxs[:split_idx//5]
        '''create subsets'''
        datasets['train'] = torch.utils.data.Subset(dataset, trainIdxs)
        datasets['test']  = torch.utils.data.Subset(dataset, valIdxs)
        print("number of training samples", len(datasets['train']))
        #print(datasets['train'].dataset)
        dataloader_tr  = torch.utils.data.DataLoader(datasets['train'], batch_size=batch_size, \
                                                                                                shuffle=True, num_workers=workers)
        dataloader_val = torch.utils.data.DataLoader(datasets['test'], batch_size=batch_size, \
                                                                                                shuffle=True, num_workers=workers)
        dataloader = {'train': dataloader_tr, 'test' : dataloader_val}

        #Loss function
        print("Initializing loss function")
        # loss = loss_pep()
        loss = loss_cls

        #Correlation in prediction
        # corr = Pearson_Correlation()

        #Load the existing Model
        if resume == 'T':
                try:
                        checkpoint_dir = f'{c_point_dir}/*'
                        f_list = glob.glob(checkpoint_dir)
                        latest_checkpoint = max(f_list, key=os.path.getctime)
                        print("Resuming Existing state of Pretrained Model")
                        checkpoint = torch.load(latest_checkpoint)
                        model.load_state_dict(checkpoint['model_state_dict'])
                        cur_epoch  = checkpoint['epoch']
                        cur_loss   = checkpoint['loss']
                        print("Loading Done, Loss: {}, Current_epoch: {}".format(cur_loss, cur_epoch))
                except:
                        print("Loading Existing state of Pretrained Model Failed ! ")
                        print("Initializing training from epoch 0, PRESS c to continue")
                        import pdb; pdb.set_trace()

        #Loading the existing Hyperparameters
        if hyper_param == 'T':
                try:
                        checkpoint_dir = f'{c_point_dir}/*'
                        f_list = glob.glob(checkpoint_dir)
                        latest_checkpoint = max(f_list, key=os.path.getctime)
                        checkpoint = torch.load(latest_checkpoint)
                        print("Loading existing hyper-parameters")
                        optimizer.load_state_dict(checkpoint['optimizer'])
                        scheduler.load_state_dict(checkpoint['scheduler'])
                except:
                        print("Failed to load existing hyper-parameters")
                        print(" Initializing from epoch 0, PRESS c to continue")

        #Initialization done, Start training loop
        #parameters
        params = {'cur_epoch': cur_epoch,
                  'epochs'   : epochs,
                  'phase'    : phase,
                  'fps'      : fps,
                  'label'    : label,
                  'data'     : data,
                  'sync'     : sync,
                  'batch'    : batch_size}

        training_loop(args, model, optimizer, scheduler, dataloader, loss, **params)

#Save Checkpoint
def save_checkpoint(state, filename):
        checkpoint_path = f'../model_checkpoints_r50/{filename}'
        torch.save(state, checkpoint_path)
        return
        
def training_loop(args, model, optimizer, scheduler, dataloader, loss, **params):
        #training vars
        best_train_loss  = 100
        mean_train_loss  = 100
        best_epoch_train = 0
        #test vars
        best_test_loss   = 100  
        mean_test_loss   = 100
        best_epoch_test  = params['cur_epoch']

        plot_train_loss = []
        plot_test_loss  = []
        plot_train_acc  = []
        plot_test_acc   = []
        
        # correct = 0

        for epoch in range(params['cur_epoch'], params['epochs']):
                optimizer.zero_grad()
                train_loss = 0.0
                train_acc  = 0.0
                train_confidence = 0.0

                print('Epoch {}/{}'.format(epoch, params['epochs']-1))

                if params['phase'] == 'train':
                        model.train()   #set model to training mode
                        #Iterate over data
                        for iteration, data in enumerate(tqdm(dataloader['train'])):
                                running_loss = 0.0
                                running_acc  = 0.0
                                label_predictions = []
                                stress_predictions = []
                                try:
                                        inputs = data['data'].cuda().float()
                                        labels = data['label']
                                        s_label = data['s_label']
                                except:
                                        print("Data read error, number of frames too less to read")
                                        continue
                                print("input: ", inputs.shape, "label: ", labels.shape, iteration)
                                b_size, num_frames, ch, h, w = inputs.shape
                                
                                #forward, track history if only in train
                                with torch.set_grad_enabled(params['phase'] == 'train'):
                                        #only for thermal dataset, as video are to large for GPU memory
                                        #feeding params['fps'] frames at once
                                        for idx in range(0, num_frames, params['fps']):
                                                label_idx = idx*1
                                                mini_input = inputs[:,idx:idx+params['fps'],:,:]
                                                mini_label = labels[:,label_idx:label_idx+(params['fps']),:]
                                                #if section of video less then fps seconds, then drop the rest of the video
                                                if params['fps'] > len(mini_label.squeeze()): continue
                                                # mini_out, out2   = model(mini_input) # TODO: mini_out is ISTI, get rid of it
                                                out2 = model(mini_input) # TODO: mini_out is ISTI, get rid of it
                                                # breakpoint()
                                                print("prediction = ", out2)

                                                # s_label	 = s_label.to(mini_out.device).to(torch.float32)
                                                # s_label_new = torch.zeros(2).to(mini_out.device)
                                                # s_label_new[s_label] = 1
                                                # loss_total = nn.BCELoss(mini_out, s_label_new)

                                                # loss_total = loss(None, out2, None, s_label) # dummy arguments for now
                                                # onehot = one_hot(torch.tensor(s_label), num_classes=2).to(device=mini_input.device, dtype=mini_input.dtype).squeeze(0)
                                                loss_total = loss(out2, s_label, mini_input.device)

                                                if(math.isnan(loss_total.item())):
                                                        print("gradient explosion or vanished, updated learning rate")
                                                cur_loss = loss_total
                                                running_loss  = running_loss + loss_total.item()
                                                cur_loss.backward()
                                                print("Local loss: ", cur_loss.item())
                                                
                                                #training accuracy
                                                # pep_preds       = predict_pep(mini_out)
                                                # correlation = corr.pearson_correlation(pep_preds, mini_label)
                                                # running_acc = running_acc + abs(correlation.item())
                                                stress_predictions.append(out2)
                                        
                                        running_loss = running_loss
                                        train_loss = train_loss +  running_loss

                                        #whole video stress prediction
                                        stress_predictions = torch.cat(stress_predictions)
                                        # breakpoint()
                                        stress_predictions = stress_predictions >= 0
                                        # stress_predictions[stress_predictions >= 0] = 1
                                        # stress_predictions[stress_predictions < 0] = 0

                                        overall_prediction = 0
                                        confidence = torch.sum(stress_predictions)/len(stress_predictions) # confidence that video is of stressed participant

                                        if confidence >= 0.5:
                                                print(f"STRESS DETECTED IN THE SUBJECT: {s_label}, CONFIDENCE: {confidence}")
                                                overall_prediction = 1
                                        else:
                                            print(f"NO-STRESS DETECTED: {s_label}, CONFIDENCE: {1 - confidence}")
                                            confidence = 1 - confidence

                                        train_confidence += confidence

                                        if overall_prediction == s_label:
                                            train_acc += 1
                                                
                                        #save_prediction(labels.data, label_predictions, iteration)

                                        if (iteration+1)%2 == 0:
                                                optimizer.step()
                                                optimizer.zero_grad()
                                                scheduler.step()

                                #mean training loss
                                print("iteration", iteration)
                                mean_train_loss = train_loss/(iteration+1)
                                mean_train_acc = train_acc/(iteration+1)

                                cur_training_vars = {'Training_loss': mean_train_loss,
                                                     'Train_acc': mean_train_acc,
                                                     'Phase': params['phase'],
                                                     'epoch': epoch+1,
                                                     'iteration': iteration+1,
                                                     'Leaning Rate': scheduler.get_lr()
                                                     }
                                best_training_vars= {'Best_train_loss': best_train_loss,
                                                                         'Min Loss epoch' : best_epoch_train
                                                                        }

                                if args.use_wandb:
                                        wandb.log(cur_training_vars)

                                print("Current Training Vars: ", cur_training_vars, "Best Training Vars: "\
                                                                                                                                        , best_training_vars)
                        try:
                                mean_train_loss = mean_train_loss.item()
                        except:
                                mean_train_loss = mean_train_loss
                        plot_train_loss.append(mean_train_loss)
                        if mean_train_loss < best_train_loss:
                                best_train_loss  = mean_train_loss
                                best_epoch_train = epoch+1

                #Validating the model
                test_loss = 0
                test_acc = 0
                for iteration, data in enumerate(tqdm(dataloader['test'])):
                        running_loss   = 0.0
                        running_acc    = 0.0
                        model.eval()

                        try:
                                inputs = data['data'].cuda().float()
                                labels = data['label']
                                s_label = data['s_label']
                        except:
                                print("Data read error, corrupted data")
                                continue

                        print("Test input: ", inputs.shape, "label: ", labels.shape, iteration)
                        b_size, num_frames, ch, h, w = inputs.shape
                        with torch.no_grad():
                                predictions = []
                                for idx in range(0, num_frames, params['fps']):
                                        mini_input = inputs[:,idx:idx+params['fps'],:,:]
                                        mini_label = labels[:,idx:idx+params['fps'],:]
                                        if params['fps'] > len(mini_label.squeeze()): continue  
                                        #output generation
                                        out2   = model(mini_input)
                                        
                                        #loss computation
                                        # loss_total = loss(mini_out, out2, mini_label, s_label)
                                        loss_total = loss(out2, s_label, mini_input.device)
                                        cur_loss   = loss_total.item()
                                        running_loss = running_loss + cur_loss
                                        print("Local loss : ", cur_loss)

                                        #predictions
                                        predictions.append(out2)
                                        # pep_preds  = predict_pep(mini_out)
                                        # correlation= corr.pearson_correlation(pep_preds, mini_label)
                                        # cur_corr   = correlation.item()
                                        # running_acc = running_acc + abs(cur_corr)

                                predictions = torch.cat(predictions) >= 0
                                test_loss = test_loss + running_loss

                                confidence = torch.sum(predictions) / len(predictions) # confidence score for whether video is of stressed individual
                                overall = 0

                                if confidence >= 0.5:
                                    overall = 1
                                    print(f"STRESS PREDICTED, CONFIDENCE: {confidence}")
                                else:
                                    confidence = 1 - confidence
                                    print(f"NO STRESS PREDICTED, CONFIDENCE: {confidence}")

                                if overall == s_label:
                                    test_acc  += 1
                                # test_acc = test_acc + running_acc

                                if ((s_label == 1) == overall_prediction):
                                    TP +=1

                                if ((s_label == 1) != overall_prediction):
                                    FP +=1

                                if ((s_label == 0) == overall_prediction):
                                    TN +=1

                                if ((s_label == 0) != overall_prediction):
                                    FN +=1

                                precision = TP/(TP+FP)
                                recall = TP/(TP+FN)
                                tpr = recall
                                tnr = TN/(TP+FP)
                                acc = (TP+TN)/(TP+TN+FP+FN)
                                bal_acc = (TPR+TNR)/2

                        #mean test loss and acc
                        mean_test_loss = test_loss / (iteration+1)
                        mean_test_acc  = test_acc / (iteration+1)

                        cur_test_vars = {'Test_loss': mean_test_loss,
                                         'Test_accuracy': mean_test_acc,
                                         'Precision': precision,
                                         'Recall': recall,
                                         'TNR': tnr,
                                         'Classification Accuracy': acc,
                                         'Balanced Accuracy': bal_acc,
                                         'Phase': 'Test',
                                         'epoch': epoch+1,
                                         'iteration': iteration+1
                                         }
                        best_test_vars= {'Best_test_loss':best_test_loss,
                                                         'Min Loss epoch':best_epoch_test
                                                        }
                        if args.use_wandb:
                                wandb.log(cur_test_vars)
                        print("Current Test Vars: ", cur_test_vars, "Best Test Vars: ", best_test_vars)

                try:
                        mean_test_loss = mean_test_loss.item()
                        mean_test_acc  = mean_test_acc.item()
                except:
                        mean_test_loss = mean_test_loss
                        mean_test_acc  = mean_test_acc

                plot_test_loss.append(mean_test_loss)
                plot_test_acc.append(mean_test_acc)
                if mean_test_loss < best_test_loss:
                        best_test_loss  = mean_test_loss
                        best_epoch_test = epoch+1
                        time_stamp               = datetime.now().strftime("%Y_%m_%d_%H_%M_%s")

                        save_checkpoint({'model_state_dict': model.state_dict(),
                                                         'epoch'                   : epoch + 1,
                                                         'loss'                    : best_test_loss,
                                                         'optimizer'       : optimizer.state_dict(),
                                                         'scheduler'       : scheduler.state_dict()}, 
                                                         'checkpoint_{0}.pth.tar'.format(time_stamp))

        #saving loss for train and test
        loss_dump = {"train" : plot_train_loss,
                                 "test"  : plot_test_loss,
                                 "test_acc" : plot_test_acc}

        with open('../loss_dump.json', 'w') as json_file:
                json.dump(loss_dump, json_file)

if __name__ == '__main__':
        main()
