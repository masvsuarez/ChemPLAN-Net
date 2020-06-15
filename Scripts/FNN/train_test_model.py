# -*- coding: utf-8 -*-
from __future__ import division

""" 
Trains a ResNeXt Model on FEATURE Vectors

"""

__author__ = "Michael Suarez"
__email__ = "masv@connect.ust.hk"
__copyright__ = "Copyright 2019, Hong Kong University of Science and Technology"
__license__ = "3-clause BSD"

import argparse
import os
import json
import pickle
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from models.model011 import FeatureResNeXt
import torch.utils.data as utils
import numpy as np
import time

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Trains ResNeXt on FEATURE Vectors', formatter_class= argparse.ArgumentDefaultsHelpFormatter)
    # Positional arguments
    parser.add_argument('features_data_path', type=str, help='Root for Features Dict.')
    parser.add_argument('data_folder', type=str, help='Root for Data.')

    # Optimization options
    parser.add_argument('--epochs', '-e', type=int, default=20, help='Number of epochs to train.')
    parser.add_argument('--batch_size', '-b', type=int, default=256, help='Batch size.')
    parser.add_argument('--learning_rate', '-lr', type=float, default=0.01, help='The Learning Rate.') #def 0.1
    parser.add_argument('--momentum', '-m', type=float, default=0.3, help='Momentum.')
    parser.add_argument('--decay', '-d', type=float, default=0.0005, help='Weight decay (L2 penalty).')
    parser.add_argument('--schedule', type=int, nargs='+', default=[10, 15], help='Decrease learning rate at these epochs.')
    parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
    # Checkpoints
    parser.add_argument('--save', '-s', type=str, default='./outputmodels', help='Folder to save checkpoints.')
    # Architecture
    parser.add_argument('--depth', type=int, default=65, help='Model depth - Multiple of 3*no_stages (5, 10)')
    parser.add_argument('--cardinality', type=int, default=8, help='Model cardinality (group) in the DxD convolutionary layer.')
    parser.add_argument('--base_width', type=int, default=8, help='Number of channels in each group. Output of the first convolution layer. Modify stages in model.py')
    parser.add_argument('--widen_factor', type=int, default=4, help='Widen factor between every block')
    # Acceleration
    parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
    parser.add_argument('--prefetch', type=int, default=8, help='Pre-fetching threads.')
    # i/o
    parser.add_argument('--name', type=str, default='model', help='Name your model')
    parser.add_argument('--checkpoint', type=bool, default=False, help='Load from checkpoint?')
    args = parser.parse_args()

    # Init logger
    if not os.path.isdir(args.save):
        os.makedirs(args.save)
    log = open(os.path.join(args.save, args.name + '_log.txt'), 'w')
    state = {k: v for k, v in args._get_kwargs()}
    log.write(json.dumps(state) + '\n')

    # Calculate number of epochs wrt batch size
    args.epochs = args.epochs * args.batch_size // args.batch_size
    args.schedule = [x * args.batch_size // args.batch_size for x in args.schedule]
    
    #initialise Dataset
    # Binary molecules 
    test_b = pickle.load(open("%s/test_b_Vec.mtr" %args.data_folder, "rb"))
    train_b = pickle.load(open("%s/train_b_Vec.mtr" %args.data_folder, "rb"))
    test_nb = pickle.load(open("%s/test_nb_Vec.mtr" %args.data_folder, "rb"))
    train_nb = pickle.load(open("%s/train_nb_Vec.mtr" %args.data_folder, "rb"))
    # Indices of training and testing FF
    test_FF = pickle.load(open("%s/test_FF.mtr" %args.data_folder, "rb"))
    train_FF = pickle.load(open("%s/train_FF.mtr" %args.data_folder, "rb"))
    # Fragfeaturevectors
    Features_all = pickle.load(open(args.features_data_path, "rb"))
    
    # multiplicate Featurevectors
    train_FFAll = Features_all[train_FF]
    test_FFAll = Features_all[test_FF]

    print("loaded data")
    
    #reduce size of dataset for initial testing
#     cutTrain = 2**17
#     np.random.seed(0)
#     selTrain = np.random.choice(train_FF.shape[0], cutTrain, replace=False)
#     cutTest = 2**13
#     np.random.seed(0)
#     selTest = np.random.choice(test_FF.shape[0], cutTest, replace=False)
#     train_FFAll = train_FFAll[selTrain]
#     test_FFAll = test_FFAll[selTest]
#     train_b = train_b[selTrain]
#     train_nb = train_nb[selTrain]
#     test_b = test_b[selTest]
#     test_nb = test_nb[selTest]
    
    #validation split 1%
    np.random.seed(0)
    ss = np.random.choice(range(train_FFAll.shape[0]), int(0.01*train_FFAll.shape[0]), replace=False)
    val_FFAll = train_FFAll[ss]
    val_b = train_b[ss]
    val_nb = train_nb[ss]    
    train_FFAll = np.delete(train_FFAll, ss, 0)
    train_b = np.delete(train_b, ss, 0)
    train_nb = np.delete(train_nb, ss, 0)

    #normalise Featurevectors
    start = time.time()
    mean = [train_FFAll[:,i].mean() for i in range(480)]
    std = [train_FFAll[:,i].std() for i in range(480)]
    for i in range (480):
        if std[i] != 0:
            train_FFAll[:,i] = (train_FFAll[:,i] - mean[i])/std[i]
            test_FFAll[:,i] = (test_FFAll[:,i] - mean[i])/std[i]
            val_FFAll[:,i] = (val_FFAll[:,i] - mean[i])/std[i]            
        else:
            train_FFAll[:,i] = train_FFAll[:,i]
            test_FFAll[:,i] = test_FFAll[:,i]
            val_FFAll[:,i] = val_FFAll[:,i]
    end = time.time()
    print("FF normalising time: ", end-start)
    
    train_FFAll = np.resize(train_FFAll, (train_FFAll.shape[0], 1, 6, 80))
    test_FFAll = np.resize(test_FFAll, (test_FFAll.shape[0], 1, 6, 80))
    val_FFAll = np.resize(val_FFAll, (val_FFAll.shape[0], 1, 6, 80))

    #construct binary class system
    train_y_values1 = np.full((train_FFAll.shape[0],1),1)
    train_y_values0 = np.full((train_FFAll.shape[0],1),0)

    test_y_values1 = np.full((test_FFAll.shape[0],1),1)
    test_y_values0 = np.full((test_FFAll.shape[0],1),0)

    val_y_values1 = np.full((val_FFAll.shape[0],1),1)
    val_y_values0 = np.full((val_FFAll.shape[0],1),0)
    
    #convert numpy arrays into tensors
    start1 = time.time()
    train_tensor_x = torch.from_numpy(train_FFAll)
    train_tensor_x2_b = torch.stack([torch.Tensor(i) for i in train_b])
    train_tensor_y_b = torch.from_numpy(train_y_values1)
    train_tensor_x2_nb = torch.stack([torch.Tensor(i) for i in train_nb]) 
    train_tensor_y_nb = torch.from_numpy(train_y_values0)
    end1 = time.time()
    print("train data arrays2tensor: ", end1-start1)

    start2 = time.time()
    test_tensor_x = torch.from_numpy(test_FFAll)
    test_tensor_x2_b = torch.stack([torch.Tensor(i) for i in test_b])
    test_tensor_y_b = torch.from_numpy(test_y_values1)
    test_tensor_x2_nb = torch.stack([torch.Tensor(i) for i in test_nb]) 
    test_tensor_y_nb = torch.from_numpy(test_y_values0)
    end2 = time.time()
    print("test data arrays2tensor: ", end2-start2)
    
    start3 = time.time()
    val_tensor_x = torch.from_numpy(val_FFAll)
    val_tensor_x2_b = torch.stack([torch.Tensor(i) for i in val_b])
    val_tensor_y_b = torch.from_numpy(val_y_values1)
    val_tensor_x2_nb = torch.stack([torch.Tensor(i) for i in val_nb]) 
    val_tensor_y_nb = torch.from_numpy(val_y_values0)
    end3 = time.time()
    print("test data arrays2tensor: ", end3-start3)
      
    train_tensor_x = torch.cat([train_tensor_x,train_tensor_x]) 
    train_tensor_x2 = torch.cat([train_tensor_x2_b,train_tensor_x2_nb])
    train_tensor_y = torch.cat([train_tensor_y_b,train_tensor_y_nb])

    test_tensor_x = torch.cat([test_tensor_x,test_tensor_x]) 
    test_tensor_x2 = torch.cat([test_tensor_x2_b,test_tensor_x2_nb])
    test_tensor_y = torch.cat([test_tensor_y_b,test_tensor_y_nb])
    
    val_tensor_x = torch.cat([val_tensor_x,val_tensor_x]) 
    val_tensor_x2 = torch.cat([val_tensor_x2_b,val_tensor_x2_nb])
    val_tensor_y = torch.cat([val_tensor_y_b,val_tensor_y_nb])
    
    train_data = utils.TensorDataset(train_tensor_x, train_tensor_x2, train_tensor_y)
    test_data = utils.TensorDataset(test_tensor_x, test_tensor_x2, test_tensor_y)
    val_data = utils.TensorDataset(val_tensor_x, val_tensor_x2, val_tensor_y)

    train_loader = utils.DataLoader(train_data,batch_size=args.batch_size, shuffle=True, num_workers=args.prefetch, pin_memory=True)
    test_loader = utils.DataLoader(test_data,batch_size=args.batch_size, shuffle=False, num_workers=args.prefetch, pin_memory=True)    
    val_loader = utils.DataLoader(val_data,batch_size=args.batch_size, shuffle=False, num_workers=args.prefetch, pin_memory=True)   

    # Init model, criterion, and optimizer
    net = FeatureResNeXt(args.cardinality, args.depth, args.base_width, args.widen_factor)
    net = net.float()
    print(net)
    if args.ngpu > 1:
        net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))

    if args.ngpu > 0:
        net.cuda()

    optimizer = torch.optim.SGD(net.parameters(), state['learning_rate'], momentum=state['momentum'], weight_decay=state['decay'], nesterov=True)

    # train function (forward, backward, update)
    def train():
        net.train()
        loss_avg = 0.0
        for batch_idx, (X, X2, Y) in enumerate(train_loader):
            X, X2, Y = torch.autograd.Variable(X.cuda()), torch.autograd.Variable(X2.cuda()), torch.autograd.Variable(Y.cuda())
            output = net(X.float(), X2.float())       

            # forward
            output = net(X.float(), X2.float()) 

            # backward
            optimizer.zero_grad()
#LOSS FUNCTION
            loss = F.binary_cross_entropy(output, Y.float())
            print('train - loss',loss)
            loss.backward()
            optimizer.step()

            # exponential moving average
            loss_avg = loss_avg * 0.2 + float(loss) * 0.8
            print('train - loss avg', loss_avg)
        state['train_loss'] = loss_avg

    # val function (forward only)
    def val():
        net.eval()
        loss_avg = 0.0
        correct = 0
        total = 0
        for batch_idx, (X, X2, Y) in enumerate(val_loader):
            X, X2, Y = torch.autograd.Variable(X.cuda()), torch.autograd.Variable(X2.cuda()), torch.autograd.Variable(Y.cuda())
            output = net(X.float(), X2.float())
            groundT = Y
            predictT = torch.gt(output, 0.5)
            total += groundT.size(0)
            correct += (predictT.float() == groundT.float()).sum().item()
    
            loss = F.binary_cross_entropy(output, Y.float())
            print('val - loss', loss)
            # test loss average
            
            loss_avg += float(loss)
            print('val_loss_avg', loss_avg/(total/args.batch_size), correct, total, 'Val. Percent Hits', correct/total*100)
            
        state['val_loss'] = loss_avg/(total/args.batch_size)
        state['val_percent_hit'] = correct/total*100

    # test function (forward only)
    def test():
        net.eval()
        loss_avg = 0.0
        correct = 0
        total = 0
        for batch_idx, (X, X2, Y) in enumerate(test_loader):
            X, X2, Y = torch.autograd.Variable(X.cuda()), torch.autograd.Variable(X2.cuda()), torch.autograd.Variable(Y.cuda())
            output = net(X.float(), X2.float())
            groundT = Y
            predictT = torch.gt(output, 0.5)
            total += groundT.size(0)
            correct += (predictT.float() == groundT.float()).sum().item()
    
            loss = F.binary_cross_entropy(output, Y.float())
            print('test - loss', loss)
            # test loss average
            loss_avg += float(loss)
            print('test_loss_avg', loss_avg/(total/args.batch_size), correct, total, 'Percent Hits', correct/total*100)
        state['test_loss'] = loss_avg/(total/args.batch_size)
        state['percent_hit'] = correct/total*100
        
    # Main loop
    best_accuracy = 100.0
    
    if args.checkpoint == True:
        loaded_state_dict = torch.load('Data/KIN.ALL/ModelOutput/model00_cp.pytorch')        
#         temp = {}
#         for key, val in list(loaded_state_dict.items()):
#             temp[key[7:]] = val
#         loaded_state_dict = temp
        net.load_state_dict(loaded_state_dict)
        
    for epoch in range(args.epochs):
        #updates learning rate 
        if epoch in args.schedule:
            state['learning_rate'] *= args.gamma
            for param_group in optimizer.param_groups:
                param_group['lr'] = state['learning_rate']
                
        state['epoch'] = epoch
        train()
        val()
        test()
        
        #decide whether to save the model
        if state['test_loss'] < best_accuracy:
            best_accuracy = state['test_loss']
            torch.save(net.state_dict(), os.path.join(args.save, args.name + '.pytorch'))
        if epoch%20 == 0 and epoch > 0:
            torch.save(net.state_dict(), os.path.join(args.save, args.name + '_cp.pytorch'))
            #write in log file
        log.write('%s\n' % json.dumps(state))
        log.flush()
        print(state)
        print("Best accuracy: %f" % best_accuracy)
        
    torch.save(net.state_dict(), os.path.join(args.save, args.name + '_final.pytorch'))

    log.close()
