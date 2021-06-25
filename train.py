import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

import numpy as np
import random
import os, sys
import argparse

from model import MLP_Mixer
from data import get_loaders


def main(args):
    # Device Seting
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False 

    # Dataset
    trainloader, testloader, image_size, n_image_channel, n_output = get_loaders(args)

    # Model, Optimizer, and Criterion
    model = MLP_Mixer(                  # MNIST     CIFAR10     CIFAR100
        n_layers    = args.n_layers,    # 2         2           6
        n_channel   = args.n_channel,   # 20        20          128
        n_hidden    = args.n_hidden,    # 64        64          128
        n_output    = n_output,         # 10        10          100
        image_size  = image_size,       # 28        32          32
        patch_size  = args.patch_size,  # 2         4           4
        n_image_channel=n_image_channel # 1         3           3
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    # Train Proper
    for epoch in range(args.n_epochs):
        tr_loss, tr_acc = train(args, model, device, trainloader, optimizer, criterion)
        te_loss, te_acc = test(args, model, device, testloader, criterion)
        print('Epoch {:3d} | Train: loss={:.3f}, acc={:5.1f}% | Test: loss={:.3f}, acc={:5.1f}%'.format(
            epoch+1, tr_loss, tr_acc, te_loss, te_acc))
    
def train(args, model, device, trainloader, optimizer, criterion):
    model.train()
    total_loss, total_num, correct = 0, 0, 0
    for x,y in trainloader:
        batch_size = x.shape[0]
        x, y = x.to(device), y.to(device)
        if len(x.shape) == 3:
            x = x.unsqueeze(1)

        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        pred = out.argmax(dim=1, keepdim=True) 

        correct    += pred.eq(y.view_as(pred)).sum().item()
        total_loss += loss.data.cpu().numpy().item()*batch_size
        total_num  += batch_size

    acc = 100. * correct / total_num
    final_loss = total_loss / total_num
    return final_loss, acc

def test(args, model, device, testloader, criterion):
    model.eval()
    total_loss, total_num, correct = 0, 0, 0
    with torch.no_grad():
        for x,y in testloader:
            batch_size = x.shape[0]
            x, y = x.to(device), y.to(device)
            if len(x.shape) == 3:
                x = x.unsqueeze(1)

            out = model(x)
            loss = criterion(out, y)
            pred = out.argmax(dim=1, keepdim=True) 

            correct    += pred.eq(y.view_as(pred)).sum().item()
            total_loss += loss.data.cpu().numpy().item()*batch_size
            total_num  += batch_size

    acc = 100. * correct / total_num
    final_loss = total_loss / total_num
    return final_loss, acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MLP Mixer')
    parser.add_argument('--patch_size', type=int,   default=2,        metavar='PATCH_SIZE')
    parser.add_argument('--n_layers',   type=int,   default=2,        metavar='N_LAYERS')
    parser.add_argument('--n_channel',  type=int,   default=20,       metavar='N_CHANNEL')
    parser.add_argument('--n_hidden',   type=int,   default=64,       metavar='N_HIDDEN')
    
    parser.add_argument('--seed',       type=int,   default=1,        metavar='SEED')
    parser.add_argument('--dataset',    type=str,   default='mnist',  metavar='DATASET')
    parser.add_argument('--batch_size', type=int,   default=256,      metavar='BATCH_SIZE')
    parser.add_argument('--n_epochs',   type=int,   default=20,       metavar='N_EPOCHS')
    parser.add_argument('--lr',         type=float, default=1e-3,     metavar='N_EPOCHS')
    parser.add_argument('--gpu',        type=str,   default='0',      metavar='GPU')
    parser.add_argument('--num_workers',type=int,   default=8,        metavar='NUM_WORKERS')

    args = parser.parse_args()
    print('='*100)
    print('Arguments =')
    for arg in vars(args):
        print('\t'+arg+':',getattr(args,arg))
    print('='*100)

    main(args)