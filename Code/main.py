import pickle
import torch
import numpy as np
from user import User
from server import Server
from sklearn import metrics
import math
import argparse
import warnings
import sys
import faulthandler
from random import randrange
faulthandler.enable()
warnings.filterwarnings('ignore')
# torch.multiprocessing.set_sharing_strategy('file_system')

parser = argparse.ArgumentParser(description="args for FedGNN")
parser.add_argument('--lr', type=float, default = 1)
parser.add_argument('--data', default='filmtrust')
parser.add_argument('--user_batch', type=int, default=5)
parser.add_argument('--num_users',type=int,default=5)
parser.add_argument('--clip', type=float, default = 0.1)
parser.add_argument('--laplace_lambda', type=float, default = 0.1)
parser.add_argument('--negative_sample', type = int, default = 1000)
parser.add_argument('--valid_step', type = int, default = 5)
args = parser.parse_args()

user_batch = args.user_batch
lr = args.lr
num_users = args.num_users

def processing_valid_data(valid_data):
    res = []
    for key in valid_data.keys():
        if len(valid_data[key]) > 0:
            for ratings in valid_data[key]:
                item, rate, _ = ratings
                res.append((int(key), int(item), rate))
    return np.array(res)

def loss(server, valid_data):
    label = valid_data[:, -1]
    predicted = server.predict(valid_data)
    mae = sum(abs(label - predicted)) / len(label)
    rmse = math.sqrt(sum((label - predicted) ** 2) / len(label))
    return mae, rmse

# read data

# build user_list
user_list = []
user_id_list = [i for i in range(num_users)]
for u in user_id_list:
    user_list.append(User(u,user_batch,randrange(40,60)))

print(f'User list = {[u.user_id for u in user_list]}')

# build server
server = Server(user_list,user_batch,0.001)
count = 0

# train and evaluate
acc_best = float('-inf')
while 1:
    acc = 0.0
    for i in range(args.valid_step):
        print(i)
        acc = server.train()
        print(f"Accuracy at iteration {i} = {acc}")
    if acc > acc_best:
        acc_best = acc
        count = 0
    else:
        count += 1
    if count > 5:
        print('not improved for 5 epochs, stop trianing')
        break

