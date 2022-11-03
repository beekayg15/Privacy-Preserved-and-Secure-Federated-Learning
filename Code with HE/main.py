
import numpy as np
from user import User
from server import Server
from sklearn import metrics
import argparse
import warnings
import faulthandler
from random import randrange
faulthandler.enable()
warnings.filterwarnings('ignore')
# torch.multiprocessing.set_sharing_strategy('file_system')

parser = argparse.ArgumentParser(description="Arguments for federated learning")
parser.add_argument('--lr', type=float, default = 0.01)
parser.add_argument('--data_path', default='/Users/tarunvisvar/Downloads/Dataset/Handwriting//Handwriting-subset')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--num_iters', type = int, default = 5)
parser.add_argument('--num_users',type = int,default = 2)
args = parser.parse_args()

lr = args.lr
num_users = args.num_users
batch_size = args.batch_size
data_path = args.data_path
num_iters = args.num_iters

# Build user_list
user_list = []
user_id_list = [i+1 for i in range(num_users)]
for u in user_id_list:
    user_list.append(User(u,batch_size,100))

print(f'User list = {[u.user_id for u in user_list]}')

# Build server
server = Server(user_list,lr,data_path)


accuracies = []
count = 0

for i in range(num_iters):
    server_train_acc = server.train()
    print(f"\nServer train accuracy at iteration {i+1} = {server_train_acc}")
    server_test_acc = server.validate()
    print(f"Server test accuarcy at iteration {i+1}= {server_test_acc}")
    if i>0:
        if server_test_acc > accuracies[-1]:
            print(f"Accuracy improved from {accuracies[-1]} to {server_test_acc}")
            count = 0
        else:
            count += 1
    
    if count>5:
        print("No improvement for 5 iterations, stopping training ... ")
    accuracies.append(server_test_acc)

print(f"List of server accuracies during federated learning : {accuracies}")



