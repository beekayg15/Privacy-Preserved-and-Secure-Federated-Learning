import torch
import os
import numpy as np
import torch.nn as nn
import dgl
from random import sample
from multiprocessing import Pool, Manager
# from torch.multiprocessing import Pool, Manager
from model import HWRModel
import pdb
torch.multiprocessing.set_sharing_strategy('file_system')

class server():
    def __init__(self, user_list, user_batch, users, items, embed_size, lr):
        self.user_list_with_coldstart = user_list
        self.user_list = self.generate_user_list(self.user_list_with_coldstart)
        self.batch_size = user_batch
        self.user_embedding = torch.randn(len(users), embed_size).share_memory_()
        self.item_embedding = torch.randn(len(items), embed_size).share_memory_()
        self.model = model(embed_size, 1)
        self.lr = lr
        self.distribute(self.user_list_with_coldstart)

    def generate_user_list(self, user_list_with_coldstart):
        ls = []
        for user in user_list_with_coldstart:
            if len(user.items) > 0:
                ls.append(user)
        return ls

    def aggregator(self, parameter_list):
        lou = parameter_list.copy()  # los -List of users
        result = None
        if len(lou) > 0:
            result = lou.pop(0)

        else:
            return result

        layer_names = lou[0].keys()  # colleting the b=name of each layer in the model

        # calculating the average of parameters all the users
        for layer in layer_names:
            for user in lou:
                result[layer] = result[layer] + user[layer]
            result[layer] = result[layer] / len(parameter_list)

        return result

    def distribute(self, users):
        for user in users:
            user.update_local_model(self.model)

    def distribute_one(self, user):
        user.update_local_GNN(self.model)

    def predict(self, valid_data):
        # print('predict')
        users = valid_data[:, 0]
        items = valid_data[:, 1]
        res = []
        self.distribute([self.user_list_with_coldstart[i] for i in set(users)])

        for i in range(len(users)):
            res_temp = self.user_list_with_coldstart[users[i]].predict(items[i], self.user_embedding, self.item_embedding)
            res.append(float(res_temp))
        return np.array(res)

    def train_one(self, user, user_embedding, item_embedding):
        print(user)
        self.parameter_list.append(user.train(user_embedding, item_embedding))

    def train(self):

        parameter_list = []

        #Returns a random 'self.batch-size' users from user list
        users = sample(self.user_list, self.batch_size)
        # print('distribute')

        #distributes the global model to all the users
        self.distribute(users)
        # print('training')
        # p = Pool()
        # for user in users:
        #     p.apply_async(self.train_one, args=(user, self.user_embedding, self.item_embedding))
        # p.close()
        # p.join()

        #Runs train for all the users(basically trains all the users)
        #After training each user, adds the resultant parameters to the parameter list
        for user in users:
            parameter_list.append(user.train(self.user_embedding, self.item_embedding))

        # print('aggregate')
        #Aggregates all the parameters
        gradient_model, gradient_item, gradient_user = self.aggregator(parameter_list)

        #Retrieving the exisitng(old) model parameters
        ls_model_param = list(self.model.parameters())

        # print('renew')

        #Updating the existing(previous) weights with the new agegated weights
        for i in range(len(ls_model_param)):
            ls_model_param[i].data = ls_model_param[i].data - self.lr * gradient_model[i]
        self.item_embedding = self.item_embedding -  self.lr * gradient_item
        self.user_embedding = self.user_embedding -  self.lr * gradient_user
