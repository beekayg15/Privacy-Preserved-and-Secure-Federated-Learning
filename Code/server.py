import torch
import os
import numpy as np
import torch.nn as nn
from random import sample
from multiprocessing import Pool, Manager
# from torch.multiprocessing import Pool, Manager
from model import HWRModel
import pdb
torch.multiprocessing.set_sharing_strategy('file_system')

class Server:
    def __init__(self, user_list, user_batch, lr):
        self.user_list_with_coldstart = user_list
        self.user_list = self.generate_user_list(self.user_list_with_coldstart)
        self.batch_size = user_batch
        #Check whether model works without calling user_instance
        self.model = HWRModel()
        self.lr = lr
        #self.distribute(self.user_list_with_coldstart)

    def generate_user_list(self, user_list_with_coldstart):
        #Not necessary for us 
        ls = []
        for user in user_list_with_coldstart:
            ls.append(user)
        return ls

    def aggregator(self, parameter_list):
        lou = parameter_list.copy()  # los -List of users
        result = None
        if len(lou) > 0:
            result = lou.pop(0)

        else:
            return result

        layer_names = lou[0].keys()  # collecting the of each layer in the model

        # calculating the average of parameters all the users
        for layer in layer_names:
            for user in lou:
                result[layer] = result[layer] + user[layer]
            result[layer] = result[layer] / len(parameter_list)

        self.model.initialise_parameters(result)
        #Testing
        print('Bias weights at server',result['conv1.bias'])

    def distribute(self,users):
        print('Distributing model to users ... \n')
        for user in users:
            user.update_local_model(self.model)

    def predict(self, valid_data):
        # print('predict')
        #Yet to complete
        users = valid_data[:, 0]
        items = valid_data[:, 1]
        res = []
        self.distribute([self.user_list_with_coldstart[i] for i in set(users)])

        for i in range(len(users)):
            res_temp = self.user_list_with_coldstart[users[i]].predict(items[i], self.user_embedding, self.item_embedding)
            res.append(float(res_temp))
        return np.array(res)

    def train_one(self, user):
        print(user)
        self.parameter_list.append(user.train())

    def train(self):

        #Try to understand main and complete it
        #After that try to initialize a new model with the weights, or try obtaining only gradients

        parameter_list = []

        #Returns a random 'self.batch-size' users from user list 
        #Implement sample function, resolve conflict between bath_size and user_batch
        users = self.user_list
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
        avg_best_acc = 0.0
        for user in users:
            parameter_list.append(user.train())
            #print(f'\nUser : {user.user_id}, Accuracy = {user.best_accuracy}')
            avg_best_acc += user.best_accuracy
        avg_best_acc /= len(users)

        aggregated_weights = self.aggregator(parameter_list)

        return avg_best_acc




        

 
