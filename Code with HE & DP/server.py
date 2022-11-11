import torch
import pydp as dp
from pydp.algorithms.laplacian import BoundedSum, BoundedMean, Count, Max
import os
import numpy as np
import torch.nn as nn
from multiprocessing import Pool, Manager
# from torch.multiprocessing import Pool, Manager
from model import HWRModel
import tenseal as ts
from time import time
torch.multiprocessing.set_sharing_strategy('file_system')

class Server:
    def __init__(self, user_list, lr, data_path):
        self.user_list = user_list
        self.model = HWRModel(data_path)
        self.lr = lr
        self.device = torch.device('mps')

    def aggregator(self, parameter_list):

        #print("Expected added result : ",parameter_list[0]['conv1.bias'].decrypt().tolist()," + ",parameter_list[1]['conv1.bias'].decrypt().tolist()," = ",(parameter_list[0]['conv1.bias']+parameter_list[1]['conv1.bias']).decrypt().tolist())
        print("\nAggregating gradients ...")
        noise = BoundedSum(epsilon = 1.5, upper_bound = 1e-9, lower_bound = 1e-10, dtype = 'float')
        noise = noise.quick_result([0])
        print("Noise Induced :", noise, type(noise))
        lou = parameter_list.copy()  # los -List of users
        result = None
        if len(lou) > 0:
            result = lou.pop(0)

        else:
            return result

        layer_names = lou[0].keys()  # collecting the of each layer in the model
        n = torch.tensor(1/len(parameter_list))
        # calculating the average of parameters all the users
        
        start = time()
        for layer in layer_names:
            for user in lou:
                result[layer] = result[layer] + user[layer]
            result[layer] = result[layer] + torch.tensor(noise)

                #print("Encrypted added result after decryption : ",torch.FloatTensor(result[layer].decrypt().tolist()))
        #For testing
        end = time()
        print(f"Completed addition in {end-start} seconds ... ")
        print(f"Debug : Added result = {result['conv1.bias'].decrypt().tolist()}")
        

        #Differential privacy
        start = time()
        for layer in result:
            result[layer] = result[layer]*n

        end = time()

        print(f'Completed division in {end - start} seconds')
        print(f"Debug : Divided result = {result['conv1.bias'].decrypt().tolist()}")
  
        print("Aggregated gradients at server: ",result['conv1.bias'])

        return result


     
    def distribute(self,aggregated_gradients):
        print('Distributing aggregated gradients to users ... \n')
        for user in self.user_list:
            user.update_local_model(aggregated_gradients)

    def distribute_model(self,users):
        print('Distributing model to users ...')
        for user in users:
            user.get_initial_model(self.model)

    def predict(self):
        pass
    
    def validate(self):
        test_loader = self.model.load_test_dataset()
        test_count = len(iter(test_loader))*self.model.batch_size
        #print(test_count)
        self.model.model.to(self.device)
        self.model.model.eval()
        test_accuracy = 0.0
        for images,labels in test_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            outputs = self.model.model(images)
            _,predictions = torch.max(outputs.data,1)
            test_accuracy += int(torch.sum(predictions==labels.data))
        test_accuracy /= test_count
        return test_accuracy


    def train_one(self, user):
        print(user)
        self.parameter_list.append(user.train())

    def run(self):

        parameter_list = []

        users = self.user_list

        #distributes the global model to all the users

        #Runs train for all the users(basically trains all the users)
        #After training each user, adds the resultant parameters to the parameter list

        avg_best_acc = 0.0
        for user in users:
            parameter_list.append(user.train())
            avg_best_acc += user.best_accuracy
        avg_best_acc /= len(users)

        aggregated_gradients = self.aggregator(parameter_list)

        self.distribute(aggregated_gradients)

        #print("Aggregated weights at server : ",aggregated_weights['conv1.bias'])
        #self.model.initialise_parameters(aggregated_weights)

        return avg_best_acc






        

 
