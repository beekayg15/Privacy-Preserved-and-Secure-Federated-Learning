import torch
import os
import numpy as np
import torch.nn as nn
from multiprocessing import Pool, Manager
# from torch.multiprocessing import Pool, Manager
from model import HWRModel
import pdb
torch.multiprocessing.set_sharing_strategy('file_system')

class Server:
    def __init__(self, user_list, lr, data_path):
        self.user_list = user_list
        self.model = HWRModel(data_path)
        self.lr = lr
        self.device = torch.device('mps')

    def aggregator(self, parameter_list):

        #print("Expected added result : ",parameter_list[0]['conv1.bias'].decrypt().tolist()," + ",parameter_list[1]['conv1.bias'].decrypt().tolist()," = ",(parameter_list[0]['conv1.bias']+parameter_list[1]['conv1.bias']).decrypt().tolist())
        
        lou = parameter_list.copy()  # los -List of users
        result = None
        if len(lou) > 0:
            result = lou.pop(0)

        else:
            return result

        layer_names = lou[0].keys()  # collecting the of each layer in the model
        n = torch.tensor(len(parameter_list))
        # calculating the average of parameters all the users
        for layer in layer_names:
            for user in lou:
                result[layer] = result[layer] + user[layer]
            #if layer=='conv1.bias':
                #print("Encrypted added result after decryption : ",torch.FloatTensor(result[layer].decrypt().tolist()))
            
            result[layer] = result[layer] / n
        #For testing
        print("Aggregated gradients : ",result['conv1.bias'])

        #Differential privacy
        result = self.decrypt(result)


        return result

    def decrypt(self,params):
        decrypted_result = dict()
        for layer in params:
            if not layer.startswith('fc'):
                decrypted_result[layer] = torch.FloatTensor(params[layer].decrypt().tolist())
            else:
                decrypted_result[layer] = params[layer]
        return decrypted_result

     
    def distribute(self,users):
        print('Distributing model to users ... \n')
        for user in users:
            user.update_local_model(self.model)

    def predict(self):
        pass
    
    def validate(self):
        test_loader = self.model.load_dataset()[1]
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

    def train(self):

        parameter_list = []

        users = self.user_list

        #distributes the global model to all the users
        self.distribute(users)

        #Runs train for all the users(basically trains all the users)
        #After training each user, adds the resultant parameters to the parameter list

        avg_best_acc = 0.0
        for user in users:
            parameter_list.append(user.train())
            avg_best_acc += user.best_accuracy
        avg_best_acc /= len(users)

        aggregated_weights = self.aggregator(parameter_list)

        print("Aggregated weight at server : ",aggregated_weights['conv1.bias'])
        self.model.initialise_parameters(aggregated_weights)

        #Include server training

        print('Bias weights at Server : ',self.model.model.conv1.bias)

        return avg_best_acc






        

 
