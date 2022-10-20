import torch
import copy
from random import sample
import torch.nn as nn
import numpy as np
import pdb
from model import HWRModel

class User:
    def __init__(self,user_id,batch_size, local_data_percentage):
        self.user_id = user_id
        self.local_data_percentage = local_data_percentage
        self.batch_size = batch_size
        self.best_accuracy = None

    def update_local_model(self, global_model):
        self.model = copy.deepcopy(global_model)
        self.model.user_instance(self.user_id,self.batch_size,self.local_data_percentage)
        print(f'{self.user_id} received model from server .. ')
        #Check whetehr the parameters are correctly transferred
        #Check whether the model has the same parameters before the train function too.
        print('bias weights at user',self.model.model.conv1.bias)


    def predict(self, item_id, embedding_user, embedding_item):
        #Yet to complete
        return None

    def train(self):
        #train function utilizes all other functions and returns best acuracy having saved best checkpoint model
        self.best_accuracy = self.model.train(num_epochs = 5)
        print("Best accuracy = ",self.best_accuracy)
        parameters = self.model.get_parameters()
        print('bias weights af User after training',parameters['conv1.bias'])
        return parameters





        

