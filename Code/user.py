import torch
import copy
from random import sample
import torch.nn as nn
import numpy as np
import pdb
from model import HWRModel

class user():
    def __init__(self,user_id,batch_size, local_data_percentage):
        self.model = HWRModel(user_id,batch_size,local_data_percentage)

    def update_local_model(self, global_model):
        self.model = copy.deepcopy(global_model)

    def predict(self, item_id, embedding_user, embedding_item):
        #Yet to complete
        return None

    def train(self):
        #train function utilizes all other functions and returns best acuracy having saved best checkpoint model
        best_accuracy = self.model.train(num_epochs = 15)
        print("Best accuracy = ",best_accuracy)
        parameters = self.model.get_parameters()
        return parameters





        

