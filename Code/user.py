
import copy

class User:
    def __init__(self,user_id,batch_size, local_data_percentage):
        self.user_id = user_id
        self.local_data_percentage = local_data_percentage
        self.batch_size = batch_size
        self.best_accuracy = None

    def update_local_model(self, global_model):
        self.model = copy.deepcopy(global_model)
        self.model.user_instance(self.user_id,self.batch_size,self.local_data_percentage)
        print(f'\nUser {self.user_id} received model from server .. ')
        print(f'Bias weights at User {self.user_id}',self.model.model.conv1.bias)


    def predict(self):
        pass


    def train(self):
        #train function utilizes all other functions and returns best acuracy having saved best checkpoint model
        print(f"\nUser {self.user_id} starting training ... \n")
        self.best_accuracy = self.model.train(num_epochs = 5)
        print(f"\nUser {self.user_id} Best accuracy = ",self.best_accuracy)
        parameters = self.model.get_parameters()
        print(f'\nBias weights at User {self.user_id} after training',parameters['conv1.bias'])
        return parameters





        

