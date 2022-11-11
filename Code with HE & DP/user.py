
import copy
import tenseal as ts
import torch
class User:
    def __init__(self,user_id,batch_size, local_data_percentage,public_key):
        self.user_id = user_id
        self.local_data_percentage = local_data_percentage
        self.batch_size = batch_size
        self.best_accuracy = None
        self.public_key = public_key
    
    def get_initial_model(self,global_model):
        self.model = copy.deepcopy(global_model)
        self.model.user_instance(self.user_id,self.batch_size,self.local_data_percentage)
        self.initial_weights = self.model.get_model_parameters()


    def update_local_model(self, aggregated_gradients):
                #Just add ceratin attributes to the model(user_id, batch size, local data percentage)
        self.model.user_instance(self.user_id,self.batch_size,self.local_data_percentage)
        
        decrypted_grads = self.decrypt(aggregated_gradients)
        print(f"User {self.user_id} received updated gradients from server")
        print("Decrypted aggregated gradients at user : ", decrypted_grads['conv1.bias'])

        self.model.initialise_parameters(decrypted_grads)
        print(f'Bias weights at User {self.user_id} after adding gradients',self.model.model.conv1.bias)

        self.initial_weights = self.model.get_model_parameters()

       


    def predict(self):
        pass

    def encrypt(self,gradient):
        with torch.no_grad():
            gradient = ts.PlainTensor(gradient.tolist())
            encrypted_gradient = ts.ckks_tensor(self.public_key,gradient)
            return encrypted_gradient

    def decrypt(self,params):
        decrypted_result = dict()
        for layer in params:
            if not layer.startswith('fc'):
                decrypted_result[layer] = torch.FloatTensor(params[layer].decrypt().tolist())
            else:
                decrypted_result[layer] = params[layer]
        return decrypted_result

    def train(self):
        #train function utilizes all other functions and returns best acuracy having saved best checkpoint model
        print(f"\nUser {self.user_id} starting training ... \n")
        self.best_accuracy = self.model.train(num_epochs = 3)
        print(f"\nUser {self.user_id} Best accuracy = ",self.best_accuracy)
        self.model.test()
        parameters = self.model.get_best_parameters()

        
        for layer in parameters:
            if not layer.startswith('fc'):
                gradient = parameters[layer] - self.initial_weights[layer]
                parameters[layer] = self.encrypt(gradient)  
                if layer == 'conv1.bias':
                    print(f"\nUser {self.user_id} raw gradients : ",gradient)


        print(f"\nUser {self.user_id} encrypted gradients : ",parameters['conv1.bias'])
        
        return parameters





        

