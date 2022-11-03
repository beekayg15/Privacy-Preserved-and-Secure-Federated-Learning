
import copy
import tenseal as ts
import torch
class User:
    def __init__(self,user_id,batch_size, local_data_percentage):
        self.user_id = user_id
        self.local_data_percentage = local_data_percentage
        self.batch_size = batch_size
        self.best_accuracy = None



    def update_local_model(self, global_model):
        self.model = copy.deepcopy(global_model)

        #Just add ceratin attributes to the model(user_id, batch size, local data percentage)
        self.model.user_instance(self.user_id,self.batch_size,self.local_data_percentage)

        self.initial_weights = self.model.get_model_parameters()

        print(f'\nUser {self.user_id} received model from server .. ')
        print(f'Bias weights at User {self.user_id}',self.model.model.conv1.bias)


    def predict(self):
        pass



    def context(self):
        bits_scale = 26
        context = ts.context(
    ts.SCHEME_TYPE.CKKS,
    poly_modulus_degree=8192,
    coeff_mod_bit_sizes=[31, bits_scale, bits_scale, 31])
        context.global_scale = pow(2, 40)
        context.generate_galois_keys()
        return context

    def encrypt(self,gradient):
        with torch.no_grad():
            context = self.context()
            gradient = ts.PlainTensor(gradient.tolist())
            encrypted_gradient = ts.ckks_tensor(context,gradient)
            return encrypted_gradient



    def train(self):
        context = self.context()
        #train function utilizes all other functions and returns best acuracy having saved best checkpoint model
        print(f"\nUser {self.user_id} starting training ... \n")
        self.best_accuracy = self.model.train(num_epochs = 5)
        print(f"\nUser {self.user_id} Best accuracy = ",self.best_accuracy)
        parameters = self.model.get_best_parameters()

        for layer in parameters:
            if not layer.startswith('fc'):
                if layer=='conv1.bias':
                    print("User's gradient : ",parameters[layer] - self.initial_weights[layer])
                parameters[layer] = self.encrypt(parameters[layer] - self.initial_weights[layer])        

        print("Gradients at user : ",parameters['conv1.bias'])
        
        return parameters





        

