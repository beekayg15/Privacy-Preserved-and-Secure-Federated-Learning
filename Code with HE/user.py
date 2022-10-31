
import torch
from model import HWRModel
import tenseal as ts
class User:
    def __init__(self,user_id,batch_size, local_data_percentage,data_path,lr):
        self.user_id = user_id
        self.model = HWRModel(data_path,lr)
        self.model.user_instance(user_id,batch_size,local_data_percentage)
        self.local_data_percentage = local_data_percentage
        self.batch_size = batch_size
        self.best_accuracy = None

    def get_weights_from_server(self, result):
        result = self.decrypt(result)
        self.model.initialise_parameters(result)
        print(f'\nUser {self.user_id} received model from server .. ')
        print(f'Bias weights at User {self.user_id}',self.model.model.conv1.bias)
    
    def decrypt(self,result):
        for layer in result:
            if not layer.startswith('fc'):
                print("At user ",result[layer])
                try:
                    result[layer] = torch.FloatTensor(result[layer].decrypt().tolist()).to(torch.device('mps'))
                except:
                    print(f"Layer {layer} not decrypted")
        return result



    def predict(self):
        pass

    def context(self):
        context = ts.context(ts.SCHEME_TYPE.CKKS, 8192, coeff_mod_bit_sizes=[60, 40, 40, 60])
        context.global_scale = pow(2, 40)
        context.generate_galois_keys()
        return context


    def encrypt(self,parameters):
        context = self.context()
        for layer in parameters:
            #parameters[layer] = torch.FloatTensor(parameters[layer].tolist())
            if not layer.startswith('fc'):
                parameters[layer] = torch.FloatTensor(parameters[layer].tolist())
                parameters[layer] =  ts.ckks_tensor(context,parameters[layer])
            print(f"Encrypted parameters for {layer}")
        return parameters
        


    def train(self):
        #train function utilizes all other functions and returns best acuracy having saved best checkpoint model
        print(f"\nUser {self.user_id} starting training ... \n")
        self.best_accuracy = self.model.train(num_epochs = 1)
        print(f"\nUser {self.user_id} Best accuracy = ",self.best_accuracy)
        parameters = self.model.get_parameters()
        print(f'\nBias weights at User {self.user_id} after training',parameters['conv1.bias'])
        parameters = self.encrypt(parameters)
        print('Encrypted parameters')
        return parameters





        

