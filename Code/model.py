import torch, torchvision
from torch.nn import Module,Sequential,Linear,Conv2d,BatchNorm2d,ReLU,MaxPool2d
from torch.utils.data import DataLoader,random_split
import pathlib
import torch.nn as nn
from torch.optim import Adam
from torchvision import transforms,datasets
from glob import glob
import pickle

#glob module can be used for file name matching

class ConvNet(nn.Module):
    def __init__(self,num_classes=3):
        super(ConvNet,self).__init__()

        
        self.conv1=nn.Conv2d(in_channels=3,out_channels=12,kernel_size=3,stride=1,padding=1)
        self.bn1=nn.BatchNorm2d(num_features=12)
        self.relu1=nn.ReLU()
        self.pool=nn.MaxPool2d(kernel_size=2)
        
        self.conv2=nn.Conv2d(in_channels=12,out_channels=20,kernel_size=3,stride=1,padding=1)
        self.relu2=nn.ReLU()
        
        self.conv3=nn.Conv2d(in_channels=20,out_channels=32,kernel_size=3,stride=1,padding=1)
        self.bn3=nn.BatchNorm2d(num_features=32)
        self.relu3=nn.ReLU()
        
        self.fc=nn.Linear(in_features=75 * 75 * 32,out_features=num_classes)

        
    def forward(self,input):
        output=self.conv1(input)
        output=self.bn1(output)
        output=self.relu1(output)
            
        output=self.pool(output)
            
        output=self.conv2(output)
        output=self.relu2(output)
            
        output=self.conv3(output)
        output=self.bn3(output)
        output=self.relu3(output)
            
        output=output.view(-1,32*75*75)

        output=self.fc(output)
            
        return output


class HWRModel:
    def __init__(self,user_id,batch_size,local_data_percentage,data_path = '/Users/tarunvisvar/Downloads/Dataset/Handwriting/Handwriting-subset'):
        self.user_id = user_id
        self.dest_file = 'best_checkpoint_{}.model'.format(user_id)
        self.batch_size = batch_size
        self.train_path = data_path + '/Train'
        self.test_path = data_path + '/Test'
        self.local_data_percentage = local_data_percentage
    
    def preprocess(self,resize=150):
        transformer = transforms.Compose(
            [
                transforms.Resize(resize),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
            ]
        )
        return transformer   

    def get_model(self):
        model = ConvNet(num_classes = 3)
        optimizer = Adam(model.parameters(), lr=0.001)
        loss_func = nn.CrossEntropyLoss()

        return (model,optimizer,loss_func)

    def get_dataset(self,path):
        img_dataset = datasets.ImageFolder(path,transform = self.preprocess())
        total_data_count = len(glob(path+"/**/*.png"))
        shared_data_count = int((self.local_data_percentage/100)*total_data_count)
        local_dataset,rem_dataset = random_split(img_dataset,(shared_data_count,total_data_count-shared_data_count))

        print("{}/{} images taken".format(shared_data_count,total_data_count))
        return local_dataset


    def load_dataset(self):
        train_loader = DataLoader(
    self.get_dataset(self.train_path),
    batch_size=batch_size, shuffle=True)

        test_loader = DataLoader(
    self.get_dataset(self.test_path),
    batch_size=batch_size, shuffle=True) 

        return(train_loader,test_loader)
        
    def train(self,num_epochs=10):
        model,optimizer,loss_func = self.get_model()
        best_accuracy = 0.0
        train_loader,test_loader = self.load_dataset()
        train_count=len(glob(self.train_path+'/**/*.png'))
        test_count=len(glob(self.test_path+'/**/*.png'))

        
        for epoch in range(num_epochs):
            model.train()
            #Model will be in training mode and takes place on training dataset
            train_loss = 0.0
            train_accuracy = 0.0
            for images,labels in train_loader:
                optimizer.zero_grad()
                outputs = model(images) 

                loss = loss_func(outputs,labels)
                loss.backward() # backpropagation
                optimizer.step() # Updates the weights

                train_loss += loss.data*batch_size
                _,predictions = torch.max(outputs.data,1)
                train_accuracy+=int(torch.sum(predictions==labels.data))
            train_accuracy /= train_count
            train_loss /= train_count
            print('Epoch: '+str(epoch)+' Train Loss: '+str(train_loss)+' Train Accuracy: '+str(train_accuracy))

            model.eval()
            test_accuracy = 0.0
            for images,labels in test_loader:
                outputs = model(images)
                _,predictions = torch.max(outputs.data,1)
                test_accuracy += int(torch.sum(predictions==labels.data))
            test_accuracy /= test_count
            print("Test accuracy =  ",str(test_accuracy))

            if test_accuracy>best_accuracy:
                
                torch.save(model,self.dest_file)
                best_accuracy=test_accuracy

        return best_accuracy
            
    def get_parameters(self):
        loaded_model = torch.load(self.dest_file)
        params = dict()
        for name,parameters in loaded_model.named_parameters():
            params[name] = parameters
            
        return params

if __name__ == '__main__':
    data_path = '/Users/tarunvisvar/Downloads/Dataset/Handwriting/Handwriting-subset'
    batch_size = 64
    local_data_percentage = 40
    parameter_list = [] # For testing aggregator function developed by Shasaank
    for i in range(5):
        mymodel = HWRModel(i,batch_size,local_data_percentage,data_path=data_path)
        mymodel.train(num_epochs = 10)
        parameters = mymodel.get_parameters()
        parameter_list.append(parameters)
    with open('parameter_list.bin','wb') as f:
        pickle.dump(parameter_list,f)
    



   

        

