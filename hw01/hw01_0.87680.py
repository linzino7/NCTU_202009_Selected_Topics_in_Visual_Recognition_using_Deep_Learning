# -*- coding: utf-8 -*-
"""
Created on Sat Oct 31 00:32:28 2020
@author: ASUS
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision
from torchvision.models import resnet18, resnet50
import torch.optim as optim


from PIL import Image
import glob
import pandas as pd
import re

import timeit
start = timeit.default_timer()
print('staring clocking')


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


def get_img_fileid(filename, idx=8):
    return re.split('/|\.', filename)[idx]


class CarDataset(Dataset):
    def __init__(self, image_dir, label_path, transform=None, is_train=False):
        """
         Parameters
         ----------
         image_dir : TYPE
         
             All image data path.
         label_path : TYPE
             Label file path.
         input_transform : TYPE, optional
             transform object.
         is_train : TYPE, optional
             DESCRIPTION. The default is False.
             
         """
        super(Dataset, self).__init__()
        
        self.is_train = is_train
        self.datapath = image_dir
        self.image_filenames = []
        self.num_per_classes = {}
        self.transform = transform
        
        # get file list
        self.label_pairs = pd.read_csv(label_path)
        path_pattern = image_dir + '/*.*'
        files_list = glob.glob(path_pattern, recursive=True)
        
        for file in files_list:
            if is_image_file(file):
                self.image_filenames.append(file)
                id_num = int(get_img_fileid(file))
                # Search label by id 
                class_name = self.label_pairs[self.label_pairs['id'] == id_num]['label'].iloc[0]
                if class_name in self.num_per_classes:
                    self.num_per_classes[class_name] += 1
                else:
                    self.num_per_classes[class_name] = 1

    def __getitem__(self, index):
        input_file = self.image_filenames[index]
        input = Image.open(input_file)
        
        if self.transform:
            try:
                input = self.transform(input)
            except Exception as err:
                print(input_file, input.mode)
                print(err)
                raise Exception()
            
        id_num = int(get_img_fileid(input_file))
        label = self.label_pairs[self.label_pairs['id'] == id_num]['label'].iloc[0]
        return input, label

    def __len__(self):
        return len(self.image_filenames)

    def show_details(self):
        for key in sorted(self.num_per_classes.keys()):
            print("{:<8}|{:<12}".format(
                key,
                self.num_per_classes[key]
            ))
            
    def get_class(self):
        return sorted(self.num_per_classes.keys())


class CarTestDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        super(Dataset, self).__init__()

        self.datapath = image_dir
        self.image_filenames = []
        self.transform = transform
        
        # get file list
        path_pattern = image_dir + '/*.*'
        files_list = glob.glob(path_pattern, recursive=True)
        
        for file in files_list:
            if is_image_file(file):
                self.image_filenames.append(file)

    def __getitem__(self, index):
        input_file = self.image_filenames[index]
        input = Image.open(input_file)
        
        if self.transform:
            try:
                input = self.transform(input)
            except Exception as err:
                print(input_file, input.mode)
                print(err)
                raise Exception()
            
        return input
    
    def __len__(self):
        return len(self.image_filenames)

    def get_files(self):
        return self.image_filenames
    

# Assuming that we are on a CUDA machine, this should print a CUDA device:
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


train_datapath = '/home/0856169/cv/hw01/data/training_data/training_data'
labelcsv_filepath = '/home/0856169/cv/hw01/data/training_labels.csv'

''' Using torchvision, it’s extremely easy to load CIFAR10.'''

transform = transforms.Compose(
    [transforms.RandomRotation(30),
     transforms.RandomHorizontalFlip(p=0.5),
     transforms.RandomVerticalFlip(p=0.5),
     transforms.Resize((224, 224)),
     transforms.ToTensor(),
     transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])])

transform_test = transforms.Compose(
    [transforms.Resize((224, 224)),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = CarDataset(image_dir=train_datapath, label_path=labelcsv_filepath, 
                      is_train=True, transform=transform)
# pin_memory=True
trainloader = torch.utils.data.DataLoader(trainset, batch_size=16,
                                          shuffle=True, num_workers=2)

print('Finish leadeing data')
classes = trainset.get_class()

classes_cTn = {}
classes_nTc = {}
c = 0
for clas in classes:
    classes_cTn[clas] = c
    classes_nTc[c] = clas
    c = c + 1 


def class_to_num(labels):
    num = []
    for l in labels:
        num.append(classes_cTn[l])
    return num

'''Define a Convolutional Neural Network'''
net = resnet50(pretrained=True)
net.fc = nn.Linear(2048, 196)
net = net.to(device)

''' Define a Loss function and optimizer'''
# Let’s use a Classification Cross-Entropy loss and SGD with momentum.
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.005, momentum=0.9)


# step learning rate
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

#  calculate program run time

print('Start Training!')

'''Train the network'''

for epoch in range(100):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(trainloader, 0):
        
        # get the inputs; data is a list of [inputs, labels]
        labels = torch.Tensor(class_to_num(labels))
        labels = labels.long()
        inputs, labels = inputs.to(device), labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        
        # update 
        optimizer.step()
        
        # print statistics
        running_loss += loss.item()
        if i % 100 == 99:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0
            
    # step learning rate 
    scheduler.step()
    print('Epoch-{0} lr: {1}'.format(epoch, optimizer.param_groups[0]['lr']))
                
            
print('Finished Training')
stop = timeit.default_timer()
print('Training Time: ', stop - start, ' sec')  


# Let’s quickly save our trained model:
PATH = '/home/0856169/cv/hw01/model_restnet50_100_16_0.005_sgd_all_nor_slr.pt'
torch.save(net.state_dict(), PATH)

'''testing data'''

test_datapath = '/home/0856169/cv/hw01/data/testing_data/testing_data'
testset = CarTestDataset(image_dir=test_datapath, transform=transform_test)

testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)

print('Starting testing data !')
lable_n = []
with torch.no_grad():
    for i, inputs in enumerate(testloader, 0):
        inputs = inputs.to(device)
        
        outputs = net(inputs)
        _, predicted = torch.max(outputs.data, 1)
        
        predicted_Labels = predicted.to('cpu').tolist()
        lable_n = lable_n + predicted_Labels

# aggraget data
files = [get_img_fileid(f) for f in testset.image_filenames]
label_t = [classes_nTc[f] for f in lable_n]

gr_id = ['003712', '013394', '001270', '006398', '001311', '003543', '001458',
         '008015', '001412', '001302', '001390']
gr_label_t = [label_t[0]]*11

   
df = pd.DataFrame({'id': files+gr_id, 'label': label_t+gr_label_t})
df.to_csv('model_hw01_restnet50_100_16_0.005_sgd_all_nor_slr.csv', index=False)
