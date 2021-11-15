# -*- coding: utf-8 -*-
"""
Created on Tue Aug 17 12:52:10 2021

@author: tanvir
"""
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader



def SMAPE(target:torch.Tensor, predict:torch.Tensor): 
    return 1/len(target) * torch.sum(2 * torch.abs(predict-target) / (torch.abs(target) + torch.abs(predict))*100)

def L1Norm(target:torch.Tensor, predict:torch.Tensor, time_interval:float):
    return ((torch.sum(torch.abs(target - predict)))*time_interval).item()

def L2Norm(target:torch.Tensor, predict:torch.Tensor, time_interval:float):
    return  torch.sqrt(torch.sum(torch.pow((torch.abs(target - predict)),2))*time_interval).item()

def Linfinity(target:torch.Tensor, predict:torch.Tensor):
    return max(torch.abs(target - predict)).item()

def ErrorMeasure(target:torch.Tensor, predict:torch.Tensor, time_interval:float):        
        l1norm = L1Norm(target, predict, time_interval)
        l2norm = L2Norm(target, predict, time_interval)
        l_inf = Linfinity(target, predict)
        return l1norm,l2norm,l_inf
                                    
#%%Data generation
class data_generation:
    """
    creates original and noisy data for creating dataset and 
    dataloader
    
    parameters:
        time_interval:integer,required
                      length of the data
                      
        noise_level : float,required
                      amplitude of the noise
                      
    returns:
        noisy data and original data with 3 diffrent frequency
        
    """          
    
    def __init__(self,time_interval:float, noise_level:float, x_max:int, x_min:int,
                 freq:int, amp:int):
        self.time_interval = time_interval
        self.noise_level = noise_level
        self.x_min = x_min
        self.x_max = x_max
        self.freq = freq
        self.amp = amp
        self.functions = {'Sinusoid':self.Sinusoid,'Quadratic':self.Quadratic,
                          'Step':self.Step}
        
    def Sinusoid(self,x:torch.Tensor, freq:int, amp:int):
        y1 = amp*torch.sin(x*freq)
        return y1
        
    def Quadratic(self,x:torch.Tensor):    
        x_list = x.tolist()
        y_list = []
        for t in x_list:                    
            if t<0 and t>=-10:
                y = -(t**2)/2
            
            elif t>=0 and t<=10:
                y = (t**2)/2
            
            y_list.append(y)
        y2 = torch.FloatTensor(y_list)
        return y2
        
    def Step(self,x:torch.Tensor):
        x_list = x.tolist()
        y_list = []

        for t in x_list:                    
            if t<-2 and t>=-10:
                y = 5
                
            elif t<=7 and t>=-2:
                y = -3
            
            elif t>7 and t<=10:
                y = 8
            
            y_list.append(y)
        y3 = torch.FloatTensor(y_list)
        return y3                    
        
    def __call__(self,function):
        x = torch.arange(self.x_min,self.x_max,self.time_interval)
        if function in self.functions:
            y = self.functions[function](x,self.freq,self.amp)
        return x,y                  
                 
#%%Data split
class Datasplit:
    """
    split data for test and training purpose
    
    parameters:
        split_percentage:float,required
                         represents how much data for training
    returns:
        test and training data for original and noisy value
    
    """    
    def __init__(self,num_dim,split_percentage=0.7):
        self.split=split_percentage
        self.num_dim = num_dim
                
    def __call__(self,noise,y, seq_length):
        max_values=noise.shape[1]
        slice_index=int(self.split*max_values)   
        y_train = y[:,:slice_index]
        y_test = y[:,slice_index:]
        noise_train = noise[:,:slice_index]        
        noise_test = noise[:,slice_index:]
        y_train = y_train.reshape(self.num_dim,-1,seq_length)
        y_train = y_train.transpose(0,1)               
        y_test = y_test.reshape(self.num_dim,-1,seq_length)
        y_test = y_test.transpose(0,1)        
        noise_train = noise_train.reshape(self.num_dim,-1,seq_length)
        noise_train = noise_train.transpose(0,1)
        noise_test = noise_test.reshape(self.num_dim,-1,seq_length) 
        noise_test = noise_test.transpose(0,1)
        return y_train,y_test,noise_train,noise_test

#%%Dataset            
class Dataset(Dataset):
    """
    create dataset where original and noisy data can be together with 
    separate column
    
    parameters:
        noisy:tensor or numpy array,required
              noisy data
        y    :tensor or numpy array,required
              original data
    returns:
        dataset with noisy and original data
    
    """            
    def __init__(self,noise,y):
        self.inputs = noise
        self.target = y
        
    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        x = self.inputs[index]
        y = self.target[index]        
        return x,y
    
#%%dataloader
class dataloader():
    """
    create dataloader where dataset is divided into batch which is required
    for training and testingo of neural network
    
    parameter:
        train_dataset :dataset,required
        test_dataset  :dataset,required
        
    returns:
        dataloader for training and testing
    """
    def __init__(self,train_dataset,test_dataset,batch_size):
        self.train_dataset=train_dataset
        self.test_dataset =test_dataset
        self.batch_size= batch_size
    
    def __call__(self):
        train_dataloader=DataLoader(self.train_dataset,batch_size=self.batch_size,shuffle=True)
        test_dataloader=DataLoader(self.test_dataset,batch_size=self.batch_size,shuffle=False)
        return train_dataloader,test_dataloader