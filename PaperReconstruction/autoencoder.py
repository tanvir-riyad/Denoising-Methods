# -*- coding: utf-8 -*-
"""
Created on Tue Aug 17 12:31:43 2021

@author: tanvir
"""

import torch
import torch.nn as nn
import HelperPlot
import modules
from statistics import mean,stdev
import HelperFunction
import optuna
from optuna.trial import TrialState
from torch.autograd import Variable
import pandas as pd
import os


class Autoencoder(nn.Module):
    """
    model of an autoencoder that includs encoder and decoder
    
    parameters:
        inp_channel : int ,required
                      number of features
    
     """   
    
    def __init__(self,kernel,inp_channels = 80):        
        super(Autoencoder, self).__init__()
        self.inp_channels = inp_channels
        self.Encoder = torch.nn.Sequential()
        self.Decoder = torch.nn.Sequential()
        
        self.Encoder.add_module("c1", nn.Conv1d(inp_channels, 128, kernel_size=kernel,stride=1))
        self.Encoder.add_module("Relu1",nn.ReLU())
        self.Encoder.add_module("c2", nn.Conv1d(128, 64, kernel_size=kernel,stride=1))        
        self.Encoder.add_module("Relu2",nn.ReLU())
        self.Encoder.add_module("c3", nn.Conv1d(64, 8, kernel_size=kernel,stride=1))
        self.Encoder.add_module("Relu3",nn.ReLU())
        self.Encoder.add_module("c4", nn.Conv1d(8, 2, kernel_size=kernel,stride=1))
        self.Encoder.add_module("Relu4",nn.ReLU())
        
        self.Decoder.add_module("d1", nn.ConvTranspose1d(2,8,kernel_size=kernel,stride=1))
        self.Decoder.add_module("Relu5",nn.ReLU())
        self.Decoder.add_module("d2", nn.ConvTranspose1d(8,64,kernel_size=kernel,stride=1))
        self.Decoder.add_module("Relu5",nn.ReLU())
        self.Decoder.add_module("d3", nn.ConvTranspose1d(64,128,kernel_size=kernel,stride=1))
        self.Decoder.add_module("Relu6",nn.ReLU())
        self.Decoder.add_module("f1", nn.Flatten(start_dim=1))
        linear_input = self.get_output_feature(inp_channels,window_length)                
        self.Decoder.add_module("fc1", nn.Linear(linear_input, 16000))

    def get_output_feature(self,inp_channels,window_length):
        batch_size = 16
        inp = Variable(torch.rand(batch_size, inp_channels,window_length))
        output_feature = self.forward_feature(inp)
        linear_input = output_feature.shape[1]
        return linear_input
                       
    def forward_feature(self, x):
        x = self.Encoder(x)
        x = self.Decoder(x) 
        return x
    
    def forward(self,x):
        x = self.forward_feature(x)
        x1 = x.reshape(-1,80,200)                
        return x1

#%%train autoencoder
class Train_AutoEncoder(nn.Module):
    """
    A class for training and testing autoencoder model and also visualizing the test,
    trainging losses and output of orginal,noisy and denoised signal
    
    parameters:
          model   :  autoencoder model
        criterion :  loss functions
            lr     : learning rate 
    returns:
        test and training losses,min losses,epoch at which min loss occurs
    """
        
    def __init__(self,model,criterion,lr):
        super(Train_AutoEncoder, self).__init__()        
        self.model = model
        self.criterion = criterion
        self.lr = lr
        self.test_loss = 0
        self.print_results = True
        self.optimizer = torch.optim.Adam(model.parameters(),lr=self.lr)
        #self.store = Storages.FigureStorage("Results")
        self.store_traindata = HelperFunction.DataStorage(['Epoch', 'Batch','train_loss'], average_window=2, 
                                                          show=1, line=1, header=5000, step=1, precision=3, name = "")         
        self.store_test_data = HelperFunction.DataStorage(['Epoch','Batch','test_loss'],average_window=2, 
                                                          show=1, line=1, header=5000, step=1, precision=3, name = "")
            
    
    def Train(self, train_dataloader, model,epoch): 
        """
        training autoencoder
        
        parameters:
            train_dataloader:dataloader,required
            model : required
            
        return:
            train_error
        
        """
        self.model.train()    
        predictions=[]
        train_error = 0
        n_iteration = len(train_dataloader)
        for i, data in enumerate(train_dataloader):
            inputs, target = data
            inputs,target = inputs.to(device),target.to(device)
            self.optimizer.zero_grad()               
            pred = self.model(inputs)
            loss = self.criterion(pred, target)
            loss.backward()
            self.optimizer.step()
            train_error += loss.item()
            predictions.append(pred)

            with open(HelperPlot.train_file,"a",encoding ="utf-8") as f:
                f.writelines([str(epoch),'\t', str(i),'\t',str(loss.item()),'\n'])
            self.store_traindata.Store([epoch, i, loss])

                        
        train_loss =  train_error/n_iteration
        #self.error.append(train_loss)
        return train_loss,model          

    def test(self, test_dataloader,model,epoch):
        """
        testing autoencoder
        
        parameters:
            test_dataloader: dataloader,required
            model : required
        return:
            test_error
        
        """
        self.model.eval()
        with torch.no_grad():
            test_error = 0
            test_iteration=len(test_dataloader)
            pred = []
            trg = []
            noise = []
            for i,(inputs,target) in enumerate(test_dataloader):
                pred_t = self.model(inputs.to(device))
                noise.append(inputs)
                pred.append(pred_t)
                trg.append(target.to(device))   
                tst_loss = self.criterion(pred_t, target.to(device))
                test_error+= tst_loss.item()
                with open(HelperPlot.filepath,"a",encoding ="utf-8") as f:
                    f.writelines([str(epoch),'\t', str(i),'\t',str(tst_loss.item()),'\n'])
                #self.store_test_data.Store([epoch, i, test_error])
            noise=torch.cat(noise,dim=0)
            pred = torch.cat(pred, dim = 0)        
            trg = torch.cat(trg, dim = 0)
            test_loss = test_error/test_iteration
            return noise, pred, trg, test_loss        
    
def result_reshape(noise,target,pred,number_of_signal):
    noise_plot = noise.transpose(0,1)
    noise_plot = noise_plot.reshape(number_of_signal,-1)
    pred_plot = pred.transpose(0,1)
    pred_plot = pred_plot.reshape(number_of_signal,-1)
    trg_plot = target.transpose(0,1)  
    trg_plot = trg_plot.reshape(number_of_signal,-1)
    return noise_plot,pred_plot,trg_plot
    
def repeat_calucalation(y,noisy_data,pred_data):        
        l1,l2,linf = modules.ErrorMeasure(y, noisy_data, time_interval) #noisy data measurement
        noise_output.append([l1,l2,linf])
        L1,L2,L_inf = modules.ErrorMeasure(y, pred_data,time_interval)
        ae_output.append([L1,L2,L_inf])
        return noise_output,ae_output

def objective(trial):

    lr = trial.suggest_float("lr", 1e-5, 1e-1, log = True)
    kernel = trial.suggest_int('kernel',2,6,log = True)
    batch_size = trial.suggest_int('batch_size',8,16, log = True)
   
    model = Autoencoder(kernel).to(device)  
    #optimizer = torch.optim.Adam(model.parameters(),lr=lr)
    train_test_loop = Train_AutoEncoder(model,nn.MSELoss(),lr)
    #test_error = train_test_loop(model,train_dataloader, test_dataloader,n_epochs,number_of_signal)
    test_losses = []
    test_pred=[]
    train_losses = []
    with open(HelperPlot.filepath,"a",encoding ="utf-8") as f:
        f.writelines(['epoch','\t', 'batch','\t','loss','\n'])
    with open(HelperPlot.train_file,"a",encoding ="utf-8") as f:
        f.writelines(['epoch','\t', 'batch','\t','loss','\n'])
    
    
    for epoch in range(0, epochs):                                                                   
        train_loss, model = train_test_loop.Train(train_dataloader, model,epoch)
        train_losses.append(train_loss)
        torch.save(model.state_dict(), './model.pt')                                                       
        noise,pred, trg, test_loss = train_test_loop.test(test_dataloader, model,epoch)
        test_losses.append(test_loss)
        test_pred.append(pred)
        trial.report(test_loss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
    
    #return test_losses
    min_loss = min(test_losses)
    min_loss_index = test_losses.index(min_loss)
    #min_loss_epoch = (min_loss_index+1)
    min_loss_pred = test_pred[min_loss_index]
    noise_plot,pred_plot,trg_plot = result_reshape(noise, trg, min_loss_pred, number_of_signal)
           
    for signal in range(number_of_signal):
        noise_output,ae_output = repeat_calucalation(trg_plot[signal,:],
                                noise_plot[signal,:],pred_plot[signal,:])

    Noise_Errors = list(map(mean,zip(*noise_output)))
    Noise_deviations = list(map(stdev,zip(*noise_output)))
    ae_Errors = list(map(mean,zip(*ae_output)))
    ae_deviations = list(map(stdev,zip(*ae_output)))    
   
    #%%print result    
    if print_results == True:        
        HelperPlot.Print_in_Console(Noise_Errors, Noise_deviations,"noise")
        HelperPlot.Print_in_Console(ae_Errors, ae_deviations, functions)
     
     #%%Plotting Result    
    if plot_result == True:
        HelperPlot.Subplot3(trg_plot[min_loss_index,:], noise_plot[min_loss_index,:], pred_plot[min_loss_index,:], freq, 
                        'Autoencoder', 'Autoencoder of '+functions +' of frequency '+str(5)+',',
                        image_format)
        HelperPlot.Subplot2(train_losses, test_losses, "train and test error", image_format, "autoencoder")
    return test_losses[-1]


# Use GPUs if avaiable
if torch.cuda.is_available():
    print("CUDA Available")
    device = torch.device('cuda:0')
else:
    print('CUDA Not Available')
    device = torch.device('cpu')                                                                            
    
#%%main method
if __name__ =="__main__":
    
    #%%parameters
    noise_level = 0.1      
    time_interval = 0.001     
    x_min = -10
    x_max = 10
    epochs = 5   
    lr = 0.001
    window_length = 200
    batch_size = 16
    number_of_signal = 80
    #freq = torch.randn(number_of_signal,len(x)   
    image_format ='png'
    print_results = True
    save_results = False
    plot_result = True
    noise_output = []
    ae_output = []           
    functions = "Sinusoid"
    kernel = 3
    func = lambda x: amp*torch.sin(freq*x)   
    
    #%% parameter storage
    parameter_store = HelperPlot.Saveparameters('Parameters:'+'\n'+'noise_level =  0.1'+'\n'+
                                                'time_interval =  0.001'+'\n'+
                                                'epochs = 5'+'\n'+
                                                'lr = 0.001'+'\n'+
                                                'window_length  = 200'+'\n'+
                                                'batch_size = 16'+'\n'+
                                                'number_of_signal = 80'+'\n'+
                                                'kernel = 3'+'\n')
    

    
    
    
    #%%data generation
    x = torch.arange(x_min,x_max,time_interval)
    freq = 5 #torch.randn(number_of_signal,len(x))+4.5
    amp = 1 # torch.rand(number_of_signal,len(x))
    data = func(x)
    y =torch.zeros(number_of_signal,len(x))
    y = y + data
    noise = torch.randn(number_of_signal,len(x))*(noise_level)
    noisy_data = y + noise
        
    # reshape to test in 2D, 3D
    #noisy_data = noisy_data.reshape(number_of_signal,10,-1)    
    dim = len(noisy_data.shape)  
                

    #%% split the data
    datasplit = modules.Datasplit(number_of_signal)
    y_train,y_test,noise_train,noise_test = datasplit(noisy_data,y,window_length)   
    
    #%% create the dataloaders   
    train_dataset = modules.Dataset(noise_train, y_train)
    test_dataset = modules.Dataset(noise_test, y_test)
    dataloaders = modules.dataloader(train_dataset, test_dataset, batch_size)    
    train_dataloader,test_dataloader=dataloaders()
        
    #%%Autoencoder implementation+
    print("Autoencoder method begins....\n\n")
# =============================================================================
#     model=Autoencoder(kernel)       
#     #HelperPlot.Saveparameters(str(model)+"\n")
#     train_test_loop = Train_AutoEncoder(model,nn.MSELoss(),lr)
#     test_error = train_test_loop(model,train_dataloader, test_dataloader,n_epochs,number_of_signal)
# =============================================================================
    
    #return test_error #noise_plot,pred_plot,target_plot,index  
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=2)
    study.trials_dataframe()
    # Find number of pruned and completed trials
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    # Display the study statistics
    print("\nStudy statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    trial = study.best_trial
    print("Best trial:")
    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
        
    
# =============================================================================
#     # Save results to csv file
#     df = study.trials_dataframe().drop(['datetime_start', 'datetime_complete', 'duration'], axis=1)  # Exclude columns
#     df = df.loc[df['state'] == 'COMPLETE']        # Keep only results that did not prune
#     df = df.drop('state', axis=1)                 # Exclude state column
#     df = df.sort_values('value')                  # Sort based on accuracy
#     df.to_csv('optuna_results.csv', index=False)  # Save to csv file
# 
#     # Display results in a dataframe
#     print("\nOverall Results (ordered by accuracy):\n {}".format(df))
# =============================================================================

    # Find the most important hyperparameters
    most_important_parameters = optuna.importance.get_param_importances(study, target=None)

    # Display the most important hyperparameters
    print('\nMost important hyperparameters:')
    for key, value in most_important_parameters.items():
        print('  {}:{}{:.2f}%'.format(key, (15-len(key))*' ', value*100))
                  
       
# =============================================================================
#     for signal in range(number_of_signal):
#         noise_output,ae_output = repeat_calucalation(target_plot[signal,:],
#                                     noise_plot[signal,:],pred_plot[signal,:])
# 
#     Noise_Errors = list(map(mean,zip(*noise_output)))
#     Noise_deviations = list(map(stdev,zip(*noise_output)))
#     ae_Errors = list(map(mean,zip(*ae_output)))
#     ae_deviations = list(map(stdev,zip(*ae_output)))    
#    
#     #%%print result    
#     if print_results == True:        
#         HelperPlot.Print_in_Console(Noise_Errors, Noise_deviations,"noise")
#         HelperPlot.Print_in_Console(ae_Errors, ae_deviations, functions)
#         
#         
#     #%%Saving Result    
#     if save_results == True:
#         HelperPlot.SaveFilterResult("Autoencoders of "+ functions,Noise_Errors, Noise_deviations)
#         HelperPlot.SaveFilterResult("Autoencoders of "+ functions,ae_Errors, ae_deviations)
# 
#     #%%Plotting Result    
#     if plot_result == True:
#         HelperPlot.Subplot3(target_plot[index,:], noise_plot[index,:], pred_plot[index,:], freq, 
#                             'Autoencoder', 'Autoencoder of '+functions +' of frequency '+str(5)+',',
#                             image_format)
#         HelperPlot.Subplot2(train_error, test_error, "train and test error", image_format, "autoencoder")
#          
#         
#         
#     #%% store last value of test and train data in parameterstorage    
#           
#     print("autoencoder method ends....")
# 
# =============================================================================
