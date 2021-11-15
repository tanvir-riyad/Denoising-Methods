# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 11:27:16 2021

@author: tanvir
"""
import modules
import torch
import HelperPlot
from statistics import mean,stdev

#%% 
class MovingAverage(object):
    """
    moving average class
    
    """ 
    dimension = [1,2,3]
    def __init__(self, window_size:int, stride:int):
        assert window_size > 0, "Window_size must be > 0."
        assert stride > 0, "Stride must be > 0."        
        
        self.window_size = window_size
        self.stride = stride
        self.padding = self.window_size //2
        self.ma_result = []
      
                                      
    def MyMovingAverage(self,noise:torch.Tensor):
        """
        perform moving averge of given input
        
        avgpool1d works for 3d input
        
        """  
        assert len(noise.shape) == 3,'dimension must be 3.'
        moving_average = torch.nn.AvgPool1d(self.window_size,self.stride,
                                            padding=self.padding,
                                            count_include_pad = False)
        return moving_average(noise)    
                                   
    def __call__(self,noise:torch.Tensor,number_of_signal):        
        assert len(noise.shape) in self.dimension,'dimension must be > 0 & < 4.'
        dim = len(noise.shape)
        if dim == 1:            
            noise_3d = noise.unsqueeze(0).unsqueeze(0)
            moving_averages = self.MyMovingAverage(noise_3d)
            moving_averages = moving_averages.squeeze(0).squeeze(0)
            self.ma_result.append(moving_averages)
            
        if dim == 2:
            for signal in range(number_of_signal):
                noise_3d = noise[signal,:].unsqueeze(0).unsqueeze(0)
                moving_averages = self.MyMovingAverage(noise_3d)
                moving_averages = moving_averages.squeeze(0).squeeze(0)
                self.ma_result.append(moving_averages)
                         
        elif dim == 3:
            noise = noise.reshape(number_of_signal,-1)
            for signal in range(number_of_signal):
                noise_3d = noise[signal,:].unsqueeze(0).unsqueeze(0)
                moving_averages = self.MyMovingAverage(noise_3d)
                moving_averages = moving_averages.squeeze(0).squeeze(0)
                self.ma_result.append(moving_averages)
                
        return self.ma_result,dim
                                                     
#%%
if __name__ =="__main__":
    
    #%%parameters
    noise_level = 0.1      
    time_interval = 0.001    
    window_size = 5
    stride = 1 
    x_min = -10
    x_max = 10 
    number_of_signal = 80
    freq = 5   
    image_format ='png'
    print_results = True
    save_results = False
    plot_result = False
    noise_output = []
    ma_output = []
    amp = 1           
    functions = "Sinusoid"
    dimension = [1,2,3]
    func = lambda x: amp*torch.sin(freq*x)
    
    #%%data generation
    x = torch.arange(x_min,x_max,time_interval)                     
    y = func(x)
    noise = torch.randn(len(x))*(noise_level)
    if len(noise.shape) == 2:        
        y = y.unsqueeze_(0)
    noisy_data = y + noise
                
    # reshape to test in 2D, 3D
    #noisy_data = noisy_data.reshape(number_of_signal,10,-1)    
    
 
                       
    #%%Moving average implementation    
    print("Moving average method begins....\n\n")
    def repeat_calucalation(y,noisy_data,pred_data):        
        l1,l2,linf = modules.ErrorMeasure(y, noisy_data, time_interval) #noisy data measurement
        noise_output.append([l1,l2,linf])       
        L1,L2,L_inf = modules.ErrorMeasure(y, pred_data,time_interval)
        ma_output.append([L1,L2,L_inf])
        return noise_output,ma_output
    
    ma_test =  MovingAverage(window_size,stride)
    moving_averages,dim = ma_test(noisy_data,number_of_signal)
    moving_averages = torch.vstack(moving_averages)
    if dim == 1:
        moving_averages = moving_averages.squeeze(0)
        for signal in range(number_of_signal):
            noise_output,ma_output = repeat_calucalation(y,noisy_data,
                                           moving_averages)        
    
    elif dim == 2:
        for signal in range(number_of_signal):
            noise_output,ma_output = repeat_calucalation(y[0,:], 
                                                noisy_data[signal,:],
                                                moving_averages[signal,:])
    elif dim == 3:
        noisy_data = noisy_data.reshape(number_of_signal,-1)
        for signal in range(number_of_signal):
            noise_output,ma_output = repeat_calucalation(y[0,:], 
                                                noisy_data[signal,:],
                                                moving_averages[signal,:])
            

    Noise_Errors = list(map(mean,zip(*noise_output)))
    Noise_deviations = list(map(stdev,zip(*noise_output)))
    ma_Errors = list(map(mean,zip(*ma_output)))
    ma_deviations = list(map(stdev,zip(*ma_output)))
    
    #%%print result    
    if print_results == True:        
        HelperPlot.Print_in_Console(Noise_Errors, Noise_deviations,"noise")
        HelperPlot.Print_in_Console(ma_Errors, ma_deviations, functions)
        
    #%%Saving Result    
    if save_results == True:
        HelperPlot.SaveFilterResult("moving average of "+ functions,Noise_Errors, Noise_deviations)
        HelperPlot.SaveFilterResult("moving average of "+ functions,ma_Errors, ma_deviations)

    #%%Plotting Result    
    if plot_result == True:
        HelperPlot.Subplot3(y, noise, moving_averages ,freq,'Moving Average', 
                            'Moving Average of '+functions +' of frequency '+str(freq[0])+',',image_format)
        
    print("moving average ends")     
