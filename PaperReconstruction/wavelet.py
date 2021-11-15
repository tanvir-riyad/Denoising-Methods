# -*- coding: utf-8 -*-
"""
Created on Sun Sep 12 22:43:58 2021

@author: Riyad
"""

import torch
import HelperPlot
import modules
import pyyawt
from numpy import float64
import numpy as np
import warnings
from statistics import mean,stdev

warnings.filterwarnings("ignore")

class Wavelet(object):
    
    dimension = [1,2,3]
    def __init__(self,thresh_select:str, SORH_select:str, scaling:str, levels:int):
        assert levels > 0
                 
        self.thresh_select = thresh_select
        self.SORH_select = SORH_select
        self.scaling = scaling
        self.levels = levels
        self.wavelet_result = []        
        
    def MyWavelet(self,noise:np.ndarray,wavelet:str):
        """
        inp must be 1D

        """   
        assert len(noise.shape) == 1,'dimesnion must be 1 for wavelet filter'            
        res,_,_ = pyyawt.denoising.wden(noise,self.thresh_select, self.SORH_select,
                                               self.scaling,levels,wavelet)
        denoised = torch.from_numpy(res)
        return denoised
    
    def __call__(self,noise:np.ndarray, wavelet:str,number_of_signal:int):
         assert len(noise.shape) in self.dimension,'dimension must be > 0 & < 4.'
         dim = len(noise.shape)
         if dim == 1:             
             wavelet_filter = self.MyWavelet(noise,wavelet)
             self.wavelet_result.append(wavelet_filter)

         elif dim == 2:
             for signal in range(number_of_signal):
                 wavelet_filter = self.MyWavelet(noise[signal,:],wavelet)
                 self.wavelet_result.append(wavelet_filter)
                 
         elif dim == 3:
             noise = noise.reshape(number_of_signal,-1)
             for signal in range(number_of_signal):
                 wavelet_filter = self.MyWavelet(noise[signal,:],wavelet)
                 self.wavelet_result.append(wavelet_filter)
                                   
         return self.wavelet_result,dim
        
if __name__ =="__main__":
    
    #%%parameters
    x_max = 10
    x_min = -10
    noise_level = 0.1         
    time_interval = 0.001
    freq = 5   
    number_of_signal = 80
    thresh_select = 'sqtwolog'
    SORH_select = 's'
    scaling = 'mln'
    levels = 5
    image_format ='png'
    wavelet = ['db8','coif2','db1']
    functions = 'Sinusoid'       
    print_results = True
    save_results = False
    plot_result = False
    noise_output = []
    wavelet_output = []
    amp = 1
    dimension = [1,2,3]
    func = lambda x: amp*torch.sin(freq*x)               
             
    #%%data generation
    x = torch.arange(x_min,x_max,time_interval)                     
    y = func(x)
    noise = torch.randn(number_of_signal,len(x))*(noise_level)
    if len(noise.shape) == 2:        
        y = y.unsqueeze_(0)        
    noisy_data = y + noise
       
    #reshape to to test in 2D,3D
    #noisy_data = noisy_data.reshape(number_of_signal,10,-1)
    dim = len(noisy_data.shape)
          
    #%%FFT implementation
    print("Wavelet methods begins....\n\n")
    assert dim in dimension,'dimension must be > 0 & < 4.'
    def repeat_calucalation(y,noisy_data,pred_data):        
        l1,l2,linf = modules.ErrorMeasure(y, noisy_data, time_interval) #noisy data measurement
        noise_output.append([l1,l2,linf])        
        L1,L2,L_inf = modules.ErrorMeasure(y, pred_data,time_interval)
        wavelet_output.append([L1,L2,L_inf])
        return noise_output,wavelet_output
    
    wavelet_test = Wavelet(thresh_select,SORH_select,scaling,levels)
    inp = noisy_data.numpy()
    inp = float64(inp)
    wavelet_filter,dim = wavelet_test(inp,wavelet[0],number_of_signal)
    wavelet_filter = torch.vstack(wavelet_filter)
        
    if dim == 1:
        wavelet_filter = wavelet_filter.squeeze(0)
        for signal in range(number_of_signal):
            noise_output,wavelet_output = repeat_calucalation(y, noisy_data,wavelet_filter)
            
    elif dim == 2:
        for signal in range(number_of_signal):
            noise_output,wavelet_output = repeat_calucalation(y[0,:],
                                                                noisy_data[signal,:],
                                                                wavelet_filter[signal,:])
    
    elif dim == 3:
        noisy_data = noisy_data.reshape(number_of_signal,-1)
        for signal in range(number_of_signal):
            noise_output,wavelet_output = repeat_calucalation(y[0,:],
                                                                    noisy_data[signal,:],
                                                                    wavelet_filter[signal,:])

    Noise_Errors = list(map(mean,zip(*noise_output)))
    Noise_deviations = list(map(stdev,zip(*noise_output)))
    wavelet_Errors = list(map(mean,zip(*wavelet_output)))
    wavelet_deviations = list(map(stdev,zip(*wavelet_output)))        
    
     #%%print result    
    if print_results == True:        
        HelperPlot.Print_in_Console(Noise_Errors, Noise_deviations,"noise")
        HelperPlot.Print_in_Console(wavelet_Errors, wavelet_deviations,functions)
    
    #%%Saving Result    
    if save_results == True:
        HelperPlot.SaveFilterResult("Wavelet of "+ functions,Noise_Errors, Noise_deviations)
        HelperPlot.SaveFilterResult("Wavelet of "+ functions,wavelet_Errors, wavelet_deviations)
    
    #%%Plotting Result    
    if plot_result == True:
        HelperPlot.Subplot3(y, noise, wavelet_filter ,freq,'Wavelet', 
                            'Wavelet of '+functions+' of frequency '+str(freq)+',',image_format)
    print("wavelet method ends")