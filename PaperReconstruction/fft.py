# -*- coding: utf-8 -*-
"""
Created on Tue Aug 17 12:30:07 2021

@author: tanvir
"""
import torch
import HelperPlot
import modules
from statistics import mean,stdev
    
class FFT():
    """
    class for doign Fast fourier transform
    freq: 
    
    """ 
    dimension = [1,2,3]
    def __init__(self, time_interval:float):            
        self.time_interval = time_interval
        self.fft_result = []
                            
    def MyFFT(self,inp:torch.Tensor, fmax:float):
        """
        perform fft of the given input
        inp: has to be 1D
        fmax: maximum frequency of lowpass
        """
        assert len(inp.shape) == 1,'dimension must be 1'
        freqpass = torch.fft.fftn(inp)
        freqfft = torch.fft.fftfreq(len(inp), self.time_interval)        
        freqpass[torch.abs(freqfft) > fmax] = 0  # setting freq above fmax to 0
        DenoisedSignal = torch.real(torch.fft.ifft(freqpass))
        return DenoisedSignal
        
    def __call__(self,noise:torch.Tensor, fmax:float, number_of_signal:int):
        assert len(noise.shape) in self.dimension,'dimension must be > 0 & < 4.'
        dim = len(noise.shape)
        if dim == 1:           
            fft_filter = self.MyFFT(noise,fmax)
            self.fft_result.append(fft_filter)
            
        elif dim == 2:
            for signal in range(number_of_signal):
                fft_filter = self.MyFFT(noise[signal,:],fmax)
                self.fft_result.append(fft_filter)
        
        elif dim == 3:
            noise = noise.reshape(number_of_signal,-1)
            for signal in range(number_of_signal):
                fft_filter = self.MyFFT(noise[signal,:],fmax)
                self.fft_result.append(fft_filter)                                             
        return self.fft_result,dim                    
        
#%%main method
if __name__ =="__main__":
    
    #%%parameters
    noise_level = 0.1     
    time_interval = 0.001    
    x_min = -10
    x_max = 10 
    number_of_signal = 80
    freq = 5  
    print_results = True
    save_results = False
    plot_result = False
    functions = 'Sinusoid'
    noise_output = []
    fft_output = []
    amp = 1    
    fft_fmax = [5.0,500.0,170.0]    
    image_format ='png'
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
    #%%FFT implementation
    print("FFT method begins....\n\n")
    def repeat_calucalation(y,noisy_data,pred_data):        
        l1,l2,linf = modules.ErrorMeasure(y, noisy_data, time_interval) #noisy data measurement
        noise_output.append([l1,l2,linf])
        L1,L2,L_inf = modules.ErrorMeasure(y,pred_data,time_interval)
        fft_output.append([L1,L2,L_inf])
        return noise_output,fft_output
    
    fft_test =  FFT(time_interval)
    fft_filter, dim = fft_test(noisy_data,fft_fmax[0], number_of_signal)
    fft_filter = torch.vstack(fft_filter)
   
    if dim == 1:
        fft_filter = fft_filter.squeeze(0)
        for signal in range(number_of_signal):
            noise_output,fft_output = repeat_calucalation(y,noisy_data,
                                                          fft_filter)
    elif dim == 2:
        for signal in range(number_of_signal):
            noise_output,fft_output = repeat_calucalation(y[0,:],
                                                        noisy_data[signal,:],
                                                        fft_filter[signal,:])
    elif dim == 3:
        noisy_data = noisy_data.reshape(number_of_signal,-1)
        for signal in range(number_of_signal):
            noise_output,fft_output = repeat_calucalation(y[0,:],
                                                        noisy_data[signal,:],
                                                        fft_filter[signal,:])   
        
    Noise_Errors = list(map(mean,zip(*noise_output)))
    Noise_deviations = list(map(stdev,zip(*noise_output)))
    fft_Errors = list(map(mean,zip(*fft_output)))
    fft_deviations = list(map(stdev,zip(*fft_output)))
     #%%print result    
    if print_results == True:        
        HelperPlot.Print_in_Console(Noise_Errors, Noise_deviations,"noise")
        HelperPlot.Print_in_Console(fft_Errors, fft_deviations,functions)
    
    #%%Saving Result    
    if save_results == True:
        HelperPlot.SaveFilterResult("FFT of "+ functions,Noise_Errors, Noise_deviations)
        HelperPlot.SaveFilterResult("FFT of "+ functions,fft_Errors, fft_deviations)
    
    #%%Plotting Result    
    if plot_result == True:
        HelperPlot.Subplot3(y, noise, fft_filter ,freq,'FFT', 
                            'FFT of '+functions+' of frequency '+str(freq)+',',image_format)        
    print("FFT method ends")
     