# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 21:10:34 2021

@author: tanvir
"""
import time
import matplotlib.pyplot as plt
import os
import pathlib


date = time.strftime("%d-%m-%Y")
datetime = time.strftime("%d-%m-%YT%H_%M")
DateTimeSec = time.strftime("%d-%m-%YT%H_%M_%S")
cwd = pathlib.Path().resolve()
path = cwd.parent.joinpath("Results/results of " + datetime)
path.mkdir(parents=True,exist_ok=True)
filepath = os.path.join(path,("test_result," + DateTimeSec + ".txt"))
train_file = os.path.join(path,("train_result," + DateTimeSec + ".txt"))
def CustomePlot():
    """
    customize the plot
    
    """
            
    font = {'style'  : 'normal',
            'family' : 'Times New Roman',
            'weight' : 'bold',
            'size'   :  20
           }
    line = {'linewidth' : 3,
            'linestyle' : 'solid',
            'color'     : 'blue'
           }
    text = {'color'  : 'black'
           }
    axes = {'titlesize'  : 20,
            'labelsize'  : 20,
            'grid'       : False             
              }
    xtick = {'top'       : True,
            'direction'  : 'in',
            'labelsize'  : 16
            }
    ytick = {'right'       : True,
             'direction'  : 'in',
             'labelsize'  : 16
            }
    grid  = {'color' : 'black',
             'linewidth' : 0.5,             
            }
    legend = {'loc'      : 'upper right',
              'fontsize' : 'small',
              'fancybox' : True
             }
    figure = {'titlesize' : 'large',
              'titleweight': 'normal',
              'figsize'   : (8,6)
              }
    plt.rc('font',**font)
    plt.rc('text',**text)
    plt.rc('figure',**figure)
    #plt.rc('grid',**grid)
    plt.rc('axes',**axes)
    plt.rc('lines',**line)
    plt.rc('xtick',**xtick)
    plt.rc('ytick',**ytick)    
    plt.rc('legend',**legend)
    return plt

def Subplot3(arg1,arg2,arg3,freq,filter_name,filename,file_format):
    
    """
    plot a figure which have 3 subplots and save the figure to working directory 
    
    paramters:
        arg1 : first variables to plot
        arg2:  second vairables to plot
        arg3:  third variables to plot
        freq:  frequency
        filter_name :  the method used to denoise
        filename : name of the file to be saved
        file_format : file format to be saved
        
    """
         
    CustomePlot().subplot(3,1,1)
    CustomePlot().title(filter_name+' with frequency '+ str(freq)) 
    CustomePlot().xlabel("Time")
    CustomePlot().ylabel("Amplitude")
    
    CustomePlot().plot(arg1,label='Original')
    CustomePlot().legend()
    CustomePlot().subplot(3,1,2)
    CustomePlot().xlabel("Time")
    CustomePlot().ylabel("Amplitude")
    
    CustomePlot().plot(arg2,label='noisy')
    CustomePlot().legend()
    CustomePlot().subplot(3,1,3)
    CustomePlot().xlabel("Time")
    CustomePlot().ylabel("Amplitude")
    
    CustomePlot().plot(arg3,label='denoised')
    CustomePlot().legend()
    inner_path = path.joinpath(filter_name)
    inner_path.mkdir(parents=True, exist_ok=True)
    DateTimeSec = time.strftime("%d-%m-%YT%H_%M_%S")
    plt.savefig(os.path.join(inner_path, filename + DateTimeSec + '.'+ file_format),format=file_format)
    CustomePlot().close()

def Subplot2(arg1,arg2,filename,file_format,filter_name):
    """
    plot a figure which have 2 subplots and save the figure to working directory
    
    parameters:
        arg1 : first variable to plot
        arg2 : second variable to plot
        filename : file name to be saved
        file_format : file format to be saved
        
    """
        
    CustomePlot().subplot(2,1,1)
    CustomePlot().plot(arg1)
    CustomePlot().title("training loss")
    CustomePlot().subplot(2,1,2)
    CustomePlot().plot(arg2)
    CustomePlot().title("test loss")
    inner_path = path.joinpath(filter_name)
    inner_path.mkdir(parents=True, exist_ok=True)
    DateTimeSec = time.strftime("%d-%m-%YT%H_%M_%S")
    CustomePlot().savefig(os.path.join(inner_path, filename + DateTimeSec +'.'+file_format),format=file_format)
    CustomePlot().close()                     
              
    
def SaveFilterResult(Filter,mean,std):
    """
    save the result of denoised method in working directory
    
    """
    signal = "Signal: "+Filter + "\n\n"    
    l1_error = "L1 norm error     : " + str(round(mean[0],3)) +" "+ str(u"\u00B1") +" "+ str(round(std[0],3))+"\n"
    l2_error = "L2 norm error     : " + str(round(mean[1],3)) + " "+ str(u"\u00B1")+ " "+ str(round(std[1],3))+"\n"
    l_inf_error = "L Infinity error  : "+str( round(mean[2],3)) + " " + str( u"\u00B1") +" " + str(round(std[2],3))+"\n\n\n"
    #smape_error = "smape error       : "+str(round(smape,3))+"\n\n\n"    
    with open(filepath,"a",encoding ="utf-8") as f:
        f.writelines([signal,l1_error, l2_error,l_inf_error])
        
    
def SaveTestTrainResult(train_error,test_error,min_loss_epoch,min_loss):
    """
    save the autoencoder training and testing losses ,minimum losses and epoch at which 
    minimum losses occurs in working directory
    
    """
    train_loss = "training loss        : " + str(round(train_error,3))+"\n"
    test_loss = "test loss            : " +str(round(test_error,3))+"\n"
    min_epoch_loss = "minimum loss at epoch : " + str(min_loss_epoch) + " \nand the loss is      : " + str(min_loss)+"\n\n\n"
    with open(filepath,"a",encoding ="utf-8") as f:
         f.writelines([train_loss,test_loss,min_epoch_loss])
         
def Saveparameters(text):
    with open(filepath,"a",encoding ="utf-8") as f:
        f.writelines([text])
        
def SaveTrainResult(train_error):
    """
    save the autoencoder training and testing losses ,minimum losses and epoch at which 
    minimum losses occurs in working directory
    
    """
    train_loss = "training loss        : " + str(train_error)+"\n"
    
    with open(filepath,"a",encoding ="utf-8") as f:
         f.writelines([train_loss])
         
def Print_in_Console(mean,std,name_of_function):
    
    print("Error value of " + name_of_function)
    print("L1 norm   : ",round(mean[0],3), u"\u00B1" , round(std[0],3))
    print("L2 norm   : ",round(mean[1],3), u"\u00B1" , round(std[1],3))
    print("L infinity: ",round(mean[2],3), u"\u00B1" , round(std[2],3),"\n\n")