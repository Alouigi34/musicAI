# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 23:58:35 2024

@author: alberto
"""

_path_ = r"C:\Users\alberto\Desktop\diplom_music\new_complete_scenario\after_summer\new_workflow\AI_assisted-composition\v5"

import os
from mido import MidiFile
import numpy as np
import string 
import numpy as np
import tensorflow as tf
print(tf. __version__)
import keras
import time



import numpy as np
import tensorflow as tf
from timeit import default_timer as timer


class LiteModel:
    
    @classmethod
    def from_file(cls, model_path):
        return LiteModel(tf.lite.Interpreter(model_path=model_path))
    
    @classmethod
    def from_keras_model(cls, kmodel):
        converter = tf.lite.TFLiteConverter.from_keras_model(kmodel)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS] 
        converter._experimental_lower_tensor_list_ops = False
        tflite_model = converter.convert()
        return LiteModel(tf.lite.Interpreter(model_content=tflite_model))

    @classmethod
    def from_keras_model_and_save(cls, kmodel,name,folder_name):
        converter = tf.lite.TFLiteConverter.from_keras_model(kmodel)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS] 
        converter._experimental_lower_tensor_list_ops = False
        tflite_model = converter.convert()
        #tflite_modelx = tflite_model.serialize()
        # Save LiteModel to a .tflite file
        with open(folder_name+"/model"+str(name)+".tflite", "wb") as f:
            f.write(tflite_model)
        return LiteModel(tf.lite.Interpreter(model_content=tflite_model))
    
    def __init__(self, interpreter):
        self.interpreter = interpreter
        self.interpreter.allocate_tensors()
        input_det = self.interpreter.get_input_details()[0]
        output_det = self.interpreter.get_output_details()[0]
        self.input_index = input_det["index"]
        self.output_index = output_det["index"]
        self.input_shape = input_det["shape"]
        self.output_shape = output_det["shape"]
        self.input_dtype = input_det["dtype"]
        self.output_dtype = output_det["dtype"]
        
    def predict(self, inp):
        inp = inp.astype(self.input_dtype)
        count = inp.shape[0]
        out = np.zeros((count, self.output_shape[1]), dtype=self.output_dtype)
        for i in range(count):
            self.interpreter.set_tensor(self.input_index, inp[i:i+1])
            self.interpreter.invoke()
            out[i] = self.interpreter.get_tensor(self.output_index)[0]
        return out
    
    def predict_single(self, inp):
        """ Like predict(), but only for a single record. The input data can be a Python list. """
        inp = np.array([inp], dtype=self.input_dtype)
        self.interpreter.set_tensor(self.input_index, inp)
        self.interpreter.invoke()
        out = self.interpreter.get_tensor(self.output_index)
        return out[0]



# Load models:
    
temp=LiteModel.from_file(_path_+"/tflite_models1/model3.tflite")
    
    
lmodels = []
lmodels2 = []
lmodels3 = []   
for i in range(89):
    folder_name = "tflite_models1"
    lmodels.append(LiteModel.from_file(_path_+"/"+folder_name+"/model"+str(i)+".tflite"))
for i in range(89):   
    folder_name = "tflite_models2"
    lmodels2.append(LiteModel.from_file(_path_+"/"+folder_name+"/model"+str(i)+".tflite"))
for i in range(89):  
    folder_name = "tflite_models3"
    lmodels3.append(LiteModel.from_file(_path_+"/"+folder_name+"/model"+str(i)+".tflite"))




################################################################################
################################################################################
# Note inputs
#datax = numpy.loadtxt('result_array.csv')[1:10000]
datax = np.loadtxt('summary_notes_reference_piece.csv')
#all_notes = datax.astype(int)   

####################### Data Normalization ################################
# explicit function to normalize array
def normalize(arr, t_min, t_max):
    norm_arr = []
    diff = t_max - t_min
    diff_arr = max(arr) - min(arr) 
    characteristic_max= max(arr)
    characteristic_min=min(arr)  
    if np.isnan(diff_arr) == False:
        for i in arr:
            temp = (((i - min(arr))*diff)/diff_arr) + t_min
            norm_arr.append(temp)
    if np.isnan(diff_arr) == True:
        for i in arr:
            norm_arr.append(0.0)     
    return norm_arr,characteristic_max,characteristic_min

all_notes = datax.copy()
characteristics_max=[]
characteristics_min=[]
#Normalize each feature indipendently:
for i in range(89):
    range_to_normalize = (0,1)
    all_notes[:,i],characteristic_max,characteristic_min = normalize(all_notes[:,i],
                                range_to_normalize[0],
                                range_to_normalize[1])
    characteristics_max.append(characteristic_max)
    characteristics_min.append(characteristic_min)
 
characteristics_min2=characteristics_min
characteristics_min3=characteristics_min
characteristics_max2=characteristics_max
characteristics_max3=characteristics_max



buffer_length1 = 5
buffer_length2 = 15
buffer_length3 = 5
characteristics_buff = 89

"""
#########################################################################################################
#########################################################################################################
Step B: Now that we have ready the model(s) we can create the suggestions for the user input
-> step B1 : Showcase the suggestion routine
#########################################################################################################
#########################################################################################################
"""
#example 1 is predictions on random user choices




####################### Data Normalization ################################
# explicit function to normalize array
def normalize(arr, t_min, t_max):
    norm_arr = []
    diff = t_max - t_min
    diff_arr = max(arr) - min(arr) 
    characteristic_max= max(arr)
    characteristic_min=min(arr)  
    if np.isnan(diff_arr) == False:
        for i in arr:
            temp = (((i - min(arr))*diff)/diff_arr) + t_min
            norm_arr.append(temp)
    if np.isnan(diff_arr) == True:
        for i in arr:
            norm_arr.append(0.0)     
    return norm_arr,characteristic_max,characteristic_min



##### De-normalize Prediction:
####################### Data De-Normalization ################################
# explicit function to denormalize array
def denormalize(arr, t_min, t_max):
    diff = 1
    diff_arr = t_max - t_min  
    if np.isnan(diff_arr) == False:
            temp = t_min+((arr-0)*diff_arr/diff)
    return temp
    




example = 1

if example == 1:
    ################################################   example input 2 (simple melody)
    #initialize:
    user_input_time_step = []
    for i in range(5):
        user_input_time_step.append(np.zeros(89))
    
    # add some random notes (with hit power). 
    #time index 1
    user_input_time_step[0][40] = 40
    #the duration:
    user_input_time_step[0][88] = 50
    
    #time index 2
    user_input_time_step[1][40] = 50
    #the duration:
    user_input_time_step[1][88] = 200
    
    #time index 3
    user_input_time_step[2][41] = 45
    #the duration:
    user_input_time_step[2][88] = 50
    
    #time index 4
    user_input_time_step[3][42] = 42
    #the duration:
    user_input_time_step[3][88] = 180
    
    #time index 5
    user_input_time_step[4][40] = 60
    #the duration:
    user_input_time_step[4][88] = 50


    ################################################   example input 2 (simple melody)
    #initialize:
    user_input_time_step2 = []
    for i in range(15):
        user_input_time_step2.append(np.zeros(89))
    
    # add some random notes (with hit power). 
    #time index 1
    user_input_time_step2[0][40] = 40
    #the duration:
    user_input_time_step2[0][88] = 50
    
    #time index 2
    user_input_time_step2[1][40] = 50
    #the duration:
    user_input_time_step2[1][88] = 200
    
    #time index 3
    user_input_time_step2[2][41] = 45
    #the duration:
    user_input_time_step2[2][88] = 50
    
    #time index 4
    user_input_time_step2[3][42] = 42
    #the duration:
    user_input_time_step2[3][88] = 180
    
    #time index 5
    user_input_time_step2[4][40] = 60
    #the duration:
    user_input_time_step2[4][88] = 50

    #time index 6
    user_input_time_step2[5][40] = 40
    #the duration:
    user_input_time_step2[5][88] = 50
    
    #time index 7
    user_input_time_step2[6][40] = 50
    #the duration:
    user_input_time_step2[6][88] = 200
    
    #time index 8
    user_input_time_step2[7][41] = 45
    #the duration:
    user_input_time_step2[7][88] = 50
    
    #time index 9
    user_input_time_step2[8][42] = 42
    #the duration:
    user_input_time_step2[8][88] = 180
    
    #time index 10
    user_input_time_step2[9][40] = 60
    #the duration:
    user_input_time_step2[9][88] = 50

    #time index 11
    user_input_time_step2[10][40] = 40
    #the duration:
    user_input_time_step2[10][88] = 50
    
    #time index 12
    user_input_time_step2[11][40] = 50
    #the duration:
    user_input_time_step2[11][88] = 200
    
    #time index 13
    user_input_time_step2[12][41] = 45
    #the duration:
    user_input_time_step2[12][88] = 50
    
    #time index 14
    user_input_time_step2[13][42] = 42
    #the duration:
    user_input_time_step2[13][88] = 180
    
    #time index 15
    user_input_time_step2[14][40] = 60
    #the duration:
    user_input_time_step2[14][88] = 50 
    
    
    
        
    
##### Normalize the inserted values:
normalized_user_input_time_step = np.array(user_input_time_step.copy())
if example != 3:
    characteristics_new_max=[]
    characteristics_new_min=[]
    #Normalize each feature indipendently:
    for i in range(89):
        range_to_normalize = (0,1)
        normalized_user_input_time_step[:,i],characteristic_max,characteristic_min = normalize(normalized_user_input_time_step[:,i],
                                    range_to_normalize[0],
                                    range_to_normalize[1])
        characteristics_new_max.append(characteristic_max)
        characteristics_new_min.append(characteristic_min)   
    normalized_user_input_time_step=np.array(normalized_user_input_time_step)    
    normalized_user_input_time_step[np.isnan(normalized_user_input_time_step)] = 0


normalized_user_input_time_step2 = np.array(user_input_time_step2.copy())
if example != 3:
    characteristics_new_max2=[]
    characteristics_new_min2=[]
    #Normalize each feature indipendently:
    for i in range(89):
        range_to_normalize = (0,1)
        normalized_user_input_time_step2[:,i],characteristic_max,characteristic_min = normalize(normalized_user_input_time_step2[:,i],
                                    range_to_normalize[0],
                                    range_to_normalize[1])
        characteristics_new_max2.append(characteristic_max)
        characteristics_new_min2.append(characteristic_min)   
    normalized_user_input_time_step2=np.array(normalized_user_input_time_step2)    
    normalized_user_input_time_step2[np.isnan(normalized_user_input_time_step2)] = 0


normalized_user_input_time_step3 = np.array(user_input_time_step.copy())
if example != 3:
    characteristics_new_max3=[]
    characteristics_new_min3=[]
    #Normalize each feature indipendently:
    for i in range(89):
        range_to_normalize = (0,1)
        normalized_user_input_time_step3[:,i],characteristic_max,characteristic_min = normalize(normalized_user_input_time_step3[:,i],
                                    range_to_normalize[0],
                                    range_to_normalize[1])
        characteristics_new_max3.append(characteristic_max)
        characteristics_new_min3.append(characteristic_min)   
    normalized_user_input_time_step3=np.array(normalized_user_input_time_step3)    
    normalized_user_input_time_step3[np.isnan(normalized_user_input_time_step3)] = 0





##### Predict:   # AI model 1
normalized_user_input_time_step_reshaped = normalized_user_input_time_step.reshape((1, buffer_length1, characteristics_buff))    
predictNextSequence=[]
for i in range(0,89): #for each characteristic
    temp = lmodels[i].predict(normalized_user_input_time_step_reshaped)
    temp[temp<0.01] = 0
    predictNextSequence.append(temp)
    #real.append(output_data[i][index,:,:])
    predicted = (np.transpose(predictNextSequence, axes = (2,1,0)))
#print(predicted)    

    
##### Predict:   # AI model 2
normalized_user_input_time_step_reshaped2 = normalized_user_input_time_step2.reshape((1, buffer_length2, characteristics_buff))    
predictNextSequence=[]
for i in range(0,89): #for each characteristic
    temp = lmodels2[i].predict(normalized_user_input_time_step_reshaped2)
    temp[temp<0.01] = 0
    predictNextSequence.append(temp)
    #real.append(output_data[i][index,:,:])
    predicted2 = (np.transpose(predictNextSequence, axes = (2,1,0)))
#print(predicted2)    
    

##### Predict:   # AI model 3
normalized_user_input_time_step_reshaped3 = normalized_user_input_time_step3.reshape((1, buffer_length3, characteristics_buff))    
predictNextSequence=[]
for i in range(0,89): #for each characteristic
    temp = lmodels3[i].predict(normalized_user_input_time_step_reshaped3)
    temp[temp<0.01] = 0
    predictNextSequence.append(temp)
    #real.append(output_data[i][index,:,:])
    predicted3 = (np.transpose(predictNextSequence, axes = (2,1,0)))
#print(predicted2)    
    




final_predicted_numpy_actual = predicted.copy()
#De-Normalize each feature indipendently:
for i in range(89):
    value = (final_predicted_numpy_actual[0][0][i])
    if example != 3:
        final_predicted_numpy_actual[0][0][i] = denormalize(value,
                                    characteristics_new_min[i],
                                    characteristics_new_max[i])
    if example == 3:
        final_predicted_numpy_actual[0][0][i] = denormalize(value,
                                    characteristics_min[i],
                                    characteristics_max[i])
        
final_predicted_numpy_actual=np.array(final_predicted_numpy_actual)    
final_predicted_numpy_actual[np.isnan(final_predicted_numpy_actual)] = 0
final_predicted_numpy_actual[final_predicted_numpy_actual<4] = 0
final_predicted_numpy_actual=np.round(final_predicted_numpy_actual)   
######################################################################
print('')
print("Note predictions - 1st model:")
np.set_printoptions(formatter={'int': '{:04d}'.format})
#print(final_predicted_numpy_actual[0][0])
for i in range(0,90,10): # print 10 rows and 10 coloumns at a time
    print(final_predicted_numpy_actual[0][0][i:i+10])
print('')





final_predicted_numpy_actual2 = predicted2.copy()
#De-Normalize each feature indipendently:
for i in range(89):
    value = (final_predicted_numpy_actual2[0][0][i])
    if example != 3:
        final_predicted_numpy_actual2[0][0][i] = denormalize(value,
                                    characteristics_new_min2[i],
                                    characteristics_new_max2[i])
    if example == 3:
        final_predicted_numpy_actual2[0][0][i] = denormalize(value,
                                    characteristics_min2[i],
                                    characteristics_max2[i])
        
final_predicted_numpy_actual2=np.array(final_predicted_numpy_actual2)    
final_predicted_numpy_actual2[np.isnan(final_predicted_numpy_actual2)] = 0
final_predicted_numpy_actual2[final_predicted_numpy_actual2<4] = 0
final_predicted_numpy_actual2=np.round(final_predicted_numpy_actual2)   
######################################################################
print("Note predictions - 2nd model:")
#print(final_predicted_numpy_actual2[0][0])   
for i in range(0,90,10): # print 10 rows and 10 coloumns at a time
    print(final_predicted_numpy_actual2[0][0][i:i+10])
print('')




final_predicted_numpy_actual3 = predicted3.copy()
#De-Normalize each feature indipendently:
for i in range(89):
    value = (final_predicted_numpy_actual3[0][0][i])
    if example != 3:
        final_predicted_numpy_actual3[0][0][i] = denormalize(value,
                                    characteristics_new_min3[i],
                                    characteristics_new_max3[i])
    if example == 3:
        final_predicted_numpy_actual3[0][0][i] = denormalize(value,
                                    characteristics_min3[i],
                                    characteristics_max3[i])
        
final_predicted_numpy_actual3=np.array(final_predicted_numpy_actual3)    
final_predicted_numpy_actual3[np.isnan(final_predicted_numpy_actual3)] = 0
final_predicted_numpy_actual3[final_predicted_numpy_actual3<4] = 0
final_predicted_numpy_actual3=np.round(final_predicted_numpy_actual3)   
######################################################################
print("Note predictions - 3rd model:")
#print(final_predicted_numpy_actual3[0][0])   
for i in range(0,90,10): # print 10 rows and 10 coloumns at a time
    print(final_predicted_numpy_actual3[0][0][i:i+10])
print('')



    
    