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
import pyaudio
import numpy as np
import time
import wave
import matplotlib.pyplot as plt
import math
import pygame
import piano_lists as pl
from pygame import mixer

pygame.init()
pygame.mixer.set_num_channels(50)

font = pygame.font.Font('assets/SwanseaItalic.ttf', 48)
medium_font = pygame.font.Font('assets/SwanseaItalic.ttf', 28)
small_font = pygame.font.Font('assets/SwanseaItalic.ttf', 16)
real_small_font = pygame.font.Font('assets/SwanseaItalic.ttf', 10)
fps = 60
timer = pygame.time.Clock()
WIDTH = int(1 * 52 * 35)
HEIGHT = 1050


screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Three Pianos")

# Define the size of each piano surface
piano_width = WIDTH
piano_height = int(HEIGHT // 3)  # Adjust the height to fit two pianos vertically
piano_height2 = int(HEIGHT // 1.5)  # Adjust the height to fit two pianos vertically

screen = pygame.display.set_mode([WIDTH, HEIGHT])


left_oct = 4
right_oct = 5

left_hand = pl.left_hand
right_hand = pl.right_hand
piano_notes = pl.piano_notes
white_notes = pl.white_notes
black_notes = pl.black_notes
black_labels = pl.black_labels

pygame.display.set_caption("Piano")


# Define colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
def draw_piano(surface, whites, blacks, whites_intensities, blacks_intensities):
    surface.fill(WHITE)  # Clear the surface
    white_rects = []
    for i in range(52):
        rect = pygame.draw.rect(surface, 'white', [i * 35, 0, 35, 300], 0, 2)
        white_rects.append(rect)
        pygame.draw.rect(surface, 'black', [i * 35, 0, 35, 300], 2, 2)
        #################################################################
        ### This draws letters on the white notes 
        key_label = small_font.render(white_notes[i], True, 'black')
        surface.blit(key_label, (i * 35 + 3, 270))
        #################################################################

    skip_count = 0
    last_skip = 2
    skip_track = 2
    black_rects = []
    for i in range(36):
        rect = pygame.draw.rect(surface, 'black', [23 + (i * 35) + (skip_count * 35), 0, 24, 200], 0, 2)
    
        #################################################################
        ### This draws a green square on the currently pressed black keys        
        for q in range(len(blacks)):
            if blacks[q] == i:
                if blacks[q] > 0:
                    if blacks_intensities[q]>0 and blacks_intensities[q]<50:
                        pygame.draw.rect(surface, (0,255,0), [23 + (i * 35) + (skip_count * 35), 0, 24, 200], 2, 2)
                    if blacks_intensities[q]>50 and blacks_intensities[q]<150:
                        pygame.draw.rect(surface, (255,0,0), [23 + (i * 35) + (skip_count * 35), 0, 24, 200], 2, 2)
                    if blacks_intensities[q]>150:
                        pygame.draw.rect(surface, (0,0,255), [23 + (i * 35) + (skip_count * 35), 0, 24, 200], 2, 2)
                        
        #################################################################

        #################################################################
        ### This draws letters on the black notes 
        key_label = real_small_font.render(black_labels[i], True, 'white')
        surface.blit(key_label, (25 + (i * 35) + (skip_count * 35), 180))
        #################################################################
        black_rects.append(rect)
        skip_track += 1
        if last_skip == 2 and skip_track == 3:
            last_skip = 3
            skip_track = 0
            skip_count += 1
        elif last_skip == 3 and skip_track == 2:
            last_skip = 2
            skip_track = 0
            skip_count += 1

    #################################################################
    ### This draws a green square on the currently pressed white keys
    for i in range(len(whites)):
        if whites[i] > 0:
            j = whites[i]
            if whites_intensities[i]>0 and whites_intensities[i]<50:
                    pygame.draw.rect(surface, (0,255,0), [j * 35, 200, 35, 100], 2, 2)
            if whites_intensities[i]>50 and whites_intensities[i]<150:
                    pygame.draw.rect(surface, (255,0,0), [j * 35, 200, 35, 100], 2, 2)
            if whites_intensities[i]>150:
                    pygame.draw.rect(surface, (0,0,255), [j * 35, 200, 35, 100], 2, 2)
                        
            
    #################################################################

    return white_rects, black_rects, whites, blacks





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
    







if __name__=="__main__":


    for trep_imes in range(100):
 
        
            '''
            ###########################################################            
            SIMULATE NEW INPUT (MIC OR ASSISTANT)
            ###########################################################
            '''
            random_number0= int(30*np.random.rand())
            random_number1= int(30*np.random.rand())
            random_number2= int(20*np.random.rand())
            random_number3= int(40*np.random.rand())
            

            example = 1
            
            if example == 1:
                ################################################   example input 2 (simple melody)
                #initialize:
                user_input_time_step = []
                for i in range(5):
                    user_input_time_step.append(np.zeros(89))
                
                # add some random notes (with hit power). 
                #time index 1
                user_input_time_step[0][random_number0] = 40
                #the duration:
                user_input_time_step[0][88] = 50
                
                #time index 2
                user_input_time_step[1][random_number1] = 50
                #the duration:
                user_input_time_step[1][88] = 200
                
                #time index 3
                user_input_time_step[2][41] = 45
                #the duration:
                user_input_time_step[2][88] = 50
                
                #time index 4
                user_input_time_step[3][random_number2] = 42
                #the duration:
                user_input_time_step[3][88] = 180
                
                #time index 5
                user_input_time_step[4][random_number3] = 60
                #the duration:
                user_input_time_step[4][88] = 50
            
            
                ################################################   example input 2 (simple melody)
                #initialize:
                user_input_time_step2 = []
                for i in range(15):
                    user_input_time_step2.append(np.zeros(89))
                
                # add some random notes (with hit power). 
                #time index 1
                user_input_time_step2[0][random_number0] = 40
                #the duration:
                user_input_time_step2[0][88] = 50
                
                #time index 2
                user_input_time_step2[1][random_number1] = 50
                #the duration:
                user_input_time_step2[1][88] = 200
                
                #time index 3
                user_input_time_step2[2][41] = 45
                #the duration:
                user_input_time_step2[2][88] = 50
                
                #time index 4
                user_input_time_step2[3][random_number2] = 42
                #the duration:
                user_input_time_step2[3][88] = 180
                
                #time index 5
                user_input_time_step2[4][random_number3] = 60
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
                
                
                
                    
            '''
            ###########################################################            
            NORMALIZATION OF THE INPUT
            AND PREDICTION PROCESS FROM EACH MODEL
            ###########################################################            
            '''                
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
            
            
            
            '''
            ###########################################################            
            DRAW PROPOSALS OF ONE MODEL AT THE PIANO
            (CONSIDER TO HAVE THREE PIANOS IN PARALLEL)
            ###########################################################            
            '''               
            ## Step 1: Convert note predictions into white_notes and black_notes idexes
            
            # Define the indices of the black keys
            black_key_indices = [1, 4, 6, 9, 11, 13, 16, 18, 21, 23, 25, 28, 30, 33, 35, 37, 40, 42, 45, 47, 49, 52, 54, 56, 59, 61, 63, 66, 68, 71, 73, 75, 78, 80, 82, 85, 87]
            
            # Create boolean masks for black and white keys
            black_keys_mask = np.zeros(88, dtype=bool)
            black_keys_mask[black_key_indices] = True
            white_keys_mask = ~black_keys_mask
            
            # Filter values for black and white keys
            piano_keys = final_predicted_numpy_actual[0][0][:-1]
            black_keys = piano_keys[black_keys_mask]
            white_keys = piano_keys[white_keys_mask]
            
            
            #white_notes and black_notes idexes:
            black_notes_1 =  list(np.nonzero(black_keys)[0])        
            white_notes_1 =  list(np.nonzero(white_keys)[0])          
            blacks_intensities1 = black_keys[black_notes_1]
            whites_intensities1 = white_keys[white_notes_1]


            # Filter values for black and white keys
            piano_keys = final_predicted_numpy_actual2[0][0][:-1]
            black_keys = piano_keys[black_keys_mask]
            white_keys = piano_keys[white_keys_mask]
            
            #white_notes and black_notes idexes:
            black_notes_2 =  list(np.nonzero(black_keys)[0])        
            white_notes_2 =  list(np.nonzero(white_keys)[0])          
            blacks_intensities2 = black_keys[black_notes_2]
            whites_intensities2 = white_keys[white_notes_2]
            
            # Filter values for black and white keys
            piano_keys = final_predicted_numpy_actual3[0][0][:-1]
            black_keys = piano_keys[black_keys_mask]
            white_keys = piano_keys[white_keys_mask]
            
            #white_notes and black_notes idexes:
            black_notes_3 =  list(np.nonzero(black_keys)[0])        
            white_notes_3 =  list(np.nonzero(white_keys)[0])          
            blacks_intensities3 = black_keys[black_notes_3]
            whites_intensities3 = white_keys[white_notes_3]                    
            

            
            timer.tick(fps)
            screen.fill('gray')
            
            # Create two surfaces for two pianos
            piano1_surface = pygame.Surface((WIDTH , HEIGHT))
            piano2_surface = pygame.Surface((WIDTH , HEIGHT))
            piano3_surface = pygame.Surface((WIDTH , HEIGHT))


                
                
                        
            # Call the draw_piano function for each piano surface
            print("*****************")

            print()
            print(white_notes_1,whites_intensities1)
            print(black_notes_1, blacks_intensities1)
            print(white_notes_2,whites_intensities2)
            print(black_notes_2, blacks_intensities2)
            print(white_notes_3,whites_intensities3)
            print(black_notes_3, blacks_intensities3)

            print("*****************")
            
            amplitude_scaling=3
            
            result_1 = draw_piano(piano1_surface, white_notes_1, black_notes_1, amplitude_scaling*whites_intensities1, amplitude_scaling*blacks_intensities1)
            result_2 = draw_piano(piano2_surface, white_notes_2, black_notes_2, amplitude_scaling*whites_intensities2, amplitude_scaling*blacks_intensities2)
            result_3 = draw_piano(piano3_surface, white_notes_3, black_notes_3, amplitude_scaling*whites_intensities3, amplitude_scaling*blacks_intensities3)
          


                
                
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False
    
            # Blit the piano surfaces onto the main screen
            screen.blit(piano1_surface, (0, 0))  # Position the first piano at the top-left corner
            screen.blit(piano2_surface, (0, piano_height))  # Position the second piano below the first one
            screen.blit(piano3_surface, (0, piano_height2))  # Position the second piano below the first one
                    
            
            pygame.display.flip()
            
            
            time.sleep(2)
            
            
    pygame.quit()
    
    
    

                    
                