# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 23:58:35 2024

@author: alberto
"""

_path_ = r"C:\Users\alberto\Desktop\diplom_music\new_complete_scenario\after_summer\new_workflow\AI_assisted-composition\v5"

## Create models subfolder  1st AI
import os 
# checking if the directory exist or not. 
if not os.path.exists(_path_ +r"\AImodel1"):      
    # if the directory is not present  then create it. 
    os.makedirs(_path_ +r"\AImodel1") 
    
## Create models subfolder  2nd AI
import os 
# checking if the directory exist or not. 
if not os.path.exists(_path_ +r"\AImodel2"):      
    # if the directory is not present  then create it. 
    os.makedirs(_path_ +r"\AImodel2") 


"""
#########################################################################################################
#########################################################################################################
Step A: Training phase
-> step A1 : Read any midi fille and convert it to suitable csv fille
#########################################################################################################
#########################################################################################################
"""

from mido import MidiFile
import numpy as np
import string   
   
mid = MidiFile(_path_+r'\reference_piece.mid', clip=True)
print(mid)

def MidiStringToInt(midstr):
    Notes = [["C"],["C#","Db"],["D"],["D#","Eb"],["E"],["F"],["F#","Gb"],["G"],["G#","Ab"],["A"],["A#","Bb"],["B"]]
    answer = 0
    i = 0
    #Note
    letter = midstr.split('-')[0].upper()
    for note in Notes:
        for form in note:
            if letter.upper() == form:
                answer = i
                break;
        i += 1
    #Octave
    answer += (int(midstr[-1]))*12
    return answer

NOTES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
OCTAVES = list(range(11))
NOTES_IN_OCTAVE = len(NOTES)
errors = {
    'program': 'Bad input, please refer this spec-\n'
               'http://www.electronics.dit.ie/staff/tscarff/Music_technology/midi/program_change.htm',
    'notes': 'Bad input, please refer this spec-\n'
             'http://www.electronics.dit.ie/staff/tscarff/Music_technology/midi/midi_note_numbers_for_octaves.htm'
}
def number_to_note(number: int) -> tuple:
    octave = number // NOTES_IN_OCTAVE
    assert octave in OCTAVES, errors['notes']
    assert 0 <= number <= 127, errors['notes']
    note = NOTES[number % NOTES_IN_OCTAVE]

    return note, octave

def msg2dict(msg):
    result = dict()
    if 'note_on' in msg:
        on_ = True
    elif 'note_off' in msg:
        on_ = False
    else:
        on_ = None
    result['time'] = int(msg[msg.rfind('time'):].split(' ')[0].split('=')[1].translate(
        str.maketrans({a: None for a in string.punctuation})))

    if on_ is not None:
        for k in ['note', 'velocity']:
            result[k] = int(msg[msg.rfind(k):].split(' ')[0].split('=')[1].translate(
                str.maketrans({a: None for a in string.punctuation})))
    return [result, on_]


def switch_note(last_state, note, velocity, on_=True):
    # piano has 88 notes, corresponding to note id 21 to 108, any note out of this range will be ignored
    result = [0] * 88 if last_state is None else last_state.copy()
    if 21 <= note <= 108:
        result[note-21] = velocity if on_ else 0
    return result


def get_new_state(new_msg, last_state):
    new_msg, on_ = msg2dict(str(new_msg))
    new_state = switch_note(last_state, note=new_msg['note'], velocity=new_msg['velocity'], on_=on_) if on_ is not None else last_state
    return [new_state, new_msg['time']]

def track2seq(track):
    # piano has 88 notes, corresponding to note id 21 to 108, any note out of the id range will be ignored
    result = []
    last_state, last_time = get_new_state(str(track[0]), [0]*88)
    for i in range(1, len(track)):
        new_state, new_time = get_new_state(track[i], last_state)
        if new_time > 0:
            result += [last_state]*new_time
        last_state, last_time = new_state, new_time
    return result

def mid2arry(mid, min_msg_pct=0.1):
    tracks_len = [len(tr) for tr in mid.tracks]
    min_n_msg = max(tracks_len) * min_msg_pct
    # convert each track to nested list
    all_arys = []
    for i in range(len(mid.tracks)):
        if len(mid.tracks[i]) > min_n_msg:
            ary_i = track2seq(mid.tracks[i])
            all_arys.append(ary_i)
    # make all nested list the same length
    max_len = max([len(ary) for ary in all_arys])
    for i in range(len(all_arys)):
        if len(all_arys[i]) < max_len:
            all_arys[i] += [[0] * 88] * (max_len - len(all_arys[i]))
    all_arys = np.array(all_arys)
    all_arys = all_arys.max(axis=0)
    # trim: remove consecutive 0s in the beginning and at the end
    sums = all_arys.sum(axis=1)
    ends = np.where(sums > 0)[0]
    return all_arys[min(ends): max(ends)]



result_array = mid2arry(mid)
import matplotlib.pyplot as plt
plt.plot(range(result_array.shape[0]), np.multiply(np.where(result_array>0, 1, 0), range(1, 89)), marker='.', markersize=1, linestyle='')
plt.title("spring_no2_adagio_gp.mid")
plt.show()



"check:"
"piano has 88 notes, corresponding to note id 21 to 108"
note_offset_fix=21
time_index=1
audio_start=1
audio_end=500
for i in range(audio_start,audio_end):
    print("time index: ", i)
    for j in range(88):   
        if result_array[i,j]>0:
            print("note:",number_to_note(j+note_offset_fix),j+note_offset_fix,"velocity:",result_array[i,j])


#np.savetxt('result_array.csv', result_array)


"""
Convert result_array to have one more coloumn that says how many times we have 
the same row (same note combinations) and delete dublicates - v2
"""
row_durations=[]
count = 0
for i in range(len(result_array)-1):
    if np.array_equal(result_array[i+1,:], result_array[i,:]) == True:
        count = count + 1
    else:
        count = count + 1
        row_durations.append(count)
        count = 0
    if i == len(result_array)-2:
        count = count + 1
        row_durations.append(count)
        count = 0        
        
new_array = np.zeros((len(row_durations),89)) 
duration_sum = 0       
for i in range(len(row_durations)):
    duration_sum = duration_sum + row_durations[i]
    new_array[i,0:88]=(result_array[duration_sum-1,:])
    new_array[i,88]=row_durations[i]
summary_notes_array=new_array.astype(int)




"""export notes to csv fille:"""
np.savetxt('summary_notes_reference_piece.csv', summary_notes_array)
# "to check reloading:"
datax = np.loadtxt('summary_notes_reference_piece.csv')
datax=datax.astype(int)   

    

"to check efficient reconstuction to initial result_array from summary_notes_array:"
#preallocate:
no_of_rows = np.sum(datax[:,88])    
reconstructed_result_array=np.zeros((no_of_rows,89))
temp=datax
indexed_sum=0
for i in range(len(datax[:,0])):
    new_element = temp[i,:].reshape(1,-1) 
    reconstructed_result_array[indexed_sum:indexed_sum+datax[i,88],:]=new_element 
    indexed_sum = indexed_sum + datax[i,88]
    
reconstructed_result_array = np.delete(reconstructed_result_array,88,1)  
reconstructed_result_array = reconstructed_result_array.astype(int)
"check for correct reconstruction:"
print("##########")
initial_array = mid2arry(mid)
print("correct reconstruction?:",np.array_equal(reconstructed_result_array, initial_array))
print("##########") 


" To recreate song in midi and check if it is the same (but reduced for piano)"
recreate_midi = True
if recreate_midi == True:
    import mido
    
    def arry2mid(ary, tempo=500000):
        # get the difference
        new_ary = np.concatenate([np.array([[0] * 88]), np.array(ary)], axis=0)
        changes = new_ary[1:] - new_ary[:-1]
        # create a midi file with an empty track
        mid_new = mido.MidiFile()
        track = mido.MidiTrack()
        mid_new.tracks.append(track)
        track.append(mido.MetaMessage('set_tempo', tempo=tempo, time=0))
        # add difference in the empty track
        last_time = 0
        for ch in changes:
            if set(ch) == {0}:  # no change
                last_time += 1
            else:
                on_notes = np.where(ch > 0)[0]
                on_notes_vol = ch[on_notes]
                off_notes = np.where(ch < 0)[0]
                first_ = True
                for n, v in zip(on_notes, on_notes_vol):
                    new_time = last_time if first_ else 0
                    track.append(mido.Message('note_on', note=n + 21, velocity=v, time=new_time))
                    first_ = False
                for n in off_notes:
                    new_time = last_time if first_ else 0
                    track.append(mido.Message('note_off', note=n + 21, velocity=0, time=new_time))
                    first_ = False
                last_time = 0
        return mid_new
    
    
    mid_new = arry2mid(reconstructed_result_array, 545455)
    mid_new.save('reference_piece_piano_version.mid')



"""
#########################################################################################################
#########################################################################################################
Step A: Training phase
-> step A2 : Train AI model 1 
#########################################################################################################
#########################################################################################################
"""

import numpy as np
import tensorflow as tf
print(tf. __version__)
import keras
import time
#tf.compat.v1.disable_eager_execution()

#Before do anything else do not forget to reset the backend for the next iteration (rerun the model)
keras.backend.clear_session()#####################################

# Clear all previously registered custom objects
keras.saving.get_custom_objects().clear()

@keras.saving.register_keras_serializable(package="MyLayers")
class MultidimensionalLSTM(tf.keras.Model):
    def __init__(self, hidden_size, hidden_size2, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        
        #self.lstm = tf.keras.layers.LSTM(hidden_size,activation='silu', return_sequences=True)        
        self.lstm = tf.keras.layers.LSTM(hidden_size,activation='relu', return_sequences=True)        
        self.dense = tf.keras.layers.Dense(output_size,activation='linear')
    
    def call(self, inputs):
        x = self.lstm(inputs)  
        #x = self.linear(x)   
        x = self.dense(x[:, -1, :])
        return x

@keras.saving.register_keras_serializable(package="MyLayers")
class TCN(tf.keras.Model):
    def __init__(self, num_channels, kernel_size, output_size):
        super(TCN, self).__init__()
        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.output_size = output_size

        self.conv1d_layers = []
        if isinstance(num_channels, int):  # Handling single integer case
            num_channels = [num_channels]
        for i, channels in enumerate(num_channels):
            self.conv1d_layers.append(
                tf.keras.layers.Conv1D(channels, kernel_size, padding='causal', activation='relu', dilation_rate=2**i)
            )
        self.dense = tf.keras.layers.Dense(output_size, activation='linear')

    def call(self, inputs):
        x = inputs
        for layer in self.conv1d_layers:
            x = layer(x)
        x = self.dense(x[:, -1, :])
        return x

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
    
all_notes=np.array(all_notes)    
all_notes[np.isnan(all_notes)] = 0
######################################################################


######################################################################
##############  INPUT DATA
######################################################################
#This is the time buffer (how many time samples will be used):
buffer_length=5
#This is the ammount of the characteristics choosen starting from the all_characteristics index:
characteristics_buff=89
#This is the len of the characteristics: 
#(choose 89 if duration should be also considered)
#(choose 88 if duration shouldn't be considered)
#(choose 41 to include characteristics between 41-characteristics_buff:41
all_characteristics=89
inputs=np.full((len(all_notes)-buffer_length,buffer_length,characteristics_buff), 0.0)

# Separate to inputs-outputs
for time_index in range(len(all_notes)-buffer_length): #number of changes
    input_temp=all_notes[time_index:time_index+buffer_length,all_characteristics-characteristics_buff:all_characteristics]
    inputs[time_index,0:buffer_length,0:characteristics_buff]=(input_temp)

# len(all_notes) samples, each with buffer_length time steps and 89 features
input_data1 = inputs

################################################################################
################################################################################ 

######################################################################
##############  OUTPUT DATA
######################################################################

output_data1=[]
for i in range(89):
    #This is the time buffer (how many time samples will be used):
    buffer_length=5
    outputs=np.full((len(all_notes)-buffer_length,1,1), 0.0)
    
    SPESIFIC_CHARACTERISTIC_TO_PREDICT = i
    
    # Separate to inputs-outputs
    for time_index in range(len(all_notes)-buffer_length): #number of changes
        output_temp=all_notes[time_index+buffer_length:time_index+buffer_length+1,SPESIFIC_CHARACTERISTIC_TO_PREDICT]
        outputs[time_index,0:1,0]=(output_temp) 
    
    # len(all_notes) samples, each with 1 time step and 89 features
    output_data1.append(outputs)
################################################################################
################################################################################   


################################################################################
################################################################################   
################################################################################
################################################################################   
retrain=False
################################################################################
################################################################################   
################################################################################
################################################################################   

if retrain == True:
    model=[]
    for i in range(89):  #Default : 89. Use <89 to learn for less characteristics
        model.append(MultidimensionalLSTM(hidden_size=150,hidden_size2=328, output_size=1))
        '''
        #://www.tensorflow.org/api_docs/python/tf/keras/losses
        '''
        #model.compile(optimizer="adam", loss="mse")
        model[i].compile(optimizer=tf.keras.optimizers.Adam(0.01), loss=tf.keras.losses.MeanSquaredError(), metrics=['accuracy'])
        #model.compile(optimizer=tf.keras.optimizers.Adam(0.01), loss=tf.keras.losses.BinaryCrossentropy(), metrics=['accuracy'])
        
        print("################ Characteristic: ",i,"  ########################")
        # We train the model using:
        # output_data1[i] --> coresponds to a 1d list of all the values of the i characteristic
        # input_data11 --> coresponds to a 3d matrix 
        model[i].fit(input_data1, output_data1[i], epochs=40)
        print("################  end  ########################")
        print("###############################################")
    
    
    
    #Save ML models
    for i in range(0,89):
       path=_path_+r"\AImodel1\\"
       model[i].save(path+"custom_model"+str(i)+".keras")

    #Load ML models
    loaded_models=[]
    for i in range(0,89):
       path=_path_+r"\AImodel1\\"
       loaded_models.append(keras.models.load_model(path+"custom_model"+str(i)+".keras"))
        
        
    #Check 
    # print("if following are equal means that loaded models are the same with the saved")
    # test_data = input_data[184,:,:]
    # test_data = test_data.reshape((1, buffer_length, characteristics_buff))   
    # print(model[1].predict(test_data, verbose=1))
    # print(loaded_models[1].predict(test_data, verbose=1))


if retrain == False:
    #Load ML models
    loaded_models=[]
    for i in range(0,89):
       path=_path_+r"\AImodel1\\"
       loaded_models.append(keras.models.load_model(path+"custom_model"+str(i)+".keras"))
        
            

"To check if the predictions are same with the initial song"
check_prediction_accuracy = False
if check_prediction_accuracy == True:
    t0 = time.time()
    final_predicted=[]
    for index in range(0,20): #for each state change (maximum value: (row_durations - buffer_length))
        predictNextSequence=[]
        real=[]
        test_data = input_data1[index,:,:]
        test_data = test_data.reshape((1, buffer_length, characteristics_buff))
        for i in range(0,89): #for each characteristic
            temp = loaded_models[i].predict(test_data, verbose=1)
            temp[temp<0.01] = 0
            predictNextSequence.append(temp)
            real.append(output_data1[i][index,:,:])
        # print('real:')
        # print(np.round(real,4))
        # print('predicted:')
        # print(np.round(predictNextSequence,4))
        # final_predicted.append(np.round(predictNextSequence,4))
        final_predicted.append(np.transpose(predictNextSequence, axes = (2,1,0)))
        #print("#################################################################")
        print("index: ",index)
    t1 = time.time()
    print('prediction duration=',t1-t0)
    

    
    # ######################################################################
    ### SUGGESTIONS:
    ### 1. LINE 432 should have length 734 to predict the whole piece
    ### 2. LINE 394 change epochs to arround 50-130 for improved accuracy
    # ######################################################################
    
    
    # Restore final_predicted to have the same format with all_notes
    final_predicted_numpy = np.array(final_predicted)
    final_predicted_numpy = final_predicted_numpy[:,0,0,:]
    
    # Denormalize in order to reconstruct the piece
    ####################### Data De-Normalization ################################
    # explicit function to denormalize array
    def denormalize(arr, t_min, t_max):
        denorm_arr = []
        diff = 1
        diff_arr = t_max - t_min  
        if np.isnan(diff_arr) == False:
            for i in arr:
                temp = t_min+((i-0)*diff_arr/diff)
                denorm_arr.append(temp)           
        return denorm_arr
    
    final_predicted_numpy_actual = final_predicted_numpy.copy()
    #De-Normalize each feature indipendently:
    for i in range(89):
        final_predicted_numpy_actual[:,i] = denormalize(final_predicted_numpy[:,i],
                                    characteristics_min[i],
                                    characteristics_max[i])
        
    final_predicted_numpy_actual=np.array(final_predicted_numpy_actual)    
    final_predicted_numpy_actual[np.isnan(final_predicted_numpy_actual)] = 0
    final_predicted_numpy_actual[final_predicted_numpy_actual<4] = 0
    final_predicted_numpy_actual=np.round(final_predicted_numpy_actual)
    
    ######################################################################
    
    
    # """export predicted notes to csv fille:"""
    print("Exporting to csv. Note that the exported file has buffer_length less elements than the original")
    np.savetxt('final_predicted_AI_from_training.csv', final_predicted_numpy_actual)




"""
#########################################################################################################
#########################################################################################################
Step A: Training phase
-> step A3 : Train AI model 2 
#########################################################################################################
#########################################################################################################
"""

#Before do anything else do not forget to reset the backend for the next iteration (rerun the model)
keras.backend.clear_session()#####################################

# Clear all previously registered custom objects
keras.saving.get_custom_objects().clear()

@keras.saving.register_keras_serializable(package="MyLayers")
class MultidimensionalLSTM(tf.keras.Model):
    def __init__(self, hidden_size, hidden_size2, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        
        #self.lstm = tf.keras.layers.LSTM(hidden_size,activation='silu', return_sequences=True)        
        self.lstm = tf.keras.layers.LSTM(hidden_size,activation='relu', return_sequences=True)        
        self.dense = tf.keras.layers.Dense(output_size,activation='linear')
    
    def call(self, inputs):
        x = self.lstm(inputs)  
        #x = self.linear(x)   
        x = self.dense(x[:, -1, :])
        return x

@keras.saving.register_keras_serializable(package="MyLayers")
class TCN(tf.keras.Model):
    def __init__(self, num_channels, kernel_size, output_size):
        super(TCN, self).__init__()
        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.output_size = output_size

        self.conv1d_layers = []
        if isinstance(num_channels, int):  # Handling single integer case
            num_channels = [num_channels]
        for i, channels in enumerate(num_channels):
            self.conv1d_layers.append(
                tf.keras.layers.Conv1D(channels, kernel_size, padding='causal', activation='relu', dilation_rate=2**i)
            )
        self.dense = tf.keras.layers.Dense(output_size, activation='linear')

    def call(self, inputs):
        x = inputs
        for layer in self.conv1d_layers:
            x = layer(x)
        x = self.dense(x[:, -1, :])
        return x

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
characteristics_max2=[]
characteristics_min2=[]
#Normalize each feature indipendently:
for i in range(89):
    range_to_normalize = (0,1)
    all_notes[:,i],characteristic_max,characteristic_min = normalize(all_notes[:,i],
                                range_to_normalize[0],
                                range_to_normalize[1])
    characteristics_max2.append(characteristic_max)
    characteristics_min2.append(characteristic_min)
    
all_notes=np.array(all_notes)    
all_notes[np.isnan(all_notes)] = 0
######################################################################


######################################################################
##############  INPUT DATA
######################################################################
#This is the time buffer (how many time samples will be used):
buffer_length=15
#This is the ammount of the characteristics choosen starting from the all_characteristics index:
characteristics_buff=89
#This is the len of the characteristics: 
#(choose 89 if duration should be also considered)
#(choose 88 if duration shouldn't be considered)
#(choose 41 to include characteristics between 41-characteristics_buff:41
all_characteristics=89
inputs=np.full((len(all_notes)-buffer_length,buffer_length,characteristics_buff), 0.0)

# Separate to inputs-outputs
for time_index in range(len(all_notes)-buffer_length): #number of changes
    input_temp=all_notes[time_index:time_index+buffer_length,all_characteristics-characteristics_buff:all_characteristics]
    inputs[time_index,0:buffer_length,0:characteristics_buff]=(input_temp)

# len(all_notes) samples, each with buffer_length time steps and 89 features
input_data2 = inputs

################################################################################
################################################################################ 

######################################################################
##############  OUTPUT DATA
######################################################################

output_data2=[]
for i in range(89):
    #This is the time buffer (how many time samples will be used):
    outputs=np.full((len(all_notes)-buffer_length,1,1), 0.0)
    
    SPESIFIC_CHARACTERISTIC_TO_PREDICT = i
    
    # Separate to inputs-outputs
    for time_index in range(len(all_notes)-buffer_length): #number of changes
        output_temp=all_notes[time_index+buffer_length:time_index+buffer_length+1,SPESIFIC_CHARACTERISTIC_TO_PREDICT]
        outputs[time_index,0:1,0]=(output_temp) 
    
    # len(all_notes) samples, each with 1 time step and 89 features
    output_data2.append(outputs)
################################################################################
################################################################################   


################################################################################
################################################################################   
################################################################################
################################################################################   
retrain=False
################################################################################
################################################################################   
################################################################################
################################################################################   

if retrain == True:
    model=[]
    for i in range(89):  #Default : 89. Use <89 to learn for less characteristics
        model.append(MultidimensionalLSTM(hidden_size=150,hidden_size2=328, output_size=1))
        '''
        #://www.tensorflow.org/api_docs/python/tf/keras/losses
        '''
        #model.compile(optimizer="adam", loss="mse")
        model[i].compile(optimizer=tf.keras.optimizers.Adam(0.01), loss=tf.keras.losses.MeanSquaredError(), metrics=['accuracy'])
        #model.compile(optimizer=tf.keras.optimizers.Adam(0.01), loss=tf.keras.losses.BinaryCrossentropy(), metrics=['accuracy'])
        
        print("################ Characteristic: ",i,"  ########################")
        model[i].fit(input_data2, output_data2[i], epochs=40)
        print("################  end  ########################")
        print("###############################################")
    
    
    
    #Save ML models
    for i in range(0,89):
       path=_path_+r"\AImodel2\\"
       model[i].save(path+"custom_model"+str(i)+".keras")

    #Load ML models
    loaded_models2=[]
    for i in range(0,89):
       path=_path_+r"\AImodel2\\"
       loaded_models2.append(keras.models.load_model(path+"custom_model"+str(i)+".keras"))
        
        
    #Check 
    # print("if following are equal means that loaded models are the same with the saved")
    # test_data = input_data[184,:,:]
    # test_data = test_data.reshape((1, buffer_length, characteristics_buff))   
    # print(model[1].predict(test_data, verbose=1))
    # print(loaded_models[1].predict(test_data, verbose=1))


if retrain == False:
    #Load ML models
    loaded_models2=[]
    for i in range(0,89):
       path=_path_+r"\AImodel2\\"
       loaded_models2.append(keras.models.load_model(path+"custom_model"+str(i)+".keras"))
        
            

"To check if the predictions are same with the initial song"
check_prediction_accuracy = False
if check_prediction_accuracy == True:
    t0 = time.time()
    final_predicted=[]
    for index in range(0,20): #for each state change (maximum value: (row_durations - buffer_length))
        predictNextSequence=[]
        real=[]
        test_data = input_data2[index,:,:]
        test_data = test_data.reshape((1, buffer_length, characteristics_buff))
        for i in range(0,89): #for each characteristic
            temp = loaded_models[i].predict(test_data, verbose=1)
            temp[temp<0.01] = 0
            predictNextSequence.append(temp)
            real.append(output_data2[i][index,:,:])
        # print('real:')
        # print(np.round(real,4))
        # print('predicted:')
        # print(np.round(predictNextSequence,4))
        # final_predicted.append(np.round(predictNextSequence,4))
        final_predicted.append(np.transpose(predictNextSequence, axes = (2,1,0)))
        #print("#################################################################")
        print("index: ",index)
    t1 = time.time()
    print('prediction duration=',t1-t0)
    

    
    # ######################################################################
    ### SUGGESTIONS:
    ### 1. LINE 432 should have length 734 to predict the whole piece
    ### 2. LINE 394 change epochs to arround 50-130 for improved accuracy
    # ######################################################################
    
    
    # Restore final_predicted to have the same format with all_notes
    final_predicted_numpy = np.array(final_predicted)
    final_predicted_numpy = final_predicted_numpy[:,0,0,:]
    
    # Denormalize in order to reconstruct the piece
    ####################### Data De-Normalization ################################
    # explicit function to denormalize array
    def denormalize(arr, t_min, t_max):
        denorm_arr = []
        diff = 1
        diff_arr = t_max - t_min  
        if np.isnan(diff_arr) == False:
            for i in arr:
                temp = t_min+((i-0)*diff_arr/diff)
                denorm_arr.append(temp)           
        return denorm_arr
    
    final_predicted_numpy_actual = final_predicted_numpy.copy()
    #De-Normalize each feature indipendently:
    for i in range(89):
        final_predicted_numpy_actual[:,i] = denormalize(final_predicted_numpy[:,i],
                                    characteristics_min[i],
                                    characteristics_max[i])
        
    final_predicted_numpy_actual=np.array(final_predicted_numpy_actual)    
    final_predicted_numpy_actual[np.isnan(final_predicted_numpy_actual)] = 0
    final_predicted_numpy_actual[final_predicted_numpy_actual<4] = 0
    final_predicted_numpy_actual=np.round(final_predicted_numpy_actual)
    
    ######################################################################
    
    
    # """export predicted notes to csv fille:"""
    print("Exporting to csv. Note that the exported file has buffer_length less elements than the original")
    np.savetxt('final_predicted_AI_from_training.csv', final_predicted_numpy_actual)






"""
#########################################################################################################
#########################################################################################################
Step A: Training phase
-> step A4 : Train AI model 3
#########################################################################################################
#########################################################################################################
"""

#Before do anything else do not forget to reset the backend for the next iteration (rerun the model)
keras.backend.clear_session()#####################################

# Clear all previously registered custom objects
keras.saving.get_custom_objects().clear()

@keras.saving.register_keras_serializable(package="MyLayers")
class MultidimensionalLSTM(tf.keras.Model):
    def __init__(self, hidden_size, hidden_size2, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        
        #self.lstm = tf.keras.layers.LSTM(hidden_size,activation='silu', return_sequences=True)        
        self.lstm = tf.keras.layers.LSTM(hidden_size,activation='relu', return_sequences=True)        
        self.dense = tf.keras.layers.Dense(output_size,activation='linear')
    
    def call(self, inputs):
        x = self.lstm(inputs)  
        #x = self.linear(x)   
        x = self.dense(x[:, -1, :])
        return x

@keras.saving.register_keras_serializable(package="MyLayers")
class TCN(tf.keras.Model):
    def __init__(self, num_channels, kernel_size, output_size):
        super(TCN, self).__init__()
        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.output_size = output_size

        self.conv1d_layers = []
        if isinstance(num_channels, int):  # Handling single integer case
            num_channels = [num_channels]
        for i, channels in enumerate(num_channels):
            self.conv1d_layers.append(
                tf.keras.layers.Conv1D(channels, kernel_size, padding='causal', activation='relu', dilation_rate=2**i)
            )
        self.dense = tf.keras.layers.Dense(output_size, activation='linear')

    def call(self, inputs):
        x = inputs
        for layer in self.conv1d_layers:
            x = layer(x)
        x = self.dense(x[:, -1, :])
        return x

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
characteristics_max3=[]
characteristics_min3=[]
#Normalize each feature indipendently:
for i in range(89):
    range_to_normalize = (0,1)
    all_notes[:,i],characteristic_max,characteristic_min = normalize(all_notes[:,i],
                                range_to_normalize[0],
                                range_to_normalize[1])
    characteristics_max3.append(characteristic_max)
    characteristics_min3.append(characteristic_min)
    
all_notes=np.array(all_notes)    
all_notes[np.isnan(all_notes)] = 0
######################################################################


######################################################################
##############  INPUT DATA
######################################################################
#This is the time buffer (how many time samples will be used):
buffer_length=5
#This is the ammount of the characteristics choosen starting from the all_characteristics index:
characteristics_buff=89
#This is the len of the characteristics: 
#(choose 89 if duration should be also considered)
#(choose 88 if duration shouldn't be considered)
#(choose 41 to include characteristics between 41-characteristics_buff:41
all_characteristics=89
inputs=np.full((len(all_notes)-buffer_length,buffer_length,characteristics_buff), 0.0)

# Separate to inputs-outputs
for time_index in range(len(all_notes)-buffer_length): #number of changes
    input_temp=all_notes[time_index:time_index+buffer_length,all_characteristics-characteristics_buff:all_characteristics]
    inputs[time_index,0:buffer_length,0:characteristics_buff]=(input_temp)

# len(all_notes) samples, each with buffer_length time steps and 89 features
input_data3 = inputs

################################################################################
################################################################################ 

######################################################################
##############  OUTPUT DATA
######################################################################

output_data3=[]
for i in range(89):
    #This is the time buffer (how many time samples will be used):
    outputs=np.full((len(all_notes)-buffer_length,1,1), 0.0)
    
    SPESIFIC_CHARACTERISTIC_TO_PREDICT = i
    
    # Separate to inputs-outputs
    for time_index in range(len(all_notes)-buffer_length): #number of changes
        output_temp=all_notes[time_index+buffer_length:time_index+buffer_length+1,SPESIFIC_CHARACTERISTIC_TO_PREDICT]
        outputs[time_index,0:1,0]=(output_temp) 
    
    # len(all_notes) samples, each with 1 time step and 89 features
    output_data3.append(outputs)
################################################################################
################################################################################   


################################################################################
################################################################################   
################################################################################
################################################################################   
retrain=False
################################################################################
################################################################################   
################################################################################
################################################################################   

if retrain == True:
    model=[]
    for i in range(89):  #Default : 89. Use <89 to learn for less characteristics
        model.append(TCN(num_channels=20, kernel_size=58, output_size=1))
        '''
        #://www.tensorflow.org/api_docs/python/tf/keras/losses
        '''
        #model.compile(optimizer="adam", loss="mse")
        model[i].compile(optimizer=tf.keras.optimizers.Adam(0.01), loss=tf.keras.losses.MeanSquaredError(), metrics=['accuracy'])
        #model.compile(optimizer=tf.keras.optimizers.Adam(0.01), loss=tf.keras.losses.BinaryCrossentropy(), metrics=['accuracy'])
        
        print("################ Characteristic: ",i,"  ########################")
        model[i].fit(input_data3, output_data3[i], epochs=40)
        print("################  end  ########################")
        print("###############################################")
    
    
    
    #Save ML models
    for i in range(0,89):
       path=_path_+r"\AImodel3\\"
       model[i].save(path+"custom_model"+str(i)+".keras")

    #Load ML models
    loaded_models3=[]
    for i in range(0,89):
       path=_path_+r"\AImodel3\\"
       loaded_models3.append(keras.models.load_model(path+"custom_model"+str(i)+".keras"))
        
        
    #Check 
    # print("if following are equal means that loaded models are the same with the saved")
    # test_data = input_data[184,:,:]
    # test_data = test_data.reshape((1, buffer_length, characteristics_buff))   
    # print(model[1].predict(test_data, verbose=1))
    # print(loaded_models[1].predict(test_data, verbose=1))


if retrain == False:
    #Load ML models
    loaded_models3=[]
    for i in range(0,89):
       path=_path_+r"\AImodel3\\"
       loaded_models3.append(keras.models.load_model(path+"custom_model"+str(i)+".keras"))
        
            

"To check if the predictions are same with the initial song"
check_prediction_accuracy = False
if check_prediction_accuracy == True:
    t0 = time.time()
    final_predicted=[]
    for index in range(0,20): #for each state change (maximum value: (row_durations - buffer_length))
        predictNextSequence=[]
        real=[]
        test_data = input_data3[index,:,:]
        test_data = test_data.reshape((1, buffer_length, characteristics_buff))
        for i in range(0,89): #for each characteristic
            temp = loaded_models[i].predict(test_data, verbose=1)
            temp[temp<0.01] = 0
            predictNextSequence.append(temp)
            real.append(output_data3[i][index,:,:])
        # print('real:')
        # print(np.round(real,4))
        # print('predicted:')
        # print(np.round(predictNextSequence,4))
        # final_predicted.append(np.round(predictNextSequence,4))
        final_predicted.append(np.transpose(predictNextSequence, axes = (2,1,0)))
        #print("#################################################################")
        print("index: ",index)
    t1 = time.time()
    print('prediction duration=',t1-t0)
    

    
    # ######################################################################
    ### SUGGESTIONS:
    ### 1. LINE 432 should have length 734 to predict the whole piece
    ### 2. LINE 394 change epochs to arround 50-130 for improved accuracy
    # ######################################################################
    
    
    # Restore final_predicted to have the same format with all_notes
    final_predicted_numpy = np.array(final_predicted)
    final_predicted_numpy = final_predicted_numpy[:,0,0,:]
    
    # Denormalize in order to reconstruct the piece
    ####################### Data De-Normalization ################################
    # explicit function to denormalize array
    def denormalize(arr, t_min, t_max):
        denorm_arr = []
        diff = 1
        diff_arr = t_max - t_min  
        if np.isnan(diff_arr) == False:
            for i in arr:
                temp = t_min+((i-0)*diff_arr/diff)
                denorm_arr.append(temp)           
        return denorm_arr
    
    final_predicted_numpy_actual = final_predicted_numpy.copy()
    #De-Normalize each feature indipendently:
    for i in range(89):
        final_predicted_numpy_actual[:,i] = denormalize(final_predicted_numpy[:,i],
                                    characteristics_min[i],
                                    characteristics_max[i])
        
    final_predicted_numpy_actual=np.array(final_predicted_numpy_actual)    
    final_predicted_numpy_actual[np.isnan(final_predicted_numpy_actual)] = 0
    final_predicted_numpy_actual[final_predicted_numpy_actual<4] = 0
    final_predicted_numpy_actual=np.round(final_predicted_numpy_actual)
    
    ######################################################################
    
    
    # """export predicted notes to csv fille:"""
    print("Exporting to csv. Note that the exported file has buffer_length less elements than the original")
    np.savetxt('final_predicted_AI_from_training.csv', final_predicted_numpy_actual)




"""
#########################################################################################################
#########################################################################################################
Step A: Training phase
-> step A5 : Convert Tensorflow models to tensorflow lite models for 100x faster prediction
#########################################################################################################
#########################################################################################################
"""
# references:
#https://micwurm.medium.com/using-tensorflow-lite-to-speed-up-predictions-a3954886eb98
#https://github.com/tensorflow/tensorflow/issues/53101

##############  Transform models to ltlite models for faster pprediction speed
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
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


### first we need to do some warm up by utilising the initial keras models
buffer_length1 = 5
buffer_length2 = 15
buffer_length3 = 5
if True:
    ################################################   example input 3 (training song's pattern)
    #initialize:
    user_input_time_step1 = []
    for i in range(buffer_length1):
        user_input_time_step1.append(np.zeros(89))
    # add some random notes (with hit power). 
    index_check = 45
    user_input_time_step1 =  input_data1[index_check,:,:]
if True:
    ################################################   example input 3 (training song's pattern)
    #initialize:
    user_input_time_step2 = []
    for i in range(buffer_length2):
        user_input_time_step2.append(np.zeros(89))
    # add some random notes (with hit power). 
    index_check = 45
    user_input_time_step2 =  input_data2[index_check,:,:]
if True:
    ################################################   example input 3 (training song's pattern)
    #initialize:
    user_input_time_step3 = []
    for i in range(buffer_length3):
        user_input_time_step3.append(np.zeros(89))
    # add some random notes (with hit power). 
    index_check = 45
    user_input_time_step3 =  input_data3[index_check,:,:]

        
##### Normalize the inserted values:
normalized_user_input_time_step1 = np.array(user_input_time_step1.copy())
normalized_user_input_time_step_reshaped1 = normalized_user_input_time_step1.reshape((1, buffer_length1, characteristics_buff))    
normalized_user_input_time_step2 = np.array(user_input_time_step2.copy())
normalized_user_input_time_step_reshaped2 = normalized_user_input_time_step2.reshape((1, buffer_length2, characteristics_buff))
normalized_user_input_time_step3 = np.array(user_input_time_step3.copy())
normalized_user_input_time_step_reshaped3 = normalized_user_input_time_step3.reshape((1, buffer_length3, characteristics_buff))
for i in range(89):
    temp = loaded_models[i].predict(normalized_user_input_time_step_reshaped1, verbose=0)
    temp2 = loaded_models2[i].predict(normalized_user_input_time_step_reshaped2, verbose=0)
    temp3 = loaded_models3[i].predict(normalized_user_input_time_step_reshaped3, verbose=0)


### Then we can convert to actual ltlite models
lmodels = []
lmodels2 = []
lmodels3 = []
import os 
if not os.path.exists(_path_ +r"\tflite_models1"):      
    os.makedirs(_path_ +r"\tflite_models1") 
if not os.path.exists(_path_ +r"\tflite_models2"):      
    os.makedirs(_path_ +r"\tflite_models2") 
if not os.path.exists(_path_ +r"\tflite_models3"):      
    os.makedirs(_path_ +r"\tflite_models3")     
for i in range(89):
    folder_name = "tflite_models1"
    lmodels.append(LiteModel.from_keras_model_and_save(loaded_models[i],i,folder_name))
for i in range(89):   
    folder_name = "tflite_models2"
    lmodels2.append(LiteModel.from_keras_model_and_save(loaded_models2[i],i,folder_name))
for i in range(89):  
    folder_name = "tflite_models3"
    lmodels3.append(LiteModel.from_keras_model_and_save(loaded_models3[i],i,folder_name))


#check:
# temp=LiteModel.from_file(_path_+"/tflite_models3/model43.tflite")
# temp2=lmodels3[43]

# temp_predict = temp.predict(normalized_user_input_time_step_reshaped)
# temp2_predict = temp2.predict(normalized_user_input_time_step_reshaped)

# print(temp_predict)
# print(temp2_predict)
# Correct results!
