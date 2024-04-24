## MusicAI
This repository provides some functions for real time computed assisted composition.

AI models are trained based on a reference_piece.mid.
This piece acts as the assistant composer for the actual performance.

train_test_on_piece_whole_process.py provides a conceptual overview: There AI models are trained based on the piece. Later on they are used to make predictions based on the user input.

# The software is aimed to be used as follow:
Step 1: Provide a reference .mid fille.

Step 2: Train the AI models using the provided function train_from_midi_song_and_create_models.py. This function currently supports the creation of 3 AI models that are saved as kerras models as well as tflite models. The conversion is necessary for using the models in real time afterwards.

(Step 3a: Run use_trained_models_to_make_suggestions.py function. This function provides suggestions to a performer based on her/his choices. The suggestions of the 3 AI models are printed.)

Step 3b: Run read_input_and_use_trained_models_to_make_suggestions.py. This function provides suggestions to a performer in real time based on her/his choices. The suggestions of the 3 AI models are presented as notes in 3 vertically positioned keyboards. This GUI provides note suggestions as well as dynamic range suggestions. 

Information: The read_input_and_use_trained_models_to_make_suggestions.py function makes suggestions based on the user input. This input provides information about the notes the performer played in a time frame. The suggestions are based on this information. The input can be provided by hand, created randomly, or by a microphone that reads the performers choices. Differently, an assistant to the current performer can observe the notes played, insert the selections inside the program , which will then generate the proposal. This process continues indefinitely. 

The input format is based on the AI models. These are time-series based models that expect a predefined number of timestamped information. For this example, the two AI models expect a sequence of 5 chords, while the third a sequence of 15 chords. Also time information is inherited in the models input too. The proposal of each model is always a chord. In the future this could be extended to multiple notes or chords.

Additional functions:
read_from_mic_and_extract_notes_real_time_and_draw_on_keyboard.py. This function transform the input microphone reading to notes. It uses an FFT based identification process. It works well when 2 or 3 frequencies are presented. Problems occur with more frequencies. This function is provided as an illustration of how a microphone could read the current performers' choices. Improvements are required and an assistant could be used instead.

![alt text](https://github.com/Alouigi34/musicAI/blob/main/example.png)

