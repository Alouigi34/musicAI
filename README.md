## MusicAI
This repository provides some functions for real time computed assisted composition.

AI models are trained based on a reference_piece.mid.
This piece acts as the assistant composer for the actual performance.

train_test_on_piece_whole_process.py provides a conceptual overview: There AI models are trained based on the piece. Later on they are used to make predictions based on the user input.

# The software is aimed to be used as foolows:
Step 1: Provide a reference .mid fille.

Step 2: Train the AI models using the provided function train_from_midi_song_and_create_models.py. This function currently supports the creation of 3 AI models that are saved as kerras models as well as tflite models. The conversion is necessary for using the models in real time afterwards.

Step 3: Run use_trained_models_to_make_suggestions.py function. This function provides suggestions to a performer in real time based on her/his choices. The suggestions of the 3 AI models are presented as notes in 3 vertically positioned keyboards. This GUI provide note suggestions as well as dynamic suggestions. 

Information: The use_trained_models_to_make_suggestions.py function makes suggestions based on the user input. This input provides information about the notes the performer played in a time frame. The suggestions are based on this information. The input can be provided by hand, created randomly, or by a microphone that reads the performers choices. Differently, an assistant to the current performer can observe the notes played, insert the selections inside the program , which will then generate the proposal. This process continues indefinitely. 

The input format is based on the AI models. THese are time-series based models that expect a predefined number of timestamped information.


