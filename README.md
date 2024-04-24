## MusicAI
This repository provides some functions for real time computed assisted composition.

AI models are trained based on a reference_piece.mid.
This piece acts as the assistant composer for the actual performance.

# train_test_on_piece_whole_process.py provides a conceptual overview: There AI models are trained based on the piece. Later on they are used to make predictions based on the user input.

# The software is aimed to be used as foolows:
Step 1: Provide a reference .mid fille.
Step 2: Train the AI models using the provided function train_from_midi_song_and_create_models.py. This function currently supports the creation of 3 AI models that are saved as kerras models as well as tflite models. The conversion is necessary for using the models in real time afterwards.
Step 3:


