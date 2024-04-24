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
WIDTH = 52 * 35
HEIGHT = 400
screen = pygame.display.set_mode([WIDTH, HEIGHT])
white_sounds = []
black_sounds = []
active_whites = []
active_blacks = []
left_oct = 4
right_oct = 5

left_hand = pl.left_hand
right_hand = pl.right_hand
piano_notes = pl.piano_notes
white_notes = pl.white_notes
black_notes = pl.black_notes
black_labels = pl.black_labels

pygame.display.set_caption("Piano")



def draw_piano(whites, blacks):
    white_rects = []
    for i in range(52):
        rect = pygame.draw.rect(screen, 'white', [i * 35, HEIGHT - 300, 35, 300], 0, 2)
        white_rects.append(rect)
        pygame.draw.rect(screen, 'black', [i * 35, HEIGHT - 300, 35, 300], 2, 2)
        #################################################################
        ### This draws letters on the white notes 
        key_label = small_font.render(white_notes[i], True, 'black')
        screen.blit(key_label, (i * 35 + 3, HEIGHT - 20))
        #################################################################

    skip_count = 0
    last_skip = 2
    skip_track = 2
    black_rects = []
    for i in range(36):
        rect = pygame.draw.rect(screen, 'black', [23 + (i * 35) + (skip_count * 35), HEIGHT - 300, 24, 200], 0, 2)
    
        #################################################################
        ### This draws a green squarre on the currently pressed black keys        
        for q in range(len(blacks)):
            if blacks[q][0] == i:
                if blacks[q][1] > 0:
                    pygame.draw.rect(screen, 'green', [23 + (i * 35) + (skip_count * 35), HEIGHT - 300, 24, 200], 2, 2)
                    blacks[q][1] -= 1
        #################################################################

        #################################################################
        ### This draws letters on the black notes 
        key_label = real_small_font.render(black_labels[i], True, 'white')
        screen.blit(key_label, (25 + (i * 35) + (skip_count * 35), HEIGHT - 120))
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
    ### This draws a green squarre on the currently pressed white keys
    for i in range(len(whites)):
        if whites[i][1] > 0:
            j = whites[i][0]
            pygame.draw.rect(screen, 'green', [j * 35, HEIGHT - 100, 35, 100], 2, 2)
            whites[i][1] -= 1
    #################################################################



    return white_rects, black_rects, whites, blacks




def freq_to_note(frequency):
    """
    Convert frequency to corresponding piano note.
    """
    if frequency == 0:
        return None

    # Dictionary mapping note names to frequencies
    notes = {
        "A0": 27.5, "A#0": 29.135, "B0": 30.868,
        "C1": 32.703, "C#1": 34.648, "D1": 36.708, "D#1": 38.891, "E1": 41.203, "F1": 43.654, "F#1": 46.249, "G1": 48.999, "G#1": 51.913,
        "A1": 55.0, "A#1": 58.27, "B1": 61.735,
        "C2": 65.406, "C#2": 69.296, "D2": 73.416, "D#2": 77.782, "E2": 82.407, "F2": 87.307, "F#2": 92.499, "G2": 97.999, "G#2": 103.826,
        "A2": 110.0, "A#2": 116.541, "B2": 123.471,
        "C3": 130.813, "C#3": 138.591, "D3": 146.832, "D#3": 155.563, "E3": 164.814, "F3": 174.614, "F#3": 184.997, "G3": 195.998, "G#3": 207.652,
        "A3": 220.0, "A#3": 233.082, "B3": 246.942,
        "C4": 261.626, "C#4": 277.183, "D4": 293.665, "D#4": 311.127, "E4": 329.628, "F4": 349.228, "F#4": 369.994, "G4": 391.995, "G#4": 415.305,
        "A4": 440.0, "A#4": 466.164, "B4": 493.883,
        "C5": 523.251, "C#5": 554.365, "D5": 587.33, "D#5": 622.254, "E5": 659.255, "F5": 698.456, "F#5": 739.989, "G5": 783.991, "G#5": 830.609,
        "A5": 880.0, "A#5": 932.328, "B5": 987.767,
        "C6": 1046.502, "C#6": 1108.731, "D6": 1174.659, "D#6": 1244.508, "E6": 1318.51, "F6": 1396.913, "F#6": 1479.978, "G6": 1567.982, "G#6": 1661.219,
        "A6": 1760.0, "A#6": 1864.655, "B6": 1975.533,
        "C7": 2093.005, "C#7": 2217.461, "D7": 2349.318, "D#7": 2489.016, "E7": 2637.021, "F7": 2793.826, "F#7": 2959.955, "G7": 3135.964, "G#7": 3322.438,
        "A7": 3520.0, "A#7": 3729.31, "B7": 3951.066,
        "C8": 4186.009
    }
    # Initialize dictionaries for black and white keys
    black_keys_dict = {}
    white_keys_dict = {}
    
    # Initialize counters for black and white keys
    black_index = 0
    white_index = 0
    
    # Iterate over the notes dictionary and assign indexes to black and white keys
    for note_name, freq in sorted(notes.items(), key=lambda x: x[1]):
        # Check if the note name contains '#' indicating a black key
        if '#' in note_name:
            black_keys_dict[black_index] = freq
            black_index += 1
        else:
            white_keys_dict[white_index] = freq
            white_index += 1
    
    # Print the dictionaries
    #print("Black Keys Dictionary:")
    #print(black_keys_dict)
    #print("\nWhite Keys Dictionary:")
    #print(white_keys_dict)
    
    
    # Find the closest note frequency
    closest_note = min(notes.items(), key=lambda x: abs(x[1] - frequency))
    #return closest_note[0]

    if '#' in closest_note[0]:
        # Return the index of the black keys dictionary and the type of note
        return next((index for index, freq in black_keys_dict.items() if freq == closest_note[1]), None), "black" , closest_note[0]
    else:
        # Return the index of the white keys dictionary and the type of note
        return next((index for index, freq in white_keys_dict.items() if freq == closest_note[1]), None), "white", closest_note[0]


def frequencies_to_notes(frequencies):
    """
    Convert list of frequencies to corresponding piano notes.
    """
    unique_notes = set()
    notes = []
    for frequency in frequencies:
        note = freq_to_note(frequency)
        if note not in unique_notes:
            unique_notes.add(note)
            notes.append(note)
    return notes

p = pyaudio.PyAudio()

for i in range(p.get_device_count()):
    print(p.get_device_info_by_index(i))

p.terminate()

audio = pyaudio.PyAudio()
print("----------------------record device list---------------------")
info = audio.get_host_api_info_by_index(0)
numdevices = info.get('deviceCount')
for i in range(0, numdevices):
        if (audio.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
            print("Input Device id ", i, " - ", audio.get_device_info_by_host_api_device_index(0, i).get('name'))

print("-------------------------------------------------------------")

audio.terminate()

index = 1

# open stream
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = int(44100/4) # RATE / number of updates per second
RECORD_SECONDS = 60

# use a Blackman window
window = np.blackman(CHUNK)

last_time = time.time()
last_notes = []

def soundPlot(stream):
    global last_time
    global last_notes

    t1 = time.time()
    data = stream.read(CHUNK, exception_on_overflow=False)
    waveData = wave.struct.unpack("%dh"%(CHUNK), data)
    npArrayData = np.array(waveData)
    indata = npArrayData * window

    fftData = np.abs(np.fft.rfft(indata))
    fftTime = np.fft.rfftfreq(CHUNK, 1. / RATE)

    ascending_indices = fftData[1:].argsort() + 1    
    descending_indices = ascending_indices[::-1]
    all_frequencies = []    

    for which in descending_indices[0:8]:   
        if which != len(fftData)-1:
            y0, y1, y2 = np.log(fftData[which-1:which+2:])
            x1 = (y2 - y0) * .5 / (2 * y1 - y2 - y0)
            thefreq = (which + x1) * RATE / CHUNK
        else:
            thefreq = which * RATE / CHUNK
        all_frequencies.append(thefreq)

    notes = frequencies_to_notes(all_frequencies)
    #print("Detected notes:", notes)

    durations = [time.time() - last_time] * len(notes)
    if last_notes == notes:
        durations = [0] * len(notes)
    #print("Notes durations:", durations)

    last_time = time.time()
    last_notes = notes

    return notes, durations
    

if __name__=="__main__":
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=RATE, input=True, frames_per_buffer=CHUNK, input_device_index=index)

    plt.ion()
    fig = plt.figure(figsize=(10,8))
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        notes_heard,durations_heard = soundPlot(stream)
        print("Detected notes:", notes_heard)
        print("Notes durations:", durations_heard)
        
        black_notes_ = []
        white_notes_ = []
        for i in range(len(notes_heard)):
            if notes_heard[i][1] == "white":
                white_notes_.append(notes_heard[i][0])
            if notes_heard[i][1] == "black":
                black_notes_.append(notes_heard[i][0])      
        
        print(black_notes_)    
        print(white_notes_)    
        
        timer.tick(fps)
        screen.fill('gray')
        
        white_keys, black_keys, active_whites, active_blacks = draw_piano(active_whites, active_blacks)
        #print(active_whites)
        #print(active_blacks)
        
        
        ## Draw all notes heard on keyboard
        # if white_notes_:
        #     for i in range(len(white_notes_)):
        #         active_whites.append([white_notes_[i], 3])
        # if black_notes_:        
        #     for i in range(len(black_notes_)):
        #         active_blacks.append([black_notes_[i], 3])
        
        ## Draw only most significant note heard on keyboard        
        if white_notes_:
            active_whites.append([white_notes_[0], 3])
        if black_notes_:
            active_blacks.append([black_notes_[0], 3])
            
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False


        pygame.display.flip()
    pygame.quit()



    stream.stop_stream()
    stream.close()
    p.terminate()
