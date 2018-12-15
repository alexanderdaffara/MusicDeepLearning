import midiFunctions
import pickle
import copy

from fileinput import filename
from music21 import *
import math

training_data = []
import os
 
path = '../MIDIs/160+JazzSongs'
listing = os.listdir(path)
for infile in listing:
    training_data.append(midiFunctions.convertFileToMIDIArr(os.path.join(path, infile)))

pitch_data = copy.deepcopy(training_data)
rhythm_data = copy.deepcopy(training_data)


for song in range(len(training_data)):
    for note in range(len(training_data[song])):
        pitch_data[song][note] = training_data[song][note][:12]
        rhythm_data[song][note] = training_data[song][note][14:]
        
        #print(pitch_data[song][note])
        #print(rhythm_data[song][note])
        """
        for value in range(len(pitch_data[song][note])):
            
            if(type(pitch_data[song][note][value]) is not int):
                print("input error")
        for value in range(len(rhythm_data[song][note])):
            if(type(rhythm_data[song][note][value]) is not int):
                print("input error")
        """

print(pitch_data)

with open("../Intermediates/pitch_data", "wb") as fp:
    pickle.dump(pitch_data, fp)
    
with open("../Intermediates/rhythm_data", "wb") as fp:
    pickle.dump(rhythm_data, fp)
    