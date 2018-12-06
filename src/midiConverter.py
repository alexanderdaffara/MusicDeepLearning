import midiFunctions
import pickle

from fileinput import filename
from music21 import *
import math

training_data = []
import os
 
path = '../MIDIs/JazzSongs'
listing = os.listdir(path)
for infile in listing:
    training_data.append(midiFunctions.convertFileToMIDIArr(os.path.join(path, infile)))

for i in range(len(training_data)):
    for j in range(len(training_data[i])):
        print(training_data[i][j])
        
with open("../Intermediates/training_data", "wb") as fp:
    pickle.dump(training_data, fp)
    