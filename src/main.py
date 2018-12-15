
#For Midi Conversion
import midiFunctions
import os
import pickle
import copy
from fileinput import filename
from music21 import *
import math

training_data = []
 
 
 """
 
    NOTE TO POTENTIAL GRADERS
 
    The way we normally execute our code are through the midiConverter, trainer, and midiPrinter files, so that
 we can independently reconvert MIDI, train the model parameters, etc. etc.
 
     This main function is here so that each of the parts is of the data flow is explicitly stated
 in this file, but trainer.py includes the LSTM training and MyLSTM.py includes are LSTM implementations.
 
 """
 
 #Part 1: Convert MIDI songs into vectors to be processed by LSTM
 
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
      
 #Part 2: Train our LSTM on the pre-processed data
 
 import MyLSTM

from fileinput import filename

import pickle
import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

mpDim = 12
mrDim = 2

pitch_data = []
with open("../Intermediates/pitch_data", "rb") as fp:
    pitch_data = pickle.load(fp)
    
rhythm_data = []
with open("../Intermediates/rhythm_data", "rb") as fp:
    rhythm_data = pickle.load(fp)

(mpLSTM, loss_function, optimizer) = MyLSTM.prepareLSTM(12, 1024, 12, False)
(mrLSTM, loss_function, optimizer) = MyLSTM.prepareLSTM(2, 1024, 2, False)


for i in range(len(pitch_data)):
    
    print("Training on Song %d\n" % (i))
    
    #print(rhythm_data[i])
    MyLSTM.trainLSTM(mpDim, mpLSTM, loss_function, optimizer, pitch_data[i], 5)
    MyLSTM.trainLSTM(mrDim, mrLSTM, loss_function, optimizer, rhythm_data[i], 10)
    
    
    if(i % 10 == 0):
        torch.save(mpLSTM, "../Intermediates/mpLSTMtmp")
        torch.save(mrLSTM, "../Intermediates/mrLSTMtmp")
        title = "Total Loss per Epoch"
        plt.xlabel("Epoch")
        plt.ylabel("Total Loss")
        plt.title(title)
        
        plt.plot(MyLSTM.plottyPerEpoch.times, MyLSTM.plottyPerEpoch.losses, 'ro')
        plt.show()
        print("saved!")

plt.show()

torch.save(mpLSTM, "../Intermediates/mpLSTM")
torch.save(mrLSTM, "../Intermediates/mrLSTM")

# Part 3: Use models to print to a midi file output and plot loss function over time

nextPitch = pitch_data[4][6]
nextRhythm = rhythm_data[3][0]

#print(nextPitch)
#print(nextRhythm)

nextPitch = mpLSTM(nextPitch).tolist()[0][0]
print(nextRhythm)
nextRhythm = mrLSTM(nextRhythm).tolist()[0][0]
#print(nextRhythm)


pitchArr = []
rhythmArr = []
pitchArr.append(nextPitch)
rhythmArr.append(nextRhythm)
#print()
for i in range(100):
    #print(i)
    #print(inputArr[i])
    print(mpLSTM(pitchArr[i]).tolist()[0][0])
    nextPitch = mpLSTM(pitchArr[i]).tolist()[0][0]
    nextRhythm = mrLSTM(rhythmArr[i]).tolist()[0][0]
    pitchArr.append(nextPitch)
    rhythmArr.append(nextRhythm)

#print(inputArr)

midiFunctions.printToMidi(pitchArr, rhythmArr)
