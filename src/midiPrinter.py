import midiFunctions
import MyLSTM

import pickle
from fileinput import filename
import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from music21 import *
import math


pitch_data = []
with open("../Intermediates/pitch_data", "rb") as fp:
    pitch_data = pickle.load(fp)
    
rhythm_data = []
with open("../Intermediates/rhythm_data", "rb") as fp:
    rhythm_data = pickle.load(fp)
 
    
mpLSTM = torch.load("../Intermediates/mpLSTMtmp")
mrLSTM = torch.load("../Intermediates/mrLSTMtmp")

"""
inputArr = []
oneHotArr = [0 for r in range(12)]
idx = np.random.randint(0,11)
oneHotArr[idx] = 1
rest = [0,0, 0, 0]
for i in range(4):
    rest[i] = np.random.randint(0, 100)
inputArr = inputArr + [oneHotArr + rest]
"""
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