import midiHandler
import MyLSTM

from fileinput import filename
import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from music21 import *
import math

(mLSTM, loss_function, optimizer) = MyLSTM.prepareLSTM(16, 1024, 16)
training_data = []
training_data.append(midiHandler.convertFileToMIDIArr("../MIDIs/Get Lucky.mid"))
training_data.append(midiHandler.convertFileToMIDIArr("../MIDIs/Happy - Copy.mid"))
print(training_data)

#MyLSTM.trainLSTM(16, mLSTM, loss_function, optimizer, training_data[0], 200)
MyLSTM.trainLSTM(16, mLSTM, loss_function, optimizer, training_data[1], 400)

inputArr = []
oneHotArr = [0 for r in range(12)]
idx = np.random.randint(0,11)
oneHotArr[idx] = 100
rest = [0,0,0,0]
for i in range(4):
    rest[i] = np.random.randint(0, 100)
inputArr = inputArr + [oneHotArr + rest]
inputArr[0] = mLSTM(inputArr[0]).tolist()[0][0]
inputArr[0] = training_data[0][0]

#print()
for i in range(100):
    #print(i)
    #print(inputArr[i])
    next = mLSTM(inputArr[i]).tolist()[0][0]
    next[12] = math.floor(next[12])
    next[13] = math.floor(next[13])
    inputArr.append( next )
    #print(next)

#print(inputArr)
midiHandler.printToMidi(inputArr)