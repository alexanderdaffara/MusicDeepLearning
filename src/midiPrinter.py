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


training_data = []
with open("../Intermediates/training_data", "rb") as fp:
    training_data = pickle.load(fp)
    
mLSTM = torch.load("../Intermediates/mLSTM")

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
midiFunctions.printToMidi(inputArr)