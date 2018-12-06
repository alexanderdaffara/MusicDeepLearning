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
 
    
mpLSTM = torch.load("../Intermediates/mpLSTM")
mrLSTM = torch.load("../Intermediates/mrLSTM")

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
nextPitch = pitch_data[0][0]
nextRhythm = rhythm_data[0][0]
print(nextPitch)
print(nextRhythm)

nextPitch = mpLSTM(nextPitch).tolist()[0][0]
nextRhythm = mrLSTM(nextRhythm).tolist()[0][0]
print(mrLSTM(nextRhythm))
print(nextRhythm)

pitchArr = []
rhythmArr = []

#print()
for i in range(100):
    #print(i)
    #print(inputArr[i])
    print(mpLSTM(inputArr[i][:12]).tolist()[0][0])
    nextPitch = mpLSTM(inputArr[i][:12]).tolist()[0][0]
    nextRhythm = mrLSTM(inputArr[i][14:]).tolist()[0][0]
    pitchArr.append(nextPitch)
    rhythmArr.append(nextRhythm)
"""
[0.017690028995275497, -0.023935995995998383, -0.054671451449394226, -0.003906804136931896,
     0.036537908017635345, 0.0303180068731308, 0.035989824682474136 -0.04869161918759346,
    -0.016102591529488564, 0.03156878426671028, -0.009204618632793427, 0.027298886328935623, nan, nan]
[0.015781491994857788, -0.024790210649371147, -0.05739596486091614, -0.0017393557354807854,
 0.03929348662495613, 0.027984388172626495, 0.03854452446103096, -0.048154011368751526,
  -0.015895375981926918, 0.02896677888929844, -0.010697830468416214, 0.029026683419942856]
"""
#print(inputArr)

midiFunctions.printToMidi(pitchArr, rhythmArr)