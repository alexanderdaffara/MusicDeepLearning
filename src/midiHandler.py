from fileinput import filename
import numpy
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from music21 import *

def convertFileToMIDIArr(filename):
    
    s = stream.Stream()
    mf = midi.translate.streamToMidiFile(s)
    mf.open(filename, 'rb')
    mf.read()
    #print(mf)
    
    melodyMIDI = mf.tracks[2].events
    
    melodyEvents = []
    timeSinceLastNote = 0
    unresolvedDuration = {}
    unresolvedIdx = {}
    
    for i in range(2, len(melodyMIDI) - 2, 2):
        
        timeSinceLastNote = timeSinceLastNote + melodyMIDI[i].time
        
        for val in unresolvedDuration:
            unresolvedDuration[val] = unresolvedDuration[val] + melodyMIDI[i].time
        
        pitch = melodyMIDI[i+1].pitch
        velocity = melodyMIDI[i+1].velocity
        
        if(melodyMIDI[i+1].type == 'NOTE_ON'):
            
            unresolvedDuration[pitch] = 0
            unresolvedIdx[pitch] = len(melodyEvents)
            melodyEvents.append(val2Vec(pitch, velocity, timeSinceLastNote))
            timeSinceLastNote = 0
        
        else:
            
            melodyEvents[unresolvedIdx[pitch]][15] = unresolvedDuration[pitch]
            del unresolvedIdx[pitch]
            del unresolvedDuration[pitch]
            
        #print(timeSinceLastNote)
    #mStream = converter.parse(filename)
    #show(mStream)
    
    return melodyEvents
    
def val2Vec(pitch, velocity, delay):
    toReturn = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, pitch, velocity, delay, -1]
    oneHotIdx = pitch % 12
    toReturn[oneHotIdx] = 100
    return toReturn
    
# Main Code Starts Here