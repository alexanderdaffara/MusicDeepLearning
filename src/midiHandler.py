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
    
    mf.close()
    return melodyEvents
    
def val2Vec(pitch, velocity, delay):
    toReturn = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, pitch, velocity, delay, -1]
    oneHotIdx = pitch % 12
    toReturn[oneHotIdx] = 100
    return toReturn
    
def printToMidi(inputList):
    
    s1 = stream.Stream()
    
    timePassed = 0
    
    #print(inputList)
    #print(len(inputList))
    
    for i in range(len(inputList)):
        
        timePassed = timePassed + inputList[i][14]
        #print("timePassed: %d\n" % (timePassed) )
        
        p = pitch.Pitch()
        octave = inputList[i][12] // 12 + 1
        max = inputList[i][0]
        #TODO: better combination of continuous and discrete pitch
        maxIdx = 0
        for j in range(1, 12):
            if (inputList[i][j] > max):
                max = inputList[i][j]
                maxIdx = j
        #If midi is p unconfident, we trust pitch?
        p.midi = maxIdx * octave
        n = note.Note()
        n.duration = duration.Duration( inputList[i][15] / 1280 )
        n.pitch = p
        n.volume.velocity = inputList[i][13]
        
        s1.insert((timePassed / 1280), n)
    
    s1.show('midi')
        
    