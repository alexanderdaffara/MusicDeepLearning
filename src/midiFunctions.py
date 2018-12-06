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
    toReturn[oneHotIdx] = 1
    return toReturn
    
def printToMidi(pitchList, rhythmList):
    
    s1 = stream.Stream()
    
    #randomGap = 2
    timePassed = 0
    
    #print(inputList)
    #print(len(inputList))
    
    for i in range(len(pitchList)):
        
        timePassed = timePassed + rhythmList[i][0]
        #print("timePassed: %d\n" % (timePassed) )
        
        p = pitch.Pitch()
        octave = 4
        
        #max2 = max
        #TODO: better combination of continuous and discrete pitch
        max = pitchList[i][0]
        maxIdx = 0
        #maxIdx2 = maxIdx
        for j in range(1, 12):
            if (pitchList[i][j] > max):
                #max2 = max
                max = pitchList[i][j]
                maxIdx = j
                #maxIdx2 = maxIdx
        #If midi is p unconfident, we trust pitch?
        p.midi = maxIdx + 12*octave
        
        #if((i % randomGap) == 0):
        #    p.midi = maxIdx2 * octave
        
        n = note.Note()
        n.duration = duration.Duration( rhythmList[i][1] / 1280 )
        n.pitch = p
        n.volume.velocity = 90
        
        s1.insert((timePassed / 1280), n)
    
    s1.show('midi')
    
        
    