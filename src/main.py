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
    
    mStream = converter.parse(filename)
    mStream.show('midi')
    print(mf)
    
    
    """
    parsed_midi = midi.MidiFile()
    parsed_midi.open(filename, "rb")
    parsed_midi.read()
    parsed_midi.close()
    parsed_midi.open(filename, "wb")
    parsed_midi.write()
    parsed_midi.close()
    """
    
    
    
# Main Code Starts Here
convertFileToMIDIArr("../MIDIs/Untitled.mid")
