import midiHandler
import MyLSTM

from fileinput import filename
import numpy
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from music21 import *

(mLSTM, loss_function, optimizer) = MyLSTM.prepareLSTM(16, 16, 16)

training_data = midiHandler.convertFileToMIDIArr("../MIDIs/Get Lucky.mid")

MyLSTM.trainLSTM(16, mLSTM, loss_function, optimizer, training_data, 100)

print(mLSTM(training_data[0]))



mt = midi.MidiTrack(1)
me1 = midi.MidiEvent(mt)
me1.type = 'NOTE_ON'
me1.channel = 3
me1.time = 200
me1.pitch = 60
me1.velocity = 120


