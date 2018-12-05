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

print(midiHandler.convertFileToMIDIArr("../MIDIs/Get Lucky.mid"))


