import midiFunctions
import pickle

from fileinput import filename
from music21 import *
import math

training_data = []
training_data.append(midiFunctions.convertFileToMIDIArr("../MIDIs/Get Lucky.mid"))
training_data.append(midiFunctions.convertFileToMIDIArr("../MIDIs/Happy - Copy.mid"))

with open("../Intermediates/training_data", "wb") as fp:
    pickle.dump(training_data, fp)
    