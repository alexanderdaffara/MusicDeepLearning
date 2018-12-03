from music21 import *

s = stream.Stream()
mf = midi.translate.streamToMidiFile(s)
mf.open('/MIDIs/happy.mid', 'wb')
