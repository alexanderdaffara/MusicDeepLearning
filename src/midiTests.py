from music21 import *

#s = stream.Stream()
#mf = midi.translate.streamToMidiFile(s)
#mf.open('/MIDIs/happy.mid', 'wb')

mf.open("../MIDIs/output.mid")
mf.write()
mt = midi.MidiTrack(1)
me1 = midi.MidiEvent(mt)
me1.type = 'NOTE_ON'
me1.channel = 3
me1.time = 200
me1.pitch = 60
me1.velocity = 120
