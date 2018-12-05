from music21 import *

#s = stream.Stream()
#mf = midi.translate.streamToMidiFile(s)
#mf.open('/MIDIs/happy.mid', 'wb')
"""
s = stream.Stream()
mf = midi.translate.streamToMidiFile(s)

mt = midi.MidiTrack(1)

dt = midi.MidiEvent(mt)
dt.type = 'DeltaTime'
dt.time = 0
dt.channel = None

me1 = midi.MidiEvent(mt)
me1.type = 'NOTE_ON'
me1.channel = 3
me1.time = 200
me1.pitch = 60
me1.velocity = 120

dt1 = midi.MidiEvent(mt)
dt1.type = 'DeltaTime'
dt1.time = 200
dt1.channel = None

me2 = midi.MidiEvent(mt)
me2.type = 'NOTE_OFF'
me2.channel = 3
me2.time = 200
me2.pitch = 60
me2.velocity = 120

mt.events.append(me1)
mt.events.append(dt)
mt.events.append(me2)
mt.events.append(dt1)
mf.tracks.append(mt)

print(mf)

mf.open("../MIDIs/output.mid", 'wb')
mf.write()
mf.close
"""

s1 = stream.Stream()
#s1.show('midi')
s1.write('midi', "../MIDIs/output.mid")