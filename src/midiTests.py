from music21 import *

"""
[0, 0, 100, 0, 0, 0, 0, 0, 0, 0, 0, 0, 74, 102, 0, 2560],

[0, 100, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 73, 102, 2560, 512]

1280 == quarter note for all MIDI

"""
s1 = stream.Stream()

p = pitch.Pitch()
p.midi = 74
n = note.Note()
n.duration = duration.Duration( 2560 / 1280 )
n.pitch = p
n.volume.velocity = 120

s1.insert(0.0, n)

p = pitch.Pitch()
p.midi = 73
n = note.Note()
n.duration = duration.Duration( 512 / 1280)
n.pitch = p
n.volume.velocity = 102

s1.insert((2560 / 1280), n)

p = pitch.Pitch()
p.midi = 74
n = note.Note()
n.duration = duration.Duration( 512 / 1280)
n.pitch = p
n.volume.velocity = 102

s1.insert((3072 / 1280), n)

s1.show('midi')

