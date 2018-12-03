from music21 import *

n = note.Note("Ab3")
n.duration.type = 'quarter'
#n.show()

littleMelody = converter.parse("tinynotation: 3/4 g4 a b c d b8 g a2")
littleMelody.show('midi')

#change