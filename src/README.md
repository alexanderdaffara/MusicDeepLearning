
hLSTM, mLSTM = setUpLSTMs(magicNumbers)

for( every file ):
    reset hidden layers
    inputArr1, inputArr2 = convertFileToInputArray()

    #CUDA/Graphics Card speed-up?

    trainHarmonyLSTM( hLSTM, inputArray2)
    trainMelodyLSTM( mLSTM, inputArray1, inputArray2)

noteArr1 = ""
noteArr2 = ""

for i in range(tickLimit):
    #Returns noteArr2 appended with new prediction using hLSTM
    noteArr2 = printHarmony(hLSTM, noteArr2)

for i in range(tickLimit):
    noteArr1 = printMelody(mLSTM, noteArr2, noteArr1)

playMIDI(noteArr1, noteArr2)

# convertFileToInputArr == vectorization of music
# trainMelodyLSTM() == how to factor in chords?
# DO we transpose to the same key?
 Ideas: If multiple vectorizations, GANy thing; Music2Vec
