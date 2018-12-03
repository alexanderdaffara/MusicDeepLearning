"""

hLSTM, mLSTM = setUpLSTMs(magicNumbers)

for( every file ):
    reset hidden layers
    convertFileToInputArray()

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

 Ideas: If multiple vectorizations, GANy thing; Music2Vec

"""
# BUT z knows something extra.
#print(z.grad_fn)

s = z.sum()
s.backward()
#print(x.grad)
#print(y.grad)

lin = nn.Linear(5, 3)  # maps from R^5 to R^3, parameters A, b
# data is 2x5.  A maps from 5 to 3... can we map "data" under A?
data = torch.randn(2, 5)
#print(data)
#print(lin(data))  # yes


data = torch.randn(2, 2)
#print(data)
#print(F.relu(data))

data = torch.randn(10, 2)
#print(data)
#print(F.softmax(data, dim=0))
#print(F.softmax(data, dim=0).sum())  # Sums to 1 because it is a distribution!
#print(F.log_softmax(data, dim=0))  # theres also log_softmax

i = torch.randn(1,9)
#print(i)
#print(i.view(1, 1, -1))


lstm = nn.LSTM(3, 3)  # Input dim is 3, output dim is 3
inputs = [torch.randn(1, 3) for _ in range(5)]  # make a sequence of length 5

# initialize the hidden state.
hidden = (torch.randn(1, 1, 3),
          torch.randn(1, 1, 3))
for i in inputs:
    # Step through the sequence one element at a time.
    # after each step, hidden contains the hidden state.
    out, hidden = lstm(i.view(1, 1, -1), hidden)
    print("LSTM")
    print(out)
    print(hidden)
    
#inputs = torch.cat(inputs).view(len(inputs), 1, -1)
#out, hidden = lstm(inputs, hidden)


