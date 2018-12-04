# Author: Robert Guthrie

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

INPUT_DIM = 3
HIDDEN_DIM = 256
OUT_DIM = 3

#training_data = [[i, i+1, i+2] for i in range(10)]
training_data = [ [1,2,3], [3, 2, 1], [6, 4, 2], [2, 4, 6], [1,2,3], [3, 2, 1], [6, 4, 2], [2, 4, 6], [1,2,3], [3, 2, 1], [6, 4, 2], [2, 4, 6]]

print(training_data)

class LSTMPredictor(nn.Module):

    def __init__(self, input_dim, hidden_dim):
        super(LSTMPredictor, self).__init__()
        self.hidden_dim = hidden_dim

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(input_dim, hidden_dim)
         # The linear layer that maps from hidden state space to tag space
        self.hidden2out = nn.Linear(hidden_dim, 3)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (torch.zeros(1, 1, self.hidden_dim),
                torch.zeros(1, 1, self.hidden_dim))

    def forward(self, inputVec):
        lstm_out, self.hidden = self.lstm(
            inputVec.view(1, 1,-1), self.hidden)
        out_space = self.hidden2out(lstm_out).view(1, 1, OUT_DIM)
        #tag_scores = F.log_softmax(out_space, dim=1)
        return out_space
    

model = LSTMPredictor(INPUT_DIM, HIDDEN_DIM)
loss_function = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=.001)

for epoch in range(1000):  # again, normally you would NOT do 300 epochs, it is toy data
    for i in range(len(training_data) - 1):
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()

        # Also, we need to clear out the hidden state of the LSTM,
        # detaching it from its history on the last instance.
        model.hidden = model.init_hidden()

        # Step 2. Get our inputs ready for the network, that is, turn them into
        # Tensors of word indices.
        input = torch.FloatTensor(training_data[i])
        target = torch.FloatTensor(training_data[i+1]).view(1, 1, 3)

        # Step 3. Run our forward pass.
        prediction = model(input)
        print(input)
        print(target)
        print(prediction)
        

        # Step 4. Compute the loss, gradients, and update the parameters by
        #  calling optimizer.step()
        loss = loss_function(prediction, target)
        print(loss)
        print()
        loss.backward()
        optimizer.step()

with torch.no_grad():
    print(model(torch.FloatTensor(training_data[0])))