# Author: Alexander Daffara and Aditya Chandrasekar

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.utils.clip_grad as clip



class LSTMPredictor(nn.Module):


    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTMPredictor, self).__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        
        self.lstm = nn.LSTM(input_dim, hidden_dim)
        self.hidden2out = nn.Linear(hidden_dim, self.output_dim)
        self.hidden = self.init_hidden()

    def init_hidden(self):

        return (torch.zeros(1, 1, self.hidden_dim),
                torch.zeros(1, 1, self.hidden_dim))

    def forward(self, inputVec):
        #print("inputVec =")
        #print(inputVec)
        
        inputVec = torch.FloatTensor(inputVec).view(1, 1, -1)
        #print(inputVec)
        
        lstm_out, self.hidden = self.lstm(
            inputVec, self.hidden)
        
        squashifier = nn.Tanh()  
        out_space = squashifier(lstm_out)

        out_space = self.hidden2out(out_space).view(1, 1, self.output_dim)
        
        #print(out_space)
        #sub_list = torch.clone(out_space[:, :, :12])
        #sub_list = F.softmax(sub_list, dim=2)
        #new_out = torch.cat([ sub_list, out_space[:, :, 12:] ], dim = 2)
        #print(new_out)
        #clip.clip_grad_value_(out_space, .5)
        
        return out_space
    
class LSTMPredictorSoftmax(nn.Module):


    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTMPredictorSoftmax, self).__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim


        self.lstm = nn.LSTM(input_dim, hidden_dim)

        self.hidden2out = nn.Linear(hidden_dim, self.output_dim)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        
        return (torch.zeros(1, 1, self.hidden_dim),
                torch.zeros(1, 1, self.hidden_dim))

    def forward(self, inputVec):
        
        inputVec = torch.FloatTensor(inputVec).view(1, 1, -1)
        
        lstm_out, self.hidden = self.lstm(
            inputVec, self.hidden)
        
        squashifier = nn.Tanh()  
        out_space = squashifier(lstm_out)
        
        out_space = self.hidden2out(out_space).view(1, 1, self.output_dim)
        
        new_out = F.softmax(out_space, dim=2)
        
        return new_out
    
def prepareLSTM(inputDim, hiddenDim, outputDim, isSoftMax):
    
    if (isSoftMax == True):
        model = LSTMPredictorSoftmax(inputDim, hiddenDim, outputDim)
        #print("softmax input")
        #print(inputDim)
    else:
        model = LSTMPredictor(inputDim, hiddenDim, outputDim)
        #print("not softmax inputDim")
        #print(inputDim)
        
    loss_function = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=.005)
    return (model, loss_function, optimizer)

class plotter():
    times = []
    losses = []
    ticker = 0

#plotty = plotter()
plottyPerEpoch = plotter()

def trainLSTM(output_dim, model, loss_function, optimizer, training_data, epochs):
    
    #print(training_data)

    
    training_data = torch.FloatTensor(training_data)
    
    #.view(len(training_data), 1, output_dim)
    
    #print(training_data)
    
    
    
    for epoch in range(epochs):
        totalEpochLoss = 0

        for i in range(len(training_data) - 1):
            
            model.zero_grad()

            model.hidden = model.init_hidden()
    
            input = torch.FloatTensor(training_data[i]).view(1, 1, model.output_dim)
            target = torch.FloatTensor(training_data[i+1]).view(1, 1, model.output_dim)
            
            #print(target)
    

            prediction = model(input)
            
            #print(prediction)
            #print(input)
            #print(target)
            
            loss = loss_function(target, prediction)
            
            #print(loss)
            
            #plotty.times.append(plotty.ticker)
            #plotty.ticker = plotty.ticker + 1
            #plotty.losses.append(loss.item())
            
            totalEpochLoss = totalEpochLoss + loss.item()
            
            loss.backward()
            optimizer.step()
    
        plottyPerEpoch.times.append(plottyPerEpoch.ticker)
        plottyPerEpoch.ticker = plottyPerEpoch.ticker + 1
        plottyPerEpoch.losses.append(totalEpochLoss)
    return

#with torch.no_grad():
#    print(model(torch.FloatTensor(training_data[0])))