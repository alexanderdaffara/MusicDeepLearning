import MyLSTM

from fileinput import filename

import pickle
import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

mpDim = 12
mrDim = 2

pitch_data = []
with open("../Intermediates/pitch_data", "rb") as fp:
    pitch_data = pickle.load(fp)
    
rhythm_data = []
with open("../Intermediates/rhythm_data", "rb") as fp:
    rhythm_data = pickle.load(fp)

(mpLSTM, loss_function, optimizer) = MyLSTM.prepareLSTM(12, 1024, 12, True)
(mrLSTM, loss_function, optimizer) = MyLSTM.prepareLSTM(2, 1024, 2, False)

for i in range(len(pitch_data)):
    
    print("Training on Song %d\n" % (i))
    
    #print(rhythm_data[i])
    MyLSTM.trainLSTM(mpDim, mpLSTM, loss_function, optimizer, pitch_data[i], 4)
    MyLSTM.trainLSTM(mrDim, mrLSTM, loss_function, optimizer, rhythm_data[i], 4)
    
    if(i % 10 == 0):
        torch.save(mpLSTM, "../Intermediates/mpLSTMtmp")
        torch.save(mrLSTM, "../Intermediates/mrLSTMtmp")
        print("saved!")

torch.save(mpLSTM, "../Intermediates/mpLSTM")
torch.save(mrLSTM, "../Intermediates/mrLSTM")