import MyLSTM

from fileinput import filename

import pickle
import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

training_data = []
with open("../Intermediates/training_data", "rb") as fp:
    training_data = pickle.load(fp)

print(training_data)

(mLSTM, loss_function, optimizer) = MyLSTM.prepareLSTM(16, 256, 16)

MyLSTM.trainLSTM(16, mLSTM, loss_function, optimizer, training_data[0], 200)
MyLSTM.trainLSTM(16, mLSTM, loss_function, optimizer, training_data[1], 200)

torch.save(mLSTM, "../Intermediates/mLSTM")