import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init


class Net(nn.Module):
    #layers is array that contain 3 element，they are l1，l2，l3's input size，l4's output size is 5 (5 types)
    def __init__(self,layers,type=2,component=1):
        super(Net, self).__init__()
        # the input size：districts 34 *years 4 + 5(extra features after onehotilized)
        self.l1 = nn.Linear(34*4*component, layers[0]) #
        self.l2 = nn.Linear(layers[0], layers[1]) # layers[0]:l1's input size
        self.l3 = nn.Linear(layers[1], layers[2]) # layers[1]:l2's input size
        self.l4 = nn.Linear(layers[2], type) # layer[2]:l3's input size
        #self.dropout = nn.Dropout(p=0.15)

    def forward(self, x):
        x = F.relu(self.l1(x))
        #x = self.dropout(x)
        x = F.relu(self.l2(x))
        #x = self.dropout(x)
        x = F.relu(self.l3(x))
        #x = self.dropout(x)
        # activation function :softmax,here we use log_softmax which'll match the NLLLoss function，combine them we get the same effect as 
        # softmax+crossentropy
        return F.log_softmax(self.l4(x), dim=1)

    # weight initialization
    def _initialize_weights(self):
        # print(self.modules())
        for m in self.modules():
            #print(m)
            if isinstance(m, nn.Linear):
                # print(m.weight.data.type())
                # input()
                # m.weight.data.fill_(1.0)
                torch.nn.init.xavier_uniform_(m.weight, gain=1)
                #print(m.weight)
class CNN(nn.Module):
    def __init__(self,type):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential( #input shape (1,34,4)
            nn.Conv2d(in_channels=1, 
                      out_channels=1, 
                     kernel_size=2, #filter size
                     stride=1, #filter step
                     padding=0 
                     ), 
            nn.Dropout(0.5), 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2) #2x2采样，output shape (16,14,14)
              
        )
        self.conv2 = nn.Sequential(nn.Conv2d(1, 1, 2, 1, 0), #output shape (32,7,7)
                                  nn.Dropout(0.5), 
                                  nn.ReLU(),
                                  nn.MaxPool2d(2))
        self.l3 = nn.Linear(1*32*2,100) 
        self.l4 = nn.Linear(100,type)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)#flat (batch_size, 32*7*7)
        x = self.l3(x)
        x = self.l4(x)
        return F.log_softmax(self.l4(x), dim=1)
    
    def _initialize_weights(self):
    for m in self.modules():
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight, gain=1)
