import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init


class Net(nn.Module):
    #layers is array that contain 3 element，they are l1，l2，l3's input size，l4's output size is 5 (5 types)
    def __init__(self,layers,type=2,component=1):
        super(Net, self).__init__()
        # the input size：districts 34 *years 4 + 5(extra features after onehotilized)

        self.bn_input = nn.BatchNorm1d(136, momentum=0.5)
        self.l1 = nn.Linear(34*4*component, layers[0]) #
        
        self.bn1 = nn.BatchNorm1d(100, momentum=0.5) 
        self.l2 = nn.Linear(layers[0], layers[1]) # layers[0]:l1's input size
        
        self.bn2 = nn.BatchNorm1d(100, momentum=0.5) 
        self.l3 = nn.Linear(layers[1], layers[2]) # layers[1]:l2's input size
        
        self.bn3 = nn.BatchNorm1d(5, momentum=0.5) 
        self.l4 = nn.Linear(layers[2], type) # layer[2]:l3's input size
         
        #self.dropout = nn.Dropout(p=0.1)
        

    def forward(self, x):
        x = self.bn_input(x)
        x = F.relu(self.l1(x))
        #x = self.dropout(x)
          
        x = self.bn1(x)
        x = F.relu(self.l2(x))
        #x = self.dropout(x)
        
        x = self.bn2(x)
        x = F.relu(self.l3(x))
        #x = self.dropout(x)

        # activation function :softmax,here we use log_softmax which'll match the NLLLoss function，combine them we get the same effect as 
        # softmax+crossentropy
        x = self.l4(x)
        x = self.bn3(x)
        return F.log_softmax(x, dim=1)

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
