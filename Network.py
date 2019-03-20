import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init


class Net(nn.Module):
    #layers is array that contain 3 element,they are l1,l2,l3's input size,l4's output size is 5 (5 types)
    def __init__(self,layers=[],classnum=5,component=1,batchnorm=False,dropout=False):
        super(Net, self).__init__()
        # the input size districts 34 *years 4 + 5(extra features after onehotilized)

        layers.append(classnum)
        self.layers = layers
 
        if dropout:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = lambda x: x 

        inputd = 136
        self.fc = []
        self.bn = []

        for i in range(len(layers)):
            outputd = layers[i]
            self.fcl = nn.Linear(inputd,outputd)
            self.fc.append(self.fcl)
            if batchnorm:
                self.bnl = nn.BatchNorm1d(outputd,momentum=batchnorm,track_running_stats=True)
            else:
                exec(self.bnl = lambda x: x)
            self.bn.append(bnl)
            inputd = outputd
            
    def forward(self,x):

        for i in range(len(self.layers)):
            x = self.fc[i](x)
            x = self.bn[i](x)
            x = self.dropout(x)
            if i != len(self.layers)-1:
                x = F.relu(x)

        # activation function :softmax,here we use log_softmax which'll match the NLLLoss function, combine them we get the same effect as softmax+crossentropy
        return F.log_softmax(x)

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
    def __init__(self,type,batchnorm=False,dropout=False):
        super(CNN, self,).__init__()
        self.conv1 = nn.Sequential( #input shape (1,34,4)
            nn.Conv2d(in_channels=1,
                      out_channels=1,
                      kernel_size=2, #filter size
                      stride=1, #filter step
                      padding=0
                      ),
            #nn.Dropout(0.5), 
            nn.BatchNorm2d(1, momentum=0.1),
            nn.ReLU(),
            #nn.MaxPool2d(kernel_size=2)

        )
        '''self.conv2 = nn.Sequential(nn.Conv2d(1, 1, 2, 1, 0),
                                  #nn.Dropout(0.3), 
                                  nn.ReLU(),
                                  #nn.MaxPool2d(2)
        )'''
        if batchnorm:
            self.bn2 = nn.BatchNorm1d(100, momentum=batchnorm)
            self.bn3 = nn.BatchNorm1d(type, momentum=batchnorm)
        else:
            self.bn2 = lambda x: x
            self.bn3 = lambda x: x
        if dropout:
            self.dropout = nn.dropout(dropout)
        else:
            self.dropout = lambda x: x
             
        self.l3 = nn.Linear(1*33*3,100)

        self.l4 = nn.Linear(100,type)
        
    def forward(self, x):
        #print(f'shapebefore:{x.shape}')

        x = x.view(-1,1,34,4) # the dimension is related to the convolution channel
        #print(f'shapeafter:{x.shape}') 
        x = self.conv1(x)

        #x = self.conv2(x)
        #print(f'shapeafterconv:{x.shape}')

        x = x.view(-1,99)
        #print(f'shapeafterview:{x.shape}')
        #x = x.view(x.size(0), -1) # flat

        x = self.l3(x)
        x = self.bn2(x)
        x = self.dropout(x)
        x = F.relu(x)

        x = self.l4(x)
        x = self.bn3(x)
        x = self.dropout(x)
        
        return F.log_softmax(x, dim=1)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
               torch.nn.init.xavier_uniform_(m.weight, gain=1)

