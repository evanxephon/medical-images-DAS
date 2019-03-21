import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init


class Net(nn.Module):
    #layers is array that contain 3 element，they are l1，l2，l3's input size，l4's output size is 5 (5 types)
    def __init__(self,layers=[],classnum=5,component=1,batchnorm=False,dropout=False):
        super(Net, self).__init__()
        # the input size districts 34 *years 4 + 5(extra features after onehotilized)
        
        self.relevance_score_output_layer = None
        self.tensor_of_each_layer = []
        
        layers.append(classnum)
        self.layers = layers
        
        if dropout:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = lambda x: x 

        inputd = 34 * 4
        self.fc = []
        self.bn = []

        for i in range(len(layers)):
            outputd = layers[i]
            fcl = nn.Linear(inputd,outputd)

            setattr(self,f'fc{i}',fcl)
            self.fc.append(fcl)
            if batchnorm:
                bnl = nn.BatchNorm1d(outputd,momentum=batchnorm,track_running_stats=True)
            else:
                bnl = lambda x: x
            setattr(self,f'bn{i}',bnl)
            self.bn.append(bnl)
            inputd = outputd

    def forward(self, x, test=False):
        
        for i in range(len(self.layers)):
            if test:
                self.tensorofeachlayer.append(x.view(-1).numpy())
            x = self.fc[i](x)
            x = self.bn[i](x)
            x = self.dropout(x)
            x = F.relu(x)
        
        # restore the relevance score of the output layer where we test
        self.relevance_score_output_layer = F.relu(x)
        
        # activation function :softmax,here we use log_softmax which'll match the NLLLoss function, combine them we get the same effect as softmax+crossentropy
        return F.log_softmax(x, dim=1)
    
    
    def relprop(self):
        
        relevance_score_of_each_layer = {}
        relevance_score = self.relevance_score_output_layer
        
        # add the output layer's relevance score to the list
        relevance_score_of_each_layer['output-layer-relevance-score'] = self.relevance_score_output_layer
       
        # get each layer's parameter in reverse order
        parameters_reverse = []
        for name,param in model.named_parameters():
            if 'fc' in name and 'weight' in name:
                paremeters_reverse = param.append(parameters_reverse)
        
        # caculate each layer's revelance score , 
        # assuming the input layer dimension is i and output layer dimension is j
        for i in range(len(self.labels)):
            
            # get positive weight
            positive_weight = F.relu(parameters_reverse[i])
            
            # after this, we get a j-dim-column-vector, adding the 1e-9 to keep the precision    i-dim-column-vector  *  j*i-matrix
            # the tensor in the "self.tensor_of_each_layer" is reversed, so we should use the index backwards
            sum_posi_weights = np.dot(parameters_reverse[i], self.tensor_of_each_layer[-(i+1)]) + 1e-9
            
            # this is a numpy element-wise operation, we'll get a j-dim-column-vector            j-dim-column-vector / j-dim-column-vector
            s_coeffecient = relevance_score / sum_posi_weights
            
            # we'll get a i-dim-column-vector                                                    j-dim-column-vector  *   j*i-matrix
            c_coeffecient = np.dot(positive_weight.T, s_coeffecient)
            
            # we get the previous layer's relevance score by using a numpy element-wise operation again   i-dim-v  *   i-dim-v 
            relevance_score = self.tensor_of_each_layer[-(i+1)] * c_coeffecient
            
            relevance_score_of_each_layer[f'l{len(self.labels) - i}-layer-relevance-score'] = relevance_score
            
        return revelance_score_of_each_layer
        
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
        super(CNN, self,).__init__()
        self.conv1 = nn.Sequential( #input shape (1,34,4)
            nn.Conv2d(in_channels=1,
                      out_channels=1,
                      kernel_size=2, #filter size
                      stride=1, #filter step
                      padding=0
                      ),
            #nn.Dropout(0.5), 
            nn.BatchNorm2d(1, momentum=0.5),
            nn.ReLU(),
            #nn.MaxPool2d(kernel_size=2)

        )
        '''self.conv2 = nn.Sequential(nn.Conv2d(1, 1, 2, 1, 0),
                                  #nn.Dropout(0.3), 
                                  nn.ReLU(),
                                  #nn.MaxPool2d(2)
        )'''
        self.l3 = nn.Linear(1*33*3,100)
        self.bn2 = nn.BatchNorm1d(100, momentum=0.5)

        self.l4 = nn.Linear(100,type)
        self.bn3 = nn.BatchNorm1d(type, momentum=0.5)
        
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

        x = self.l4(x)
        x = self.bn3(x)
        
        return F.log_softmax(x, dim=1)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
               torch.nn.init.xavier_uniform_(m.weight, gain=1)

