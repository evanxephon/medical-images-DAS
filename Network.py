import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init
import numpy as np


class Net(nn.Module):
    #layers is array that contain 3 element，they are l1，l2，l3's input size，l4's output size is 5 (5 types)
    def __init__(self,layers=[],classnum=5,component=1,batchnorm=False,dropout=False,cnn=False):
        super(Net, self).__init__()
        # the input size districts 34 *years 4 + 5(extra features after onehotilized)
        
        self.relevance_score_output_layer = None
        self.tensor_of_each_layer = []
        
        layers.append(classnum)
        self.layers = layers

        self.cnn = cnn
        
        if dropout:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = lambda x: x 
          
        inputd = 34 * 4

        self.cv = []
        self.fc = []
        self.bn = []
        
        # CNN
        if cnn:
            for i in range(len(cnn)):
                conv = nn.Sequential( #input shape (1,34,4)
                                           nn.Conv2d(
                                           in_channels=cnn[i][0],
                                           out_channels=cnn[i][1],
                                           kernel_size=cnn[i][2], #filter size
                                           stride=cnn[i][3], #filter step
                                           padding=cnn[i][4]
                                           ),
           
                                           #nn.Dropout(dropout),  
                                           nn.BatchNorm2d(1, momentum=batchnorm),
                                           nn.ReLU(),
                                           #nn.MaxPool2d(kernel_size=2)
                                           )
                setattr(self,f'conv{i}',conv)
                self.cv.append(conv)
            inputd = int((34 - cnn[i][2] + 1 + 2 * cnn[i][4]) * (4 - cnn[i][2] + 1 + 2 * cnn[i][4]) / (cnn[i][3]))

        self.dimafterconv = inputd
        
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

    def forward(self, x):
        
        if self.cnn:
            x = x.view(-1,self.cnn[0][0],34,4)
            for i in range(len(self.cv)):
                x = self.cv[i](x)
            x = x.view(-1,self.dimafterconv)
            
        for i in range(len(self.layers)):
            if x.shape[0] == 1:
                self.tensor_of_each_layer.append(x.view(-1).cpu().detach().numpy())
            x = self.fc[i](x)
            x = self.bn[i](x)
            x = self.dropout(x)
            x = F.relu(x)
        
        # restore the relevance score of the output layer where we test
        if x.shape[0] == 1:
            self.relevance_score_output_layer = F.relu(x).view(-1).cpu().detach().numpy()
        
        # activation function :softmax,here we use log_softmax which'll match the NLLLoss function, combine them we get the same effect as softmax+crossentropy
        return F.log_softmax(x, dim=1)
    
    
    def relprop(self):
        
        relevance_score_of_each_layer = {}
        relevance_score = self.relevance_score_output_layer
        self.tensor_of_each_layer.reverse()
        
        
        # add the output layer's relevance score to the list
        relevance_score_of_each_layer['output-layer-relevance-score'] = self.relevance_score_output_layer
       
        # get each layer's parameter in reverse order
        parameters = []
   
        for name, param in self.named_parameters():
            if 'fc' in name and 'weight' in name:
                parameters.append(param.data.cpu().detach().numpy())
#         print(parameters)
        parameters.reverse()
#         print(len(parameters))
#         print(len(self.tensor_of_each_layer))
        
        # caculate each layer's relevance score , 
        # assuming the input layer dimension is i and output layer dimension is j
        for i in range(len(self.layers)):
            
            # get positive weight
            positive_weight = np.abs(parameters[i])
            
            # after this, we get a j-dim-column-vector, adding the 1e-9 to keep the precision    i-dim-column-vector  *  j*i-matrix
            sum_posi_weights = np.dot(positive_weight[i], self.tensor_of_each_layer[i]) + 1e-9
            
            # this is a numpy element-wise operation, we'll get a j-dim-column-vector            j-dim-column-vector / j-dim-column-vector
            s_coeffecient = relevance_score / sum_posi_weights
            
            # we'll get a i-dim-column-vector                                                    j-dim-column-vector  *   j*i-matrix
            c_coeffecient = np.dot(positive_weight.T, s_coeffecient)
            
            # we get the previous layer's relevance score by using a numpy element-wise operation again   i-dim-v  *   i-dim-v 
            relevance_score = self.tensor_of_each_layer[i] * c_coeffecient
            
            relevance_score_of_each_layer[f'l{len(self.layers) - i}-layer-relevance-score'] = relevance_score
            
        return relevance_score_of_each_layer
        
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
