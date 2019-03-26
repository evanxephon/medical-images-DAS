import numpy as np

# z+ rule's implementation
def rplus(layers, tensor_of_each_layer, current_relevance_score, parameters, relevance_score_of_each_layer):

# caculate each layer's relevance score , 
# assuming the input layer dimension is i and output layer dimension is j

    for i in range(len(layers)):

        # get positive weight
        positive_weight = np.abs(parameters[i])

        # after this, we get a j-dim-column-vector, adding the 1e-9 to keep the precision    i-dim-column-vector  *  j*i-matrix
        sum_posi_weights = np.dot(positive_weight[i], tensor_of_each_layer[i]) + 1e-9

        # this is a numpy element-wise operation, we'll get a j-dim-column-vector            j-dim-column-vector / j-dim-column-vector
        s_coeffecient = relevance_score / sum_posi_weights

        # we'll get a i-dim-column-vector                                                    j-dim-column-vector  *   j*i-matrix
        c_coeffecient = np.dot(positive_weight.T, s_coeffecient)

        # we get the previous layer's relevance score by using a numpy element-wise operation again   i-dim-v  *   i-dim-v 
        relevance_score = tensor_of_each_layer[i] * c_coeffecient

        relevance_score_of_each_layer[f'l{len(layers) - i}-layer-relevance-score'] = relevance_score
            
    return relevance_score_of_each_layer
