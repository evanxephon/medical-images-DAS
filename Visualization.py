import pickle
import matplotlib.pyplot as plt

def heatmap(data):
    for relevance_score in data:
        # the other layer's relevance score   
        output_layer_relevance_score = relevance_score['score']['output-layer-relevance-score']
        l4_layer_relevance_score = relevance_score['score']['l4-layer-relevance-score']
        l3_layer_relevance_score = relevance_score['score']['l3-layer-relevance-score']
        l2_layer_relevance_score = relevance_score['score']['l2-layer-relevance-score']
        
        # the input layer's relevance score
        l1_layer_relevance_score = relevance_score['score']['l1-layer-relevance-score']
        l1_layer_relevance_score = l1_layer_relevance_score.reshape([4,34])
        
        inputdata = relevance_score['data']
        inputdata = inputdata.reshape([4,34])
        
        label = relevance_score['label']
        
        fig = plt.figure()
        plt.figure(figsize=(34,4), dpi=120)
        
        plt.subplots_adjust(left=None, bottom=0.1, right=None, top=None,
                wspace=1, hspace=1)
        
        # plot the input data's heatmap
        inputheatmap = plt.subplot(2,1,1)    
        inputheatmap.set_title(f'heatmap of raw data, label:{label}')
        inputheatmap.set_xlabel('Districts')
        inputheatmap.set_ylabel('Years')
        inputheatmap.set_xticks(range(34))
        inputheatmap.set_yticks(range(4))
        
        imshow(inputdata,cmap=plt.cm.gray)
        
        # plot the input data's relevance score's heatmap
        relevancescoreheatmap = plt.subplot(2,1,2)
        relevancescoreheatmap.set_xlabel('Districts')
        relevancescoreheatmap.set_ylabel('Years')
        relevancescoreheatmap.set_xticks(range(34))
        relevancescoreheatmap.set_yticks(range(4))
        
        relevancescoreheatmap.set_title('heatmap of input layer relevance score')
        
        imshow(l1_layer_relevance_score,cmap=plt.cm.gray)
        
if __name__ == '__main__':
    with open('relevance_scores.pk','rb') as f:
        relevance_scores = pickle.load(f)
        heatmap(relevance_scores)
