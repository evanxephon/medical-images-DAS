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
        
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                wspace=None, hspace=1)
        
        plt.figure(figsize=(34, 1),dpi=120)

        lobemap = plt.subplot(3, 1, 1)
        lobemap.set_title('6 lobes')
        lobemap.set_xticks(range(34))
        lobemap.set_yticks(range(1))
        lobemap.set_ylabel('Lobes')

        lobemapdata = np.array(list([1 for _ in range(9)]) + list([2 for _ in range(11)]) + list([3 for _ in range(4)]) + list([4 for _ in range(5)]) + list([5 for _ in range(4)]) + [6])
        
        lobemapdata = lobemapdata.reshape([1, 34])
        lobemap.imshow(lobemapdata, cmap=plt.cm.gray)
        
        # lobe map
        order = [2,6,7,9,15,16,30,33,34,4,12,14,17,18,19,20,24,27,28,32,5,11,13,21,8,22,25,29,31,3,10,23,26,35]
        order = [ x-1 for x in order]
        
        # plot the input data's heatmap
        
        plt.figure(figsize=(34, 4),dpi=120)
        inputheatmap = plt.subplot(3,1,2)    
        inputheatmap.set_title(f'heatmap of raw data, label:{label}')
        inputheatmap.set_xlabel('Districts')
        inputheatmap.set_ylabel('Years')
        inputheatmap.set_xticks(range(34))
        inputheatmap.set_yticks(range(4))
        
        imshow(inputdata,cmap=plt.cm.gray)
        
        # plot the input data's relevance score's heatmap
        
        plt.figure(figsize=(34, 4),dpi=120)
        relevancescoreheatmap = plt.subplot(3,1,3)
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
