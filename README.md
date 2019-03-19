# Data Augmentation Strategy in Medical Field
Implementation of some of strategies on data augmentation in medical field

## Background
医疗领域使用深度学习的一个问题是数据量太少。图像数据的数据增强已有很实用的策略并被广泛使用，但是非图像数据的数据增强还没有。而我们现在有的数据比较特殊，它不是图像数据，但是数据之间仍存在位置上的关系。我们为这类数据设计了几种数据增强策略，并通过实验证明它们是否有效。  
  
The lack of data is an essential problem in the field of medical analysis. While the technique of data augmentation has already been wildly applied into image processing, there is few instances of non-image data augmention application. What we deal within this project is the data of some residual scores of raw MRI image. Every single entry of the data is a sort of measurement of a certain region of brain so that the data itself remains the relationship between different lobes of brain. To recognise the hidden pattern of the data by trainning our classical DNN as a classifier, we proposed several different strategies of data augmentation so we can have sufficient trainning data.

## The Data 
数据是人脑MRI扫描结果，一共有34个扫描的区域，被扫描后每个区域会给出一个数值，每个患者也就是一个记录给出TA四年的数据，也就是一共有34 * 4 个features，数据的label为一个分类结果，一共有五种，在这里就标为ABCDE。34个扫描区域还被分为6个lobes，每个lobes包括不同的区域。还有三个额外的feature包括患者性别，设备品牌等。  

The experimental data is the residual scores of the raw MRI image which mainly covers 34 regions of brain. Every region yields a numerical measurement as every alcoholism subject has a record for 4 years. Thus our original input data order will be at 34 * 4. We now having a data set with 505 samples and 5 labels (e.g., ctrl,d1,d2,etc.) . For additional information, these 34 region has been divided into 6 lobes based on clinical studies along with 3 other features (e.g., age,gender,scanner) of the every single records.

## Data Augmentation Strategy (DAS)

### Adherent
最初的策略，是将X个同类别的数据直接组合到一起来生成新的数据。该方法可以轻松生成大量数据，X的数量如果超过4，生成该策略所有可能数据就已经没有可能了。这个方法的问题，我认为是通过改变网络结构，我们能得到同样的效果。  

Our very first strategy is to directly adhere several data together to generate a new one with bigger order according on the parameter of multiplier λ we choose. This strategy can easily generate massive new data set, somehow the problem is that not only the generated data set is inflated too fast and the order of every new data will equal to original order * λ which will cause dimension explosion even with a fairly small λ.

### Kernel

#### Fixed Kernel Substitution
选择X * Y大小的kernel，将kernel中选定的区域替换成其他记录同样区域的数据。这种方法同样可以生成大量数据。我们原本数据为一维的长度为136的数据，在kernel替换策略中，我们需先将数据转换成4 * 34的二维数据。在这里我们在不同year之间的数据替换可能能保存疾病的一种变化趋势。  

By choosing a fixed kernel with X * Y order for all original data cross different groups, we then substitute the entry of the kernel with other records from the subject of same group label. Repeat that process till every data has been covered. Apparently, we can deduce that the size of generated data set will be linear by X, Y： N = (L-X+1) * (W-Y+1), where L=34, W=4 in our case.

#### Fixed Kernel Substitution for Different Lobes
脑科学中将大脑一定区域归为一类，这每一类就称为一个lobe。因此为每一个lobe采用不同的kernel来生成数据显得非常合理，因为在同lobe的区域可能有更多的联系，替换生成的数据更接近该类别。  

According to neuroscience knowledge, all the MRI image of human brain can be divided into several lobes which includes variant regions. Applying this theory, we can rearrange and divide our original residual scores matrix (which is our data format) into different areas representing different lobes. In this case, we will then choose different order of kernel between different lobes and substituting them with the entries from other subject's measurements from same group label.  All other strategy are also developed inspired by this logic. 

#### Dynamic Kernels Substitution for Different Lobes
对应每一种类别的数据生成，采用不同的kernels。这样做的原因主要是为了解决数据的类别不均匀的问题，数据不均衡会导致训练出来的分类器是naive的--预测结果总是为比例最高的类别即可保证训练集的loss很低，同时如果采用是同样分布的测试集，该分类器的准确率Accuracy也会很高，但是一些类别的Precision和Recall会为0，这样的分类器显然不是我们想要的。其解决方式之一即使提供类别均衡的训练数据，而我们在数据增强的过程中正好可以做到这点。

This strategy requires dynamic kernels between different lobes. Based on the algorithm, the benefit will bring us a more balanced generated data set which could be very crucial to train a non-naive classifier.

#### Kernel Accumlation
将指定区域的数据加上其他记录同样区域的数据。这样做的理论基础我没有很理解，但是不妨碍我们多做几次尝试。

We proposed this strategy by accumlating the value of data of different subjects within same group label through the kernels. The reason is that if the difference between the implicit pattern and noise is monotonic, then the accumlation move will enhance the trend of pattern so the recognition and classification will be much easier.  

## Code
机器学习框架主要采用了Pytorch。目前的代码都为重构过的版本，虽然仍然存在很多软件开发中常见的问题，但比起之前已经好用很多了。  

We're using Pytorch as our main package. This part is basically self-explainable by the name of the file itself.

### Preprocess.py
对数据做最初的预处理，原始数据为.mat格式，读取后，经过一些操作返回.csv文件。

### Augmentation.py
实现了所有的数据增强策略，在前面说到的基础上，还实现了指定生成数量的实现版本，因为一些策略能产生的数据的量会超出我们的计算能力，因此生成一部分数据进行训练。

### Dataset.py
将生成的数据和测试数据组合好，得到一个可以被网络调用的DataLoader实例。

### Network.py
一个简单的多层全连接的神经网络的实现。我们可以手动调整各种超参数。
又新加入卷积神经网络的版本。虽然卷积神经网络多用于图像。但是我认为卷积操作能提取脑部不同区域在年份之间的变化，需实验验证。
Update: 加入卷积后并未得到更好的准确率，同时对非图像数据使用CNN结构不合理。

### Train.py
这是我们进行训练的主程序，我们可以在训练时调整各种超参数和设置。

### RandomForest.py
使用随机森林对原始数据进行了训练，因为随机森林是一种十分适合数据量很少的情况下的机器学习算法。使用随机森林，我们能确保自己没有跑偏，得到连训练原始数据都不如的结果。

We are applying Random Forest algorithm to our original data to make sure that augmentation strategy would not bring us an even worse accuracy than the uninflated set.

### Modules.py & Utils.py
对现有网络实现深度泰勒分解和相关度指数的两个基础库程序，以获得对输入矩阵中单个元素对最终预测结果的影响程度大小，进而实现对分类器的解释。  

The fundamental function library of generating a heatmap of input matrix (which could be an interpretation of our prediciton), applying the algorithms of deep taylor decomposition or layer-wise relevance propagation. 

## Challenges and Solutions So Far
### Imbalanced Data
不平衡的数据集可能导致得到的分类器是Naive的，即预测结果始终为占比最多的类别，这确实是训练误差最小的分类器，同时如果测试集保持同样的分布，那么在该数据集上的准确率也会很高。但是对于不平衡的数据集，Accuracy准确率是一种非常片面的衡量标准，应该使用各个类别的Precision精准率和Recall召回率来衡量。
目前关于不平衡数据的解决方法有很多种观点，可以通过上采样，下采样或生成数据的方式来改变数据的分布，也可以对损失函数加权等等。改变数据分布的方式是有弊端的，丧失训练集中关于数据的分布的信息。
目前关于训练集不平衡的问题，我们目前采用的方式仍然是改变数据的分布，已我们的数据增强策略为主，上下采样为辅来得到平衡的数据集。对于测试集，我们特意选择了平衡的数据，为的是希望仅使用Accuracy准确率就能衡量分类器的优劣。  

A naive classifier with a certain high overall accuracy can easily be deducted/trained by an imbalanced data set. Other than using the benchmark of Precision, Recall and F1-Measure to evaluate our model, we can also apply methods like up-sampling, sub-sampling or adding weights to loss function etc. to change the distribution of the data trending to more balance way. Somehow we may risk losing some critical information among the original distribution of the data.

We will split our original data to make the test data set to be well balanced, the rest part will be involved into the DAS.

### Insufficient Test Data
我们可以生成足够多的数据来训练，并通过观察分类器在测试集之上的效果来判断我们的策略是否有效。但是，存在一个问题：测试集的数据数量很少。我们将原始数据分成训练集的种子（用来生成训练集）和测试集。测试集的数量只有100个，与此同时我们用来训练的数据经过生成后会达到百万级别。仅仅通过在100个数据的表现，来证明我们策略有效，其中存在很大的随机性，是非常没有说服力。同时我们是不能对测试数据进行生成的操作，不然就有一丝‘循环论证’的意味了——先假定我们的策略就能生成完美的真实数据，然后再去证明它。  
解决测试数据少的问题，我们可以采取交叉验证。有多种交叉验证的方法，如K折交叉验证，留一交叉验证和留P交叉验证等等。它们主要的思想就是使用多次不同划分的训练和测试集来训练。验证集一般是用来评估模型预测的好坏和选择模型及其对应的参数。但是在我们的情况，我认为采用交叉验证的方法可以减少我们测试集过少带来的随机性的影响。 
因为算力的原因，只能采用K折交叉验证，但是目前并不急着去实现代码和训练，因为需要先得到高的准确率，才需要考虑我们的结果的说服力。
目前已经在使用交叉验证来充分利用数据，让每一个数据都可以被用到训练之中。  

We use K-fold cross validation to avoid the overfit of the trainning process and also to increase the usage of the original data which is a rather small set.

### Overfitting
截至目前，我得到的五分类器的最好准确率为30%，但同时分类器在训练集之上的准确率能达到80%。这明显是有过拟合的问题。
应对过拟合最根本的解决方法即加大训练数据的量。但是在我们在做的已经是生成训练数据了，在达到生成数据的数量瓶颈后。现在能做的是改变网络结构或者在之上应用一些常见的减轻过拟合的Tricks。
#### Regularization
L1正则化被证明不适合，本身即是用来得到稀疏的权值的一种做法。我们使用后发现会大大影响在训练集上的准确率。  
L2正则化有一定效果，虽然没有得到超过30%的准确率，但是使用后确实提高了分类器在测试集之上的准确率。
Update: 仅使用L2正则化的效果并不好，需要结合batchnorm来一起使用。
当然也可能是我们没能找到好的比例，比例过高，模型将不会从数据中学习到东西，而比例过低，正则化将没有效果。
#### Dropout
Dropout的思想是在每次反向传播时，随机屏蔽一定比例的权值，也就是不做计算和更新权值。这样做的有助于提高模型泛化性，在CNN中被广泛运用。  
实践后发现有一定效果，但是对训练集的准确率影响很大，关于屏蔽比例这个超参数还需要多训练试试。
#### Batch Normalization
批标准化被用来解决梯度消失的问题，可以加快训练速度，对解决过拟合现象也有帮助。  
批标准化即在每一层输入进入非线性的激活函数前，做一个标准化的操作，将输入向量变成均值为0，方差为1。变换的具体操作是：某个神经元对应的原始的激活x通过减去mini-Batch内m个实例获得的m个激活x求得的均值E(x)并除以求得的方差Var(x)来进行转换。经过变换后，为防止网络的表达能力下降，对变换后的输出做如下的scale和shift操作：y = ax+b,参数a和b同为需要训练的参数。
我在网络中加入BatchNorm后，准确率有所上升，并且训练速度有所上升，但并未突破30%。
### Training
训练一直得不到理想的准确率，各种调参，我们的最好结果只是维持在25%，有时会波动到30%左右，但是对于100个数据的测试集来说，这样的波动是很常见的。所以应该还是按照收敛后的稳定准确率来算，甚至说这样也不够有说服力，我们需要取不同训练数据和测试数据划分的分类器的平均稳定准确率。
## Optimization
### Optimization on Training Process
#### Classical Tricks
训练速度的优化其实并不需要我们去操心太多，有很多普适性的优化Trick都已经被设置成默认的，想要使用只需要添加几行代码就可以了。
像是我们的训练使用的自适应学习率策略是加上momentum动量，结合随机梯度下降的方式，在code中我只需要一句话就能设置好。  
又比如参数初始化，我使用的著名的KaiMingHe-weight-initialize, 何恺明用论文证明了，他那样的初始参数能让训练收敛的更快。我没有去关注论文，但是依然用了他的方法。
BatchNorm也是被用来提高训练速度的。但是和Dropout的相性不好，不能同时使用。
#### Tuning the Hyperparameter
深度学习被一些人戏称为炼丹，就是因为这超参数调优。我们的网络的超参数有这么些：隐藏层数目及输入输出向量维度，学习率，自适应学习率对应的动量比例，L2正则的lambda值，Dropout的比例，BatchNorm的动量值等等。索性对我们来说，重点并不是要调出完美的超参数配置而是适当调整得到一个还不错的准确率即可。毕竟我们的重点是证明生成策略有效并解释我们的数据。
### Optimization on Code
#### Time Complexity and Data Generating Speed
最初的生成策略实现函数每生成新的一行数据，就将它加到我们最终输出的表后面，在实现中，我的代码也采用了这个逻辑。但是代码跑起来很慢，最初并没有觉得有什么问题，毕竟也是百万量级的数据生成，函数的时间复杂度也很高，包括了多个循环，这样的耗时也可能是合理的。  
但是我们在一次生成数据的过程中，发现了函数的每一个循环的耗时在不断的增大，按照我实现的逻辑，每个循环之间并不应该有耗时上大的区别。检查后发现，就是这个把新数据加到表后面的操作实现的方式有问题，每加上一些数据，表就变得越长，将一行数据加到表的后方的操作就要耗费越长的时间，因为这个操作是要将表本身读取到内存之中的，当表的长度很长的时候，生成新数据的耗时已经算不了什么了，绝大数的时间都被用来读取表了，所以我们并不应该直接来这样操作，而是要每一个大的循环新建一个表作为中继，把新的数据都先加到它的后面，然后在一个大的循环结束后，将这个中继中的数据一次性的加入到我们最终输出的表之中，这样我们就减少了大量的读取一个长表的操作。百万数据的生成时间也从原来的几天变成几小时即可。
#### Code Refactor
虽然我们的代码数量不是很多，但是在后期，我依然体会到了代码缺少组织结构所带来的种种不便。作为一名软件工程专业的学生，写出简洁，易复用，易扩展的代码确实也是应该有的觉悟。将重构后的代码分成独立的模块，每个模块之间的依赖减少，有助于之后的开发之中加入新的东西。在软件开发课程中所学到的东西，果然还是要自己动手做一遍才知道怎么做和为什么这么做。
## Validation
### Cross Validation
为了减少认为划分数据带来的随机性，进行交叉验证，同时因为算力的限制，不使用留一验证，K折验证等等方式，而使用不完全的留N验证，即划分N和X-N的组合，只取一部分来进行分类器的训练。做交叉验证的一个好处是让本来就不多的数据得到充分的利用，我们能将更大比例的数据用于训练集的生成。
# To be continued




