# Lecture 2：Word Vectors 2 and Word Senses
### Global Vectors for Word Representation
LSA、HAL等方法能有效地利用了全局的统计特征，主要捕获词语相似的地方。但是在推理、次优向量空间表现不太理想。 而shallow
window-based的方法，如skip-gram、CBOW，利用了固定窗口的信息，学习word
vector。在词语相似度方面表现不错，但缺少利用全局词语共现的统计信息。
GloVe由一个加权最小二乘模型组成，该模型训练全局词与词的共现计数，从而有效地利用统计数据。![ff0ca08a9ade5bb1834b9f698f61d6dd](picture/Lecture2/B01FDC2A-BCA4-4EF4-A8E9-6EAEB82507D6.png)
最小二乘的目标：
![d666edc558330078f041783e59d7d8ff](picture/Lecture2/0A48FE0E-0E82-4AA9-AEA0-5162569A8F45.png)
损失函数为：
![fe07ca150182bb1c0c5faf3eaad62982](picture/Lecture2/371DD99A-4F9E-4525-91BF-8B1E82903BBA.png)
放宽到整个预料集：
![b834cafc228bad818044786d3b1dc7eb](picture/Lecture2/7F3D627A-3FD6-4F12-82C2-A7F7B79E1799.png)
上述公式中的Xij表现词i和词j的共现次数，而对于公式中的Q是需要标准化的，但是这样在真个预料集中花费大量的求和计算。
因此使用最小二乘的方法来替换：![c2d7564e9cc77980f9c53f62344a769d](picture/Lecture2/C5994C2E-AF0C-4A81-999C-57BADF30B31C.png)
但Xij通常是一个比较大的值，这使得优化变得十分困难。 因此对P和Q进行log转换：
![f9375b0c88d2269a693a4048c7bd8696](picture/Lecture2/7442489D-EE0F-4E22-9A7D-7974EC931516.png)
![f43d34348baf4c90beb1b24962ccfc24](picture/Lecture2/B432A763-ABE9-4C60-B313-8D9C59F770F8.png)
另外，权重Xi不能保证一定是最佳的参数。 因此引入了更加通用、且可以自由选择所依赖上下文词语特征的权重函数：
![8c612a19fa0ad186e780afd07aa0e66e](picture/Lecture2/432B3696-1C32-498E-92C0-64E2D18A4A76.png)

### 怎么评估词向量的质量？
#### 内在方法：
* 应用在一个特殊/中间，子任务进行评估
* 模型是否足够快地训练完
* 对理解系统有帮助
* 除非与真正的任务建立联系，否则不知道是否有用

#### 外在方法：
* 在一个真实任务中进行评估
* 需要花费很长时间训练
* 不清楚是子系统的问题，还是它的交互作用，还是其他的子系统
* 如果用一个子系统替换了另外一个子系统，导致了模型acc有提高


##### 内在方法评估例子：词向量推理
使用cosine similarity的方法进行计算，例如queen-king=actress-actor。使用一下公式来计算两个单词之间的相似度。
![0348f4af1fa177b9731d48cbfb88048e](picture/Lecture2/F8ABFC46-4D05-4BAF-B3F9-6C5F5D681464.png)


推理规则：A:b::c:?
  
语义词向量=》Chicago : Illinois : : Houston : Texas

语义词向量=》Abuja : Nigeria : : Accra : Ghana

语法词向量=》bad : worst : : big : biggest

##### 内在方法评估微调例子：类比评估
通过调节word vector embedding的超参，来看看其实怎么影响vector的质量
结论：
* 推理任务表现强依赖与word embedding的模型
* 大语料会提高模型表现
* word vector的维度越低，模型表现越差

##### 内在方法评估例子：相关性评估
人工在一个固定的单词上评估两个单词之间的相似性标度(比如0-10)，同时使用余弦相似度进行计算比较。在多个数据集上进行测试。
若人类标注的分数与cosine计算的分数相似，那么可认为模型时有效的。

##### Training for Extrinsic Tasks
一般的word embedding都是会被应用到其他下游的任务，这里我们按照这个思路来分析。

###### retraining word vector
图1是使用预训练好的word embedding做词分类时的效果，而图2是对word embedding进行retrain后的词分类效果。相对而言效果变差了。
* 如果retrain数据集太小，那word vector不应该被训练。有可能会导致原始的word vector发生改变，甚至减低原始vector的效果。
* 如果retrain数据集太大，那word vector可以再训练。
![ce4b8838e6ccd904de6580a0d7552ca4](picture/Lecture2/110DEEC9-B127-4BF7-9343-5850027B645C.png)
![f6bad56547ef17c2cffb5aa41b5917a3](picture/Lecture2/EF9BFE49-81F2-46D3-BE72-80C37C6611AB.png)


###### Window Classification

窗口小的情况下，模型在句法测试表现较好

窗口大的情况下，模型在语义测试表现较
