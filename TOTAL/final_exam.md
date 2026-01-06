奔放的试卷
总的来说考察的是非常全面广泛的，基本上每一个模型，每一个方法都考察到了（考的很细，不是突击复习可以做到的）（每周建议花大概一天时间来复习，考前至少准备5天地毯式复习）。
细节部分是工程上的（数据处理和正则化，归一化单独复习），对模型的解读理解倒是不多（大部分是主观问答题）（李沐的动手深度学习有大量工程上的实现，有助于提高对模型内部矩阵计算的理解）。
基本没有计算题，个人感觉真的是为了考试而考试。
单选题（40：20*2）
Dropout 如何克服overfitting （考了三个Dropout 相关的，特点，区别，影响，与batch norm的关系）
各种 norm的使用场景考察。
数据预处理的考察（归一化对于training set 和test set的操作）。
这些处理操作对模型训练的影响。
cnn相关层级的使用与定义。
VAE的特殊使用场景（每一个模型的应用场景要尽可能的熟悉）。
RNN的矩阵形状（判断隐藏层的矩阵形状）（好好看课件理解原理）。
CNN的层级形状（给数据计算下一层）
反向传播的定义，还有一个场景题。
VAE？
其他部分不是没考，是不记得题目了！！！

问答题：（课件上的全部公式，全部模型都要手动带着值反向传播）
21.（18）可以手算（唯一的计算题）
1.前向传播计算。
2.带正则化的Loss计算。
3.反向传播（传播到前一层）（注意L1，L2的求导公式）

22.CNN层级描述（名称列举），降低计算开销要选什么层+原因。

2？RNN部分，RNN的基于时间步的反向传播的描述，LSTM优势描述，GRN的门是什么，注意力机制如何解决fix-seq问题的（课件上有几个问题，long history之类的，定义的很死，但是要记忆。）

2？.Adversral example如何生成，defense如何实现的，结构上的特点？

25.
1.GAN两个部分如何相互影响，相互工作？
2.GAN的实现描述，包含LOSS默写，求导公式默写（两个）（个人认为极度无聊没有任何意义，因为根本没有用上）
3.AE的结构描述，对比PCA的最大优势在什么地方。
4.GAN 和 VAE 在生成图片质量上的比较，还有一个training stability上的比较。


复习课内容
- History:single larger perception (不同模型提出的时间列表)
- BP is very important（要带计算器）（chain rule：可能要复习一下高数的内容？）
- Loss （MSE），权重计算
- 优化方法：动量/ADAGard /RMSprop（对比他们的不同）
- 正则化方法：
  - augmentation（数据争强）
  - L1/L2正则化的使用
  - Dropout（how to perform）（training  ，testing）
CNN
  - Fundamental
    - motivation--（比较与mlp的优势在什么地方）（weight sharing sparsity）（三个）
    - 发展历史中的几个模型，一直到Res，DES （考察层级结构）（涉及归一化，输入，特征，batch norms）
RNN
- 动机
- Vanle RNN ---缺点，原因 --- LSTM（改进）
- LSTM 3个门（I F O）（门的权重和对应的更新公式）（如何解决上面的模型的问题的）
- 应用。
Auto Encode（Manfold learning）
- PCA/AE（vanats）（三种对应的AE）（回答PCA和AE直接的不同）
Imagine generation
- 回答VAE对比AE的不同（V指的是什么，loss是什么，结构）
- GAN（对比GAN和VAE）（两个部分，Generator，Pisdrimation）（how to trian GAN，step by step ，描述的时候要加上loss）
Adversral example
- 如何生成 adv example
- 不同的methods（FGSM）（T。。）（Least liekly class）
- defence
  - （robust training ）（Data 。。）
  - Adv detection 
  - Design highly-nonliner
GCN
- 如何设计的---layer（给一个图，让我们设计一个Layer）
Attention
- 动机
- 类别：普通的，self的，有mask的（目的是什么，为什么要使用mask）
- 使用的好处是什么
