## 相关连接
- [相应的codelab链接](https://codelabs.developers.google.com/codelabs/tfjs-training-regression/index.html?hl=zh-cn#0)
- [神经网络结构](https://developers.google.com/machine-learning/crash-course/introduction-to-neural-networks/anatomy?hl=zh-cn)
- 分层的神经网络结构和激活函数运行机制[《神经网络、流图和拓扑》](http://colah.github.io/posts/2014-03-NN-Manifolds-Topology/)

## 密集层
全连接层中的一种，每个神经元都要接受上一层的大量输入因此得名。通常用于实现神经网络中的非线性变换和分类功能。例如在密集层中使用激活函数，提供非线性变换，提高神经网络表达能力。或者在分类模型最后一层，用于预测类别标签。

## 隐藏层
在输入层和输出层中间的层都叫隐藏层。隐藏层的设计直接影响神经网络的性能和泛化能力。常见的可设置项及其影响：
- **数量**：隐藏层数量越多，神经网络的表达能力越好，同时也越复杂越难训练；
- **宽度（神经元数量、节点数）**：神经元越多表达能力越好，同时也越复杂越难训练；
- **激活函数**：引入非线性函数可增加神经网络表达能力，适配不同的非线性问题，常见的激活函数：Sigmoid、ReLU、Tanh；
