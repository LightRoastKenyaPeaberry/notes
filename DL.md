

# pytorch classification

## LeNet 

### 亮点

CNN的雏形

### 结构

![image-20230404204512671](DL.assets/image-20230404204512671.png)



## AlexNet

### 亮点

* 首次使用GPU
* 使用ReLu
* 使用LRN局部响应归一化
* 在全连接层的前两层使用了Dropout随机失活神经元，以减少过拟合。

padding=int 上下左右

​				 tuple （a,b) 上下a, 左右b

nn.ZeroPad2d((1,2,1,2))



### 结构

![image-20230404205005306](DL.assets/image-20230404205005306.png)





## VGG

### 亮点

如下图

### 结构

![image-20230404205047270](DL.assets/image-20230404205047270.png)

![image-20230404205138701](DL.assets/image-20230404205138701.png)

![image-20230404205316426](DL.assets/image-20230404205316426.png)







## GoogLeNet

### 亮点

* 引入Inception结构（融合不同尺度的特征信息）
* 使用1x1的卷积核进行降维以及映射处理
* 添加两个辅助分类器帮助训练
* 丢弃全连接层，使用平均池化层（大大减少模型参数）



### 结构

![image-20230404205636906](DL.assets/image-20230404205636906.png)

![image-20230404205814407](DL.assets/image-20230404205814407.png)

<img src="DL.assets/7da7c823b81e4a63854a2897106b3b91.jpg" alt="在这里插入图片描述" style="zoom:67%;" />



## ResNet

### 亮点

* 超深的网络结构(>1000)
* 提出residual模块
* 使用Batch Normalization加速训练（丢弃Dropout)



首先，由于1.梯度消失or梯度爆炸； 2. 退化(degradation)问题； 

卷积层和池化层并不是越多越好

解决上述两个问题：

1. 数据标准化 权重初始化  BN
2. 残差的结构

<img src="DL.assets/image-20230404195611880.png" alt="image-20230404195611880"  />

### 结构

![image-20230404200247802](DL.assets/image-20230404200247802.png)

==注意每一种不同的Block在交接时，需要虚线处理下才能用==

![image-20230419173747123](DL.assets/image-20230419173747123.png)

![image-20230404201833471](DL.assets/image-20230404201833471.png)

常用的迁移学习方式：

* 载入权重后训练所有参数
* 载入权重后之训练最后几层参数
* 载入权重后在原网络基础上再添加一层全连接层，仅训练最后一层。



### ResNeXt

![image-20230404211446369](DL.assets/image-20230404211446369.png)





## MobileNet



### 亮点

+ Depthwise Convolution(大大减少运算量和参数数量)
+ 增加超参数α（控制卷积核个数）、β（控制图像的分辨率）

传统卷积：

+ kernel channel==input channel
+ kernel number==output channel

DW卷积：

+ kernel channel ==1
+ input channel == kernel number == output channel

Depthwise Separable Conv :

DW + PW (Pointwise Conv)

PW: 普通卷积核，大小为1



### 结构



==version1==

![image-20230421193157632](DL.assets/image-20230421193157632.png)



==version2==

亮点：

+ inverted residuals (倒残差结构)
+ linear bottlenecks

![image-20230421193418338](DL.assets/image-20230421193418338.png)



![image-20230421194110287](DL.assets/image-20230421194110287.png)



![image-20230421194452623](DL.assets/image-20230421194452623.png)



==version3==

亮点：

+ 更新block(bneck)
+ 使用NAS(neural architecture search)搜索参数
+ 重新设计耗时层机构

 ![image-20230421201105134](DL.assets/image-20230421201105134.png)

![image-20230421201259040](DL.assets/image-20230421201259040.png)

![image-20230421201521464](DL.assets/image-20230421201521464.png)

![image-20230421201858008](DL.assets/image-20230421201858008.png)

![image-20230421201915519](DL.assets/image-20230421201915519.png)





## ShuffuleNet



提出了channel shuffle 的思想

其中的unit 全是GConv和DWConv



![image-20230428170037268](DL.assets/image-20230428170037268.png)



![image-20230428201449018](DL.assets/image-20230428201449018.png)





<img src="DL.assets/image-20230429160348233.png" alt="image-20230429160348233" style="zoom:67%;" />



==代码问题==

由于换了一个train.py，导致花费了很长时间。

尽管已经载入了迁移学习的参数，第一个epoch得到的准确率不会超过50%， 这和教学视频不符合。

用了以前的train.py不管是载入参数，还是无迁移学习训练，效果都不好，初始几个epoch的准确率在20+%，而且还上不去。

大概是model.py里没有initiate weights的问题。 估计Up把参数初始化放到他新改的函数步骤里了。

需要看train_with_multi_GPU这一集

明天改。



载入初始权重还是有错误，估计是model.py编写的时候出了什么问题。暂先放弃。



$\uparrow$绝对是自己写的model有问题 







## EfficientNet

### 亮点

==compound scaling: channels, layers and resolution    $\uparrow$==

似乎和mobilenet差不多。。。

<img src="DL.assets/image-20230501201608715.png" alt="image-20230501201608715" style="zoom:50%;" />

### 结构



![image-20230501202257739](DL.assets/image-20230501202257739.png)

![image-20230501202337285](DL.assets/image-20230501202337285.png)



![image-20230501202757276](DL.assets/image-20230501202757276.png)

![image-20230501203513960](DL.assets/image-20230501203513960.png)



## EfficientNetV2



### 亮点

 

+ 引入Fused-MBConv模块
+ 引入渐进式学习策略（训练更快）



针对v1版本的问题：

+ 训练图像尺寸很大时，训练速度非常慢
+ 在网络浅层使用DWConv的速度会很慢
+ 同等的放大每个stage是次优的

<img src="DL.assets/image-20230503165147431.png" alt="image-20230503165147431" style="zoom:67%;" />



<img src="DL.assets/image-20230503165716267.png" alt="image-20230503165716267" style="zoom:67%;" />



==渐进式学习==

<img src="DL.assets/image-20230503171735672.png" alt="image-20230503171735672" style="zoom:67%;" />





### 结构

![image-20230503170033993](DL.assets/image-20230503170033993.png)



![image-20230503170856737](DL.assets/image-20230503170856737.png)



![image-20230503203714130](DL.assets/image-20230503203714130.png)





## Vision Transformer

### 亮点

可能在于把NPL的模型拿来用了吧= =；

self-attention & multi-head self-attention的东西见笔记本

### 结构



<img src="DL.assets/image-20230504151526451.png" alt="image-20230504151526451" style="zoom:67%;" />

<img src="DL.assets/image-20230504152008006.png" alt="image-20230504152008006" style="zoom:67%;" />

==注意： 位置编码是一个跟token形状一样的Tensor(Vector?)，将二者按位置相加，最终得到进入Transformer Encoder的输入。==



<img src="DL.assets/image-20230504155737886.png" alt="image-20230504155737886" style="zoom:67%;" />

==layer normalization 实际上和batch normalization 相似，只不过前者常用于NPL， 后者常用于image processing==



<img src="DL.assets/image-20230504160236848.png" alt="image-20230504160236848" style="zoom:67%;" />



==整体结构框架==



![vit-b/16](DL.assets/20210704124600507.png)





## Swin-Transfomer

### 亮点

+ surpass all the previous state-of-the-art by a large margin(2~3)
+ 使用了W-MSA，减少计算量

![image-20230507112159240](DL.assets/image-20230507112159240.png)

关于这个计算量怎么来的： [ Swin-Transformer网络结构详解_swin transformer_太阳花的小绿豆的博客-CSDN博客](https://blog.csdn.net/qq_37541097/article/details/121119988)



+ S(Shifted)W-MSA，和W-MSA成对搭配使用，来实现window之间信息的交互

+ relative position



### 结构

![image-20230507103854880](DL.assets/image-20230507103854880.png)



#### Patch Merging

<img src="DL.assets/image-20230507105315122.png" alt="image-20230507105315122" style="zoom: 50%;" />





patch merging 起到将输入长宽减半，通道双倍的作用

<img src="DL.assets/image-20230507110832811.png" alt="image-20230507110832811" style="zoom:50%;" />

#### 关于SW-MSA的问题：

偏移窗口后得到的划分块大小不一致：padding(块数增加会导致更多的计算量) or reconstruct（不相邻的小块现在组成一个大快，但彼此的q不需要对方的k）

reconstruct: masked-MSA    (大块里的小块qkv计算的时候会有另外一个小块，但与另一块的计算结果会-100，经过softmax就置为0了，等价于没和另一个小块交流)   （ps: 矩阵的加减在gpu看来都没啥计算量的）

<img src="DL.assets/image-20230507121415314.png" alt="image-20230507121415314" style="zoom: 67%;" />

<img src="DL.assets/image-20230507121439161.png" alt="image-20230507121439161" style="zoom:67%;" />



<img src="DL.assets/image-20230507113415412.png" alt="image-20230507113415412" style="zoom:50%;" />



#### relative position



<img src="DL.assets/image-20230510192911301.png" alt="image-20230510192911301" style="zoom: 67%;" />

<img src="DL.assets/image-20230510192930679.png" alt="image-20230510192930679" style="zoom:67%;" />



<img src="DL.assets/image-20230510193013627.png" alt="image-20230510193013627" style="zoom: 67%;" />

<img src="DL.assets/image-20230510193042911.png" alt="image-20230510193042911" style="zoom:67%;" />

<img src="DL.assets/image-20230510193118850.png" alt="image-20230510193118850" style="zoom:67%;" />



==最后训练的是relative position bias table里的参数==



#### 参数

![image-20230507134107068](DL.assets/image-20230507134107068.png)



#### 代码的结构

| def              | class                |
| ---------------- | -------------------- |
| drop_path_f      | DropPath             |
| window_partition | PatchEmbed           |
| window_reverse   | PatchMerging         |
|                  | Mlp                  |
|                  | WindowAttention      |
|                  | SwinTransformerBlock |
|                  | BasicLayer           |
|                  | SwinTransformer      |

```mermaid
graph LR;
SwinTransformer-->BasicLayer
SwinTransformer-->PatchEmbed
BasicLayer-->self.create_mask
BasicLayer--> SwinTransformerBlock
BasicLayer-->PatchMerging
BasicLayer-->A[window_partition]
SwinTransformerBlock-->DropPath
SwinTransformerBlock-->Mlp
SwinTransformerBlock-->WindowAttention
SwinTransformerBlock-->B[window_partition]
SwinTransformerBlock-->window_reverse
DropPath-->drop_path_f

```



## ConvNeXt

### 亮点

transformer 的策略能否用在卷积神经网络里，让后者变得更为有效？

于是乎在以下方面做了探索

+ Macro design
+ ResNeXt
+ Inverted bottleneck
+ Large kernel size
+ Various layer-wise Micro designs

<img src="DL.assets/image-20230511133856825.png" alt="image-20230511133856825" style="zoom:67%;" />





### 结构



![image-20230511133926917](DL.assets/image-20230511133926917.png)









## MobileViT

### 亮点

+ light-weight， general-purpose, mobile-friendly
+ pure Transfomer model  issues: 
  + 参数多，要求算力高
  + 缺少空间归纳偏置--->绝对位置，相对位置
  + 迁移到其他任务比较繁琐<--- 由位置编码导致的
  + 训练困难
+ 是cnn和transformer的混合





### 结构

![image-20230515190420533](DL.assets/image-20230515190420533.png)

![image-20230515190452312](DL.assets/image-20230515190452312.png)

<img src="DL.assets/image-20230515193443323.png" alt="image-20230515193443323" style="zoom:50%;" />

这里做self-attention的时候，划分的格子也叫做patch。



<img src="DL.assets/image-20230515193634209.png" alt="image-20230515193634209" style="zoom:50%;" />



![image-20230515194114346](DL.assets/image-20230515194114346.png)



# pytorch object detection

+ 评价指标

```
IoU metric: bbox
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.322
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.635
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.289
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.040
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.195
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.383
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.331
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.453
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.461
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.113
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.318
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.530
```

![image-20230919143208728](./DL.assets/image-20230919143208728.png)





## ssd

+ faster rcnn的问题
  + 对小目标检测效果很差
  + 模型大，检测速度慢

### 网络结构

![image-20231013174602449](./DL.assets/image-20231013174602449.png)





## yolov1

#### 整体思想

<img src="./DL.assets/image-20231015193451099.png" alt="image-20231015193451099" style="zoom:50%;" />

<img src="./DL.assets/image-20231015193509261.png" alt="image-20231015193509261" style="zoom:50%;" />

<img src="./DL.assets/image-20231015193527507.png" alt="image-20231015193527507" style="zoom:50%;" />



#### 网络结构

<img src="./DL.assets/image-20231015193601764.png" alt="image-20231015193601764" style="zoom:50%;" />



+ 损失函数

![img](./DL.assets/clip_image002.png)

<img src="./DL.assets/image-20231015194523193.png" alt="image-20231015194523193" style="zoom:50%;" />
