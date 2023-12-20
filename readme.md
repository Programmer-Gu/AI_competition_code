# README

[toc]



## 一、代码运行命令

### 1.环境配置：

```
pip install -r requirements.txt
```

### 2.运行命令：

```split: select from val, testa and testb``` \
```save_dir: path to save your results```

```python
CUDA_VISIBLE_DEVICES=0 python main.py \
      --input_file 'data/{split}.jsonl' \
      --detector_file 'data/dets_dict.json' \
      --image_root 'data/train2014' \
      --cache_path '{save_dir}/cache' \
      --results_path 'code/result/{split}.json'\
      --sam_pt_path 'code/sam_pt'\
      --sam_checkpoint 'sam模型权重路径'\
```

>运行命令说明：
>
>```python
>--input_file  data/{split}.jsonl    数据文件
>--detector_file 'data/dets_dict.json'  检测框文件
>--image_root 'data/train2014'   图片路径
>--cache_path '{save_dir}/cache' 缓存路径
>--results_path 'code/result/{split}.json'  结果保存路径
>--sam_pt_path 'code/sam_pt' sam生成的缓存保存路径
>--sam_checkpoint 'sam模型权重路径(一定要设置)'
>```
>
>例如：
>
>```python
>python main.py --input_file 'data/val/annos.jsonl' --detector_file 'data/dets_dict.json' --image_root 'data/val/images' --cache_path 'code/result/cache' --results_path 'code/result/result.json' --sam_pt_path 'code/sam_pt' --sam_checkpoint 'E:\全球校园人工智能算法精英大赛\算法挑战赛文件\SAM\SAM模型权重\sam_vit_h_4b8939.pth'
>```
>
>

# 算法流程图

## 1. 算法流程总览

![image-20231202184403820](C:\Users\31602\AppData\Roaming\Typora\typora-user-images\image-20231202184403820.png)

## 2. 算法流程说明

### 2.1 模型选择

1. clip使用ViT-B32与RN50X6权重参数
2. SAM使用samhuge权重参数

### 2.2 模型模块

#### 2.2.1 综合信息模块

该模块将题目中出现的多段文本描述做拼接处理，合成一段具有综合语义信息的长语句，将其与所有锚框中的图片生成相似度，生成综合语义内容信息相似度$S$。

#### 2.1.2 reclip语法信息模块

传统reclip基于语法信息生成语法树，通过节点（名词）之间的语法联系，以此完成对方位的解析任务。这种方法能够有效提升单个语句对于方位的辨识任务。但是在本次任务中，面对多个包含不同语义信息的语句指向同一目标的任务，如果沿用reclip而采用一个句子对一个目标的方式会导致语义信息的缺失，无法综合衡量目标与周围环境的详细信息。因此我们提出将多个指向同一目标的语句通过同义词检测的方式，将相似的节点合并，使得相似节点能够获取其它节点的后缀信息，以此完成语法树的合并。在检测到同义词出现在多个语法树上时，我们会减低所在树的最终权重，以此避免重复词语带来的语法树冗余信息的主导。

对于一段语句，reclip将句子中的名词作为顶点，节点之间的关系作为边，构建语法树，得到对应语法树的顶点集$V = \{v_{1},v_{2} \dots v_{n}\}$，边集为$E=\{\{v_{1},v_{2}\},\{v_{2},v_{3}\}, \dots\}$，为了找出不同语法树之间名词相似的节点，我们将所有语法树的节点置于同一个集合，顶点集合表示为：
$$
V_{nouns} = [v_{11},v_{12}. \dots v_{ij}],(i \le j)
$$
对集合内部所有名词做编码，将其转到同一向量中，集合表示为：
$$
En_{nouns}=[en_{11},en_{12} \dots en_{ij} ],(i \le j)
$$
对编码集合两两元素做相似度计算，得出两两节点的相似度：
$$
Sim_{ij \times ij} = sim(En_{nouns},En_{nouns})
$$
当$Sim[pq,ij] (p \ne i)$的值大于$t_{threshold}$（这里的t取0.86），我们认为$pq,ij$节点是同类型的名词节点，此时将$pq,ij$节点合并为新的节点$n_{node}$，且$children(n_{node})=\{\dots \{v_{m},v_{n}\} \dots \}$，$m \in \{p,i\}$，合并流程直观表示为：

最终我们得出合并语法树后的概率计算式为：
$$
\begin{align*}
  \begin{cases}
  \text Pro(i,j)= \sum^{m}\frac{1}{N}\sum^{n}(Pro(i-1,j)R(i,j),\ \ (i = 1)  \\
  \text Pro(i,j)= \sum^{n}(Pro(i-1,j)R(i,j), \ \ others
  \end{cases}&
\end{align*}
$$
其中N代表该顶点名词的出现频次，R代表该节点名词与其它锚框内图片相似度。

#### 2.1.3  颜色信息模块

通过对于数据集结果分析，发现传统clip模型对于颜色信息识别许多情况并不准确，我们提出颜色信息模块，以此修正颜色的偏差。我们会将图片转换到hsv空间，通过色调、饱和度、明度信息筛选出与对应颜色文本匹配的色彩区域mask，之后对色彩mask内部区域与标准rgb色值求出偏差，最后对每一个框求出色彩区域比重和颜色偏差均值，将上面信息转换为概率值相乘，得出综合色彩面积与色彩信息的色彩置信概率。

首先将图片从rgb空间转换到hsv空间表示为$Img_{h,s,v}$，对图片中指定颜色的hsv范围表示为$Range_{h,s,v}$，$Mask_{color}$计算为：
$$
Mask_{color}(i,j) = Img_{h}(i,j) \in Range_{h} \ \wedge \ Img_{s}(i,j) \in Range_{s} \ \wedge \ Img_{v}(i,j) \in Range_{v}
$$
计算rgb颜色距离，我们使用如下公式表示：
$$
Distance_{color} = \sqrt{(2 + \frac{r_{\text{mean}}}{256}) \times (R^2) + 4 \times (G^2) + (2 + \frac{255 - r_{\text{mean}}}{256}) \times (B^2)}
$$
之后我们对$Mask_{color}(i,j)$与$Distance_{color}$求均值，分别代表颜色的面积占比值和颜色的差距平均。所有锚框的面积占比值和颜色的差距平均值表示为$Ce,Cd$ 我们得出，该锚框指定颜色的概率计算公式为：
$$
CPro(i) = 0.8softmax(\frac{Ce(i) - \mu_{Ce}}{\sigma_{Ce}}) \times 0.8softmax(\frac{Cd(i) - \mu_{Cd}}{\sigma_{Cd}})+0.3
$$


#### 2.1.4 绝对方位模块

reclip在面对许多包含绝对方位计算缺陷较为明显，这是由于reclip在绝对方位计算往往是针对单个名词，没有考虑全局信息，这导致该部分位置信息会被兄弟节点通过加和的方式削弱，到达根节点时无法正确反映物体绝对位置信息。为此，我们检测句子中关键的方位词语，通过判断方位词前后是否含有名词来判断该方位词是否是绝对方位，一旦确定方位情况，会将反方向锚框全部过滤。

#### 2.1.5 标签筛选模块

在面对同类型图片通过同类之间的相互关系（如2nd，3rd等排序信息），reclip便无法处理该类问题。在检索到序数词后，我们通过找出句子的主语与所有锚框图片做匹配，如果匹配结果大于一定阈值，我们便认为，锚框内部的图片是同一种类型的对象。这样我们通过逻辑顺序直接从该同类对象中选取出对应的结果得出答案。



### 2.3 综合说明

在得到四个模块的概率信息后，我们会将这四部分信息相乘得到最终的结果，最终选择概率最高的值对应的锚框作为这些句子的同一目标答案。



## 三、参数的选择和定义

> 此处只有官方给出参数的一些值的设定

| 参数               | 值    |
| ------------------ | ----- |
| box_area_threshold | 0.045 |
| enlarge_boxes      | 0.2   |
| blur_std_dev       | 100   |
| baseline_threshold | 2     |
| temperature        | 5     |



