# VA-IDN
Variable Augmented Network for Invertible Decolorization (基于辅助变量增强的可逆彩色图像灰度化)

**Authors**: LIAO Yifan, LI Zihao, WU Chunhua, WANG Guoyou, LIU Qiegen*  

廖一帆, 李子豪, 伍春花, 汪国有, 刘且根. 基于辅助变量增强的可逆彩色图像灰度化[J]. 电子与信息学报. doi: 10.11999/JEIT221205

彩色图像灰度化是一种被广泛应用于各个领域的图像压缩方式，但很少有研究关注彩色图像与灰度图像之间的相互转换技术。该文运用深度学习，创新性地提出了一种基于辅助变量增强的可逆彩色图像灰度化方法。该方法使用变量增强技术来保证输出与输入变量通道数相同以满足网络的可逆特性。具体来说，该方法通过可逆神经网络的正向过程实现彩色图像灰度化，逆向过程实现灰度图像的色彩复原。将所提方法在VOC2012, NCD和Wallpaper数据集上进行定性和定量比较。实验结果表明，所提方法在评价指标上均获得了更好的结果。无论是在全局还是局部，生成图像都可以最大程度地保留亮度、颜色对比度和结构相关性等特征。

## Visulization of the performance of VA-IDN
![](./Fig/Fig1.jpg)  
Gcs、Ledecolor和VA-IDN（从(b)到(d)）在NCD、VOC2012和Wallpaper数据集（从上到下）上图像灰度化效果

## The Flowchart of VA-IDN
![](./Fig/Fig2.jpg)  

### Other Related Projects

  * Variable Augmented Network for Invertible Modality Synthesis and Fusion  [<font size=5>**[Paper]**</font>](https://ieeexplore.ieee.org/abstract/document/10070774)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/iVAN)    
  
 * Variable augmentation network for invertible MR coil compression  [<font size=5>**[Paper]**</font>](https://www.sciencedirect.com/science/article/abs/pii/S0730725X24000225)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/VAN-ICC)         

 * Virtual coil augmentation for MR coil extrapoltion via deep learning  [<font size=5>**[Paper]**</font>](https://www.sciencedirect.com/science/article/abs/pii/S0730725X22001722)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/VCA)    

  * Synthetic CT Generation via Invertible Network for All-digital Brain PET Attenuation Correction  [<font size=5>**[Paper]**</font>](https://arxiv.org/abs/2310.01885)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/PET_AC_sCT)        

  * Invertible and Variable Augmented Network for Pretreatment Patient-Specific Quality Assurance Dose Prediction  [<font size=5>**[Paper]**</font>](https://link.springer.com/article/10.1007/s10278-023-00930-w)       
    
  * Variable augmented neural network for decolorization and multi-exposure fusion [<font size=5>**[Paper]**</font>](https://www.sciencedirect.com/science/article/abs/pii/S1566253517305298)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/DecolorNet_FusionNet_code)   [<font size=5>**[Slide]**</font>](https://github.com/yqx7150/EDAEPRec/tree/master/Slide)   
   
