# PSCR

This is the official repo of the paper [PSCR: Patches Sampling-based Contrastive Regression for AIGC Image Quality Assessment](http://arxiv.org/abs/2311.15556):
  ```
@misc{yuan2023pkui2iqa,
      title={PKU-I2IQA: An Image-to-Image Quality Assessment Database for AI Generated Images}, 
      author={Jiquan Yuan and Xinyan Cao and Changjin Li and Fanyi Yang and Jinlong Lin and Xixin Cao},
      year={2023},
      eprint={2311.15556},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
<hr />

> **Abstract:** *In recent years, Artificial Intelligence Generated Content (AIGC) has gained widespread attention beyond the computer science community. Due to various issues arising from continuous creation of AI-generated images (AIGI), AIGC image quality assessment (AIGCIQA), which aims to evaluate the quality of AIGIs from human perception perspectives, has emerged as a novel topic in the field of computer vision. However, most existing AIGCIQA methods directly regress predicted scores from a single generated image, overlooking the inherent differences among AIGIs and scores. Additionally, operations like resizing and cropping may cause global geometric distortions and information loss, thus limiting the performance of models. To address these issues, we propose a patches sampling-based contrastive regression (PSCR) framework. We suggest introducing a contrastive regression framework to leverage differences among various generated images for learning a better representation space. In this space, differences and score rankings among images can be measured by their relative scores. By selecting exemplar AIGIs as references, we also overcome the limitations of previous models that could not utilize reference images on the no-reference image databases. To avoid geometric distortions and information loss in image inputs, we further propose a patches sampling strategy. To demonstrate the effectiveness of our proposed PSCR framework, we conduct extensive experiments on three mainstream AIGCIQA databases including AGIQA-1K, AGIQA-3K and AIGCIQA2023. The results show significant improvements in model performance with the introduction of our proposed PSCR framework.* 
<hr />


### PSCR
The pipeline of our proposed patches sampling-based contrastive regression framework.
![PSCR](https://github.com/jiquan123/PSCR/blob/main/Fig/PSCR.png)


### Pre-trained visual backbone
For feature extraction from input images, we selected several backbone
network models pre-trained on the ImageNet dataset, including:
-  VGG16 [url](https://download.pytorch.org/models/vgg16-397923af.pth)
-  VGG19 [url](https://download.pytorch.org/models/vgg19-dcbb9e9d.pth)
-  ResNet18 [url](https://download.pytorch.org/models/resnet18-f37072fd.pth)
-  ResNet50 [url](https://download.pytorch.org/models/resnet50-0676ba61.pth)
-  InceptionV4 [url](http://data.lip6.fr/cadene/pretrainedmodels/inceptionv4-8e4777a0.pth)


### Contact
If you have any question, please contact yuanjiquan@stu.pku.edu.cn
