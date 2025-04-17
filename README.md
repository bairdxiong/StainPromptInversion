# StainPromptInversion: Unpaired Multi-Domain Histopathology Virtual Staining using Dual Path Prompted Inversion

**Official implementation of "Unpaired Multi-Domain Histopathology Virtual Staining using Dual Path Prompted Inversion"**

👨‍💻:Bing Xiong, Yue Peng, RanRan Zhang, Fuqiang Chen, JiaYe He, Wenjian Qin.

1.ShenZhen Institue of Advanced Technology
2.University of Chinese Academy of Sciences

## [Paper](http://arxiv.org/abs/2412.11106)

![image](./assert/architecture_v4.png)

## 🤳 News:
If you are confused about our work or code, feel free to contact me b.xiong@siat.ac.cn
+ 🔥 We have released the full dataset predict result of our paper for evaluation. 
+ 🎊🎊Congratulations! Our paper is accepted by AAAI'25

## 📑 Prepare for Dataset and Enviroment

### 1. Enviroment

```code
conda env create -f environment.yaml
conda activate nvpdiff
```

### 2. Download Processed Dataset
Noted：val folder is for test, but we only use H&E folder. other stain type is only use for calculate FID.
Thanks to ZhangYi noted that!

1. BaiduNetDiskDownload: 
```code
 https://pan.baidu.com/s/1WEokxDFFvWO-NiqnXb6KcQ?pwd=hepm  提取码: hepm 

```

2. GoogleDrive

```code
https://drive.google.com/file/d/1b3pY6DryQdMDl934keb1vuFg-jE3Iamf/view?usp=drive_link 
```

### 3. Download Pretrained Weight
```
modelxxxx.pt represent pretrained diffusion model. ANHIR_params_lastest.pt represent UMDST pretrained weight. lastest_net_G_A.pt represent Cyclegan pretrained weight.
```

1. BaiduNetDiskDownload: 
```code
https://pan.baidu.com/s/15CSVbS0pSYwvnmNyCTknuA?pwd=mtqu 提取码: mtqu 
```

2.GoolgeDriven
```code
https://drive.google.com/drive/folders/1lpbfL9xZmIHoVwql5PTYD4eJztGMV6Po?usp=drive_link
```
### 4. Results in this paper
1.GoolgeDriven
```code
https://drive.google.com/file/d/1VqA-djwq0xkjm_U77OLW1vCnzyZb2Tot/view?usp=sharing
```
2.BaiduNetDiskDownload:
```code
链接: https://pan.baidu.com/s/1OkZdqUXuwv8KkzUCEw7ztg?pwd=b8f5 提取码: b8f5 
--来自百度网盘超级会员v6的分享
```

# Training Your own model

**Our method is a training-free method. If you want to training on your own dataset, you can following the stage.**

## 1. pretrained diffusion model

**Attention**: When you use improved-diffusion to pretrained diffusion model, you shoud set --rescale_timesteps True

You can pretrain your own diffusion model in this repo: [improved diffusion](https://github.com/openai/improved-diffusion) and [guided diffusion](https://github.com/openai/guided-diffusion)

## 2. pretrained GAN model
Our Method based on [Cyclegan](https://github.com/junyanz/CycleGAN) and [UMDST(AAAI'22)](https://github.com/linyiyang98/UMDST). If you want to pretrain your own GAN, please follow the two repo ReadME.md file. It is clear and easy to follow.

# Test/Inference

```
bash scripts/stain2stain_sample.sh
```
**Key Code in the file(guided_diffusion/gaussian_diffusion.py  line 911-line 985). Our Method is Simple and Useful**

For HE2MAS condition choose None(style path) extra(struct path) for paper setting, under alpha=0.05 (to get a better performance in style)

For HE2PAS We choose extra(style path) extra(struct path) for paper setting, under alpha=0.55(to get a balance performance)
```code
noise,latents = diffusion.ddim_reverse_sample_loop(
            model, fake_image,  # style template reverse path
            clip_denoised=True,
            device=dist_util.dev(),
            model_kwargs=None  # condition sample : reverse path  image to noise.  choice:{None,extra,model_kwargs}
        ) 
        _,latents_source = diffusion.ddim_reverse_sample_loop(
            model, sample, #  structual template reverse path
            clip_denoised=True,
            device=dist_util.dev(),
            model_kwargs=extra  # condition sample: forward path  noise to image choice:{None,extra,model_kwargs}
        )
```

You can use None condition on forward and reverse process to get a image with minor error.As paper Fig.5(b). Based on this,you can find some rule to choose condition. 

# Evaluate Metrics📈

```code
python eval.py -source_path  xxx/source_imge  --translate_path  xxx/results  --style_gt_path  xxx/ANHIR/val/{MAS,PAS} 
```

# Disscussion📰

We tested the effect of modifying the sampling condition variables on the results, and in most cases, selecting the condition corresponding to HE during the forward process of inputting HE images and transferring the staining to the target domain showed better results. However, under some parameters, consistently using the condition variables corresponding to HE showed better results. We look forward to implementing the visual stain prompt optimization strategy in this article in a faster way. If this repo is helpful to you, please star✨ and cite us! Thanks!


## BibTex
**If this repo is helpful to you, please cite us.**

Xiong, B., Peng, Y., Zhang, R., Chen, F., He, J., & Qin, W. (2025). Unpaired Multi-Domain Histopathology Virtual Staining Using Dual Path Prompted Inversion. Proceedings of the AAAI Conference on Artificial Intelligence, 39(8), 8780-8787. https://doi.org/10.1609/aaai.v39i8.32949

## Acknowledgement
**This work was supported in part by the Ministry of Science and Technology’s key research and development program (2023YFF0723400), Shenzhen-Hong Kong Joint Lab
on Intelligence Computational Analysis for Tumor lmaging (E3G111), and the Youth Innovation Promotion Association CAS (2022365).**

**This repo is built upon [improved diffusion](https://github.com/openai/improved-diffusion) and [guided diffusion](https://github.com/openai/guided-diffusion), dataset [ANHIR](https://anhir.grand-challenge.org/) . Thanks for their amazing work and contribution !**
