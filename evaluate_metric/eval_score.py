import torch
from evaluate_metric.interact import set_logger
import os
from evaluate_metric.fid import calculate_fid_given_paths
import logging
from evaluate_metric.mse_psnr_ssim_mssim import calculate_ssim,calculate_msssim,calculate_psnr,calculate_mse,calculate_css
from datasets import image2tensor,imageresize2tensor

def calculate_l2_given_paths(path1,path2):
    file_name = os.listdir(path1)
    total = 0
    for name in file_name:
        s = imageresize2tensor(os.path.join(path1,name),256)
        name_i = name.split('.')[0]
        name = name_i + '.jpg'
        t = imageresize2tensor(os.path.join(path2,name),256)
        l2_i = torch.norm(s-t,p=2)
        total += l2_i
    return total/len(file_name)

def fid_l2_psnr_ssim(dataset,translate_path,source_path,gt_path):
    path1 = translate_path
    path2 = source_path

    fid_value = calculate_fid_given_paths(paths=[path1, gt_path],dataset = dataset)
    print('fid:{}'.format(fid_value))

    l2_distance = calculate_l2_given_paths(path1, path2)
    print('l2:{}'.format(l2_distance))

    mse = calculate_mse(path1, path2)
    print('mse:{}'.format(mse))

    psnr_value = calculate_psnr(path1, path2)
    print('psnr:{}'.format(psnr_value))

    ssim = calculate_ssim(path1, path2)
    print('ssim:{}'.format(ssim))
    return fid_value,l2_distance,mse,psnr_value,ssim

def fid_psnr_css_ssim_msssim(translate_path,source_path,gt_path):
    path1 = translate_path
    path2 = source_path
    dataset=""
    fid_value = calculate_fid_given_paths(paths=[path1, gt_path],dataset = dataset)
    print('fid:{}'.format(fid_value))

    psnr_value = calculate_psnr(path1, path2)
    print('psnr:{}'.format(psnr_value))

    ssim = calculate_ssim(path1, path2)
    print('ssim:{}'.format(ssim))

    css = calculate_css(path1, path2)
    print('css:{}'.format(css))
    
    ms_ssim=calculate_msssim(path1,path2)
    print("ms-ssim:{}".format(ms_ssim))
    
    return fid_value,psnr_value,ssim,css,ms_ssim





