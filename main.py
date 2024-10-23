#
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path as osp
import sys
import time 
import math
# add python path of VirtualStain to sys.path
parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 2)))
sys.path.insert(0, parent_path)
import argparse
import os 
from PIL import Image
import blobfile as bf
import torch as th
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW
from guided_diffusion.script_util import (
    add_dict_to_argparser,
    args_to_dict,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
)
from guided_diffusion.train_util import parse_resume_step_from_filename, log_loss_dict,TrainLoop
from guided_diffusion import dist_util, logger
from guided_diffusion.fp16_util import MixedPrecisionTrainer
from guided_diffusion.image_datasets import load_data,load_data_onehot
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.resizer import Resizer
from guided_diffusion.model.cyclegan_network import define_G,__patch_instance_norm_state_dict
from guided_diffusion.model.umdst_network import ResnetGenerator

def main():
    args = create_argparser().parse_args()
    device = th.device('cuda') if th.cuda.is_available() else th.device('cpu')
    logger.log("creating conditional and uncondtional model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    model.eval()
    
    # -------------------Feature Adapter Function(Style Template Path)----------------------------------
    logger.log("creating GAN model..")
    # umdst
    genA2B = ResnetGenerator(input_nc=3, output_nc=3, ngf=64, n_blocks=4, img_size=256).to(dist_util.dev())
    state=th.load("./pretrained_model/ANHIR_params_latest.pt")
    genA2B.load_state_dict(state['genA2B'])
    # staingan
    # genA2B=define_G(input_nc=3,output_nc=3,ngf=64,netG="resnet_9blocks",norm="instance",use_dropout=False,init_type='normal',init_gain=0.02).to(dist_util.dev())
    # state=th.load("./pretrained_model/latest_net_G_A.pth")
    # genA2B.load_state_dict(state)
    for key in list(state.keys()):  # need to copy keys here because we mutate in loop
        __patch_instance_norm_state_dict(state, genA2B, key.split('.'))
    # genA2B.load_state_dict(state)
    genA2B.eval()
    # ----------------------------------------------------------------------------------------------------


    logger.log("creating AHNIR Dataset...")
    from datasets.ahnir_dataset import AHNIR_Dataset,get_ahnir_dataloader
    from mpi4py import MPI  
    dataset=AHNIR_Dataset(root_dataset=args.data_dir,classes=["HE"],
                          img_size=args.image_size,shard=MPI.COMM_WORLD.Get_rank(),
                          num_shard=MPI.COMM_WORLD.Get_size(),class_cond=True,random_flip=False)
    dl=get_ahnir_dataloader(dataset=dataset,batch_size=args.batch_size,deterministic=True)
    
    all_images=[]
    all_labels=[]
    label_kwargs={"HE":th.tensor([0],dtype=th.int16),"MAS":th.tensor([1],dtype=th.int16),"PAS":th.tensor([2],dtype=th.int16),"PASM":th.tensor([3],dtype=th.int16)}
    i=0
    
    # for al in range(75,100,5):
    #     alpha=al/100.
    #     print("alpha:",alpha)
    #     i=0
        # alpha=0.05
    alpha=0.55
    for index,(sample,extra) in enumerate(dl):
        model_kwargs={}
        sample=sample.to(dist_util.dev())
        # sample_gray=sample_gray.to(dist_util.dev())
        sample_ = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample_ = sample_.permute(0, 2, 3, 1)
        sample_ = sample_.contiguous()
        # logger.log("input onehot {}".format(extra))
        img=Image.fromarray(sample_.cpu().numpy()[0])
        fake_image=genA2B(sample,label_kwargs[args.target_domain].long().to(dist_util.dev()),dist_util.dev()) # umdst
        # fake_image=genA2B(sample).to(dist_util.dev())  staingan
        img.save(f"./evalations/inputs_res/{i}.png")
        sample_ = ((fake_image   + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample_ = sample_.permute(0, 2, 3, 1)
        sample_ = sample_.contiguous()
        img=Image.fromarray(sample_.cpu().numpy()[0])
        img.save(f"./evalations/gan/{i}.png")
        model_kwargs['y']=label_kwargs[args.target_domain].long().to(dist_util.dev())
        extra['y']=extra['y'].long().to(dist_util.dev()) #th.tensor([1]).long().to(dist_util.dev())
        logger.log("input condition and target domain condition:",extra,model_kwargs)
        
        noise,latents = diffusion.ddim_reverse_sample_loop(
            model, fake_image,  # style template reverse path
            clip_denoised=True,
            device=dist_util.dev(),
            model_kwargs=extra  # condition sample : reverse path  image to noise.
        )
        _,latents_source = diffusion.ddim_reverse_sample_loop(
            model, sample, #  structual template reverse path
            clip_denoised=True,
            device=dist_util.dev(),
            model_kwargs=model_kwargs  # condition sample: forward path  noise to image 
        )
        # ---- key componet ----
        null_visual_prompt=diffusion.stainpromptInversion(model,noise,latents,latents_source,50,alpha=alpha,clip_denoised=True,device=dist_util.dev(),model_kwargs=model_kwargs)
        target=diffusion.VPsample(noise,model,null_visual_prompt,model_kwargs)
        
        target = ((target + 1) * 127.5).clamp(0, 255).to(th.uint8)
        target = target.permute(0, 2, 3, 1)
        target = target.contiguous()
        img=Image.fromarray(target.cpu().numpy()[0])
        if not  os.path.exists(f"./evalations/umdst/alpha{alpha}"):
            os.makedirs(f"./evalations/umdst/alpha{alpha}")
        img.save(f"./evalations/umdst/alpha{alpha}/{i}.png")
        print("save {}".format(i))
        i+=1
    
        if i==11:
            # small scale infer to get better setting of alpha
            break
        
        
        
def create_argparser():
    defaults = dict(
        data_dir="",
        image_size=256,
        num_class=4,
        batch_size=1,
        microbatch=-1,
        schedule_sampler="uniform",
        model_path="",
        uncond_model_path="",
        use_ddim=True,
        classifier_path="",
        classifier_scale=2.5,
        target_domain="MAS",

        # rescale_timesteps=True,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()