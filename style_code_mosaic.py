#from __future__ import print_function
#import argparse
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torchvision.utils as vutils
import time
import math



def generate_mosaic(net,disp_size=256,bs=4):
  device = torch.device("cuda")
  torch.manual_seed(10)

  scale = nn.AdaptiveAvgPool2d(disp_size)

  netG = net.cuda()
  with torch.no_grad():
      mix_levels = [4,8,12]

      
      z0 = [torch.randn(bs, 512).cuda()]
      target_fixed,style = netG(z0,return_latents=True)
      target_fixed = scale(target_fixed)
      target_fixed = torch.cat([torch.zeros(1,3,disp_size,disp_size),target_fixed.cpu()])
      
      
      with torch.no_grad():
          for m in mix_levels:
              for l in range(2):
                  
                  zl = [torch.randn(1, 512).cuda()]
                  target_fixed_l,style_l = netG(zl,return_latents=True)
                  
                  imgs = [scale(target_fixed_l).cpu()]
                  
                  for k in range(bs):
                      # Exercise 5: your code here. Mix style_l and style using m, by using first m dimensions (in dim=1) for style_l and remaining from style
                      imgs.append(scale(netG([style_m],input_is_latent=True)[0]).detach().cpu())
                      
                  imgs = torch.cat(imgs,dim=0).cpu()
                  target_fixed = torch.cat([target_fixed,imgs],dim=0)

  vutils.save_image(target_fixed.cpu().detach().clamp(-1,1),f'gen_mosaic.jpg',nrow=int(5),normalize=True)            
  return target_fixed
              
#vutils.save_image(target_fixed.cpu().detach().clamp(-1,1),out_folder + f'style_grid.jpg',nrow=int(5),normalize=True)                 
                