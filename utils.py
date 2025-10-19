import torch
import numpy
import numpy as np
import math
from torchvision import datasets, transforms
import os
from PIL import Image
from numpy.lib.stride_tricks import as_strided as ast
import cv2

def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)

def ssim(img1, img2):
    SSIM=ssim(img1, img2,channel_axis=-1)
    return SSIM

def psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def mse(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    return mse

def calculate_ssim(img1, img2):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')

def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

def block_view(A, block=(3, 3)):
    """Provide a 2D block view to 2D array. No error checking made.
    Therefore meaningful (as implemented) only for blocks strictly
    compatible with the shape of A."""
    # simple shape and strides computations may seem at first strange
    # unless one is able to recognize the 'tuple additions' involved ;-)
    shape = (A.shape[0] / block[0], A.shape[1] / block[1]) + block
    strides = (block[0] * A.strides[0], block[1] * A.strides[1]) + A.strides
    return ast(A, shape=shape, strides=strides)

def image_concat(images):
    width, height = images[0].size
    target_shape = (4 * width, height)
    background = Image.new(mode="RGB", size=target_shape, color='black')
    for i,img in enumerate(images):
        location = ((i) * width,0)
        background.paste(img, location)
    return background

def all_image_concat(images):
    width, height = images[0].size
    target_shape = (width, height*4)
    background = Image.new(mode="RGB", size=target_shape, color='black')
    for i,img in enumerate(images):
        location = (0,(i) * height)
        background.paste(img, location)
    return background


def img_show(inputs,outputs,mask,delta,config):
    inputs = torch.squeeze(inputs)
    outputs = torch.squeeze(outputs)
    mask = torch.squeeze(mask)
    delta=torch.squeeze(delta)
    if config == 'Clean':
        img_p = transforms.ToPILImage()(inputs.detach().cpu()).convert('RGB')
        clean_p = transforms.ToPILImage()(outputs.detach().cpu()).convert('RGB')
        clean_mask_p = transforms.ToPILImage()(mask.detach().cpu()).convert('L')
        images = [img_p,clean_p,clean_mask_p]

    elif config == 'DWV':
        adv1 = transforms.ToPILImage()(inputs.detach().cpu()).convert('RGB')
        adv1_p = transforms.ToPILImage()(outputs.detach().cpu()).convert('RGB')
        adv1_mask_p = transforms.ToPILImage()(mask.detach().cpu()).convert('L')
        images = [adv1,adv1_p,adv1_mask_p]

    elif config == 'IWV':
        adv2 = transforms.ToPILImage()(inputs.detach().cpu()).convert('RGB')
        adv2_p = transforms.ToPILImage()(outputs.detach().cpu()).convert('RGB')
        adv2_mask_p = transforms.ToPILImage()(mask.detach().cpu()).convert('L')
        delta = transforms.ToPILImage()(delta.detach().cpu()).convert('RGB')
        images = [adv2,adv2_p,adv2_mask_p,delta]

    elif config == 'RN':
        random = transforms.ToPILImage()(inputs.detach().cpu()).convert('RGB')
        random_pred = transforms.ToPILImage()(outputs.detach().cpu()).convert('RGB')
        random_mask_p = transforms.ToPILImage()(mask.detach().cpu()).convert('L')
        images = [random,random_pred,random_mask_p]

    return image_concat(images)







