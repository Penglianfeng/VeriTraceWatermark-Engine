import torch
import torchvision
import torchvision.transforms as transforms
from pytorch_fid import fid_score
import os
import cv2
import numpy as np
from utils import *
from tqdm import tqdm
from PIL import Image
#除了SSIM，都是用的文件夹图片
# 准备真实数据分布和生成模型的图像数据
real_images_folder = '/private/bisan/mist-main_Test/metrics/Lora/new'
generated_images_folder = '/private/bisan/mist-main_Test/metrics/Lora/old'


# 获取文件夹中的所有图像文件
real_image_files = [os.path.join(real_images_folder, f) for f in os.listdir(real_images_folder) if f.endswith(('png', 'jpg', 'jpeg'))]
generated_image_files = [os.path.join(generated_images_folder, f) for f in os.listdir(generated_images_folder) if f.endswith(('png', 'jpg', 'jpeg'))]

real_path = '/private/bisan/mist-main_Test/metrics/Lora/old/00045-3679344484.png'
generated_path='/private/bisan/mist-main_Test/metrics/Lora/new/00000-1979293107.png'
ssim_real_img=Image.open(real_path).convert("RGB")
ssim_generated_img=Image.open(generated_path).convert("RGB")
# 将PIL图像转换为PyTorch张量
transform = transforms.ToTensor()
real_tensor = transform(ssim_real_img).unsqueeze(0)  # 增加批次维度
generated_tensor=transform(ssim_generated_img).unsqueeze(0)
# 将图像张量转换为PIL图像
real_ssim = transforms.ToPILImage()(real_tensor.squeeze(0))
generated_ssim=transforms.ToPILImage()(generated_tensor.squeeze(0))
# 将PIL图像转换为NumPy数组
real_ssim_np = np.array(real_ssim)
generated_ssim_np=np.array(generated_ssim)
# 读取图像并转换为数组
def load_images(image_files):
    images = []
    for file in image_files:
        img = cv2.imread(file)
        if img is not None:
            images.append(img)
    return np.array(images)

# 将图像转换为数组
real_images_array = load_images(real_image_files)
generated_images_array = load_images(generated_image_files)

#MSE Score
MSE=mse(real_images_array,generated_images_array)
RMSE = np.sqrt(MSE)

#PSNR Score
PSNR=psnr(real_images_array,generated_images_array)

#SSIM Score
SSIM=calculate_ssim(real_ssim_np,generated_ssim_np)

# 加载预训练的Inception-v3模型
inception_model = torchvision.models.inception_v3(pretrained=True)

# 定义图像变换
transform = transforms.Compose([
    transforms.Resize(299),
    transforms.CenterCrop(299),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# 计算FID距离值
fid_value = fid_score.calculate_fid_given_paths([real_images_folder, generated_images_folder],
                                                 1,'cuda',2048)
print('FID value:', fid_value)
print('RMSE value:', RMSE)
print('SSIM value:', SSIM)
print('PSNR:', PSNR)

