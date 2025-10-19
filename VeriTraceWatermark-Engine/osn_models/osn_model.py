import torch.nn as nn
from . import scseunet
import numpy as np
import torch 
from gzhu import *

class OSN_Net(nn.Module):

    def __init__(self,tmodel,object):
        super(OSN_Net,self).__init__()
        self.name ='OSN_Net'

        imagenet_templates_small_style = ['a painting']
        imagenet_templates_small_object = ['a photo']
        if object:
            imagenet_templates_small = imagenet_templates_small_object
        else:
            imagenet_templates_small = imagenet_templates_small_style
        input_prompt = [imagenet_templates_small[0] for i in range(1)]
        self.condition=input_prompt
        # for Facebook
        # resRatio = 0.02
        # qf =92
        # for WeChat 
        # resRatio = 0.05
        # qf = 58 
        # for qq
        resRatio = 0.3
        qf = 85
        self.UNet = scseunet.SCSEUnet(seg_classes=3,backbone_arch='seresnext50',resRatio=resRatio,qf=qf)
        self.UNet =torch.nn.DataParallel(self.UNet).cuda() 
        modelpath='/private/bisan/RobustOSNAttack_Test/se_resnext50_32x4d-a260b3a4.pth'
        #modelpath = '/private/bisan/zzl2/RobustOSNAttack/scseunet_facebook.pth'
        self.modelname = 'scseunet'
        #self.UNet.load_state_dict(torch.load(modelpath))
        self.UNet.load_state_dict(torch.load(modelpath),False)
        print("load {}".format(modelpath))


        self.target_model = tmodel
      
    def forward(self, x,target_info,mode,rate,target_size,components):
       
        # self.mean = torch.Tensor(np.array([0.5, 0.5, 0.5])[:, np.newaxis, np.newaxis])
        # self.std = torch.Tensor(np.array([0.5, 0.5, 0.5])[:, np.newaxis, np.newaxis])
        # height,width = x.shape[-2],x.shape[-1]
        # self.mean = self.mean.expand(3, int(height), int(width)).cuda()
        # self.std = self.std.expand(3, int(height), int(width)).cuda()
        # x = (x-self.mean)/self.std #x has to normalized to [-1,1]
        x1 = self.UNet(x)
        # x2 = x1 * self.std + self.mean
        # x3 = torch.clamp(x2,0.,1.)
        # x3 = x3.astype(np.float32) / 127.5 - 1.0
        
        outputs = self.test(x1,target_info,mode,rate,target_size,components)

        return outputs
    
    def get_components(self, x, no_loss=False):
        """
        Compute the semantic loss and the encoded information of the input.
        :return: encoded info of x, semantic loss
        """

        z = self.target_model.get_first_stage_encoding(self.target_model.encode_first_stage(x)).to(device)
        c = self.target_model.get_learned_conditioning(self.condition)
        if no_loss:
            loss = 0
        else:
            loss = self.target_model(z, c)[0]
        return z, loss

    def pre_process(self, x, target_size):
        processed_x = torch.zeros([x.shape[0], x.shape[1], target_size, target_size]).to(device)
        trans = transforms.RandomCrop(target_size)
        for p in range(x.shape[0]):
            processed_x[p] = trans(x[p])
        return processed_x

    def test(self, x, target_info,mode,rate,target_size,components=False):
        """
        Compute the loss based on different mode.
        The textural loss shows the distance between the input image and target image in latent space.
        The semantic loss describles the semantic content of the image.
        :return: The loss used for updating gradient in the adversarial attack.
        """
        fn = nn.MSELoss(reduction="sum")
        zx, loss_semantic = self.get_components(x, True)
        zy, _ = self.get_components(target_info, True)
        if mode != 1:
            _, loss_semantic = self.get_components(self.pre_process(x, target_size))
        if components:
            return fn(zx, zy), loss_semantic
        if mode == 0:
            return - loss_semantic
        elif mode == 1:

            return fn(zx, zy)
        else:
            return fn(zx, zy) - loss_semantic * rate

class OSN_Model(nn.Module):
    def __init__(self,target_model,object):
        super(OSN_Model,self).__init__()
        self.model = OSN_Net(target_model,object)
        self.model = torch.nn.DataParallel(self.model).cuda() 

    def forward(self, x,target_info,mode,rate,target_size,components=False):
        self.model.eval()
        outputs = self.model(x,target_info,mode,rate,target_size,components)
        return outputs



            
    

