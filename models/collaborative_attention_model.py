from collections import OrderedDict
import torch
from torch import nn
from .base_model import BaseModel
from . import networks
import numpy as np
import time

class CollaborativeAttentionModel(BaseModel):
    """ This class implements the pix2pix model, for learning a mapping from input images to output images given paired data.

    The model training requires '--dataset_mode aligned' dataset.
    By default, it uses a '--netG unet256' U-Net generator,
    a '--netD basic' discriminator (PatchGAN),
    and a '--gan_mode' vanilla GAN loss (the cross-entropy objective used in the orignal GAN paper).

    pix2pix paper: https://arxiv.org/pdf/1611.07004.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For pix2pix, we do not use image buffer
        The training objective is: GAN Loss + lambda_L1 * ||G(A)-B||_1
        By default, we use vanilla GAN loss, UNet with batchnorm, and aligned datasets.
        """
        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(norm='batch', netG='mh_resnet_6blocks', dataset_mode='aligned')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        self.n_input_modal = opt.n_input_modal
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:  # during test time, only load G
            self.model_names = ['G']
        # define networks (both generator and discriminator)
        self.attention_type = opt.attention_type
        self.netG = networks.define_MHG(opt.attention_type, opt.n_input_modal, opt.input_nc+opt.n_input_modal+1, opt.output_nc, opt.ngf, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            self.netD = networks.define_D(opt.n_input_modal*(opt.input_nc+opt.n_input_modal+1) + opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)


        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
    
            self.criterionL2 = torch.nn.MSELoss()
            self.criterionKL = torch.nn.KLDivLoss()
    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        self.real_A = input['A'].to(self.device)
        self.real_B = input['B'].to(self.device)
        self.real_B_no_mask = input['B'][:, :self.opt.input_nc].to(self.device)
        self.modal_names = [i[0] for i in input['modal_names']]

    def forward(self, train=False):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        if train:
            self.fake_B, self.encoder_features, self.fused_features = self.netG(self.real_A, True)
        else:
            self.fake_B = self.netG(self.real_A, train)  # G(A)

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        # Real
        real_AB = torch.cat((self.real_A, self.real_B_no_mask), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB) 

        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B_no_mask) * self.opt.lambda_L1
        # combine loss and calculate gradients
        self.loss_G = self.loss_G_GAN + self.loss_G_L1
        if hasattr(self, 'sr'):
            self.loss_SR_intra = 0
            self.loss_SR_inter = 0
            sr_encoder_features = self.sr.get_features(self.real_B_no_mask)

            if 'single_modal_attention' in self.attention_type:
                for i in range(len(self.encoder_features)):
                    self.loss_SR_intra += self.criterionL2(sr_encoder_features[2], self.encoder_features[i][2])
                    self.loss_G += self.loss_SR_intra * 0.1
                
            if 'multi_modal_attention' in self.attention_type:
                self.loss_SR_inter = self.criterionL2(sr_encoder_features[-1], self.fused_features)
                self.loss_G += self.loss_SR_inter * 0.1
        self.loss_G.backward()


    def optimize_parameters(self):
        self.forward(True)  # compute fake images: G(A)
        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()     # set D's gradients to zero
        self.backward_D()                # calculate gradients for D
        self.optimizer_D.step()          # update D's weights
        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # udpate G's weights

    def compute_visuals(self):
        """Calculate additional output images for tensorboard visualization"""
        pass

    def get_current_visuals(self):
        modal_imgs = []
        for i in range(self.n_input_modal):
            modal_imgs.append(self.real_A[:, i*(self.n_input_modal+1+self.opt.input_nc):i*(self.n_input_modal+1+self.opt.input_nc)+self.opt.input_nc, :, :])
        modal_imgs.append(self.real_B_no_mask)
        visual_ret = OrderedDict()
        for name, img in zip(self.modal_names, modal_imgs):
            visual_ret[name] = img
        visual_ret['fake_' + self.modal_names[-1]] = self.fake_B

        return visual_ret

    def add_srmodel(self, sr_model):
        self.sr = sr_model
        self.sr.set_requires_grad(self.sr.netG, False)
        if 'single_modal_attention' in self.attention_type:
            self.loss_names.append('SR_inter')
        if 'multi_modal_attention' in self.attention_type:
            self.loss_names.append('SR_intra')

    def get_encoder_features(self):
        encoder_features, attention_features = self.netG.module.get_features(self.real_A)
        if attention_features is not None:
            attention_features = torch.chunk(attention_features, self.opt.n_input_modal, dim=1)
        if hasattr(self, 'sr'):
            sr_encoder_features = self.sr.get_features(self.real_B)
        else:
            sr_encoder_features = None
        return encoder_features, attention_features, sr_encoder_features
