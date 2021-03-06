"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import torch
import os
import torch.nn.functional as Fu
import torchvision
import numpy as np
from tools.utils import auto_init_args, weight_init, choice_exclude,\
                        get_visdom_connection, save_image, gradient_penalty

from torch import nn
from torch.nn.utils.spectral_norm import spectral_norm
from torch.nn.parameter import Parameter
from models.relate_helpers import AdaIngen_bg, AdaIngen_obj


class RELATEStatOrder(torch.nn.Module):
    def __init__(self,
                 x_max=5.,
                 y_max=5.,
                 n_objects=2,
                 backgd_dim=30,
                 obj_dim=30,
                 offset_mlp=0.3,
                 ablation_xy=False,
                 custom_param_groups=True,
                 loss_weights={
                    'l_gen': 1.,
                    'l_gen_eval': 1.,
                    'l_disc': 1.,
                    'l_xy': 0.,
                    'l_style_disc': 0.,
                    'l_style_gen': 0.,
                    'l_style_gen_eval': 0.,
                    'l_gradient': 0.,
                 },
                 log_vars=[
                    'objective',
                    'l_gen',
                    'l_gen_eval',
                    'l_disc',
                    'l_style_disc',
                    'l_style_gen',
                    'l_style_gen_eval',
                    'l_xy',
                    'l_gradient'
                    ],
                 **kwargs):
        super(RELATEStatOrder, self).__init__()

        # autoassign constructor params to self
        auto_init_args(self)

        # Use gpu if available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.latent_dims = [self.backgd_dim, self.obj_dim]
        self.grad_pen = (loss_weights.get('l_gradient',0)>0.)
        
        self.zbg = Parameter(0.02*torch.randn([1, 4*64, 16, 16],
                                              device=self.device))
        self.zfg = Parameter(0.02*torch.randn([1, 8*64, 4, 4],
                                              device=self.device))

        # Init generator
        self.generator = GAN_gen(latent_dims=self.latent_dims)

        # Init discriminator
        self.discriminator = GAN_disc(pos_dim=2)

        # Position predictors
        self.mlp_xy = nn.Sequential(nn.Linear(sum(self.latent_dims)+2, 128), nn.LeakyReLU(negative_slope=0.2),
                                    nn.Linear(128, 64), nn.LeakyReLU(negative_slope=0.2),
                                    nn.Linear(64, 2))

        self.mlp_xy_rec = nn.Sequential(nn.Linear(sum(self.latent_dims), 128), nn.LeakyReLU(negative_slope=0.2),
                                        nn.Linear(128, 64), nn.LeakyReLU(negative_slope=0.2),
                                        nn.Linear(64, 2))

        # BCELoss init
        self.loss_bce = torch.nn.BCEWithLogitsLoss()

        # Init weights
        self.apply(weight_init)

    def forward(self, gt_images, labels=None, it=0,
                exp_dir='', gen=False, **kwargs):
        self.preds = {}

        if not hasattr(self, 'batch_size'):
            self.batch_size = gt_images.size(0)

        # Run main network
        loss, gen_im = self.run_model(gt_images, gen,
                                  validation=(kwargs['trainmode'] == 'val'))

        # fill in the dict of predictions
        self.preds['gen_images'] = gen_im

        if gen:
            self.preds['l_gen_eval'] = loss['l_gen_eval']
            self.preds['l_style_gen_eval'] = loss['l_style_gen_eval']

        if not gen or kwargs['trainmode'] == 'val':
            self.preds['l_disc'] = loss['l_disc']
            self.preds['l_gen'] = loss['l_gen']
            self.preds['l_style_disc'] = loss['l_style_disc']
            self.preds['l_style_gen'] = loss['l_style_gen']

        if 'xy' in loss.keys():
            self.preds['l_xy'] = loss['xy']

        if 'l_gradient' in loss.keys():
            self.preds['l_gradient'] = loss['l_gradient']

        # finally get the optimization objective using self.loss_weights
        self.preds['objective'] = self.get_objective(self.preds)

        # Save images: one batch during training and all at eval time
        if kwargs['trainmode'] == 'val':
            os.makedirs(exp_dir+'/images_val', exist_ok=True)
            for i in range(gen_im.size(0)):
                save_image(exp_dir+'/images_val/%06u.jpg' % (it*self.batch_size+i),
                           torch.clamp(gen_im[i] + 0.5, 0., 1.))

        return self.preds.copy(), self.preds['objective']

    def get_objective(self, preds):
        losses_weighted = [preds[k] * float(w) for k, w in
                           self.loss_weights.items()
                           if k in preds and w != 0.]
        if not hasattr(self, '_loss_weights_printed') or\
                not self._loss_weights_printed:
            print('-------\nloss_weights:')
            for k, w in self.loss_weights.items():
                print('%20s: %1.2e' % (k, w))
            print('-------')
            self._loss_weights_printed = True
        loss = torch.stack(losses_weighted).sum()
        return loss
    
    def _get_param_groups(self):
        return {'disc': [{'params': self.discriminator.parameters()}],
                'gen': [{'params': self.generator.parameters()},
                        {'params': self.zbg},
                        {'params': self.zfg},
                        {'params': self.mlp_xy.parameters()},
                        {'params': self.mlp_xy_rec.parameters()}]}

    def visualize(self, visdom_env_imgs, trainmode,
                  preds, stats, clear_env=False,
                  exp_dir=None, show_gt=True):
        viz = get_visdom_connection(server=stats.visdom_server,
                                    port=stats.visdom_port)
        if not viz.check_connection():
            print("no visdom server! -> skipping batch vis")
            return

        if clear_env:  # clear visualisations
            print("  ... clearing visdom environment")
            viz.close(env=visdom_env_imgs, win=None)

        idx_image = 0
        title = "e%d_it%d_im%d" % (stats.epoch, stats.it[trainmode], idx_image)

        if show_gt:
            types = ('gen_images', 'gt_images')
        else:
            types = ('gen_images',)

        for image_type in types:
            image = torch.clamp(preds[image_type], -.5, .5)
            image = (image + .5)
            image = torchvision.utils.make_grid(image, nrow=8).data.cpu().numpy()
            viz.image(image, env=visdom_env_imgs,
                      opts={'title': title+"_%s" % image_type})


    def run_model(self, x, gen=False, validation=False):
        batch_size = x.size(0)
        loss = {}
        # Sample different variables
        if not validation:
            self.background_vec, self.appearance_vec = 2*torch.rand([batch_size, self.latent_dims[0]], device=self.device)-1,\
                            2*torch.rand([1, batch_size, self.latent_dims[1]*self.n_objects], device=self.device)-1
        else:
            self.background_vec, self.appearance_vec = 1.5*torch.rand([batch_size, self.latent_dims[0]], device=self.device)-0.75,\
                            2*torch.rand([1, batch_size, self.latent_dims[1]*self.n_objects], device=self.device)-1

        self.tx = (2*self.x_max*torch.rand([1+int(self.ablation_xy)*(self.n_objects-1), batch_size, 1], device=self.device)-self.x_max) / 8.
        self.ty = (-1.*self.y_max*torch.rand([1+int(self.ablation_xy)*(self.n_objects-1), batch_size, 1], device=self.device)) / 8.

        # Just help the network by forcing position up
        tx, ty = self.tx, self.ty

        # Variants for ablation study
        if not self.ablation_xy:
            xy = self.mlp_xy(torch.cat((self.background_vec, self.appearance_vec[0, :, :self.latent_dims[1]].detach(),
                             torch.cat((self.tx, self.ty), 0).permute(1, 0, 2).squeeze()), 1))

            self.tx = 0.5*Fu.tanh(xy[:, :1]).permute(1, 0).unsqueeze(2) + tx
            self.ty = 0.5*Fu.tanh(xy[:, 1:]).permute(1, 0).unsqueeze(2) + ty

            self.other_style = self.appearance_vec
            self.appearance_vec = self.appearance_vec[:, :, :self.latent_dims[1]]

            for i in range(1, self.n_objects):
                new_pos = self.mlp_xy_rec(torch.cat((self.appearance_vec[i-1].detach(), self.background_vec.detach()), 1))
                self.tx = torch.cat((self.tx, self.tx[-1:, :, :].detach()+0.2*Fu.tanh(new_pos[:, :1]).permute(1, 0).unsqueeze(2)), 0)
                self.ty = torch.cat((self.ty, self.ty[-1:, :, :].detach()+0.1+self.offset_mlp*Fu.sigmoid(new_pos[:, 1:]).permute(1, 0).unsqueeze(2)), 0)

                self.appearance_vec = torch.cat((self.appearance_vec, self.other_style[:, :, i*self.latent_dims[1]:(i+1)*self.latent_dims[1]]), 0)
        else:
            self.other_style = self.appearance_vec
            self.appearance_vec = self.appearance_vec[:, :, :self.latent_dims[1]]
            for i in range(1, self.n_objects):
                self.appearance_vec = torch.cat((self.appearance_vec, self.other_style[:, :, i*self.latent_dims[1]:(i+1)*self.latent_dims[1]]), 0)

        # Sample numbers of objects
        n_obj = np.random.randint(2, self.n_objects+1, batch_size)
        for i in range(batch_size):
            self.tx[n_obj[i]:, i, 0] = -2.
            self.ty[n_obj[i]:, i, 0] = -2.

        # Run encoder first
        gen_images, gen_vec = self.generator(batch_size, [self.zbg, self.zfg],
                                             [self.background_vec, self.appearance_vec],
                                             self.tx, self.ty,
                                             select=None)

        # In case we generate a super resolution
        # if gen_images.shape[2]!=x.shape[2] or gen_images.shape[3]!=x.shape[3]:
        #     gen_images = Fu.interpolate(gen_images, size=(x.shape[2], x.shape[3]))

        # Run discriminator on real/fake images
        _, gen_logits, gen_xy, gen_style = self.discriminator(gen_images)

        # Position loss
        pos_tensor = torch.cat((self.tx.detach()[:1], self.ty.detach()[:1]), 2).permute(1, 0, 2).contiguous().view(-1, 2)
        loss['xy'] = torch.mean((gen_xy-pos_tensor)**2)

        if not gen or validation:
            _, x_logits, _, disc_style = self.discriminator(x)
            
            if self.grad_pen and not validation:
                loss['l_gradient'] = gradient_penalty(x, self.discriminator)
            
            loss['l_disc'] = self.loss_bce(x_logits, torch.ones_like(x_logits))
            loss['l_gen'] = self.loss_bce(gen_logits, torch.zeros_like(gen_logits))

            loss['l_style_disc'] = sum([self.loss_bce(z, torch.ones_like(z)) for z in disc_style])
            loss['l_style_gen'] = sum([self.loss_bce(z, torch.zeros_like(z)) for z in gen_style])

        if gen or validation:
            loss['l_gen_eval'] = self.loss_bce(gen_logits, torch.ones_like(gen_logits))
            loss['l_style_gen_eval'] = sum([self.loss_bce(z, torch.ones_like(z)) for z in gen_style])

        return loss, gen_images


class GAN_gen(nn.Module):
    def __init__(self, latent_dims=[90, 30], gf_dim=64, c_dim=3):
        super(GAN_gen, self).__init__()
        auto_init_args(self)

        # Objects and Background tensor generators
        self.bg_generator = AdaIngen_bg(latent_dims[0], gf_dim, f_dim=gf_dim*2, lrelu=True)
        self.obj_generator = AdaIngen_obj(latent_dims[1], latent_dims[0], gf_dim, f_dim=gf_dim*2, lrelu=True)

        self.deconv1 = nn.Sequential(nn.ConvTranspose2d(4*self.gf_dim, 2*self.gf_dim, 4, stride=2, padding=1), nn.LeakyReLU(negative_slope=0.2))
        self.deconv2 = nn.Sequential(nn.ConvTranspose2d(2*self.gf_dim, self.gf_dim, 4, stride=2, padding=1), nn.LeakyReLU(negative_slope=0.2))
        self.deconv3 = nn.Sequential(nn.ConvTranspose2d(self.gf_dim, self.gf_dim, 3, stride=1, padding=1), nn.LeakyReLU(negative_slope=0.2))
        self.deconv4 = nn.Sequential(nn.ConvTranspose2d(self.gf_dim, self.gf_dim, 4, stride=2, padding=1), nn.LeakyReLU(negative_slope=0.2))
        self.deconv5 = nn.Sequential(nn.ConvTranspose2d(self.gf_dim, self.c_dim, 3, stride=1, padding=1))

        self.upsample_net = nn.Sequential(*[getattr(self, 'deconv%u' % i) for i in range(1, 6)])

    def forward(self, batch_size, const_tensor, z, t_x, t_y, select=None):
        n_objects = z[1].size(0)

        all_objects = []
        for i in range(n_objects):
            all_objects.append(self.obj_generator(const_tensor[1], z[1][i],
                                                  t_x[i], t_y[i], z0=z[0]))

        h_BG = self.bg_generator(const_tensor[0], z[0])
        all_objects.append(h_BG)
        h2 = torch.max(torch.stack(all_objects, 0), dim=0)[0]
        output = self.upsample_net(h2)

        gen_vec = None
        if select is not None:

            out = self.upsample_net(all_objects[0])
            gen_vec = [Fu.tanh(out)]

            for i in range(1, len(all_objects)):
                out = self.upsample_net(Fu.relu(all_objects[i]))
                gen_vec.append(Fu.tanh(out))

        return Fu.tanh(output), gen_vec


class GAN_disc(nn.Module):
    def __init__(self, df_dim=64, pos_dim=90):
        super(GAN_disc, self).__init__()
        auto_init_args(self)

        def spec_conv(in_channels, out_channels, k_size=5):
            return spectral_norm(nn.Conv2d(in_channels, out_channels, k_size,
                                           stride=2, padding=k_size//2))

        self.disc_conv1 = nn.Conv2d(3, int(df_dim),
                                 5, stride=2, padding=5//2)
        self.lrelu1 = nn.LeakyReLU(negative_slope=0.2)
        self.disc_conv2 = spec_conv(int(df_dim), int(df_dim*2))
        self.inorm1 = nn.Sequential(spectral_norm(nn.InstanceNorm2d(int(df_dim*2), affine=True)), nn.LeakyReLU(negative_slope=0.2))
        self.disc_conv3 = spec_conv(int(df_dim*2), int(df_dim*4)) 
        self.inorm2 = nn.Sequential(spectral_norm(nn.InstanceNorm2d(int(df_dim*4), affine=True)), nn.LeakyReLU(negative_slope=0.2))
        self.disc_conv4 = spec_conv(int(df_dim*4), int(df_dim*8)) 
        self.inorm3 = nn.Sequential(spectral_norm(nn.InstanceNorm2d(int(df_dim*8), affine=True)), nn.LeakyReLU(negative_slope=0.2))
        self.disc_conv5 = spec_conv(int(df_dim*8), int(df_dim*16)) 
        self.inorm4 = nn.Sequential(spectral_norm(nn.InstanceNorm2d(int(df_dim*16), affine=True)), nn.LeakyReLU(negative_slope=0.2))

        # Get all linear regressors
        self.class1 = nn.Linear(int(df_dim*4), 1)
        self.class2 = nn.Linear(int(df_dim*8), 1)
        self.class3 = nn.Linear(int(df_dim*16), 1)
        self.class4 = nn.Linear(int(df_dim*32), 1)

        self.dh4 = spectral_norm(nn.Linear(int(df_dim*16*16), 1))
        self.enc = spectral_norm(nn.Linear(int(df_dim*16*16), self.pos_dim))

    def forward(self, x):
        out = self.lrelu1(self.disc_conv1(x))
        out = self.disc_conv2(out)

        l1 = self.class1(torch.cat([torch.mean(out, [2, 3]),
                         torch.var(out.view(out.size(0), out.size(1), -1), 2)], 1))
        out = self.disc_conv3(self.inorm1(out))

        l2 = self.class2(torch.cat([torch.mean(out, [2, 3]),
                         torch.var(out.view(out.size(0), out.size(1), -1), 2)], 1))
        out = self.disc_conv4(self.inorm2(out))

        l3 = self.class3(torch.cat([torch.mean(out, [2, 3]),
                         torch.var(out.view(out.size(0), out.size(1), -1), 2)], 1))
        out = self.inorm3(out)

        style_list = [l1, l2, l3]

        out = self.disc_conv5(out)
        l4 = self.class4(torch.cat([torch.mean(out, [2, 3]),
                         torch.var(out.view(out.size(0), out.size(1), -1), 2)], 1))
        out = self.inorm4(out)
        style_list.append(l4)

        out = out.view(out.size(0), -1)

        # Returning logits to determine whether the images are real or fake
        h4 = self.dh4(out)
        cont_vars = self.enc(out)

        return Fu.sigmoid(h4), h4, Fu.tanh(cont_vars),\
            style_list