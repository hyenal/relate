"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import os
import torch
import torch.nn.functional as Fu
import torchvision
import numpy as np
from tools.utils import auto_init_args, weight_init, choice_exclude,\
     get_visdom_connection, save_gif, gradient_penalty

from torch import nn
from torch.nn.utils.spectral_norm import spectral_norm
from torch.nn.parameter import Parameter
from models.relate_helpers import AdaIngen_obj, AdaIngen_bg, NPE


class RELATEVideo(torch.nn.Module):
    def __init__(self,
                 x_max=5.,
                 y_max=5.,
                 n_objects=2,
                 backgd_dim=30,
                 obj_dim=30,
                 seq_len=5,
                 fixed_objs=False,
                 obj_size=6,
                 past=3,
                 len_ev=None,
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
                     'l_gradient',
                     ],
                 **kwargs):
        super(RELATEVideo, self).__init__()

        # autoassign constructor params to self
        auto_init_args(self)

        # Use gpu if available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.latent_dims=[self.backgd_dim, self.obj_dim]
        self.grad_pen=(loss_weights.get('l_gradient',0)>0.)

        self.zbg = Parameter(0.02*torch.randn([1, 4*64, 16, 16], device=self.device))
        self.zfg = Parameter(0.02*torch.randn([1+self.fixed_objs*(self.n_objects-1), 8*64,
                             self.obj_size, self.obj_size], device=self.device))

        # Init generator
        self.generator = GAN_gen(latent_dims=[self.latent_dims[0],self.latent_dims[1]+2])

        # Init discriminator
        self.discriminator  = GAN_disc(pos_dim=2, first=self.seq_len)

        # Init Gamma
        self.Gamma = NPE(self.latent_dims[1]+(self.past-1)*self.latent_dims[1]*(1-int(self.fixed_objs))+2*(self.past-1),
                         self.latent_dims[0], out_dim=self.latent_dims[1]*(1-int(self.fixed_objs)))

        # Init misc MLPs
        self.mlp_speed = nn.Sequential(nn.Linear(self.latent_dims[1] + 2 + self.latent_dims[0], 128), nn.LeakyReLU(negative_slope=0.2),
                                       nn.Linear(128, 128), nn.LeakyReLU(negative_slope=0.2), nn.Linear(128,  2*past), nn.Tanh())

        self.mlp_xy = NPE(self.latent_dims[1], self.latent_dims[0])

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

        # finally get the optimization objective using self.loss_weights
        self.preds['objective'] = self.get_objective(self.preds)

        # Save images: one batch during training and all at eval time
        if kwargs['trainmode'] == 'val':
            os.makedirs(exp_dir+'/videos_val', exist_ok=True)
            for i in range(gen_im.size(0)):
                save_gif(exp_dir+'/videos_val/%06u.gif' %
                         (it*self.batch_size+i),
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
                        {'params': self.Gamma.parameters()},
                        {'params': self.mlp_xy.parameters()},
                        {'params': self.mlp_speed.parameters()}]}

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
            for i in range(2):
                image = torch.clamp(preds[image_type][:, - i], -.5, .5)
                image = (image + .5)
                image = torchvision.utils.make_grid(image, nrow=8).data.cpu().numpy()
                viz.image(image, env=visdom_env_imgs,
                          opts={'title': title+"_%s_%d" % (image_type, - i)})

    def run_model(self, x, gen=False, validation=False):
        batch_size = x.size(0)
        loss = {}

        # Sample different variables
        if not validation:
            self.background_vec, self.appearance_vec = 2*torch.rand([batch_size, self.latent_dims[0]], device=self.device)-1,\
                        2*torch.rand([1, batch_size, self.n_objects*self.latent_dims[1]], device=self.device)-1
        else:
            self.background_vec, self.appearance_vec = 1.5*torch.rand([batch_size, self.latent_dims[0]], device=self.device)-0.75,\
                        2*torch.rand([1, batch_size, self.n_objects*self.latent_dims[1]], device=self.device)-1

        self.tx = (2*self.x_max*torch.rand([self.n_objects*2, batch_size, 1], device=self.device)-self.x_max) / 8.
        self.ty = (2*self.y_max*torch.rand([self.n_objects*2, batch_size, 1], device=self.device)-self.y_max) / 8.

        if not self.fixed_objs:
            # Sample n_objects - numbers of objects
            n_obj = np.random.randint(0, self.n_objects, batch_size)
            choice = [np.random.choice(self.n_objects, n_obj[i], replace=False)
                      for i in range(batch_size)]
            select_object = torch.from_numpy(np.array([choice_exclude(self.n_objects, exc) for exc in choice])).long()
            for i in range(batch_size):
                for j in choice[i]:
                    self.tx[j, i, 0] = -2.
                    self.ty[j, i, 0] = -2.
        else:
            n_obj = self.n_objects
            choice = [[] for i in range(batch_size)]
            select_object = torch.from_numpy(np.array([choice_exclude(self.n_objects, exc) for exc in choice])).long()
            self.appearance_vec = torch.ones_like(self.appearance_vec)

        # Compute adjustment to position
        xy = self.mlp_xy(torch.cat((self.appearance_vec.view(batch_size, self.n_objects, self.latent_dims[1]).permute(1,0,2),\
                self.tx[:self.n_objects], self.ty[:self.n_objects], self.tx[:self.n_objects]*0, self.ty[:self.n_objects]*0),2), self.background_vec)
        tx, ty = self.tx, self.ty

        self.tx = 0.5*xy[:, :, :1] + tx[:self.n_objects]
        self.ty = 0.5*xy[:, :, 1:] + ty[:self.n_objects]

        self.tx = torch.clamp(self.tx.reshape(self.n_objects,batch_size,1).contiguous()[:,:,:],-.8,.8) - 1.*(self.tx.reshape(self.n_objects,batch_size,1).contiguous()[:,:,:] < -1.5).float()
        self.ty = torch.clamp(self.ty.reshape(self.n_objects,batch_size,1).contiguous()[:,:,:],-.8,.8) - 1.*(self.tx.reshape(self.n_objects,batch_size,1).contiguous()[:,:,:] < -1.5).float()

        self.appearance_vec = self.appearance_vec.view(batch_size, self.n_objects, self.latent_dims[1]).permute(1,0,2)

        # Adjust velocity
        for i in range(self.n_objects):
            v_n = self.mlp_speed(torch.cat((self.appearance_vec[i], self.tx[i].detach(), self.ty[i].detach(), self.background_vec),1)) * (self.tx[i] > -1.5).float()
            if i == 0:
                self.v = v_n.unsqueeze(0)
            else:
                self.v = torch.cat((self.v, v_n[None]),0)

        self._vx, self._vy = [self.v[:,:,i:i+1] for i in range(self.past)], [self.v[:,:,self.past+i:self.past+i+1] for i in range(self.past)]        

        tx_tmp = self.tx
        ty_tmp = self.ty

        self.appearance_vec = [self.appearance_vec for _ in range(self.past)]

        len_s = self.len_ev if validation else 20
        for i in range(len_s):
            new_sv = self.Gamma(torch.cat((torch.cat(self.appearance_vec[-self.past+(self.past-1)*int(self.fixed_objs):],2), tx_tmp, ty_tmp, torch.cat(self._vx[-self.past:],2), torch.cat(self._vy[-self.past:],2)),2), self.background_vec)
            self._vx.append(new_sv[:, :, :1])
            self._vy.append(new_sv[:, :, 1:2])
            tx_tmp = tx_tmp + self._vx[-1]
            ty_tmp = ty_tmp + self._vy[-1]
            if not self.fixed_objs:
                self.appearance_vec.append(0*new_sv[:,:,2:]+self.appearance_vec[-1])
            else:
                self.appearance_vec.append(self.appearance_vec[-1])
        self._vx = torch.stack(self._vx[self.past-1:],0)
        self._vy = torch.stack(self._vy[self.past-1:],0)
        self.appearance_vec = torch.stack(self.appearance_vec[self.past-1:],0)

        self._vx[0], self._vy[0] = 0*self._vx[0], 0*self._vy[0]        
        self._vx = self._vx.cumsum(0)
        self._vy = self._vy.cumsum(0)

        # Randomly select starting point
        select_start = torch.from_numpy(np.random.randint(0,20-self.seq_len-1, batch_size)).long().view(1,1,batch_size,1).repeat(1,self.n_objects,1,1)
        if torch.cuda.is_available():
            select_start = select_start.cuda()
        if validation:
            select_start = (select_start*0 + 1.).long()
        range_p = self.seq_len if self.len_ev is None else self.len_ev
        self._vx = torch.cat([torch.gather(self._vx, 0, select_start+i) for i in range(range_p)], 0)
        self._vy = torch.cat([torch.gather(self._vy, 0, select_start+i) for i in range(range_p)], 0)
        self.appearance_vec = torch.cat([torch.gather(self.appearance_vec, 0, select_start.repeat(1,1,1,self.latent_dims[1])+i) for i in range(range_p)],0)

        # Run encoder first for each positions
        gen_images, gen_vec = [], []
        for i in range(range_p):
            gen_images_t, gen_vec_t = self.generator(batch_size, [self.zbg, self.zfg], [self.background_vec, torch.cat([self.appearance_vec[i],0*(self.tx+self._vx[:i+1].sum(0)).detach(),0*(self.ty+self._vy[:i+1].sum(0)).detach()],2)],\
                self.tx+self._vx[i], self.ty+self._vy[i], select=[select_object] if not i else None)
            gen_images.append(gen_images_t)
            gen_vec.append(gen_vec_t)

        # In case we generate a super resolution
        # if gen_images.shape[2]!=x.shape[2] or gen_images.shape[3]!=x.shape[3]:
        #     gen_images = Fu.interpolate(gen_images, size=(x.shape[2], x.shape[3]))

        # Position predictor
        # self.discriminator.dh4.hidden = self.discriminator.dh4.init_hidden(batch_size)
        _, _, gen_xy, _ = self.discriminator(torch.cat((gen_vec[0],gen_vec[0].repeat(1,self.seq_len-1,1,1).detach()),1))
        pos_tensor = torch.cat((self.tx.detach()[select_object,torch.arange(batch_size)]+self._vx[0][select_object,torch.arange(batch_size)],\
            self.ty.detach()[select_object,torch.arange(batch_size)]+self._vy[0][select_object,torch.arange(batch_size)]),1).contiguous().view(-1,2)
        loss['xy'] = torch.mean((gen_xy-pos_tensor)**2) 


        # Run discriminator on real/fake images
        _, gen_logits, _, gen_style = self.discriminator(torch.cat((gen_images[:self.seq_len]),1))
        
        if not gen or validation:
            _, x_logits, _, disc_style   = self.discriminator(torch.cat(([x[:,i] for i in range(self.seq_len)]), 1))

            loss['l_disc'] = self.loss_bce(x_logits, torch.ones_like(x_logits))
            loss['l_gen'] = self.loss_bce(gen_logits, torch.zeros_like(gen_logits))

            loss['l_style_disc'] = sum([self.loss_bce(z, torch.ones_like(z)) for z in disc_style])
            loss['l_style_gen'] = sum([self.loss_bce(z, torch.zeros_like(z)) for z in gen_style])

        if gen or validation:
            loss['l_gen_eval'] = self.loss_bce(gen_logits, torch.ones_like(gen_logits))
            loss['l_style_gen_eval'] = sum([self.loss_bce(z, torch.ones_like(z)) for z in gen_style])

        return loss, torch.stack(gen_images,0).permute(1,0,2,3,4)

class GAN_gen(nn.Module):
    def __init__(self, latent_dims=[90,30], gf_dim=64, c_dim=3):
        super(GAN_gen, self).__init__()
        auto_init_args(self)

        s_h, s_w = 64, 64
        s_h2, s_w2 = 32, 32
        s_h4, s_w4 = 16, 16

        self.bg_generator = AdaIngen_bg(latent_dims[0], gf_dim,
                                        f_dim=gf_dim*2, lrelu=True)
        self.obj_generator = AdaIngen_obj(latent_dims[1], latent_dims[0], gf_dim,
                                          f_dim=gf_dim*2, lrelu=True)
 
        self.deconv1 = nn.Sequential(nn.ConvTranspose2d(4*self.gf_dim, 2*self.gf_dim, 4, stride=2, padding=1), nn.LeakyReLU(negative_slope=0.2))
        self.deconv2 = nn.Sequential(nn.ConvTranspose2d(2*self.gf_dim, self.gf_dim, 4, stride=2, padding=1), nn.LeakyReLU(negative_slope=0.2))
        self.deconv3 = nn.Sequential(nn.ConvTranspose2d(self.gf_dim, self.gf_dim, 3, stride=1, padding=1), nn.LeakyReLU(negative_slope=0.2))
        self.deconv4 = nn.Sequential(nn.ConvTranspose2d(self.gf_dim, self.gf_dim, 4, stride=2, padding=1), nn.LeakyReLU(negative_slope=0.2))
        self.deconv5 = nn.Sequential(nn.ConvTranspose2d(self.gf_dim, self.c_dim, 3, stride=1, padding=1))

        self.upsample_net = nn.Sequential(*[getattr(self,'deconv%u'%i) for i in range(1,6)])

    def forward(self, batch_size, const_tensor, z, t_x, t_y, select=[]):
        n_objects = z[1].size(0)
        if const_tensor[1].size(0) > 1:
            comp_func = lambda x : torch.sum(x,dim=0)
        else:
            comp_func = lambda x : torch.max(x,dim=0)[0]

        all_objects = []
        for i in range(n_objects):
            if const_tensor[1].size(0) > 1 :
                all_objects.append(self.obj_generator(const_tensor[1][i:i+1], z[1][i], t_x[i], t_y[i]))
            else:
                all_objects.append(self.obj_generator(const_tensor[1], z[1][i], t_x[i], t_y[i], z0=z[0]))

        h_BG = self.bg_generator(const_tensor[0], z[0])
        all_objects.append(h_BG)

        h2 = comp_func(torch.stack(all_objects, 0))
        output = self.upsample_net(h2)

        gen_vec = None
        if select:
            select_objs = torch.stack(all_objects[:-1], 0)[select[0], torch.arange(batch_size)]
            h2 = comp_func(torch.stack([select_objs, all_objects[-1]], 0))
            out = self.upsample_net(h2)
            gen_vec = Fu.tanh(out)

        return Fu.tanh(output), gen_vec

class GAN_disc(nn.Module):
    def __init__(self, df_dim=64, pos_dim=90, first=5):
        super(GAN_disc, self).__init__()
        auto_init_args(self)
        
        def spec_conv(in_channels, out_channels, k_size=5):
            return spectral_norm(nn.Conv2d(in_channels, out_channels,
                                 k_size, stride=2, padding=k_size//2))

        self.disc_conv1 = nn.Conv2d(3*first,int(df_dim),5,stride=2,padding=5//2)
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
        self.class1 = nn.Linear(int(df_dim*4),1)
        self.class2 = nn.Linear(int(df_dim*8),1)
        self.class3 = nn.Linear(int(df_dim*16),1)
        self.class4 = nn.Linear(int(df_dim*32),1)

        self.dh4 = spectral_norm(nn.Linear(int(df_dim*16*16), 1))
        self.enc = spectral_norm(nn.Linear(int(df_dim*16*16), self.pos_dim))


    def forward(self, x):
        out = self.lrelu1(self.disc_conv1(x))
        out = self.disc_conv2(out)

        l1 = self.class1(torch.cat([torch.mean(out,[2,3]),\
            torch.var(out.view(out.size(0), out.size(1), -1) ,2)],1))
        out = self.disc_conv3(self.inorm1(out))

        l2 = self.class2(torch.cat([torch.mean(out,[2,3]), \
            torch.var(out.view(out.size(0), out.size(1), -1) ,2)],1))
        out = self.disc_conv4(self.inorm2(out))

        l3 = self.class3(torch.cat([torch.mean(out,[2,3]), \
            torch.var(out.view(out.size(0), out.size(1), -1) ,2)],1))
        out = self.inorm3(out)

        style_list = [l1, l2, l3]

        out = self.disc_conv5(out)
        l4 = self.class4(torch.cat([torch.mean(out,[2,3]), \
            torch.var(out.view(out.size(0), out.size(1), -1) ,2)],1))
        out = self.inorm4(out)
        style_list.append(l4)

        out = out.view(out.size(0), -1)
        
        #Returning logits to determine whether the images are real or fake
        h4 = self.dh4(out)
        cont_vars = self.enc(out)

        return Fu.sigmoid(h4), h4, Fu.tanh(cont_vars), style_list