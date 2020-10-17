import torch
from torch import nn
import torch.nn.functional as Fu


class AdaIn(nn.Module):
    def __init__(self, latent_dims=128, gf_dim=64):
        super(AdaIn, self).__init__()
        self.gf_dim = gf_dim
        self.mlp = nn.Sequential(nn.Linear(latent_dims, 2*gf_dim), nn.ReLU())
        self.instNorm = nn.InstanceNorm2d(gf_dim, affine=False)

    def forward(self, x, z):
        s = self.mlp(z)
        out = self.instNorm(x)
        out = s[:, :self.gf_dim, None, None]*out\
            + s[:, self.gf_dim:, None, None]
        return out


class AdaIngen_obj(nn.Module):
    def __init__(self, latent_dims=128, latent_dims_env=1, gf_dim=64,
                 f_dim=64, lrelu=False, upsample=False):
        super(AdaIngen_obj, self).__init__()
        self.upsample = upsample


        if not lrelu:
            self.act_fn = lambda x: Fu.relu(x)
            act_fn_in = nn.ReLU
        else:
            self.act_fn = lambda x: Fu.leaky_relu(x, negative_slope=1.0)
            act_fn_in = nn.LeakyReLU(negative_slope=0.2)

        self.style1 = AdaIn(latent_dims, gf_dim*8)
        self.deconv1 = nn.Sequential(nn.ConvTranspose2d(gf_dim*8, f_dim*4, 3, stride=1, padding=1, output_padding=0), act_fn_in)
        self.style2 = AdaIn(latent_dims, 4*f_dim)
        self.deconv2 = nn.Sequential(nn.ConvTranspose2d(4*f_dim, 2*f_dim, 3+int(upsample), stride=1+int(upsample),
                                     padding=1, output_padding=0), act_fn_in)
        self.style3 = AdaIn(latent_dims, 2*f_dim)

        self.scale = nn.Sequential(nn.Linear(latent_dims_env+latent_dims+2, 32), nn.LeakyReLU(negative_slope=0.2),
                                   nn.Linear(32, 32), nn.LeakyReLU(negative_slope=0.2), nn.Linear(32, 1), nn.Tanh())

    def forward(self, const_tensor, z, tx, ty, z0=None):
        out = self.style1(const_tensor, z)
        out = self.act_fn(out)

        out = self.deconv1(out)
        out = self.style2(out, z)
        out = self.act_fn(out)

        out = self.deconv2(out)
        out = self.style3(out, z)
        out = self.act_fn(out)

        out = Fu.pad(out, pad=((16*(1+self.upsample)-out.size(2))//2, (16*(1+self.upsample)-out.size(2))//2,
                               (16*(1+self.upsample)-out.size(3))//2, (16*(1+self.upsample)-out.size(3))//2),
                     mode='constant', value=0)

        # Do the affine transform here
        offset = torch.cat((tx, ty), 1)

        rot_mat = torch.eye(2)
        if torch.cuda.is_available():
            rot_mat = rot_mat.cuda()

        rot_mat = rot_mat[None, :, :].repeat(out.size(0), 1, 1)

        if z0 is not None:
            scale = 1+2*self.scale(torch.cat([z0, z, tx, ty], 1))
            rot_mat = scale[:, :, None].repeat(1, 2, 2) * rot_mat

        rot_mat = torch.cat((rot_mat, offset[:, :, None]), 2)

        # Create grid and rotate images
        grid = Fu.affine_grid(rot_mat, out.size())
        out = Fu.grid_sample(out, grid, mode='bilinear')

        return out


class AdaIngen_bg(nn.Module):
    def __init__(self, latent_dims=128, gf_dim=64, f_dim=64, lrelu=False):
        super(AdaIngen_bg, self).__init__()

        if not lrelu:
            self.act_fn = lambda x: Fu.relu(x)
            act_fn_in = nn.ReLU
        else:
            self.act_fn = lambda x: Fu.leaky_relu(x, negative_slope=0.2)
            act_fn_in = nn.LeakyReLU(negative_slope=1.0)

        self.style1 = AdaIn(latent_dims, gf_dim*4)
        self.deconv1 = nn.Sequential(nn.ConvTranspose2d(gf_dim*4, 4*f_dim, 3, stride=1, padding=1, output_padding=0), act_fn_in)
        self.style2 = AdaIn(latent_dims, 4*f_dim)
        self.deconv2 = nn.Sequential(nn.ConvTranspose2d(4*f_dim, 2*f_dim, 3, stride=1, padding=1, output_padding=0), act_fn_in)
        self.style3 = AdaIn(latent_dims, 2*f_dim)

    def forward(self, const_tensor, z):
        out = self.style1(const_tensor, z)
        out = self.act_fn(out)

        out = self.deconv1(out)
        out = self.style2(out, z)
        out = self.act_fn(out)

        out = self.deconv2(out)
        out = self.style3(out, z)
        out = self.act_fn(out)

        return out


class NPE(nn.Module):
    def __init__(self, latent_dims, env_dim=0, history=1, out_dim=0):
        super(NPE, self).__init__()

        self.interaction = nn.Sequential(nn.Linear(history*2*(latent_dims+4), 32),
                                         nn.LeakyReLU(negative_slope=0.2), nn.Linear(32, 32),
                                         nn.LeakyReLU(negative_slope=0.2), nn.Linear(32, 32))
        self.v = nn.Sequential(nn.Linear(32+(latent_dims+4)*history+env_dim, 32),
                               nn.LeakyReLU(negative_slope=0.2), nn.Linear(32, 32),
                               nn.LeakyReLU(negative_slope=0.2), nn.Linear(32, 2+out_dim), nn.Tanh())

    def forward(self, stylexyvxy, env=None):
        n_objects = stylexyvxy.size(0)
        out = []
        for i in range(n_objects):
            curr = stylexyvxy[i]
            val = 0.
            for j in range(n_objects):
                if i != j:
                    val += (stylexyvxy[j, :, -4:-3] > -1).float()*self.interaction(torch.cat((curr, stylexyvxy[j]), 1))
            if env is None:
                out.append((stylexyvxy[i, :, -4:-3] > -1).float()*self.v(torch.cat((curr, val), 1)) - (1-(stylexyvxy[i, :, -4:-3] > -1).float())*2)
            else:
                out.append((stylexyvxy[i, :, -4:-3] > -1).float()*self.v(torch.cat((curr, val, env), 1)) - (1-(stylexyvxy[i, :, -4:-3] > -1).float())*2)
        return torch.stack(out, 0)
