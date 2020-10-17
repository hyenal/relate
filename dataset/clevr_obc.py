import os
import numpy as np
from PIL import Image
import PIL
import torch


class CLEVROBC(object):
    def __init__(self, data_root, train=True, image_size=64, **kwargs):
        self.root_dir = data_root
        self.train = train
        if train:
            self.data_dir = os.path.join(self.root_dir, 'train/')
            self.ordered = False
        else:
            self.data_dir = os.path.join(self.root_dir, 'test/')
            self.ordered = True
        self.dirs = []
        for d1 in os.listdir(self.data_dir):
            if d1[-4:] == '.png':
                self.dirs.append(os.path.join(self.data_dir, d1))

        self.image_size = image_size
        self.seed_is_set = False  # multi threaded loading
        self.N = len(self.dirs)

    def set_seed(self, seed):
        if not self.seed_is_set:
            self.seed_is_set = True
            np.random.seed(seed)

    def __len__(self):
        return self.N

    def get_seq(self, index):
        d = self.dirs[index % (len(self.dirs))]
        image_seq = []
        im = (np.asarray(Image.open(d).convert('RGB')\
              .resize((self.image_size, self.image_size), PIL.Image.LANCZOS))\
              .reshape(1, self.image_size, self.image_size, 3)\
              .astype('float32') \
              - 127.5) / 255
        image_seq.append(im)
        image_seq = np.concatenate(image_seq, axis=0)
        image_seq = torch.from_numpy(image_seq[0, :, :, :]).permute(2, 0, 1)
        return image_seq

    def __getitem__(self, index):
        self.set_seed(index)
        return {'gt_images': self.get_seq(index)}
