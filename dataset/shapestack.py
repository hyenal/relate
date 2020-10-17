import os
import numpy as np
from PIL import Image
import PIL
import torch


class ShapeStack(object):
    """Data Handler that loads ShapeStack data."""

    def __init__(self, data_root, train=True, image_size=64, seq_len=1,
                 epoch_size=100000, **kwargs):
        self.root_dir = data_root
        self.data_dir = os.path.join(self.root_dir, 'recordings/')
        self.test_dir = []
        self.train = train

        self.test_split = os.path.join(self.root_dir, 'splits/default/test.txt')
        with open(self.test_split) as fp:
            for line in fp:
                line = line.strip('\n')
                self.test_dir.append(line)

        if train:
            self.dir = [self.data_dir+f for f in os.listdir(self.data_dir)
                        if f[0] == 'e' and f not in self.test_dir]
        else:
            self.dir = [self.data_dir+f for f in self.test_dir]

        self.dirs = []
        for i in range(len(self.dir)):
            dir_name = self.dir[i]
            if os.path.exists(dir_name) and \
               int(dir_name.split('-h=')[-1][0]) <= 5:
                name = [dir_name+'/'+f for f in os.listdir(dir_name)
                        if f[-4:] == '.png' and
                        (f[-13] == '_' or int(f[-12]) < 2)]
                if len(name) > 0:
                    # Look at the h option and load only 2...
                    self.dirs.append(name)

        self.seq_len = seq_len
        self.image_size = image_size
        self.seed_is_set = False  # multi threaded loading
        self.d = 0
        self.N = int(epoch_size)

    def set_seed(self, seed):
        if not self.seed_is_set:
            self.seed_is_set = True
            np.random.seed(seed)

    def __len__(self):
        return self.N

    def get_seq(self, index):
        d = self.dirs[index % (len(self.dirs))]
        image_seq = []
        # l_seq = len([f for f in os.listdir(d) if f[-4:]=='.png'])
        # name = [f for f in os.listdir(d) if f[-4:]=='.png'][0][:36]
        if not self.train:
            np.random.seed(index)
        start = np.random.randint(0, len(d))

        # for i in range(start, start+self.seq_len):
        #     fname = '%s/%s%d.png' % (d,name,i)
        im = (np.asarray(Image.open(d[start]).resize((self.image_size,
              self.image_size), PIL.Image.LANCZOS)).reshape(1,
              self.image_size, self.image_size, 3).astype('float32')
              - 127.5) / 255

        image_seq.append(im)
        image_seq = np.concatenate(image_seq, axis=0)
        if self.seq_len == 1:
            image_seq = \
                torch.from_numpy(image_seq[0, :, :, :]).permute(2, 0, 1)
        return image_seq

    def __getitem__(self, index):
        self.set_seed(index)
        return {'gt_images': self.get_seq(index)}
