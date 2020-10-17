import numpy as np
import torch


class Dummy(object):
    '''
        Dummy dataset used for evaluation
    '''
    def __init__(self, image_size=128, epoch_size=10000,
                 seq_len=1, **kwargs):
        self.N = epoch_size
        self.image_size = image_size
        self.seq_len = seq_len

    def __len__(self):
        return self.N

    def get_seq(self, index):
        image_seq = torch.from_numpy(np.zeros([3, self.image_size,
                                     self.image_size], dtype='float32'))
        if self.seq_len > 1:
            image_seq = image_seq[None].repeat(self.seq_len, 1, 1, 1)
        return image_seq

    def __getitem__(self, index):
        return {'gt_images': self.get_seq(index)}
