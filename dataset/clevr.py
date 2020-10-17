"""
Data loader for CLEVR_v1.0
Website: https://cs.stanford.edu/people/jcjohns/clevr/
Paper: https://arxiv.org/pdf/1612.06890.pdf
Data: https://dl.fbaipublicfiles.com/clevr/CLEVR_v1.0.zip
Code: https://github.com/facebookresearch/clevr-dataset-gen
"""

import os
import json

from PIL import Image
import PIL
import numpy as np
import torch


class CLEVR_v1(object):
    """Data handler which loads the CLEVR images."""

    def __init__(
            self,
            data_root, train=True, image_size=64,
            num_objects_min=3, num_objects_max=6,
            **kwargs):
        self.root_dir = data_root
        self.image_files = []
        self.image_annotations = []
        if train:  # merge train and val splits
            for split in ['train', 'val']:
                # load file names and annotations
                self.image_files.extend([
                        os.path.join(self.root_dir, 'images', split, fn) \
                        for fn in sorted(os.listdir(os.path.join(data_root,
                                         'images', split)))
                ])
                with open(os.path.join(data_root, 'scenes', 'CLEVR_%s_scenes.json' % split)) as fp:
                    train_scenes = json.load(fp)['scenes']
                self.image_annotations.extend(train_scenes)
            # filter by number of objects
            obj_cnt_list = list(
                    zip(
                            self.image_files,
                            [len(scn['objects']) for scn
                             in self.image_annotations]
                    )
            )
            obj_cnt_filter_index = [
                    t[1] >= num_objects_min and t[1] <= num_objects_max
                    for t in obj_cnt_list
            ]
            # apply filter
            self.image_files = [self.image_files[idx] for idx, t
                                in enumerate(obj_cnt_filter_index) if t]
            self.image_annotations = [self.image_annotations[idx] for idx, t
                                      in enumerate(obj_cnt_filter_index) if t]
            print(">>> Loaded %d images from split 'train+val'." % len(self.image_files))
        else:  # load test data only
            self.image_files.extend([
                     os.path.join(self.root_dir, 'images', 'test', fn)
                     for fn in sorted(os.listdir(os.path.join(data_root, 'images', 'test')))
            ])
            print(">>> Loaded %d images from split 'test'." % len(self.image_files))
        self.image_size = image_size
        self.seed_is_set = False
        self.N = len(self.image_files)
        self.seq_len = 1

    def set_seed(self, seed):
        if not self.seed_is_set:
            self.seed_is_set = True
            np.random.seed(seed)

    def __len__(self):
        return self.N

    def get_seq(self, index):
        image_seq = []
        image_file = self.image_files[index]
        im = (np.asarray(Image.open(image_file).convert('RGB').resize(
              (self.image_size, self.image_size), PIL.Image.LANCZOS)).reshape(
              1, self.image_size, self.image_size, 3).astype(
              'float32') - 127.5) / 255
        image_seq.append(im)
        image_seq = np.concatenate(image_seq, axis=0)
        if self.seq_len == 1:
            image_seq = torch.from_numpy(image_seq[0, :, :, :]).permute(2, 0, 1)
        return image_seq

    def __getitem__(self, index):
        self.set_seed(index)
        return {'gt_images': self.get_seq(index)}
