#!/usr/bin/python
import PIL
from PIL import Image
import os, sys
import argparse
from os.path import join

def resize(path, out_path, size):
    dirs = os.listdir(path)
    for item in dirs:
        if os.path.isfile(join(path,item)):
            im = Image.open(join(path,item))
            f, e = os.path.splitext(join(path,item))
            imResize = im.resize((size,size), PIL.Image.LANCZOS)
            imResize.save(join(out_path,item))

if __name__ == '__main__':
    # Get arguments
    parser = argparse.ArgumentParser('Resize scripts main arguments')
    parser.add_argument('--image_dir', required=True, type=str, help='Image Directory')
    parser.add_argument('--output_dir', required=True, type=str, help='Output Directory')
    parser.add_argument('--size', type=int, default=128, help='To size')
    args = parser.parse_args()

    # Create output directory file
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Resize set
    resize(args.image_dir, args.output_dir, args.size)