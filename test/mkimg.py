#!/usr/bin/env python

import imageio
import glob

anim_file = 'dcgan.gif'

filenames = sorted(glob.glob('image*.png'))
images = map(lambda filename: imageio.imread(filename), filenames)
imageio.mimsave(anim_file, images, fps=3) # modify the frame duration as needed


