#!/usr/bin/env python

import imageio
import glob

anim_file = 'dcgan.gif'

with imageio.get_writer(anim_file, mode='I') as writer:
    filenames = sorted(glob.glob('image*.png'))
    last = -1
    for i, filename in enumerate(filenames):
        #frame = 2*(i**0.5)
        #if round(frame) > round(last):
        #    last = frame
        #else:
        #    continue
        image = imageio.imread(filename)
        writer.append_data(image)
