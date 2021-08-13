import numpy as np
from tqdm import tqdm
from merlinpy.datapipeline.sampling import VISTA

#import medutils

def main():
    #p_list = [132, 144, 150, 156, 162, 174, 180, 186, 192]
    p_list = [150]
    mode = 'VISTA'
    #R_list = range(2, 25)
    R_list = [8, 16, 24]
    frames = 25
    sd_list = range(10, 201, 10)
    sd_list = [42]

    for p in tqdm(p_list):
        print('nFE:', p)
        dim = np.array((1, p, 1, frames, 1))
        for R in tqdm(R_list):
            print('acceleration factor is:', R)
            for sd in tqdm(sd_list):
                print('mask: %d' % (sd/10))
                mask = VISTA(dim, R, mode, sd=sd).generate_mask()
                np.savetxt("mask_%s_%dx%d_acc%d_%d.txt" % (mode, p, frames, R, sd/10), mask, fmt="%d", delimiter=",")
                #medutils.visualization.imsave(mask, f'mask_{mode}_pe{p}_frames{frames}_acc{R}_sd{sd}.png')


if __name__ == '__main__':
    main()
