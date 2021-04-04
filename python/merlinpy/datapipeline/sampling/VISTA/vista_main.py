import numpy as np
from tqdm import tqdm
from VISTA_sampling import VISTA_


def main():
    p_list = [132, 144, 150, 156, 162, 174, 180, 186, 192]
    for p in tqdm(p_list):
        print('nFE:', p)
        dim = np.array((1, p, 1, 25, 1))

        for R in tqdm(range(2, 25)):
            print('acceleration factor is:', R)
            for sd in tqdm(range(10, 201, 10)):
                print('mask: %d' % (sd/10))
                mask = VISTA_(dim, R, 'VISTA', sd=sd)
                mask_VISTA = mask.generate_mask()
                np.savetxt("mask_VISTA_%dx%d_acc%d_%d.txt" % (p, 25, R, sd/10), mask_VISTA, fmt="%d", delimiter=",")


if __name__ == '__main__':
    main()
