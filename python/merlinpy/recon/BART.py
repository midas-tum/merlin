import os
import sys
import h5py
import numpy as np
import tempfile

def setup_bart(bart_path):
    os.environ['TOOLBOX_PATH'] = bart_path
    sys.path.append(os.path.join(bart_path, 'python'))
    try:
        from bart import bart
        return True
    except Exception:
        print(Warning("BART toolbox not setup properly or not available"))
        return False

def recon(kspace, smap, mask=None, noisy=None, bartpara='', dim='2D'):
    # kspace        raw k-space data as [batch, coils, X, Y] or [batch, coils, X, Y, Z] or [batch, coils, time, X, Y] or [batch, coils, time, X, Y, Z]
    # smap          coil sensitivity maps with same shape as kspace (or singleton dimension for time)
    # mask          subsampling including/excluding soft-weights with same shape as kspace
    # noisy         initialiaztion for reconstructed image, if None not considered
    # bartpara      BART reconstruction parameters, e.g. L1-wavelet along the spatial dimension with weighting 0.01 '-R W:7:0:0.01
    # dim           dimensionality of k-space data, '2D', '2Dt', '3D', '3Dt'

    from bart import bart
    from bart import cfl

    for i in range(kspace.shape[0]):
        kspacecurr = kspace[i, ...]
        maskcurr = mask[i, ...] if mask is not None else None
        smapcurr = smap[i, ...]
        noisycurr = noisy[i, ...] if noisy is not None else None

        if dim == '2D':
            kspacecurr = np.expand_dims(np.transpose(kspacecurr, (1, 2, 0)), 2)
            maskcurr = np.expand_dims(np.transpose(maskcurr, (1, 2, 0)), 2) if maskcurr is not None else None
            smapcurr = np.expand_dims(np.transpose(smapcurr, (1, 2, 0)), 2)
            noisycurr = np.expand_dims(np.transpose(noisycurr, (1, 2, 0)), 2) if noisycurr is not None else None
        elif dim == '2Dt':
            kspacecurr = np.expand_dims(np.expand_dims(np.transpose(kspacecurr, (2, 3, 0, 1)), 2), (4, 5, 6, 7, 8, 9))
            maskcurr = np.expand_dims(np.expand_dims(np.transpose(maskcurr, (2, 3, 0, 1)), 2), (4, 5, 6, 7, 8, 9)) if maskcurr is not None else None
            smapcurr = np.expand_dims(np.expand_dims(np.transpose(smapcurr, (2, 3, 0, 1)), 2), (4, 5, 6, 7, 8, 9))
            noisycurr = np.expand_dims(np.expand_dims(np.transpose(noisycurr, (2, 3, 0, 1)), 2), (4, 5, 6, 7, 8, 9)) if noisycurr is not None else None
        elif dim == '3D':
            kspacecurr = np.transpose(kspacecurr, (1, 2, 3, 0))
            maskcurr = np.transpose(maskcurr, (1, 2, 3, 0)) if maskcurr is not None else None
            smapcurr = np.transpose(smapcurr, (1, 2, 3, 0))
            noisycurr = np.transpose(noisycurr, (1, 2, 3, 0)) if noisycurr is not None else None
        elif dim == '3Dt':
            kspacecurr = np.expand_dims(np.transpose(kspacecurr, (2, 3, 4, 0, 1)), (4, 5, 6, 7, 8, 9))
            maskcurr = np.expand_dims(np.transpose(maskcurr, (2, 3, 4, 0, 1)), (4, 5, 6, 7, 8, 9)) if maskcurr is not None else None
            smapcurr = np.expand_dims(np.transpose(smapcurr, (2, 3, 4, 0, 1)), (4, 5, 6, 7, 8, 9))
            noisycurr = np.expand_dims(np.transpose(noisycurr, (2, 3, 4, 0, 1)), (4, 5, 6, 7, 8, 9)) if noisycurr is not None else None

        if maskcurr is not None:
            tmp = tempfile.NamedTemporaryFile(suffix='_weights')
            cfl.writecfl(tmp.name, maskcurr)
            bartpara = bartpara + ' -p ' + tmp.name

        if noisycurr is not None:
            tmp = tempfile.NamedTemporaryFile(suffix='_warmstart')
            cfl.writecfl(tmp.name, maskcurr)
            bartpara = bartpara + ' -W ' + tmp.name

        outcurr = np.squeeze(bart(1, 'pics ' + bartpara, kspacecurr, smapcurr))
        out = outcurr[np.newaxis, ...] if i == 0 else np.concatenate((out, outcurr[np.newaxis, ...]), axis=0)

    if dim == '2Dt':
        out = np.transpose(out, (0, -1, 1, 2))
    elif dim == '3Dt':
        out = np.transpose(out, (0, -1, 1, 2, 3))
    return out
