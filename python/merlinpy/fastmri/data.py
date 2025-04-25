# to build a dataframe with header information
import glob
import xmltodict
import pandas as pd
from tqdm import tqdm
import h5py
import os

__all__ = ['get_fnames',
           'get_header_information']

def get_fnames(path, csv_file=None, data_filter={}):
    # pandas filtering
    if csv_file != None:
        df = pd.read_csv(csv_file)
    else:
        df = get_header_information(path)
    for key in data_filter.keys():
        df = df[df[key].isin(data_filter[key])]
    fnames = df.filename
    fnames = [os.path.join(path, fname.split('/')[-1]) for fname in fnames]
    return fnames

def get_header_information(path):
    # generate the file names
    image_names = sorted(glob.glob(f'{path}/*.h5'))

    # init dicts for infos that will be extracted from the dataset
    img_info = {'filename' : image_names, 'acquisition' : []}
    acq_info = {'systemVendor' : [], 'systemModel' : [], 'systemFieldStrength_T' : [], 'receiverChannels' : [], 'institutionName' : [] }
    seq_info = {'TR' : [] , 'TE' : [], 'TI': [], 'flipAngle_deg': [], 'sequence_type': [], 'echo_spacing': []}
    enc_info = {'enc_x' : [], 'enc_y' : [], 'enc_z' : [], \
            'rec_x' : [], 'rec_y' : [], 'rec_z' : [], \
            'enc_x_mm' : [], 'enc_y_mm' : [], 'enc_z_mm' : [],
            'rec_x_mm' : [], 'rec_y_mm' : [], 'rec_z_mm' : [],
            'nPE' : []}
    acc_info = {'acc' : [], 'num_low_freq' : []}
    print(f'Build dataframe for {path}')
    for fname in tqdm(image_names):
        dset =  h5py.File(fname,'r')
        acq = dset.attrs['acquisition']
        if acq == 'AXT1PRE': acq = 'AXT1'
        img_info['acquisition'].append(acq)
        acc_info['acc'].append(dset.attrs['acceleration'] if 'acceleration' in dset.attrs.keys() else 0)
        acc_info['num_low_freq'].append(dset.attrs['num_low_frequency'] if 'num_low_frequency' in dset.attrs.keys() else 0)
        ismrmrd_header = dset['ismrmrd_header'][()]
        header = xmltodict.parse(ismrmrd_header)['ismrmrdHeader']
        #pprint.pprint(header)   
        for key in acq_info.keys():
            acq_info[key].append(header['acquisitionSystemInformation'][key])
        for key in seq_info.keys():
            if key in header['sequenceParameters']:
                seq_info[key].append(header['sequenceParameters'][key])
            else:
                seq_info[key].append('n/a')
        enc_info['nPE'].append(int(header['encoding']['encodingLimits']['kspace_encoding_step_1']['maximum'])+1)
        if int(header['encoding']['encodingLimits']['kspace_encoding_step_1']['minimum']) != 0:
            raise ValueError('be careful!')
        for diridx in ['x', 'y', 'z']:
            enc_info[f'enc_{diridx}'].append(header['encoding']['encodedSpace']['matrixSize'][diridx])
            enc_info[f'rec_{diridx}'].append(header['encoding']['reconSpace']['matrixSize'][diridx])
            enc_info[f'enc_{diridx}_mm'].append(header['encoding']['encodedSpace']['fieldOfView_mm'][diridx])
            enc_info[f'rec_{diridx}_mm'].append(header['encoding']['reconSpace']['fieldOfView_mm'][diridx])

    data_info = {**img_info, **acq_info, **enc_info, **acc_info, **seq_info}

    # convert to pandas
    df = pd.DataFrame(data_info)
    return df