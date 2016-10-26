import os
import numpy as np
import htkmfc
import sys
import subprocess
import h5py
from fuel.datasets.hdf5 import H5PYDataset
import re
from six.moves import cPickle

hdf5file='/misc/scratch03/reco/bhattaga/data/RSR/rsr-DEVBKG-cldnn-train.hdf5'

nspk_trn=47284
nspk_val=5000

ndp = nspk_trn+nspk_val


if os.path.exists(hdf5file):
    print 'HDF5 fie exits. Deleting...'
    command00 = "rm -r" +" "+ hdf5file
    process0 = subprocess.check_call(command00.split())

f=h5py.File(hdf5file, mode='w')

features = f.create_dataset('features', (ndp,400,382), dtype='float32')
masks = f.create_dataset('masks',(ndp,382), dtype='float32')
names = f.create_dataset('names', (ndp,), dtype='S150')
labels = f.create_dataset('labels',(ndp,), dtype='int32')

#label the dimensions
for i, label in enumerate(('batch','timesteps', 'dimension')):
      f['features'].dims[i].label = label
for i, label in enumerate(('batch','timesteps')):
      f['masks'].dims[i].label = label
for i, label in enumerate(('batch',)):
      f['labels'].dims[i].label = label
for i, label in enumerate(('batch',)):
      f['names'].dims[i].label = label

nspk_all = nspk_trn+nspk_val

split_dict = {'train': {'features': (0, nspk_trn), 'labels': (0, nspk_trn),
              'masks':(0,nspk_trn), 'names': (0, nspk_trn)},
              'valid': {'features': (nspk_trn, nspk_all), 'labels': (nspk_trn, nspk_all),
              'masks':(nspk_trn,nspk_all),'names': (nspk_trn,nspk_all)}}


DATA=np.load('/misc/scratch03/reco/bhattaga/data/RSR/rsr-mf-DEVBKG-cldnn.npy')
LABELS = np.load('/misc/scratch03/reco/bhattaga/data/RSR/rsr-mf-spkphr-labels-cldnn.npy')
fname = open('/misc/scratch03/reco/bhattaga/data/RSR/rsr-mf-BKG-names-cldnn.pkl')
NAMES = cPickle.load(fname)
MASKS = np.load('/misc/data15/reco/bhattgau/Rnn/code/Code_/rsr/newshit/DATA/rsr-mf-p1-DEVBKG-masks.npy')

features[...] = DATA
labels[...] = LABELS
masks[...] = MASKS
names[...] = NAMES

f.attrs['split'] = H5PYDataset.create_split_array(split_dict)
f.flush()
f.close()
