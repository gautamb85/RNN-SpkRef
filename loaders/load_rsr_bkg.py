import htkmfc
import sys
import numpy as np
import os
import re
from six.moves import cPickle
import h5py
from fuel.datasets.hdf5 import H5PYDataset
#count the number of chunks in a file
#bkg set

f1=open('/misc/data15/reco/bhattgau/Rnn/code/Code_/rsr/newshit/lists/seq2seq/rsr-p1-mf-devbkg.trn')
tfeats = [l.strip() for l in f1.readlines()]

f2=open('/misc/data15/reco/bhattgau/Rnn/code/Code_/rsr/newshit/lists/seq2seq/rsr-p1-mf-devbkg.val')
vfeats = [l.strip() for l in f2.readlines()]

dpth='/misc/scratch02/reco/alamja/workshop_pdt_2015/features/Nobackup/mel_fbank_fb40_fs8k_cmvn/rsr2015/VQ_VAD_HO_EPD/'

tfinames = [re.split('[/.]',l)[2] for l in tfeats]
tspk_phr = list(set([l.split('_')[0]+'_'+l.split('_')[2] for l in tfinames]))

all_feats = tfeats+vfeats

frame_count=[]
LABELS=[]
NAMES=[]
MASKS=[]

#find maxlen recording in training set
"""
for l in all_feats:
  spk_id = re.split('[/_.]',l)[2]+'_'+re.split('[/_.]',l)[4]
  lab = tspk_phr.index(spk_id)
  LABELS.append(lab)
  fname = re.split('[/.]',l)[2]
  NAMES.append(fname)

  fbfeat = os.path.join(dpth,l)
  
  if os.path.exists(fbfeat):
    ff = htkmfc.HTKFeat_read(fbfeat)
    data = ff.getall()
    
    nframes=data.shape[0]
    frame_count.append(nframes)

maxl = max(frame_count)
print maxl
"""

maxl = 382
slc=10

### pad and store in matrices
rsr_train_data = []

for l in all_feats:
  
  fbfeat = os.path.join(dpth,l)
  
  spk_id = re.split('[/_.]',l)[2]+'_'+re.split('[/_.]',l)[4]
  lab = tspk_phr.index(spk_id)
  fname = re.split('[/.]',l)[2]

  if os.path.exists(fbfeat):
    
    LABELS.append(lab)
    NAMES.append(fname)
    mask = np.zeros((382,),dtype='float32')
    
    ff = htkmfc.HTKFeat_read(fbfeat)
    data = ff.getall()
    
    nframes=data.shape[0]
    
    mask[:nframes] = 1.0
    MASKS.append(mask)

    padl = maxl - nframes
    pad = np.zeros((padl,40),dtype='float32')

    datapad = np.vstack((data,pad))
    nframes = datapad.shape[0]

    ptr=0
#give each frame a forward-backward context of 5 frames
    cnnstack = np.empty((400,nframes),dtype='float32')
    
    while ptr < nframes:

      if ptr+slc >= nframes:
        padl = ptr+slc-nframes + 1 
        pad=np.zeros((padl,40),dtype='float32')
        dslice = datapad[ptr: ptr+slc-padl]
        dslice = np.vstack((dslice,pad))
        dslice = dslice.flatten()

      else:
        dslice = datapad[ptr:ptr+slc,:]
        dslice = dslice.flatten()

      cnnstack[:,ptr]=dslice
      ptr+=1

    rsr_train_data.append(cnnstack)

rsr_train_data = np.asarray(rsr_train_data,dtype='float32')
nlen = len(rsr_train_data)

rsr_train_data = np.reshape(rsr_train_data,(nlen,382,400))
LABELS = np.asarray(LABELS,dtype='int32')

mlen = len(MASKS)
if mlen != nlen:
  print 'shit'

MASKS = np.asarray(MASKS,dtype='float32')
MASKS = np.reshape(MASKS,(mlen,382))

hdf5file='/misc/scratch03/reco/bhattaga/data/RSR/rsr-DEVBKG-cldnn-train.hdf5'

nspk_trn=47284
nspk_val=5000

ndp = nspk_trn+nspk_val


if os.path.exists(hdf5file):
    print 'HDF5 fie exits. Deleting...'
    command00 = "rm -r" +" "+ hdf5file
    process0 = subprocess.check_call(command00.split())

f=h5py.File(hdf5file, mode='w')

features = f.create_dataset('features', (ndp,382,400), dtype='float32')
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
f
