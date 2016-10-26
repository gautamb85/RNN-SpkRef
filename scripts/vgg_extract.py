import os
import lasagne
from lasagne.layers import InputLayer, DenseLayer, batch_norm, DropoutLayer, BatchNormLayer, Conv2DLayer, DimshuffleLayer, MaxPool2DLayer, ReshapeLayer, NonlinearityLayer
from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
import time
import theano
import theano.tensor as T
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import cPickle
import sys
import subprocess
import scipy.io
from fuel.datasets import IndexableDataset
from fuel.schemes import ShuffledScheme, SequentialScheme
from fuel.streams import DataStream
from lasagne.nonlinearities import TemperatureSoftmax
from collections import OrderedDict
import h5py
from fuel.datasets import H5PYDataset
from lasagne.regularization import regularize_layer_params, l1
from lasagne.init import GlorotUniform,HeNormal

#weights='/misc/scratch03/reco/bhattaga/data/i-vectors/dnn_projects/rsr/cnn-utt-TempSoftmax5/weights/s2s_rsr_minloss_valincr.npz' 
weights='/misc/scratch03/reco/bhattaga/data/i-vectors/dnn_projects/rsr/cnn-utt-TempSoftmax5-sigmoid/weights/s2s_rsr_minloss_valincr.npz'

##project path - make a new one each time
#one folder for the weigths
feat_path='/misc/scratch03/reco/bhattaga/data/i-vectors/dnn_projects/rsr/cnn-utt-TempSoftmax5-sigmoid/rsr-mf-runtime-sigmoid/'
post_path='/misc/scratch03/reco/bhattaga/data/i-vectors/dnn_projects/rsr/cnn-utt-TempSoftmax5-sigmoid/rsr-mf-runtime-posteriors-sigmoid/'


if os.path.exists(feat_path):
  print 'Project folder exits. Overwriting...'
else:
  print 'Creating Feature folders'
  command0 = "mkdir -p" +" "+ feat_path
  process = subprocess.check_call(command0.split())
  
  command1 = "mkdir -p" +" "+ post_path
  process1 = subprocess.check_call(command1.split())

network={}

X = T.tensor3(name='features',dtype='float32')

print("Building network ...")
network['input'] = InputLayer(shape=(None,382,40), input_var = X)
batchs,_,_ = network['input'].input_var.shape

#pre-activation
network['reshape'] = ReshapeLayer(network['input'],(batchs,1,382,40))

network['conv1_1'] = batch_norm(ConvLayer(network['reshape'], 64,(3,3),pad=1,flip_filters=False, W=HeNormal('relu'))) 
network['conv1_2'] = batch_norm(ConvLayer(network['conv1_1'],64,(3,3),pad=1, flip_filters=False, W=HeNormal('relu')))
network['pool1']  = MaxPool2DLayer(network['conv1_2'],2)

network['conv2_1'] = batch_norm(ConvLayer(network['pool1'], 128,(3,3),pad=1,flip_filters=False,W=HeNormal('relu'))) 
network['conv2_2'] = batch_norm(ConvLayer(network['conv2_1'],128,(3,3),pad=1,flip_filters=False,W=HeNormal('relu')))
network['pool2']  = MaxPool2DLayer(network['conv2_2'],2)

network['conv3_1'] = batch_norm(ConvLayer(network['pool2'], 256,(3,3),pad=1,flip_filters=False, W=HeNormal('relu'))) 
network['conv3_2'] = batch_norm(ConvLayer(network['conv3_1'],256,(3,3),pad=1,flip_filters=False, W=HeNormal('relu')))
network['conv3_3'] = batch_norm(ConvLayer(network['conv3_2'],256,(3,3),pad=1,flip_filters=False, W=HeNormal('relu')))
network['pool3']  = MaxPool2DLayer(network['conv3_3'],2)

network['conv4_1'] = batch_norm(ConvLayer(network['pool3'], 512,(3,3),pad=1,flip_filters=False, W=HeNormal('relu'))) 
network['conv4_2'] = batch_norm(ConvLayer(network['conv4_1'],512,(3,3),pad=1,flip_filters=False, W=HeNormal('relu')))
network['conv4_3'] = batch_norm(ConvLayer(network['conv4_2'],512,(3,3),pad=1,flip_filters=False, W=HeNormal('relu')))
network['pool4']  = MaxPool2DLayer(network['conv4_3'],2)

network['conv5_1'] = batch_norm(ConvLayer(network['pool4'], 512,(3,3),pad=1,flip_filters=False, W=HeNormal('relu'))) 
network['conv5_2'] = batch_norm(ConvLayer(network['conv5_1'],512,(3,3),pad=1,flip_filters=False, W=HeNormal('relu')))
network['conv5_3'] = batch_norm(ConvLayer(network['conv5_2'],512,(3,3),pad=1,flip_filters=False, W=HeNormal('relu')))
network['pool5']  = MaxPool2DLayer(network['conv5_3'],2)

network['fc1'] = batch_norm(DenseLayer(network['pool5'],1024,nonlinearity=lasagne.nonlinearities.rectify, W=HeNormal('relu')))
network['fc1_drop'] = DropoutLayer(network['fc1'],p=0.2)
network['fc2'] = batch_norm(DenseLayer(network['fc1_drop'],1024,nonlinearity=lasagne.nonlinearities.rectify, W=HeNormal('relu')))
network['fc2_drop'] = DropoutLayer(network['fc2'],p=0.3)
#softmax
temperature = 100.0
custom_softmax = TemperatureSoftmax(temperature)
# Output layer
softmax = custom_softmax

network['fc3'] = DenseLayer(network['fc2'],5820,nonlinearity=None)
network['prob'] = NonlinearityLayer(network['fc3'],nonlinearity=softmax)

print "Loading trained network"
with np.load(weights) as f:
   param_values=[f['arr_%d' % i] for i in range(len(f.files))]

lasagne.layers.set_all_param_values(network['prob'],param_values)

all_parameters = lasagne.layers.get_all_params(network['prob'], trainable=True)

print "Feature-Exactor Parameters"
print "-"*40
for param in all_parameters:
  print param, param.get_value().shape
print "-"*40

rvector = lasagne.layers.get_output(network['fc2_drop'], deterministic=True)
softmax_output = lasagne.layers.get_output(network['prob'], deterministic=True)

feat_extractor = theano.function([X],rvector)
posterior = theano.function([X], softmax_output)
print("Starting Feature Extraction...")

#Bkg
#dataset = H5PYDataset('/misc/data15/reco/bhattgau/Rnn/code/Code_/rsr/newshit/DATA/rsr-mf-DEVBKG.hdf5',which_sets=('BKG',))
#Runtime
dataset = H5PYDataset('/misc/data15/reco/bhattgau/Rnn/code/Code_/rsr/newshit/DATA/rsr-mf-RUNTIME.hdf5',which_sets=('runtime',))
h1=dataset.open()

bsize=512
scheme = SequentialScheme(examples=dataset.num_examples, batch_size=512)

data_stream = DataStream(dataset=dataset, iteration_scheme=scheme)

for data in data_stream.get_epoch_iterator():
  t_data,_,t_name = data
  #runtime data is padded to 376, so pad 6 more
   
  tsize = t_data.shape[0]
  pad = np.zeros((6,40),dtype='float32')
  temp=[]
  for tt in t_data:
    temp.append(np.vstack((tt,pad)))
 
  temp = np.asarray(temp,dtype='float32')
  t_data = np.reshape(temp,(tsize,382,40))
  

  rvecs = feat_extractor(t_data)
  posts = posterior(t_data)

  for rv,pst,n in zip(rvecs,posts, t_name):
    #convert to double precision
    rv = np.cast['float32'](rv)
    pv = np.cast['float32'](rv)

    utt_feat = os.path.join(feat_path, n +'.rvec')  
    post_feat = os.path.join(post_path,n + '.post')
    rv.tofile(utt_feat)
    pv.tofile(post_feat)

dataset.close(h1)



