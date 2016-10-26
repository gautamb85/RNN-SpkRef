import os
import lasagne
from lasagne.layers import InputLayer, DenseLayer, batch_norm, ConcatLayer, DropoutLayer, BatchNormLayer, Conv2DLayer, DimshuffleLayer, MaxPool2DLayer, ReshapeLayer, NonlinearityLayer,GRULayer
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
from lasagne.nonlinearities import TemperatureSoftmax

weights='/misc/scratch03/reco/bhattaga/data/i-vectors/dnn_projects/rsr/rnn-utt-lstm-deep/weights/s2s_rsr_minloss_valincr.npz'

##project path - make a new one each time
#one folder for the weigths
feat_path='/misc/scratch03/reco/bhattaga/data/i-vectors/dnn_projects/rsr/rnn-utt-lstm-deep/weights/rsr-mf-runtime/'

if os.path.exists(feat_path):
  print 'Project folder exits. Overwriting...'
else:
  print 'Creating Feature folders'
  command0 = "mkdir -p" +" "+ feat_path
  process = subprocess.check_call(command0.split())
  
network={}

X = T.tensor3(name='features',dtype='float32')
Masks = T.matrix(name='masks',dtype='float32')
labels = T.ivector(name='spk-phr-labels')

print("Building network ...")
network['input'] = InputLayer(shape=(None,None,40), input_var = X)
batchs,_,_ = network['input'].input_var.shape
#added for lstm-dropout

network['mask'] = InputLayer((None,None), input_var = Masks)


gate_parameters = lasagne.layers.recurrent.Gate(
    W_in=lasagne.init.Uniform(0.05), W_hid=lasagne.init.Uniform(0.05),
    b=lasagne.init.Constant(0.))

cell_parameters = lasagne.layers.recurrent.Gate(
    W_in=lasagne.init.Uniform(0.05), W_hid=lasagne.init.Uniform(0.05),
    W_cell=None, b=lasagne.init.Constant(0.),
    nonlinearity=lasagne.nonlinearities.tanh)

network['lstm-forward'] = lasagne.layers.recurrent.LSTMLayer(network['input'],400, mask_input=network['mask'],
                          ingate=gate_parameters, forgetgate=gate_parameters, cell=cell_parameters, outgate=gate_parameters, 
                          gradient_steps=-1, grad_clipping=100., only_return_final=False)

network['lstm-backward'] = lasagne.layers.recurrent.LSTMLayer(network['input'],400, mask_input=network['mask'],
                           ingate=gate_parameters, forgetgate=gate_parameters, cell=cell_parameters, outgate=gate_parameters, 
                           gradient_steps=-1, grad_clipping=100., only_return_final=False, backwards=True)

network['concat1'] = ConcatLayer([network['lstm-forward'],network['lstm-backward']],axis=2)

network['lstm-forward2'] = lasagne.layers.recurrent.LSTMLayer(network['concat1'],400, mask_input=network['mask'],
                          ingate=gate_parameters, forgetgate=gate_parameters, cell=cell_parameters, outgate=gate_parameters, 
                          gradient_steps=-1, grad_clipping=100., only_return_final=True)

network['lstm-backward2'] = lasagne.layers.recurrent.LSTMLayer(network['concat1'],400, mask_input=network['mask'],
                           ingate=gate_parameters, forgetgate=gate_parameters, cell=cell_parameters, outgate=gate_parameters, 
                           gradient_steps=-1, grad_clipping=100., only_return_final=True, backwards=True)


network['concat'] = ConcatLayer([network['lstm-forward2'],network['lstm-backward2']])
#added for lstm-dropout

network['fc1'] = batch_norm(DenseLayer(network['concat'],2048,nonlinearity=lasagne.nonlinearities.sigmoid))
network['drop1'] = DropoutLayer(network['fc1'],p=0.5)


#softmax
temperature = 1.0
custom_softmax = TemperatureSoftmax(temperature)
# Output layer
softmax = custom_softmax

network['fc3'] = DenseLayer(network['drop1'],5820,nonlinearity=None)
network['prob'] = NonlinearityLayer(network['fc3'],nonlinearity=softmax)

## set weights ##
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

network_output = lasagne.layers.get_output(network['prob'])
  
hidden_output = lasagne.layers.get_output(network['drop1'], deterministic=True)
feat_extractor = theano.function([X, Masks],hidden_output)

dataset = H5PYDataset('/misc/data15/reco/bhattgau/Rnn/code/Code_/rsr/newshit/DATA/rsr-mf-RUNTIME.hdf5',which_sets=('runtime',))
h1=dataset.open()

bsize=512
scheme = SequentialScheme(examples=dataset.num_examples, batch_size=512)

data_stream = DataStream(dataset=dataset, iteration_scheme=scheme)

for data in data_stream.get_epoch_iterator():
  t_data,t_mask,t_name = data
  #runtime data is padded to 376, so pad 6 more
  """ 
  tsize = t_data.shape[0]
  pad = np.zeros((6,40),dtype='float32')
  temp=[]
  for tt in t_data:
    temp.append(np.vstack((tt,pad)))
 
  temp = np.asarray(temp,dtype='float32')
  t_data = np.reshape(temp,(tsize,382,40))
  """
  
  rvecs = feat_extractor(t_data,t_mask)

  for rv,n in zip(rvecs, t_name):
    #convert to double precision
    rv = np.cast['float32'](rv)

    utt_feat = os.path.join(feat_path, n +'.rvec')  
    rv.tofile(utt_feat)

dataset.close(h1)



 

