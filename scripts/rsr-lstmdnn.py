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

def save(network, wts_path): 
  print('Saving Model ...')
  np.savez(wts_path, *lasagne.layers.get_all_param_values(network))


##project path - make a new one each time
#one folder for the weigths
project_path='/misc/scratch03/reco/bhattaga/data/i-vectors/dnn_projects/rsr/rnn-utt-lstm-dnn/'
wts_path = os.path.join(project_path,'weights')
epoch_path = os.path.join(project_path,'epoch_weights')

logfile = os.path.join(project_path,'uttRNN-train.log')

if os.path.exists(project_path):
  print 'Project folder exits. Deleting...'
  command00 = "rm -r" +" "+ project_path
  process0 = subprocess.check_call(command00.split())
      
  command0 = "mkdir -p" +" "+ project_path
  process = subprocess.check_call(command0.split())
  command1 = "mkdir -p" +" "+ wts_path
  process1 = subprocess.check_call(command1.split())
  command2 = "mkdir -p" +" "+ epoch_path
  process2 = subprocess.check_call(command2.split())
else:
  print 'Creating Project folder'
  command0 = "mkdir -p" +" "+ project_path
  process = subprocess.check_call(command0.split())
  command1 = "mkdir -p" +" "+ wts_path
  process1 = subprocess.check_call(command1.split())
  command2 = "mkdir -p" +" "+ epoch_path
  process2 = subprocess.check_call(command2.split())

network={}

X = T.tensor3(name='features',dtype='float32')
Masks = T.matrix(name='masks',dtype='float32')
labels = T.ivector(name='spk-phr-labels')

print("Building network ...")
network['input'] = InputLayer(shape=(None,382,40), input_var = X)
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
                          gradient_steps=-1, grad_clipping=100., only_return_final=True)

network['lstm-backward'] = lasagne.layers.recurrent.LSTMLayer(network['input'],400, mask_input=network['mask'],
                           ingate=gate_parameters, forgetgate=gate_parameters, cell=cell_parameters, outgate=gate_parameters, 
                           gradient_steps=-1, grad_clipping=100., only_return_final=True, backwards=True)


network['concat'] = ConcatLayer([network['lstm-forward'],network['lstm-backward']])
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

network_output = lasagne.layers.get_output(network['prob'])
  
hidden_output = lasagne.layers.get_output(network['drop1'], deterministic=True)

val_prediction = lasagne.layers.get_output(network['prob'], deterministic=True)
  
#needed for accuracy
val_acc = T.mean(T.eq(T.argmax(val_prediction, axis=1), labels), dtype=theano.config.floatX)
#training accuracy
train_acc = T.mean(T.eq(T.argmax(network_output, axis=1), labels), dtype=theano.config.floatX)

total_cost = lasagne.objectives.categorical_crossentropy(network_output, labels) #+ L1_penalty*1e-7
mean_cost = total_cost.mean()

#accuracy function
val_cost = lasagne.objectives.categorical_crossentropy(val_prediction, labels) #+ L1_penalty*1e-7
val_mcost = val_cost.mean()

#Get parameters of both encoder and decoder
all_parameters = lasagne.layers.get_all_params(network['prob'], trainable=True)

print("Trainable Model Parameters")
print("-"*40)
for param in all_parameters:
    print(param, param.get_value().shape)
print("-"*40)

all_grads = T.grad(mean_cost, all_parameters)
Learning_rate = 0.0001
learn_rate = theano.shared(np.array(Learning_rate, dtype='float32'))
lr_decay = np.array(0.1, dtype='float32')

updates = lasagne.updates.adam(all_grads, all_parameters, learn_rate)

train_func = theano.function([X, Masks, labels], [mean_cost, train_acc], updates=updates)

val_func = theano.function([X, Masks, labels], [val_mcost, val_acc])
  
#function to return the softmax posterior
posterior = theano.function([X,Masks], val_prediction)
dvector = theano.function([X,Masks], hidden_output)

X_train = np.load('/misc/data15/reco/bhattgau/Rnn/code/Code_/rsr/newshit/DATA/rsr-mf-p1-DEVBKG.npy')
X_masks = np.load('/misc/data15/reco/bhattgau/Rnn/code/Code_/rsr/newshit/DATA/rsr-mf-p1-DEVBKG-masks.npy')
X_labels = np.load('/misc/data15/reco/bhattgau/Rnn/code/Code_/rsr/newshit/DATA/rsr-mf-DEVBKG-labels.npy')


X_t = X_train[:47284,:,:]
X_m = X_labels[:47284]
X_ms = X_masks[:47284,:]

X_v = X_train[47284:,:,:]
X_mv = X_labels[47284:]
X_vs = X_masks[47284:]


train_set = IndexableDataset(indexables=OrderedDict([('features',X_t),('labels',X_m),('masks',X_ms)]), axis_labels=OrderedDict([('features',('batch','time_steps','feat_dim')),('labels',('batch',)),('masks',('batch','time_steps'))]))
valid_set = IndexableDataset(indexables=OrderedDict([('features',X_v),('labels',X_mv),('masks',X_vs)]), axis_labels=OrderedDict([('features',('batch','time_steps','feat_dim')),('labels',('batch',)),('masks',('batch','time_steps'))]))

min_val_loss = np.inf
val_prev = 1000

patience=0#patience counter
val_counter=0 
epoch=0
num_epochs=600

print("Starting training...")
    # We iterate over epochs:
while 'true':
    # In each epoch, we do a full pass over the training data:
    train_err = 0
    train_acc = 0
    train_batches = 0

    h1=train_set.open()
    h2=valid_set.open()

    scheme = ShuffledScheme(examples=train_set.num_examples, batch_size=64)
    scheme1 = SequentialScheme(examples=valid_set.num_examples, batch_size=128)

    train_stream = DataStream(dataset=train_set, iteration_scheme=scheme)
    valid_stream = DataStream(dataset=valid_set, iteration_scheme=scheme1)

    start_time = time.time()

    for data in train_stream.get_epoch_iterator():
        
        t_data,t_labs,t_mask = data
        terr,tacc = train_func(t_data,t_mask, t_labs)
        train_err += terr
        train_acc += tacc
        train_batches += 1
    
    val_err = 0
    val_acc = 0
    val_batches = 0

    for data in valid_stream.get_epoch_iterator():
        v_data,v_labs,v_mask = data    
        err,acc = val_func(v_data, v_mask, v_labs)
        val_err += err
        val_acc += acc
        val_batches += 1
    
    epoch+=1
    train_set.close(h1)
    valid_set.close(h2)
    
    print("Epoch {} of {} took {:.3f}s Learning Rate {}".format(
          epoch, num_epochs, time.time() - start_time, learn_rate.get_value()))
    print("  training loss:{:.6f}, training accuracy:{:.2f}, validation loss:{:.6f}, validation accuracy:{:.2f}".format((train_err / train_batches), 
         (train_acc/train_batches*100), (val_err / val_batches), (val_acc/val_batches*100)))
     
    flog1 = open(logfile,'ab')
    flog1.write("Epoch {} of {} took {:.3f}s Learning rate {}\n".format(
        epoch, num_epochs, time.time() - start_time, learn_rate.get_value()))
    flog1.write("  training loss:{:.6f}, training accuracy:{:.2f}, validation loss:{:.6f}, validation accuracy:{:.2f}\n".format((train_err / train_batches), 
        (train_acc/train_batches*100), (val_err / val_batches), (val_acc/val_batches*100)))
      
    flog1.write("\n")
    flog1.close()
    

    valE = val_err/val_batches
    valA = (val_acc/val_batches)*100
    
    if valE <= min_val_loss:

      #save the network parameters corresponding to this loss
      min_loss_network = network['prob'] 
      patience=0
      min_val_loss = valE
      mloss_epoch=epoch+1
      
      mname = 'uttCNN-weights-epoch-%d'%(epoch+1)
      spth = os.path.join(epoch_path,mname+'.npz')
      save(min_loss_network,spth)

    #Patience / Early stopping
    else:
      #increase the patience counter
      patience+=1
      #decrease the learning rate
      learn_rate.set_value(learn_rate.get_value()*lr_decay)
      spth = os.path.join(wts_path,'s2s_rsr_minloss_valincr.npz')
      save(min_loss_network,spth)

    if patience==5:
      break
   
    if val_prev - valE <= 0.001:
      learn_rate.set_value(learn_rate.get_value()*lr_decay)
      val_counter+=1
    
    val_prev = valE

    if val_counter==10:
      spth = os.path.join(wts_path,'s2s_rsr_minloss_valincr.npz')
      save(min_loss_network,spth)
      break #break out
    
    if epoch == num_epochs: 
      spth = os.path.join(wts_path,'s2s_rsr_minloss_nepoch.npz')
      save(min_loss_network, spth)
      break

 

