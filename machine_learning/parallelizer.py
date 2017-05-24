"""
Name: parallizer.py
Author: Ronald Kemker
Description: Parallelize Keras Models

Note:
Requires Tensorflow, Keras
https://www.tensorflow.org/
https://keras.io/
"""

import keras.backend as K
from keras.layers import Lambda, merge, concatenate
from keras.models import Model
import numpy as np
import tensorflow as tf

class Parallelizer(object):
    """
    Parallizer
    
    This takes a keras model and parallelizes it between multiple GPUs.  This
    class requires Tensorflow and Keras.
    
    Parameters
    ----------
    gpu_list : list of int, None
        If None, the model is trained on all available GPUs.
        If list of integers, the GPU is trained on the GPUs in the list
        
    Attributes
    ----------
    gpu_list : List of GPU ids that the model is trained on
    n_gpus : int, total number of GPUs that the model is trained on
    
    Notes
    -----
    Ref: https://github.com/kuza55/keras-extras/blob/master/utils/multi_gpu.py
    
    """    
    def __init__(self, gpu_list = None):
        if gpu_list is None:
            self.gpu_list = self._get_available_gpus()
            self.n_gpus = len(self.gpu_list)
        else:
            self.n_gpus = len(gpu_list)
            self.gpu_list = gpu_list

    def _get_available_gpus(self):
        """Get available GPUs."""
        from tensorflow.python.client import device_lib
        local_device_protos = device_lib.list_local_devices()
        return [int(x.name[-1]) for x in local_device_protos if x.device_type == 'GPU']

    def save_model(self, file_name, parallel_model):
        """Transform the keras model to a parallelized model
        
        Parameters
        ----------
        file_name : string, contains path to where the model should be saved
        parallel_model : parallelized keras model
        """
        parallel_model.layers[-2].save(file_name)
    
    def transform(self, model):
        """Transform the keras model to a parallelized model
        
        Parameters
        ----------
        model : keras model
        
        Returns
        -------
        output : parallelized keras model
        
        """
        def get_slice(data, idx, parts):
            is_last_slice = idx == parts - 1
            
            shape = K.shape(data)
            minibatch_size, features = shape[:1], shape[1:]
            stride = K.concatenate([minibatch_size//parts, features*0], axis=0)
            if is_last_slice:
                # feed everything else if it's the last slice
                size = K.concatenate([[-1], features], axis=0)
            else:
                size = K.concatenate([minibatch_size//parts, features], axis=0)
            begin = stride * idx
            return tf.slice(data, begin, size)

        outputs_all = [[] for i in model.outputs]

        # Place a copy of the model on each GPU
        # each getting a slice of the batch
        for idx, gpu_id in enumerate(self.gpu_list):
            with tf.device('/gpu:%d' % gpu_id):
                with tf.name_scope('tower_%d' % gpu_id) as scope:
                    inputs = []
                    # Slice each input into a piece for processing on this GPU
                    for x in model.inputs:
                        input_shape = tuple(x.get_shape().as_list())[1:]
                        slice_n = Lambda(
                            get_slice,
                            output_shape=input_shape,
                            arguments={
                                'idx': idx,
                                'parts': self.n_gpus
                            })(x)
                        inputs.append(slice_n)

                    outputs = model(inputs)

                    if not isinstance(outputs, list):
                        outputs = [outputs]

                    # Save all the outputs for merging back together later
                    for l in range(len(outputs)):
                        outputs_all[l].append(outputs[l])
       
        # all output tensors on CPU
        with tf.device('/cpu:0'):
            if self.n_gpus > 1:
            
                return Model(
                    inputs=model.inputs,
                    outputs=[
                        # merge outputs from all GPU
                        concatenate(o, axis=0, name=o[0].name.split('/')[-2])
                        for o in outputs_all
                    ]
                )
            else:
                return Model(inputs=model.inputs, outputs=outputs_all[0],
                             name=outputs_all[0][0].name.split('/')[-2])
                
if __name__ is '__main__':
    
    from keras.models import Sequential
    from keras.layers import Dense
    model = Sequential()
    model.add(Dense(64, input_shape=(16,)))
    model.add(Dense(64))
    
    par = Parallelizer(gpu_list=[1,2,3])
    model = par.transform(model)
    
