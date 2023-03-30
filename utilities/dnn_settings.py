"""
Class to set and store some important characteristics of a Keras DNN
"""

import warnings 

warnings.formatwarning = lambda msg, *args, **kwargs: f'\n{msg}\n'

class DnnSettings:
    """
    Class to set and store some important characteristics of a Keras DNN.

    :param layers: List or tuple of integers, that indicate the number of neurons in each internal dense layer.
    :type layers: list[int] or tuple[int]
    :param val_fraction: Fraction of the training dataset used for validation.
    :type val_fraction: float
    :param epochnum: Number of epochs for the training.
    :type epochnum: float
    :param learning_rate: Value given as learning rate to the Adam optimizer.
    :type learning_rate: float
    :param batch_size: Size of the batches used in the training.
    :type batch_size: int
    :param dropout: Drop probability in the AlphaDropout layer.
    :type dropout: float
    :param verbose: Set how verbose the training is on shell.
    :type verbose: int
    """

    def __init__(self, layers=(75, 60, 45, 30, 20), val_fraction=0.5,
                 epochnum=200, learning_rate=0.001, batch_size=128,
                 dropout=0, verbose=2):
        """
        Constructor method
        """
        self._layers = layers
        self._val_fraction = val_fraction
        self._epochnum = epochnum
        self._learning_rate = learning_rate
        self._batch_size = batch_size
        self._dropout = dropout
        self._verbose = verbose

    @property
    def layers(self):
        return self._layers

    @property
    def val_fraction(self):
        return self._val_fraction

    @property
    def epochnum(self):
        return self._epochnum

    @property
    def learning_rate(self):
        return self._learning_rate

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def dropout(self):
        return self._dropout

    @property
    def verbose(self):
        return self._verbose

    @layers.setter
    def layers(self, neurons_list):
        if type(neurons_list) is not list and type(neurons_list) is not tuple:
            msg = f'***WARNING*** \nLayers {neurons_list} given is NOT a list/tuple! No change will be made to it...\
\n*************\n'
            warnings.warn(msg, stacklevel=2)
        elif len(neurons_list)==0:
            msg = f'***WARNING*** \nLayers {neurons_list} given has length 0! No change will be made to it...\
\n*************\n'
            warnings.warn(msg, stacklevel=2)
        else:
            self._layers = neurons_list
        

    @val_fraction.setter
    def val_fraction(self, frac):
        if type(frac) is not float:
            msg = f'***WARNING*** \nFraction {frac} given is NOT a float! No change will be made to it...\
\n*************\n'
            warnings.warn(msg, stacklevel=2)
        elif frac<=0 or frac>=1:
            msg = f'***WARNING*** \nFraction {frac} given is NOT inside (0,1)! No change will be made to it...\
\n*************\n'
            warnings.warn(msg, stacklevel=2)
        else:
            self._val_fraction = frac

    @epochnum.setter
    def epochnum(self, epochs):
        if type(epochs) is not int:
            msg = f'***WARNING*** \nEpochs {epochs} given is NOT an integer! No change will be made to it...\
\n*************\n'
            warnings.warn(msg, stacklevel=2)
        elif epochs<=0:
            msg = f'***WARNING*** \nEpochs {epochs} given is NOT a positive integer! No change will be made to \
it...\n*************\n'
            warnings.warn(msg, stacklevel=2)
        else:
            self._epochnum = epochs
            

    @learning_rate.setter
    def learning_rate(self, lr):
        if type(lr) is not float:
            msg = f'***WARNING*** \nLearning Rate {lr} given is NOT a float! No change will be made to it...\
\n*************\n'
            warnings.warn(msg, stacklevel=2)
        elif lr<=0:
            msg = f'***WARNING*** \nLearning Rate {lr} given is NOT positive! No change will be made to it...\
\n*************\n'
            warnings.warn(msg, stacklevel=2)
        else:
            self._learning_rate = lr

    @batch_size.setter
    def batch_size(self, batch):
        if type(batch) is not int:
            msg = f'***WARNING*** \nBatch Size {batch} given is NOT an integer! No change will be made to it...\
\n*************\n'
            warnings.warn(msg, stacklevel=2)
        elif batch<=0:
            msg = f'***WARNING*** \nBatch Size {batch} given is NOT a positive integer! No change will be made \
to it...\n*************\n'
            warnings.warn(msg, stacklevel=2)
        else:
            self._batch_size = batch

    @dropout.setter
    def dropout(self, dr):
        if type(dr) is not float:
            msg = f'***WARNING*** \nDropout Rate {dr} given is NOT a float! No change will be made to it...\
\n*************\n'
            warnings.warn(msg, stacklevel=2)
        elif dr<=0:
            msg = f'***WARNING*** \nDropout Rate {dr} given is NOT positive! No change will be made to it...\
\n*************\n'
            warnings.warn(msg, stacklevel=2)
        else:
            self._dropout = dr

    @verbose.setter
    def verbose(self, verb):
        if verb in [0, 1, 2]:
            self._verbose = verb
        else:
            msg = f'***WARNING*** \nVerbosity {verb} given is NOT valid! Accepted values are {0,1,2}. No change \
will be made to it...\n*************\n'
            warnings.warn(msg, stacklevel=2)
