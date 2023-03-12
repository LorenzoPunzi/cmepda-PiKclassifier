"""
Define a class that allow to store and set quickly the most important options
for the DNN used in the analysis
"""


class DnnSettings:
    """
    Class that stores the most important options for the keras DNN algorithm

    Attributes:
        layers : list or tuple
            List of integers, that indicate the number of neurons in each
            internal dense layer
        val_fraction : float
            Fraction of the training dataset used for validation
        epochnum : int
            Number of epochs
        learning_rate : float
            Value given as learning rate to Adam optimizer
        batch_size : int
            Size of the batches in the dnn
        dropout : float
            Drop probability in the AlphaDropout layer
        verbose : int
            Control of what to print on shell during the training.
    """

    def __init__(self, layers=(75, 60, 45, 30, 20), val_fraction=0.5,
                 epochnum=200, learning_rate=0.001, batch_size=128,
                 dropout=0, verbose=2):
        """
        Initializes the class instance with default values
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
        if len(neurons_list) != 0:
            self._layers = neurons_list
        else:
            print('ERROR: the \"epochs\" value cannot be 0')

    @val_fraction.setter
    def val_fraction(self, frac):
        if frac > 0 and frac < 1:
            self._val_fraction = frac
        else:
            print('ERROR: invalid value given as \"validation fraction\"')

    @epochnum.setter
    def epochnum(self, epochs):
        if int(epochs) != 0:
            self._epochnum = int(epochs)

    @learning_rate.setter
    def learning_rate(self, lr):
        if lr > 0:
            self._learning_rate = lr
        else:
            print('ERROR: the learning rate must be > 0')

    @batch_size.setter
    def batch_size(self, batch):
        if int(batch) != 0:
            self._batch_size = int(batch)
        else:
            print('ERROR: the \"batch size\" value cannot be 0')

    @dropout.setter
    def dropout(self, dr):
        if dr >= 0:
            self._dropout = dr
        else:
            print('ERROR: dropout rate must be >= 0')

    @verbose.setter
    def verbose(self, verb):
        if verb in [0, 1, 2]:
            self._verbose = verb
        else:
            print('ERROR: uncorrect value given to \"verbose\" method')
