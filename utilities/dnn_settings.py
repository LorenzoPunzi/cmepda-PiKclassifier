"""
"""


class dnn_settings:
    """
    """

    def __init__(self, layers = [20, 20, 15, 10], epochnum = 200, learning_rate = 0.001, batch_size = 128, batchnorm = False, dropout = 0, verbose = 2, showhistory = True):

        self._layers = layers
        self._epochnum = epochnum
        self._learning_rate = learning_rate
        self._batch_size = batch_size
        self._batchnorm = batchnorm
        self._dropout = dropout
        self._verbose = verbose
        self._showhistory = showhistory

    @property
    def layers(self):
        return self._layers

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
    def batchnorm(self):
        return self._batchnorm

    @property
    def dropout(self):
        return self._dropout

    @property
    def verbose(self):
        return self._verbose

    @property
    def showhistory(self):
        return self._showhistory

    @layers.setter
    def layers(self, neurons_list):
        if not len(neurons_list) == 0:
            self._layers = neurons_list
        else:
            print('ERROR: the "epochs" value cannot be 0')

    @epochnum.setter
    def epochnum(self, epochs):
        if not int(epochs) == 0:
            self._epochnum = int(epochs)

    @learning_rate.setter
    def learning_rate(self, lr):
        if lr > 0:
            self._learning_rate = lr
        else:
            print('ERROR: the learning rate must be > 0')

    @batch_size.setter
    def batch_size(self, batch):
        if not int(batch) == 0:
            self._batch_size = int(batch)
        else:
            print('ERROR: the "batch size" value cannot be 0')

    @batchnorm.setter
    def batchnorm(self, bnorm):
        if type(bnorm) is bool:
            self._batchnorm = bnorm
        else:
            print('ERROR: "batchnorm" method must be a boolean value')

    @dropout.setter
    def dropout(self, dr):
        if dr >= 0:
            self._dropout = dr
        else:
            print('ERROR: dropout rate must be >= 0')

    @verbose.setter
    def verbose(self, verb):
        if (verb == 0 or verb == 1 or verb == 2):
            self._verbose = verb
        else:
            print('ERROR: uncorrect value given to "verbose" method')

    @showhistory.setter
    def showhistory(self, show):
        if type(show) == bool:
            self._showhistory = show
        else:
            print('ERROR: "showhistory" method must be a boolean value')
