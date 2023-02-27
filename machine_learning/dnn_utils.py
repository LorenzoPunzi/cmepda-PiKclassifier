
class dnn_settings:
    """
    """

    def __init__(self):
        self._layers = [20, 20, 15, 10]
        self._epochnum = 200
        self._batch_size = 128
        self._verbose = 1
        self._showhistory = True
        # self._plotoptions = []

    @property
    def layers(self):
        return self._layers

    @property
    def epochnum(self):
        return self._epochnum

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def verbose(self):
        return self._verbose

    @property
    def showhistory(self):
        return self._showhistory

    # Not necessary, the options are passed in the functions in another way
    # @property
    # def plotoptions(self):
    #     return self._plotoptions

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

    @batch_size.setter
    def batch_size(self, batch):
        if not int(batch) == 0:
            self._batch_size = int(batch)
        else:
            print('ERROR: the "batch size" value cannot be 0')

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

    '''
    @plotoptions.setter
    def plotoptions(self, opt_list):
        # if len(opt_list) == 3 & type(opt_list[0]) == str & type(opt_list[0]) == str & type(opt_list[0]) == str:
        self._plotoptions = opt_list
        # else:
        #     print('ERROR: uncorrect set of values given as "plotoptions"')
    '''


if __name__ == '__main__':
    a = dnn_settings()
    a.verbose = 3
    print(a.verbose)
