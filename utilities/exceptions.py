"""
Module containing custom exceptions used in the package
"""


class InvalidSourceError(ValueError):
    def __init__(self, source):
        self.source = source
        self.message = f"\'{self.source}\' is not a valid source! Only " \
            "\'.txt\' and \'.root\' are allowed, \'.root\' being the default"
        super().__init__(self.message)


class InvalidArrayGenRequest():
    def __init__(self, for_training, for_testing, length):
        if (for_training and for_testing):
            self.message = "To create both the tranining and testing " \
                           "datasets, the file paths (.root) of both the two "\
                           "MCs d.s. and the mixed d.s. are needed!"
        elif for_training:
            self.message = "To create the tranining dataset, the file paths " \
                           "(.root) of both the two MCs datasets are needed!"
        elif for_testing:
            self.message = "To create the testing dataset, only the file " \
                           "path (.root) of the mixed dataset is needed!"
        else:
            self.message = "UNEXPECTED ERROR: a deep review of the code is " \
                           "suggested."
        super().__init__(self.message)
