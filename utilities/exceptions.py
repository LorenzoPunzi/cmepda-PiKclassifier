"""
Module containing custom exceptions used in the package
"""


class InvalidSourceError(ValueError):
    """
    Exception that gives an error message if the type of the source file given
    to a function is incorrect
    """

    def __init__(self, source):
        self.source = source
        self.message = f"\'{self.source}\' is not a valid source! Only \
            \'.txt\' and \'.root\' are allowed, \'.root\' being the default"
        super().__init__(self.message)


class InvalidArrayGenRequestError(ValueError):
    """
    Exception that gives an error message if there is an incompatibility of the
    number of source file given and the type of array to import

    Parameters:
        for_training, for_testing, mixed : bool
            Flags that select which type of error message to show
    """

    def __init__(self, for_training, for_testing, mixing=False):
        if (for_training and for_testing):
            self.message = "To create both the tranining and testing \
                            datasets, the file paths (.root) of both the two \
                            MCs d.s. and the mixed d.s. are needed!"
        elif for_training:
            self.message = "To create the tranining dataset, the file paths \
                            (.root) of both the two MCs datasets are needed!"
        elif for_testing:
            self.message = "To create the testing dataset, only the file \
                            path (.root) of the mixed dataset is needed!"
        elif mixing:
            self.message = "To add the mixed-variables data, all the three \
                            root file paths are needed!"
        else:
            self.message = "UNEXPECTED ERROR: a deep review of the code is \
                            suggested."
        super().__init__(self.message)


class LoadHeadError(Exception):
    """
    Exception that gives an error message if the header file cannot be found in
    the selected path
    """

    def __init__(self, header_path):
        self.header_path = header_path
        self.message = f"Could not load header at path \'{self.header_path}\'"
        super().__init__(self.message)


class IncorrectFractionError(ValueError):
    """
    Exception that gives an error message if the given fraction of Kaons (to be
    used in the dataset generation) is non-physical.
    """

    def __init__(self, f):
        self.fraction = f
        self.message = f" {self.fraction} is not a valid value for the \
            fraction of Kaons in the mixed dataset!"
        super().__init__(self.message)


class IncorrectNumGenError(ValueError):
    """
    Exception that gives an error if the given number of events to be stored in
    the MC and mixed datasets is incompatible with the event size of the toyMCs
    """

    def __init__(self):
        self.message = "Incorrect combinations of \'num_mc\' and \'num_data\'"\
            " values: make sure that the toyMC files contain a sufficient" \
            " number of events and lower the values of these two inputs!"
        super().__init__(self.message)
