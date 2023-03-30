"""
Module containing the custom exceptions used in the package
"""


class InvalidSourceError(ValueError):
    """
    Exception that gives an error message if the type of source file given
    is incorrect.
    """

    def __init__(self, source):
        """
        :param source: Invalid source type given by the user.
        :type source: str
        """
        self.source = source
        self.message = f"\'{self.source}\' is not a valid source! Only \
            \'.txt\' and \'.root\' are allowed, \'.root\' being the default"
        super().__init__(self.message)


class InvalidArrayGenRequestError(ValueError):
    """
    Exception that gives an error message if there is an incompatibility of the
    number of source files given and the type of array to import.
    """

    def __init__(self, for_training, for_testing, mixing=False):
        """
        :param for_training: Flag indicating the user wants to generate arrays for training.
        :type for_training: bool
        :param for_testing: Flag indicating the user wants to generate arrays for testing.
        :type for_testing: bool
        :param mixing: Flag indicating the user wants to use mixed variables.
        :type mixing: bool
        """
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
    the selected path.
    """

    def __init__(self, header_path):
        """
        :param header_path: Invalid header path given by the user.
        :type header_path: str
        """
        self.header_path = header_path
        self.message = f"Could not load header at path \'{self.header_path}\'"
        super().__init__(self.message)


class IncorrectFractionError(ValueError):
    """
    Exception that gives an error message if the given fraction of signal events (to be
    used in the dataset generation) is not in the physical interval (0,1).
    """

    def __init__(self, f):
        """
        :param f: Invalid fraction given by the user.
        :type f: float
        """
        self.fraction = f
        self.message = f" {self.fraction} is not a valid value for the \
            fraction of signal events in the mixed dataset! Make sure it is in\
            the range (0,1)"
        super().__init__(self.message)


class IncorrectNumGenError(ValueError):
    """
    Exception that gives an error if the given number of events to be stored in
    the MC and mixed dataset is incompatible with the event size of the toyMCs.
    """

    def __init__(self, num_mc, num_data, num_toyb, num_toys):
        """
        :param num_mc: Invalid number of mc events requested by the user.
        :type num_mc: int
        :param num_data: Invalid number of data events requested by the user.
        :type num_data: int
        :param num_toyb: Number of events in the background toy.
        :type num_toyb: int
        :param num_toys: Number of events in the signal toy.
        :type num_toys: int
        """
        self.message = f"Invalid combinations of num_mc = {num_mc} and num_data = {num_data}"\
            ": make sure that the toyMC files contain a sufficient" \
            f" number of events and lower the values of these two inputs! Current number of events are" \
            f" {num_toyb} for the background toy and {num_toys} for the signal toy"
        super().__init__(self.message)


class IncorrectEfficiencyError(ValueError):
    """
    Exception that gives an error if the given value of efficiency is not in
    the physical interval (0,1).
    """

    def __init__(self, eff):
        """
        :param eff: Invalid efficiency/specificity given by the user.
        :type eff: float
        """
        self.message = f"{eff} is not a valid value for efficiency/specificity. Make sure it is"
        "in the range (0,1)"
        super().__init__(self.message)

class IncorrectIterableError(Exception):
    """
    Exception that gives an error if the length of a list/tuple is incorrect.
    """

    def __init__(self, input, target_length):
        """
        :param input: Invalid list/tuple given by the user.
        :type input: list or tuple
        :param target_length: Minimum length required for the list/tuple by the function.
        :type target_length: int
        """
        self.message = f"{input} given has length {len(input)}, which is less than the minumum number\
            of arguments {target_length}"
        if not (type(input)==list or type(input)==tuple):
            self.message = f"{input} given in not a list/tuple!"
        super().__init__(self.message)
