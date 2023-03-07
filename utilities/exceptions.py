"""
Module containing custom exceptions used in the package
"""

class InvalidSourceError(ValueError):

    def __init__(self,source):
        self.source = source
        self.message = f"\'{self.source}\' is not a valid source! Only \'txt\' and \'root\' are allowed, \'txt\' being the default"
        super().__init__(self.message)