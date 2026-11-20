
"""
exceptions.py

This module describes custom or user-defined exceptions
throughout the project.

Custom or user-defined exceptions are implemented to
manage dataset validation errors and satisfy the
assignment's requirements.
"""


class InvalidDatasetError(Exception):
    """
    Raised when the structure of dataset is not
    in accordance with the required format
    """

    def __init__(self, message):
        """
        Generates the exceptions with a specific error message
        """

        super().__init__(message)