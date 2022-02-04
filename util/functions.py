"""
A collection of utility functions used for analysis
"""


def find_str(string: str, pattern: str) -> list:
    """
    Find all indices of patterns in a string

    Parameters
    ----------
    string : str
        input string
    pattern : str
        string pattern to search

    Returns
    -------
    ind : list
        list of starting indices
    """
    import re
    if not pattern.isalpha():  # if the pattern contains non-alphabetic chars such as *
        pattern = "\\" + pattern

    ind = [m.start() for m in re.finditer(pattern, string)]
    return ind


def find_data_path():
    """
    Request the user to manually find dir path and return it
    """
    from pathlib import Path
    from tkinter import Tk
    from tkinter import filedialog
    root = Tk()
    root.withdraw()
    data_dir = filedialog.askdirectory()
    return Path(data_dir)


def extract_ind(timestamp, range):
    """
    Extract timestamp indices from array from the specified range

    Args:
        timestamp: array
        range: list [start end]

    Returns:
        ind: array
            index of the array
        new_array: array
            array within the range
    """
    import numpy as np
    start = range[0]
    end = range[1]

    ind = np.where((timestamp >= start) & (timestamp <= end))
    new_array = timestamp[ind]
    return ind, new_array


def normalize(array):
    """
    Normalizes an array by its average and sd
    """
    import numpy as np

    return (np.array(array) - np.average(array)) / np.std(array)


def exists(var):
    """
    Check if a variable exists

    Parameters
    ----------
    var : str
        Note that the argument should be in parenthesis
    Returns
    -------
    bool
    """

    return var in globals()

def para_interp(x, y):
    """
    Get max value by performing parabolic interpolation given three data points
    Parameters
    ----------
    x : array
    y : array

    Returns
    -------
    x_max : float
        max index
    y_max : float
        estimated max value
    """
    import numpy as np

    x = np.vstack((x ** 2, x))
    x = np.vstack((x, np.array([1, 1, 1]))).T

    x = np.linalg.inv(x)
    func = np.dot(x, y)

    x_max = -func[1] / (2 * func[0])
    y_max = (func[0] * x_max ** 2) + (func[1] * x_max) + func[2]

    return x_max, y_max