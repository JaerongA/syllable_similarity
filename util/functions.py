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


def extract_ind(timestamp, range):
    """
    Extract timestamp indices from array from the specified range

    Parameters
    ----------
    timestamp: array
        input string
    range: list
        [start end]

    Returns
    -------
    ind : array
       index of an array
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


def unique(input_list: list) -> list:
    """
    Extract unique strings from the list in the order they appear

    Parameters
    ----------
    input_list : list

    Returns
    -------
    list of unique, ordered string
    """
    seen = set()
    return [x for x in list if not (x in seen or seen.add(x))]
