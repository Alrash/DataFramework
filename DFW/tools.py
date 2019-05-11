# -*- coding: UTF-8 -*-
__author__ = 'Alrash'

import numpy as np
import h5py
import scipy.io as scio

import sys
import os
# import warnings

import copy
from collections import Iterable


def tolist(item):
    return list(item) if isinstance(item, Iterable) and not isinstance(item, str) else [item]


def totuple(item):
    return item if isinstance(item, tuple) else tuple([item]) if isinstance(item, str) else tuple(list(item))


# for example, [1] => 1, "str" => "str", ["str"] => "str"
def leave_iterable(item):
    return (list(item.values())[0] if isinstance(item, dict) else item[0]) \
        if isinstance(item, Iterable) and len(item) == 1 else item


def check_item_type(item, support_type, iter_type = None) -> bool:
    if type(item) in (support_type if isinstance(support_type, Iterable) else [support_type]):
        if isinstance(item, Iterable) and not isinstance(item, str) and iter_type is not None:
            iter_type = iter_type if isinstance(iter_type, Iterable) else [iter_type]
            for elem in item:
                if type(elem) not in iter_type:
                    return False
        return True
    else:
        return False


def remove_dict_items(items: dict, keys) -> dict:
    dict_items, keys = copy.deepcopy(items), keys if isinstance(items, Iterable) and not isinstance(items, str) else [keys]
    for key in keys:
        dict_items.pop(key)
    return dict_items


# set default config item to config var
def set_default_config(default: dict, config: dict):
    has_key = config.keys()
    for key in default.keys():
        if key not in has_key:
            config[key] = default[key]
    return config


def loadmat(filename: str, domain: list = None):
    if not os.path.exists(filename):
        return None
    else:
        try:
            # load mat file via h5py, by default
            # pass
            # data = h5py.File(filename, 'r')
            raise OSError
        except OSError:
            # load mat file via scipy.io
            data = scio.loadmat(filename)

        # return all variable
        if domain is None or len(domain) == 0:
            return data

        # select var
        select = {}
        for key in domain:
            select[key] = data[key]
        return select


# output warning message
def warning_mesg(mesg):
    # warnings.warn(mesg)
    sys.stdout.write('[Warning] %s\n' % mesg)


# print error message to stderr, and exit(status)
def err_exit_mesg(mesg, status = -1):
    sys.stderr.write('[Error] %s\n' % mesg)
    sys.exit(status)


class Join2String:
    def __init__(self, item, delimiter = ', '):
        self.__item, self.__delimiter = None, delimiter
        self.set_item(item)

    def tostring(self):
        return self.__delimiter.join(self.__item)

    # except string, int float and so on
    def set_item(self, item):
        self.__item = item
        return self


# generator database path
class DatabaseStr:
    def __init__(self, format_str = None, config = None):
        self.__format, self.__config = None, None
        self.set_format_string(format_str), self.set_config(config)

    def decode(self):
        if type(self.__config) is dict:
            string = self.__format
            for key in self.__config.keys():
                if type(self.__config[key]) not in [int, str, float]:
                    err_exit_mesg('only support int, float and str type, please check key ' + key)
                string = string.replace('{%s}' % key, self.__config[key] if type(self.__config[key]) is str else str(self.__config[key]))
        else:
            string = self.__format
        return string

    def set_format_string(self, format_str):
        self.__format = format_str
        return self

    def set_config(self, config):
        self.__config = config
        return self
