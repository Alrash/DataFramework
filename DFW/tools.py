# -*- coding: UTF-8 -*-
__author__ = 'Alrash'

import sys

import warnings


def tolist(item):
    return item if type(item) is list else [item]


# set default config item to config var
def set_default_config(default: dict, config: dict):
    has_key = config.keys()
    for key in default.keys():
        if key not in has_key:
            config[key] = default[key]
    return config


# output warning message
def warning_mesg(mesg):
    warnings.warn(mesg)


# print error message to stderr, and exit(status)
def err_exit_mesg(mesg, status = -1):
    sys.stderr.write(mesg)
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
    def __init__(self, format_str, config):
        self.__format, self.__config = None, None
        self.set_format_string(format_str), self.set_config(config)

    def decode(self):
        if type(self.__config) is dict:
            string = self.__format
            for key in self.__config.keys():
                if type(self.__config[key]) not in [int, str]:
                    err_exit_mesg('only support int and str type, please check key ' + key)
                string = string.replace('{%s}' % key, self.__config[key] if type(self.__config[key]) is str else str(self.__config[key]))
        else:
            string = self.__format
        return string

    def set_format_string(self, format_str):
        self.__format = format_str

    def set_config(self, config):
        self.__config = config
