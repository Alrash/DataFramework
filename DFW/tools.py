# -*- coding: UTF-8 -*-
__author__ = 'Alrash'

import sys


def tolist(item):
    return item if type(item) is list else [item]


# print error message to stderr, and exit(status)
def err_exit_mesg(mesg, status = -1):
    sys.stderr.write(mesg)
    sys.exit(status)


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
