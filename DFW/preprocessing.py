# -*- coding: UTF-8 -*-
__author__ = 'Alrash'

import numpy as np
import copy

from .predefine import DEFAULT_LOAD_DATABASE_CONFIG as default_load_config

from .tools import DatabaseStr, Join2String
from .tools import tolist, set_default_config
from .tools import err_exit_mesg, warning_mesg


class PreProcessing:
    def __init__(self, config):
        pass


class LoadFeatureFromMatFile:
    def __init__(self, config: dict):
        # init
        self._decode_white_list = ['num', 'name', 'isint']

        self._config = self._decode(default_load_config, config)

    def _decode(self, default: dict, config: dict):
        cfg, keys = {'name': None}, config.keys()

        # check name key
        if 'name' not in keys:
            err_exit_mesg('load feature error, please set name key in config!')
        # check name item type
        if type(config['name']) not in [list, str]:
            err_exit_mesg('"name" item only support list or string type in database config!')

        # set default config
        config = set_default_config(default, config)
        # transfer name item type to list and then set num item
        cfg['name'] = tolist(config['name'])
        cfg['num'] = len(cfg['name'])

        # match name size
        for key in set(keys) - set(self._decode_white_list):
            cfg[key] = self._match_and_fill(cfg['name'], cfg['num'], key, config[key])

        return cfg

    @ staticmethod
    def _match_and_fill(name_item: list, num: int, key: str, item):
        item_type = type(item)

        if item_type in [str, int, tuple]:
            item = [item]
        elif item_type in [list, dict]:
            pass
        elif item_type is np.ndarray:
            item = item.tolist()
        else:
            err_exit_mesg('could not support %s type in database config!' % str(item_type))

        maps = {}
        if item_type is dict:
            item_keys = item.keys()
            if 'default' not in item_keys and len(item) != num:
                err_exit_mesg('could not match length of "name" item and "%s" item, please set "default" item at least!' % key)
            if len(set(item_keys) - set(name_item)) != 0:
                warning_mesg('found unknown key set [%s]' % Join2String(set(item_keys) - set(name_item)).tostring())

            for database in name_item:
                maps[database] = item[database] if database in item_keys else copy.deepcopy(item['default'])
        else:
            if len(item) not in [1, num]:
                err_exit_mesg('could not match length of "name" item and "%s" item' % key)

            for index in range(num):
                maps[name_item[index]] = copy.deepcopy(item[0] if num == 1 else item[index])

        return maps
