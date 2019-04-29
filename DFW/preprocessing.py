# -*- coding: UTF-8 -*-
__author__ = 'Alrash'

from .predefine import DEFAULT_LOAD_DATABASE_CONFIG as default_load_config

from .tools import DatabaseStr
from .tools import err_exit_mesg
from .tools import tolist


class PreProcessing:
    def __init__(self, config):
        pass


class LoadFeatureFromMatFile:
    def __init__(self, config: dict):
        self._config = self._decode(default_load_config, config)

    def _decode(self, default: dict, config: dict):
        cfg, keys = {'name': None}, config.keys()

        # check name key
        if 'name' not in keys:
            err_exit_mesg('load feature error, please set name key in config!')

        # transfer name item type to list and then set num item
        cfg['name'] = tolist(config['name'])
        cfg['num'] = len(cfg['name'])

        return cfg

    def _match(self, name_item, item):
        pass
