# -*- coding: UTF-8 -*-
__author__ = 'Alrash'

import numpy as np

import re
import copy
from operator import itemgetter
from collections import Iterable

import os.path

from .predefine import DEFAULT_LOAD_DATABASE_CONFIG as default_load_config

from .tools import DatabaseStr, Join2String
from .tools import tolist, leave_iterable, set_default_config, check_item_type, remove_dict_items, loadmat
from .tools import err_exit_mesg, warning_mesg


class PreProcessing:
    def __init__(self, config):
        try:
            load_mat_file = LoadFeatureFromMatFile(config['database'])
        except KeyError:
            err_exit_mesg('could not find "database" item in config, please set it!')
        self._data, self._data_default_key = load_mat_file.data, load_mat_file.default_key

        # decode process item
        self.__special_key_type = {
            'cv': ([int], None),
            'training': ([int, float], None),
            'center': ([bool], None)
        }
        self._config = self._decode(default_load_config, config['precess'] if 'precess' in config.keys() else {})

    def _decode(self, default: dict, config: dict) -> dict:
        cfg, config = {}, set_default_config(default, config); keys = config.keys()
        return cfg


class LoadFeatureFromMatFile:
    def __init__(self, config: dict):
        # init
        self.__special_key_type = {
            'format': ([str, tuple], str),
            'root': ([str], None),
            'group': ([str, tuple], str),
            'group_rand': ([str, tuple], str),
            'range': ([tuple, list], int),
            'labeled': ([bool], None),
        }
        self._tuple_key_dict = {}
        self._decode_white_list = ['num', 'name', 'name_map']
        self._config = None
        self.__generate_info, self._default_key, self._data = {}, 'default', {}

        self._decode(default_load_config, config)._generate()._load()

    def _decode(self, default: dict, config: dict):
        # init
        cfg = {'name': None}

        # check name key
        if 'name' not in config.keys():
            err_exit_mesg('load feature error, please set name key in config!')
        # check name item type
        if type(config['name']) not in [list, str]:
            err_exit_mesg('"name" item only support list or string type in database config!')

        # set default config
        config = set_default_config(default, config); keys = config.keys()
        # transfer name item type to list and then set num item
        name, namemapflag = tolist(config['name']), config['name_map'] is not None
        try:
            cfg['name'] = name if not namemapflag else list(itemgetter(*name)(config['name_map']))
            for database in cfg['name']:
                if not check_item_type(database, [str, int]):
                    err_exit_mesg('the elements of "%s" key only support string type or int type!' % \
                                  ('name' if not namemapflag else 'name_map'))
        except KeyError as e:
            err_exit_mesg('could not find "%s" key in name_map, please recheck it!' % str(e))
        cfg['num'] = len(cfg['name'])

        # preprocess range item
        config['range'] = self._adjust_range_item(config['range'])

        # init name dict and record dict
        cfg.update(dict([(database, {}) for database in cfg['name']]))
        self._tuple_key_dict = dict([(database, []) for database in cfg['name']])

        # match name size and set key if the type of its item is tuple
        for key in set(keys) - set(self._decode_white_list):
            for database, value in self._match_and_fill(cfg['name'], cfg['num'], key, config[key]).items():
                cfg[database][key] = leave_iterable(value if key != 'format' else self._fix_format_item(value))
                self._tuple_key_dict[database].append(key) if isinstance(cfg[database][key], tuple) else None

        # check item type
        for database in set(cfg['name']):
            keys = self.__special_key_type.keys()
            for key in keys:
                if not check_item_type(cfg[database][key], self.__special_key_type[key][0], self.__special_key_type[key][1]):
                    err_exit_mesg('the elements of "%s" key only support %s!' % \
                                  (key, ' or '.join([str(item) for item in self.__special_key_type[key][0]])))
            for key in (set(cfg[database].keys()) - set(keys)):
                if not check_item_type(cfg[database][key], [int, float, str, tuple], [int, float, str]):
                    err_exit_mesg('the elements of "%s" key only support int, float, string and tuple type!' % key)
            # set name item
            cfg[database]['name'] = database

        # set config
        self._config = cfg
        return self

    def _generate(self):
        database_str, self.__generate_info = DatabaseStr(), {}
        for database in self._config['name']:
            self.__generate_info[database] = {}; database_str.set_format_string(self._config[database]['format'])
            self.__generate_info[database][self._default_key] = {
                'path': os.path.join(self._config[database]['root'], \
                                     database_str.set_config(\
                                         remove_dict_items(self._config[database], \
                                                           self.__special_key_type.keys())).decode()),
                'group': self._config[database]['group'],
                'group_rand': self._config[database]['group_rand'],
                'range': self._config[database]['range'],
                'labeled': self._config[database]['labeled'],
            }
        return self

    def _load(self):
        self._data = {}
        for database in self._config['name']:
            self._data[database] = {}
            for key in self.__generate_info[database].keys():
                data = loadmat(self.__generate_info[database][key]['path'], \
                               [self.__generate_info[database][key]['group'], \
                                self.__generate_info[database][key]['group_rand']])
                if data is None:
                    err_exit_mesg('could not find MAT file [%s]!' % self.__generate_info[database][key]['path'])

                # select data with range
                self._data[database][key] = self._adapt_range(data, \
                                                              self.__generate_info[database][key]['group'], \
                                                              self.__generate_info[database][key]['group_rand'], \
                                                              self.__generate_info[database][key]['range'], \
                                                              self.__generate_info[database][key]['labeled'])
        return self

    @ staticmethod
    def _adjust_range_item(drange):
        return tuple(drange) if isinstance(drange, list) else drange

    @ staticmethod
    def _adapt_range(data: np.ndarray, group_name: str, group_rand_name: str, drange: Iterable = None, labeled: bool = False):
        if drange is None or (drange[0] == -1):
            new = {'x': data[group_name][0], 'r': data[group_rand_name][0]}
        else:
            srange = range((drange[0] - 1), drange[1])
            new = {'x': data[group_name][0][srange], 'r': data[group_rand_name][0][srange]}

        for i in range(new['x'].shape[-1]):
            new['x'][i] = np.transpose(new['x'][i][:-1] if labeled else new['x'][i][:-1])

        return new

    @ staticmethod
    def _fix_format_item(format_str):
        fixed, flag = [], isinstance(format_str, Iterable) and not isinstance(format_str, str)
        for item in format_str if flag else [format_str]:
            fixed.append('%s.mat' % re.sub(r'\.[Mm][Aa][Tt]$', '', item))
        return tuple(fixed) if flag else fixed[0]

    @ staticmethod
    def _match_and_fill(name_item: list, num: int, key: str, item) -> dict:
        item_type = type(item)

        if item_type in [str, bool, int, float, tuple]:
            item = [item]
        elif item_type in [list, dict]:
            pass
        elif item_type is np.ndarray:
            item = item.tolist()
        else:
            err_exit_mesg('"%s" key could not support %s in database config!' % (key, str(item_type)))

        maps = {}
        if item_type is dict:
            item_keys = item.keys()
            if 'default' not in item_keys and len(item) != num:
                err_exit_mesg('could not match length of "name" item and "%s" item, please set "default" item at least!' % key)
            if len(set(item_keys) - set(name_item + ['default'])) != 0:
                warning_mesg('found unknown key set [%s] in "%s" item!' % \
                             (Join2String(set(item_keys) - set(name_item + ['default'])).tostring(), key))

            for database in name_item:
                maps[database] = item[database] if database in item_keys else copy.deepcopy(item['default'])
        else:
            if len(item) not in [1, num]:
                err_exit_mesg('could not match length of "name" item and "%s" item' % key)

            for index in range(num):
                maps[name_item[index]] = copy.deepcopy(item[0] if len(item) == 1 else item[index])

        return maps

    @ property
    def default_key(self):
        return self._default_key

    @ property
    def data(self):
        return self._data

    @ property
    def name(self):
        return self._config['name']
