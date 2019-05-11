# -*- coding: UTF-8 -*-
__author__ = 'Alrash'

import numpy as np
import math

import re
import copy
from operator import itemgetter
from collections import Iterable, deque

import os.path

from .predefine import DEFAULT_KEY_NAME
from .predefine import DEFAULT_LOAD_DATABASE_CONFIG as default_load_config
from .predefine import DEFAULT_PREPROCESS_DATA_CONFIG as default_preprocess_config

from .tools import DatabaseStr, Join2String
from .tools import tolist, leave_iterable, set_default_config, check_item_type, remove_dict_items, loadmat
from .tools import err_exit_mesg, warning_mesg


class PreProcessTransfer:
    def __init__(self, config):
        try:
            load_mat_file = LoadFeatureFromMatFile(config['database'])
        except KeyError:
            err_exit_mesg('could not find "database" item in config, please set it!')
        self._data, self._databases, self._data_default_key = load_mat_file.data, load_mat_file.name, load_mat_file.default_key

        # decode process item
        self.__special_key_type = {
            'cv': ([int], None),
            'train': ([int, float, tuple, list], [int, float]),
            'center': ([bool], None)
        }
        # cut => train and test
        # cv => cross validation
        self._config, self._cut_index, self._cv_index = None, {}, {}
        self._decode(default_preprocess_config, config['precess'] if 'precess' in config.keys() else {})._cut_database()

    def generator(self, no: int = 0, force: bool = False):
        pass

    def _transfer(self):
        pass

    # cut database and save it's index
    def _cut_database(self) -> 'PreProcessTransfer':
        database_index_list, database_cv_index_list = {}, {}
        for database in self._databases:
            database_index_list[database], database_cv_index_list[database] = {}, {}
            for descriptor in self._data[database]:
                # init
                database_index_list[database][descriptor], database_cv_index_list[database][descriptor], offset = {}, {}, {}
                for train_setting in self._config[database]['train']:
                    database_cv_index_list[database][descriptor][train_setting] = {}
                    offset[train_setting] = 0
                    for i in range(self._config[database]['cv']):
                        database_cv_index_list[database][descriptor][train_setting][i] = {'train': np.array([], dtype = np.int64), 'test': np.array([], dtype = np.int64)}
                # end for train_setting

                # calculate in each class
                for k in range(self._data[database][descriptor]['x'].shape[-1]):
                    origin, cv = self._cut_one_class(self._data[database][descriptor]['x'][k].shape[0], database)

                    # record with each class
                    database_index_list[database][descriptor][k] = origin
                    # record all cv information with each class
                    for train_setting in self._config[database]['train']:
                        for i in range(self._config[database]['cv']):
                            database_cv_index_list[database][descriptor][train_setting][i]['train'] = \
                                np.concatenate((database_cv_index_list[database][descriptor][train_setting][i]['train'], cv[train_setting][i]['train'] + offset[train_setting]))
                            database_cv_index_list[database][descriptor][train_setting][i]['test'] = \
                                np.concatenate((database_cv_index_list[database][descriptor][train_setting][i]['test'], cv[train_setting][i]['test'] + offset[train_setting]))
                        # end for i
                        # update for
                        offset[train_setting] += origin[train_setting]['train'].shape[-1]
                        print('class [%d]: train %d, cv train %d, cv test %d, offset %d' % \
                              (k, origin[train_setting]['train'].shape[-1], \
                               cv[train_setting][0]['train'].shape[-1], cv[train_setting][0]['test'].shape[-1],\
                               offset[train_setting]))
                    # end for train_setting
                # end for k
            # end for descriptor
        # end for database

        self._cut_index, self._cv_index = database_index_list, database_cv_index_list
        return self

    def _cut_one_class(self, data_size: int, name: str) -> tuple:
        class_index_list = dict([(train_setting, {}) for train_setting in self._config[name]['train']])
        class_cv_index_list = copy.deepcopy(class_index_list)

        for train_setting in self._config[name]['train']:
            class_index_list[train_setting], class_cv_index_list[train_setting] = \
                self._cut_data_index(data_size, train_setting, self._config[name]['cv'])

        return class_index_list, class_cv_index_list

    @ staticmethod
    # cut one class data by index
    def _cut_data_index(size: int, train_setting, cv: int, offset: int = 0) -> tuple:
        index_list = {'train': np.ndarray([]), 'test': np.ndarray([])}
        cv_index_list = dict([(i, {'train': np.ndarray([]), 'test': np.ndarray([])}) for i in range(cv)])

        # train size should be over 'parameter' cv without considering the number of data
        if train_setting == -1:
            train_size = size
        elif 0 < train_setting < 1:
            train_size = round(size * train_setting)
            if train_size < cv:
                train_size = min(size, cv)
        elif train_setting >= 1 and isinstance(train_setting, int):
            train_size = min(train_setting, size)
            if train_size < cv:
                train_size = min(cv, size)
        else:
            err_exit_mesg('"train" item should be -1 or between 0 and 1 or positive integer!')

        # set train and test index
        index_list['train'], index_list['test'] = np.arange(train_size).astype(np.int64), np.arange(train_size, size).astype(np.int64)

        # cross validation set index
        if train_size < cv:
            data_index_list = deque(range(offset, offset + train_size))
            for i in range(cv):
                test_index = data_index_list.popleft()
                cv_index_list[i]['train'], cv_index_list[i]['test'] = np.array(data_index_list, dtype = np.int64), np.array([test_index], dtype = np.int64)
                data_index_list.append(test_index)
        else:
            # # 8 instances, cv = 5 => [[0], [1, 2], [3], [4, 5], [6, 7]]
            # # 7 instances, cv = 5 => [[0], [1], [2, 3], [4], [5, 6]]
            pos = [math.floor(elem) for elem in np.linspace(offset, offset + train_size, cv + 1)]
            cv_index_group, index = np.array([np.arange(pos[i], pos[i + 1]) for i in range(cv)]), np.arange(cv)
            for i in range(cv):
                cv_index_list[i]['train'], cv_index_list[i]['test'] = np.concatenate(cv_index_group[index != i]), cv_index_group[i]

        return index_list, cv_index_list

    # decode config settings
    def _decode(self, default: dict, config: dict) -> 'PreProcessTransfer':
        # init
        cfg, config = {}, set_default_config(default, config); keys = config.keys()

        # filling
        cfg.update(dict([(database, {}) for database in self._databases]))
        for key in keys:
            for database, value in self._match_and_fill(self._databases, config[key], key).items():
                cfg[database][key] = leave_iterable(value)
                if not check_item_type(cfg[database][key], self.__special_key_type[key][0], self.__special_key_type[key][1]):
                    err_exit_mesg('the elements of "%s" key only support %s!' % \
                                  (key, ' or '.join([str(item) for item in self.__special_key_type[key][0]])))

        self._config = cfg
        return self._adapt_train_item()

    @ staticmethod
    def _match_and_fill(name: list, item, key_name: str) -> dict:
        item_type = type(item)
        if item_type in [str, int, float, bool, tuple]:
            item = [item]
        elif item_type in [dict, list]:
            pass
        elif item_type is np.ndarray:
            item = item.tolist()
        else:
            err_exit_mesg('"%s" key could not support %s in precess config!' % (key_name, str(item_type)))

        maps, num = {}, len(name)
        if item_type is dict:
            item_keys = item.keys()
            if DEFAULT_KEY_NAME not in item_keys and len(item) != num:
                err_exit_mesg('could not match length of "name" item and "%s" item, please set "default" item at least!' % key_name)
            if len(set(item_keys) - set(name + [DEFAULT_KEY_NAME])) != 0:
                warning_mesg('found unknown key set [%s] in "%s" item!' % \
                             (Join2String(set(item_keys) - set(name + [DEFAULT_KEY_NAME])).tostring(), key_name))

            for database in name:
                maps[database] = item[database] if database in item_keys else copy.deepcopy(item[DEFAULT_KEY_NAME])
        else:
            if len(item) not in [1, num]:
                err_exit_mesg('could not match length of "name" item and "%s" item' % key_name)

            for index in range(num):
                maps[name[index]] = copy.deepcopy(item[0] if len(item) == 1 else item[index])

        return maps

    def _set_one_item(self, key: str, settings):
        if isinstance(settings, dict):
            if DEFAULT_KEY_NAME in settings.keys():
                item = self._match_and_fill(self._databases, settings, key)
            else:
                item = settings

            for database, value in item.items():
                val = leave_iterable(value)
                if not check_item_type(val, self.__special_key_type[key][0], self.__special_key_type[key][1]):
                    err_exit_mesg('[run setting] the elements of "%s" key only support %s!' % \
                                  (key, ' or '.join([str(item) for item in self.__special_key_type[key][0]])))
                self._config[database][key] = val
        else:
            if len(self._databases) != 1:
                warning_mesg('the number of database is over 1, ignore to set "%s" item!' % key)
                return
            elif not check_item_type(settings, self.__special_key_type[key][0], self.__special_key_type[key][1]):
                err_exit_mesg('[run setting] the elements of "%s" key only support %s!' % \
                              (key, ' or '.join([str(item) for item in self.__special_key_type[key][0]])))

            self._config[self._databases[0]][key] = settings

        return self

    # preprocess train item
    # list -> list
    # not Iterable or tuple -> tuple
    def _adapt_train_item(self) -> 'PreProcessTransfer':
        # return item if isinstance(item, Iterable) else tuple(item)
        for database in self._databases:
            self._config[database]['train'] = self._config[database]['train'] \
                if isinstance(self._config[database]['train'], Iterable) \
                else tuple([self._config[database]['train']])
        return self

    def get_train(self, force: bool = False):
        train = {}
        for database in self._databases:
            train[database] = self._config[database]['train']
        return train if force else leave_iterable(train)

    @ property
    def train(self):
        return self.get_train()

    @ train.setter
    def train(self, setting) -> None:
        self._set_one_item('train', setting)._adapt_train_item()._cut_database()


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
        self.__generate_info, self._default_key, self._data = {}, DEFAULT_KEY_NAME, {}

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
            if DEFAULT_KEY_NAME not in item_keys and len(item) != num:
                err_exit_mesg('could not match length of "name" item and "%s" item, please set "default" item at least!' % key)
            if len(set(item_keys) - set(name_item + [DEFAULT_KEY_NAME])) != 0:
                warning_mesg('found unknown key set [%s] in "%s" item!' % \
                             (Join2String(set(item_keys) - set(name_item + [DEFAULT_KEY_NAME])).tostring(), key))

            for database in name_item:
                maps[database] = item[database] if database in item_keys else copy.deepcopy(item[DEFAULT_KEY_NAME])
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
