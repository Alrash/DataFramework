# -*- coding: UTF-8 -*-
__author__ = 'Alrash'

DEFAULT_KEY_NAME = 'default'

DEFAULT_LOAD_DATABASE_CONFIG = {
    'format': {DEFAULT_KEY_NAME: 'done_{name}.mat'},
    'name_map': None,
    'group': {DEFAULT_KEY_NAME: 'group'},
    'group_rand': {DEFAULT_KEY_NAME: 'group_rand'},
    'range': {DEFAULT_KEY_NAME: [-1, -1]},
    'labeled': {DEFAULT_KEY_NAME: False},
    'root': {DEFAULT_KEY_NAME: '.'},
    'num': 1
}

DEFAULT_PREPROCESS_DATA_CONFIG = {
    'center': True,
    'extend': False,
    'cv': 5,
    'train': {DEFAULT_KEY_NAME: 0.5}
}
