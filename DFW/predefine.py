# -*- coding: UTF-8 -*-
__author__ = 'Alrash'


DEFAULT_LOAD_DATABASE_CONFIG = {
    'format': {'default': 'done_{name}.mat'},
    'name_map': None,
    'group': {'default': 'group'},
    'group_rand': {'default': 'group_rand'},
    'range': {'default': [-1, -1]},
    'labeled': {'default': False},
    'root': {'default': '.'},
    'num': 1
}

DEFAULT_PREPROCESS_DATA_CONFIG = {
    'center': True,
    'cv': 5,
    'training': {'default': 0.5}
}
