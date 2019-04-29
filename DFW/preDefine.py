from .tools import err_exit_mesg


class DatabaseStr:
    def __init__(self, format, config):
        self.__format, self.__config = None, None
        self.set_format_string(format), self.set_config(config)

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

    def set_format_string(self, format):
        self.__format = format

    def set_config(self, config):
        self.__config = config
