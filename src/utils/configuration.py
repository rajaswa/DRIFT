import yaml
import copy
from src.utils.mapper import configmapper


def load_yaml(path):
    """
    Function to load a yaml file and
    return the collected dict(s)

    Parameters
    ----------
    path : str
        The path to the yaml config file

    Returns
    -------
    result : dict
        The dictionary from the config file
    """

    assert isinstance(path, str), "Provided path is not a string"
    try:
        f = open(path, "r")
        result = yaml.load(f, Loader=yaml.Loader)
    except FileNotFoundError as e:
        # Adding this for future functionality
        raise e
    return result


def convert_params_to_dict(params):
    dic = {}
    for k, v in params.as_dict():
        try:
            obj = configmapper.get_object("params", v)
            dic[k] = v
        except:
            print(
                f"Undefined {v} for the given key: {k} in mapper        ,storing original value"
            )
            dic[k] = v
        return value


class Config:
    """Config Class to be used with YAML configuration files

    This class can be used to address keys as attributes.
    Ensure that there are no spaces between the keys.
    Only objects of type dict can be converted to config.

    Attributes
    ----------
    _config : dict,
        The dictionary which is formed from the
        yaml file or custom dictionary

    Methods
    -------
    as_dict(),
        Return the config object as dictionary

        Possible update:
        ## Can be converted using __getattr__ to use **kwargs
        ## with the Config object directly.

    set_value(attr,value)
        Set the value of a particular attribute.
    """

    def __init__(self, *, path=None, dic=None):
        """
        Initializer for the Config class

        Needs either path or the dict object to create the config

        Parameters
        ----------
        path: str, optional
            The path to the config YAML file.
            Default value is None.
        dic : dict, optional
            The dictionary containing the configuration.
            Default value is None.
        """
        if path:
            self._config = load_yaml(path)
        elif dict:
            self._config = dic
        else:
            raise Exception("Need either path or dict object to instantiate object.")
        # self.keys = self._config.keys()

    def __getattr__(self, attr):
        """
        Get method for Config class. Helps get keys as attributes.

        Parameters
        ----------
        attr: The attribute name passed as <object>.attr

        Returns
        -------
        self._config[attr]: object or Config object
            The value of the given key if it exists.
            If the value is a dict object,
            a Config object of that dict is returned.
            Otherwise, the exact value is returned.

        Raises
        ------

        KeyError() if the given key is not defined.
        """
        if attr in self._config:
            if isinstance(self._config[attr], dict):
                return Config(dic=self._config[attr])
            else:
                return self._config[attr]
        else:
            raise KeyError(f"Key:{attr} not defined.")

    def set_value(self, attr, value):
        """
        Set method for Config class. Helps set keys in the _config.

        Parameters
        ----------
        attr: The attribute name passed as <object>.attr
        value: The value to be stored as the attr.
        """
        self._config[attr] = value

    def __str__(self):
        """Function to print the dictionary
        contained in the object."""
        return self._config.__str__()

    def __repr__(self):
        return f"Config(dic={self._config})"

    def __deepcopy__(self, memo):
        return Config(dic=copy.deepcopy(self._config))

    def as_dict(self):
        """Function to get the config as dictionary object"""
        return dict(self._config)
