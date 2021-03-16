class ConfigMapper:
    """Class for creating ConfigMapper objects.

    This class can be used to create custom configuration names using YAML files.
    For each class or object instantiated in any modules,
    the ConfigMapper object can be used either with the functions,
    or as a decorator to store the mapping in the function.

    Attributes
    ----------

    Methods
    -------

    """

    dicts = {
        "models": {},
        "trainers": {},
        "metrics": {},
        "losses": {},
        "optimizers": {},
        "schedulers": {},
        "devices": {},
        "embeddings": {},
        "params": {},
        "datasets": {},
        "preprocessors": {},
        "tokenizers": {},
    }

    @classmethod
    def map(cls, key, name):
        """
        Map a particular name to an object, in the specified key

        Parameters
        ----------
            name : str
                The name of the object which will be used.
            key : str
                The key of the mapper to be used.
        """

        def wrap(obj):
            if key in cls.dicts.keys():
                cls.dicts[key][name] = obj
            else:
                cls.dicts[key] = {}
                cls.dicts[key][name] = obj
            return obj

        return wrap

    @classmethod
    def get_object(cls, key, name):
        """"""
        try:
            return cls.dicts[key][name]
        except:
            raise NotImplementedError("Key:{name} Undefined".format(name=name))


configmapper = ConfigMapper()
