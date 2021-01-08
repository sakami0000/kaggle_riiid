import copy
from pathlib import Path
import yaml


class Config(dict): 
    """A dictionary class that allows for attribute-style access of values.
    """
    __setattr__ = dict.__setitem__

    def __getattr__(self, key):
        value = super().get(key)
        if isinstance(value, dict):
            return Config(value)
        return value

    def __deepcopy__(self, memo=None):
        """Prevent errors in the `copy.deepcopy` method.

        References
        ----------
        - https://stackoverflow.com/questions/49901590/python-using-copy-deepcopy-on-dotdict
        """
        return Config(copy.deepcopy(dict(self), memo=memo))

    @classmethod
    def load(cls, config_path: str) -> 'Config':
        """Load a config file.

        Parameters
        ----------
        config_path : str
            Path to config file.

        Returns
        -------
        Config
            Configuration parameters.
        """
        if Path(config_path).exists():
            with open(config_path) as f:
                config = cls(yaml.safe_load(f))
            return config
        else:
            raise ValueError(f'Configuration file {config_path} does not exist.')
