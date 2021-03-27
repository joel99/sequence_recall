#!/usr/bin/env python3
# Author: Joel Ye

from typing import List, Optional, Union

from yacs.config import CfgNode as CN

DEFAULT_CONFIG_DIR = "config/"
CONFIG_FILE_SEPARATOR = ","

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------
_C = CN()
_C.SEED = 0

# Name of experiment
_C.VARIANT = "test"

# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------
_C.TASK = CN()
_C.TASK.ALPHABET_SIZES = [3] # TODO auto-generate
_C.TASK.T_RANGE = [5, 10] # extrema

# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
_C.MODEL = CN()
_C.MODEL.TYPE = 'gru'
_C.MODEL.HIDDEN_SIZE = 32
_C.MODEL.HOLD = 0 # Number of timesteps after encoding before go cue is given.

# -----------------------------------------------------------------------------
# Train Config
# -----------------------------------------------------------------------------
_C.TRAIN = CN()

_C.TRAIN.BATCH_SIZE = 32
_C.TRAIN.EPOCHS = 120
_C.TRAIN.WEIGHT_DECAY = 0.0

def get_cfg_defaults():
  """Get default LFADS config (yacs config node)."""
  return _C.clone()

def get_config(
    config_paths: Optional[Union[List[str], str]] = None,
    opts: Optional[list] = None,
) -> CN:
    r"""Create a unified config with default values overwritten by values from
    :p:`config_paths` and overwritten by options from :p:`opts`.

    :param config_paths: List of config paths or string that contains comma
        separated list of config paths.
    :param opts: Config options (keys, values) in a list (e.g., passed from
        command line into the config. For example,
        :py:`opts = ['FOO.BAR', 0.5]`. Argument can be used for parameter
        sweeping or quick tests.
    """
    config = get_cfg_defaults()
    if config_paths:
        if isinstance(config_paths, str):
            if CONFIG_FILE_SEPARATOR in config_paths:
                config_paths = config_paths.split(CONFIG_FILE_SEPARATOR)
            else:
                config_paths = [config_paths]

        for config_path in config_paths:
            config.merge_from_file(config_path)

    if opts:
        config.merge_from_list(opts)

    config.freeze()
    return config

