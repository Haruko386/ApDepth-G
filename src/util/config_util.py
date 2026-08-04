# Author: Jiawei Wang
# Last modified: 2026-08-04

import os
from typing import Any, Dict, Tuple

import omegaconf
from omegaconf import OmegaConf


PFR_CONFIG_KEY = "prior_guided_farfield_rectification"
PFR_OPTION_NAMES = {
    "top_ratio",
    "prior_far_quantile",
    "ref_near_quantile",
    "min_candidate_ratio",
    "margin",
    "strength",
    "smooth_kernel",
    "prior_far_is_larger",
    "depth_far_is_larger",
    "eps",
}


def load_pfr_config(config_path: str) -> Tuple[bool, Dict[str, Any]]:
    """Load and validate the inference-time PFR configuration."""
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Runtime config not found: {config_path}")

    config = OmegaConf.load(config_path)
    if not isinstance(config, omegaconf.DictConfig):
        raise TypeError(f"Runtime config must be a YAML mapping: {config_path}")

    section = config.get(PFR_CONFIG_KEY)
    if section is None:
        return False, {}
    if not isinstance(section, omegaconf.DictConfig):
        raise TypeError(f"'{PFR_CONFIG_KEY}' must be a YAML mapping")

    pfr_config = OmegaConf.to_container(section, resolve=True)
    enabled = pfr_config.pop("enabled", False)
    if not isinstance(enabled, bool):
        raise TypeError(f"'{PFR_CONFIG_KEY}.enabled' must be true or false")

    unknown_options = set(pfr_config) - PFR_OPTION_NAMES
    if unknown_options:
        unknown = ", ".join(sorted(unknown_options))
        raise ValueError(f"Unknown options in '{PFR_CONFIG_KEY}': {unknown}")

    return enabled, pfr_config


def recursive_load_config(config_path: str) -> OmegaConf:
    conf = OmegaConf.load(config_path)

    output_conf = OmegaConf.create({})

    # Load base config. Later configs on the list will overwrite previous
    base_configs = conf.get("base_config", default_value=None)
    if base_configs is not None:
        assert isinstance(base_configs, omegaconf.listconfig.ListConfig)
        for _path in base_configs:
            assert (
                _path != config_path
            ), "Circulate merging, base_config should not include itself."
            _base_conf = recursive_load_config(_path)
            output_conf = OmegaConf.merge(output_conf, _base_conf)

    # Merge configs and overwrite values
    output_conf = OmegaConf.merge(output_conf, conf)

    return output_conf


def find_value_in_omegaconf(search_key, config):
    result_list = []

    if isinstance(config, omegaconf.DictConfig):
        for key, value in config.items():
            if key == search_key:
                result_list.append(value)
            elif isinstance(value, (omegaconf.DictConfig, omegaconf.ListConfig)):
                result_list.extend(find_value_in_omegaconf(search_key, value))
    elif isinstance(config, omegaconf.ListConfig):
        for item in config:
            if isinstance(item, (omegaconf.DictConfig, omegaconf.ListConfig)):
                result_list.extend(find_value_in_omegaconf(search_key, item))

    return result_list


if "__main__" == __name__:
    conf = recursive_load_config("config/train_base.yaml")
    print(OmegaConf.to_yaml(conf))
