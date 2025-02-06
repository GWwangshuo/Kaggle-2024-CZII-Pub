from omegaconf import OmegaConf


def get_config(config_path: str, valid_id: str, dot_list: list) -> dict:
    config_omega_from_yaml = OmegaConf.load(config_path)
    config_omega_from_args = OmegaConf.from_dotlist(dot_list)
    config_omega = OmegaConf.merge(config_omega_from_yaml, config_omega_from_args)
    config = OmegaConf.to_container(config_omega, resolve=True)  # DictConfig -> dict
    
    if valid_id is not None:
        config['metric']['valid_id'] = valid_id
        
    return config