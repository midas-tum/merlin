def fix_dict_in_wandb_config(wandb):
    """"Adapted from [https://github.com/wandb/client/issues/982]"""
    config = dict(wandb.config)
    for k, v in config.copy().items():
        if '.' in k:
            keys = k.split('.')
            if len(keys) == 2:
                new_key = k.split('.')[0]
                inner_key = k.split('.')[1]
                if new_key not in config.keys():
                    config[new_key] = {}
                config[new_key].update({inner_key: v})
                del config[k]
            elif len(keys) == 3:
                new_key_1 = k.split('.')[0]
                new_key_2 = k.split('.')[1]
                inner_key = k.split('.')[2]
                
                if new_key_1 not in config.keys():
                    config[new_key_1] = {}
                if new_key_2 not in config[new_key_1].keys():
                    config[new_key_1][new_key_2] = {}
                config[new_key_1][new_key_2].update({inner_key: v})
                del config[k]
            else: # len(keys) > 3
                raise ValueError('Nested dicts with depth>3 are currently not supported!')
    
    wandb.config = wandb.Config()
    for k, v in config.items():
        wandb.config[k] = v