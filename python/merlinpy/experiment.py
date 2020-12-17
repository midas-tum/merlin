import yaml
import argparse
import pathlib
import pprint

__all__ = ["Experiment"]

def loadYaml(cfile, experiment):
    with open(cfile, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        config = config[experiment]

        var_list = [(k, config[k]) for k in config.keys() if k.startswith('__') and k.endswith('__')]

        for var_key, var_val in var_list:
            del config[var_key]
            
        for var_key, var_val in var_list:
            for key in config.keys():
                if isinstance(config[key], str):
                    config[key] = config[key].replace(var_key, f'{var_val}')

        return config

class Experiment(object):
    def __init__(self, config=None):
        """
        Experiment:
        """

        super().__init__()

        # global
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--config', type=pathlib.Path, default='./config.yml', help='Path to the nufft/binning config')
        self.parser.add_argument('--experiment', type=str, required=True, help='Path to the nufft/binning config')
        self.parser.add_argument('--gpu', type=int, default=0, help='Path to the nufft/binning config')
        self.parser.add_argument('--resume', action='store_true', default=False)

        self.config = config

    def __getattr__(self, key):
        try:                                                                    
            return self.config[key]                                                    
        except KeyError:                                                        
            raise AttributeError(key)       

    def parse(self):
        assert not self.config
        self.args = self.parser.parse_args()
        self.config = loadYaml(self.args.config, self.args.experiment)
        # self.config['config'] = self.args.config
        # self.config['experiment'] = self.args.experiment
        # self.config['gpu'] = self.args.gpu
        # self.config['resume'] = self.args.resume
            
        if 'exp_dir' in self.config:
            self.config['exp_dir'] = pathlib.Path(self.config['exp_dir'])

        if 'out_dir' in self.config:
            self.config['out_dir'] = pathlib.Path(self.config['out_dir'])

        for arg in vars(self.args):
            if arg in self.config:
                print(f'Overriding {arg} from argparse')
            self.config[arg] = getattr(self.args, arg)

    def __repr__(self):
        ppconfig = pprint.pformat(self.config, indent=4)
        return 'Experiment config\n' + \
              f'{ppconfig}\n'