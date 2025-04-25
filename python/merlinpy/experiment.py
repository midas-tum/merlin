import yaml
import argparse
import pathlib
import pprint
import re
import os

__all__ = ["Experiment"]

def loadYaml(cfile, experiment):
    """
    [SOURCE] https://medium.com/swlh/python-yaml-configuration-with-environment-variables-parsing-77930f4273ac
    Load a yaml configuration file and resolve any environment variables
    The environment variables must have !ENV before them and be in this format
    to be parsed: ${VAR_NAME}.
    E.g.:
    database:
        host: !ENV ${HOST}
        port: !ENV ${PORT}
    app:
        log_path: !ENV '/var/${LOG_PATH}'
        something_else: !ENV '${AWESOME_ENV_VAR}/var/${A_SECOND_AWESOME_VAR}'
    :param str path: the path to the yaml file
    :param str data: the yaml data itself as a stream
    :param str tag: the tag to look for
    :return: the dict configuration
    :rtype: dict[str, T]
    """
    # pattern for global vars: look for ${word}
    pattern = re.compile('.*?\${(\w+)}.*?')
    # loader = yaml.SafeLoader  # SafeLoader broken in pyyaml > 5.2, see https://github.com/yaml/pyyaml/issues/266
    loader = yaml.Loader
    tag = "!ENV"

    # the tag will be used to mark where to start searching for the pattern
    # e.g. somekey: !ENV somestring${MYENVVAR}blah blah blah
    loader.add_implicit_resolver(tag, pattern, None)

    def constructor_env_variables(loader, node):
        """
        Extracts the environment variable from the node's value
        :param yaml.Loader loader: the yaml loader
        :param node: the current node in the yaml
        :return: the parsed string that contains the value of the environment
        variable
        """
        value = loader.construct_scalar(node)
        match = pattern.findall(value)  # to find all env variables in line
        if match:
            full_value = value
            for g in match:
                full_value = full_value.replace(
                    f'${{{g}}}', os.environ.get(g, g)
                )
            return full_value
        return value

    loader.add_constructor(tag, constructor_env_variables)

    with open(cfile, 'r') as f:
        config = yaml.load(f, Loader=loader)
        config = config[experiment]

        var_list = [(k, config[k]) for k in config.keys() if k.startswith('__') and k.endswith('__')]

        for var_key, var_val in var_list:
            del config[var_key]
            
        def replace(config, var_key, var_val):
            for key in config.keys():
                if isinstance(config[key], str):
                    config[key] = config[key].replace(var_key, f'{var_val}')
                if isinstance(config[key], dict):
                    replace(config[key], var_key, var_val)

        for var_key, var_val in var_list:
            replace(config, var_key, var_val)

    return config

class Experiment(object):
    def __init__(self, config=None):
        """
        Experiment:
        """

        super().__init__()

        # global
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--config', type=pathlib.Path, default='./config.yml', help='Path to the config file')
        self.parser.add_argument('--experiment', type=str, required=True, help='Experiment/model name')
        self.parser.add_argument('-t', '--task', type=int, default=0, help='Task ID to be performed')
        self.parser.add_argument('-n', '--name', type=str, default='run_00', help='Unique experiment name')
        self.parser.add_argument('--gpu', type=int, default=0, help='To be used GPU')
        action = self.parser.add_mutually_exclusive_group(required=False)
        action.add_argument('--tf', action='store_true', default=True, help='Use Tensorflow backend (default)')
        action.add_argument('--th', action='store_true', default=False, help='Use Pytorch backend')
        self.parser.add_argument('--resume', action='store_true', default=False, help='Resume training')
        self.parser.add_argument('--debug', action='store_true', default=False, help='Debug mode')
        action = self.parser.add_mutually_exclusive_group(required=False)
        action.add_argument('--train', action='store_true', default=True, help='Train the model')
        action.add_argument('--evaluate', action='store_true', default=False, help='Evaluate the model')
        action.add_argument('--predict', action='store_true', default=False, help='Predict the model')
        action.add_argument('--mode', type=str, default='train', help='Processing mode: ''train'' | ''evaluate'' | ''predict'' | ''zerofilling'' | ''cgsense''')

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