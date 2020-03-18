import yaml

def load_hparams(path):
    with open(path, 'r') as stream:
        try:
            defaults = yaml.safe_load(stream)['hparams']
        except yaml.YAMLError as e:
            print(e)

    return defaults
