import random


def gen_range(start, stop, step):
    return random.randint(0, int((stop - start) / step)) * step + start

class HParams():
    def __init__(self, template):
        self.template = template

    def generate(self, mutation_rate=0.1, old=None):
        hparams = {}
        for key in self.template:
            if type(self.template[key]) == dict:
                type_ = self.template[key]['type']
                if type_ == 'range':
                    new_val = gen_range(self.template[key]['min'],
                                        self.template[key]['max'],
                                        self.template[key]['step']
                                                    if 'step' in self.template[key] else 1)
                    if old is not None:
                        old_val = old[key]
                        if random.random() < mutation_rate:
                            hparams[key] = new_val
                        else:
                            hparams[key] = old_val
                    else:
                        hparams[key] = new_val
            else:
                hparams[key] = self.template[key]

        return hparams
