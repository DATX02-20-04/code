import random
from evolve.hparams import HParams

class Pool():
    def __init__(self, create_model, hparams_template):
        self.pool = []
        self.create_model = create_model
        self.hparams = HParams(hparams_template)
        self.best = None

    def generate(self, mutation_rate, old):
        hp = self.hparams.generate(mutation_rate, old)
        return (hp, self.create_model(hp))

    def populate(self, size, mutation_rate=0.1):
        self.best = None
        old = lambda i: self.pool[i][0] if 0 <= i < len(self.pool) and mutation_rate != 1 else None

        self.pool = [self.generate(mutation_rate, old(i)) for i in range(size)]
        return self

    def evaluate(self, fitness_fn):
        self.fitness = [(fitness_fn(model[1]), i)
                        for i, model in enumerate(self.pool)]
        self.fitness.sort(reverse=True)
        self.best = self.pool[self.fitness[0][1]]
        return self

    def select(self, count):
        # Always keep best one
        f, i_best = self.fitness[0]
        selected = [i_best]

        # Tournament selection
        for _ in range(count-1):
            f1, i1 = self.fitness[random.randrange(len(self.fitness))]
            f2, i2 = self.fitness[random.randrange(len(self.fitness))]
            if f1 > f2:
                selected.append(i1)
            elif f2 > f1:
                selected.append(i2)
            else:
                if random.random() < 0.5:
                    selected.append(i1)
                else:
                    selected.append(i2)

        self.pool = [self.pool[i] for i in selected]

        return self
