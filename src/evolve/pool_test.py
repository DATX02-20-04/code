import unittest
import random
import math
import numpy as np
from evolve.pool import Pool

class TestPool(unittest.TestCase):

    def test_pool_populate(self):
        pool = Pool(lambda hp: self.assertTrue('test' in hp), {
            'test': {
                'type': 'range',
                'min': 1,
                'max': 5
            }
        }).populate(10)

        self.assertEqual(len(pool.pool), 10)

    def test_pool_evaluate(self):
        pool = (Pool(lambda hp: {}, {})
                .populate(10)
                .evaluate(lambda x: random.random()))

        last = math.inf
        for f, i in pool.fitness:
            self.assertLess(f, last)
            last = f

    def test_pool_select(self):
        pool = (Pool(lambda hp: {}, {})
                .populate(10)
                .evaluate(lambda x: random.random()))

        self.assertEqual(len(pool.pool), 10)
        pool.select(100)
        self.assertEqual(len(pool.pool), 100)

    def test_pool_evolve(self):
        pool = Pool(lambda hp: hp['test'], {
            'test': {
                'type': 'range',
                'min': 0,
                'max': 10000,
            }
        })
        target = 1337
        generations = []
        for _ in range(10):
            pool.populate(500, 1)
            generation = 0
            while pool.best is None or pool.best[1] != target:
                pool = (pool.populate(500, 0.5)
                        .evaluate(lambda x: 1/(abs(x-target)+1))
                        .select(500))
                generation += 1
            generations.append(generation)
        gen_avg = np.array(generations).mean()
        print(gen_avg)
        self.assertTrue(gen_avg < 100)
