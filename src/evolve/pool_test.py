import unittest
import random
import math
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
        pool = Pool(lambda hp: hp['test']+hp['other'], {
            'test': {
                'type': 'range',
                'min': 0,
                'max': 10000,
                'step': 1,
            },
            'other': 2
        })
        for _ in range(200):
            pool = (pool.populate(200, 0.1)
                    .evaluate(lambda x: 1/(abs(x-1337)+1))
                    .select(200))
        self.assertTrue(pool.fitness[0][0]>0.1)
