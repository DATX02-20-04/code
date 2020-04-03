import unittest
from evolve.hparams import HParams

class TestHParams(unittest.TestCase):

    def test_generate(self):
        hparams = HParams({
            'int_range_test': {
                'type': 'range',
                'min': 1,
                'max': 10,
            },
            'float_range_test': {
                'type': 'range',
                'min': 0,
                'max': 1,
                'step': 0.1
            },
        }).generate()
        self.assertIsInstance(hparams['int_range_test'], int)
        self.assertIsInstance(hparams['float_range_test'], float)

