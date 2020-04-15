# This script should be used to generate a waveform by first running the music
# transformer and then generating each individual note with specgan.
# This is then combined into a single wavfile.

# load MIDI dataset

# maybe the training script should save the checkpoints to drive

import util
import models.transformer as transformer

hparams = util.load_hparams('hparams/transformer.yml')
melody = transformer.generate(hparams)

print(melody)
