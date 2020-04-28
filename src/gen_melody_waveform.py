# This script should be used to generate a waveform by first running the music
# transformer and then generating each individual note with specgan.
# This is then combined into a single wavfile.

# load MIDI dataset

# maybe the training script should save the checkpoints to drive

import util
import tensorflow as tf
import numpy as np
import librosa
import scripts.gen_tone
import models.transformer.generate as transformer
# import models.gan.generate as gan
from models.common.training import Trainer
from models.gan.model import GAN
import data.process as pro
import data.midi as M

transformer_hparams = util.load_hparams('hparams/transformer.yml')
gan_hparams = util.load_hparams('hparams/gan.yml')

melody = transformer.generate(transformer_hparams)
midi = pro.decode_midi()(melody)
# with open("test.midi", "rb") as f:
# 	midi = M.read_midi(f)
midi = midi.flatten()
print(midi)

pitches = tf.cast([a.pitch-24 for a in midi if isinstance(a, M.Midi.NoteEvent)], tf.int32)
amp     = tf.cast([a.velocity / 127 for a in midi if isinstance(a, M.Midi.NoteEvent)], tf.float32)
vel     = tf.ones_like(pitches)*2

sr = 16000
samples_per_note = 8000
tone_length = sr


gan_stats = np.load('gan_stats.npz')
gan = GAN((256, 128), gan_hparams)
gan_trainer = Trainer(None, gan_hparams)
gan_ckpt = tf.train.Checkpoint(
    step=gan_trainer.step,
    generator=gan.generator,
    discriminator=gan.discriminator,
    gen_optimizer=gan.generator_optimizer,
    disc_optimizer=gan.discriminator_optimizer,
)
gan_trainer.init_checkpoint(gan_ckpt)

def generate_tones(pitches):
    seed = tf.random.normal((len(pitches), gan_hparams['latent_size']))
    pitches = tf.one_hot(pitches, gan_hparams['cond_vector_size'], axis=1)

    samples = gan.generator([seed, pitches], training=False)
    samples = tf.reshape(samples, [-1, 256, 128])
    audio = pro.pipeline([
        pro.denormalize(normalization='specgan', stats=gan_stats),
        pro.invert_log_melspec(gan_hparams['sample_rate']),
        list,     # Stupid workaround becuase invert_log_melspec only does
        np.array, # one spectrogram at a time
    ])(samples)
    return audio

def generate_all_tones(pitches, amp):
    for a in range(0, len(pitches), 32):
        print(a, len(pitches))
        yield from generate_tones(pitches[a:a+32]) * amp[a:a+32,None]

times = [int(a.time * samples_per_note) for a in midi if isinstance(a, M.Midi.NoteEvent)]
out = tf.zeros(max(times) + tone_length)
for time, sound in zip(times, generate_all_tones(pitches, amp)):
    sound = sound[:sr]
    out += tf.pad(sound, [(time, len(out)-len(sound)-time)])

librosa.output.write_wav('gen_melody.wav', out.numpy(), sr=sr, norm=True)
