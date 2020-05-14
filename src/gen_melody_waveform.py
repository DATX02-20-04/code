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
from models.new_gan.process import load as gan_load
from models.upscaler.process import load as upscaler_load
# import models.gan.generate as gan
from models.common.training import Trainer
from models.new_gan.model import GAN
from models.new_gan.process import invert
import data.process as pro
import data.midi as M

transformer_hparams = util.load_hparams('hparams/transformer.yml')
gan_hparams = util.load_hparams('hparams/new_gan.yml')
upscaler_hparams = util.load_hparams('hparams/upscaler.yml')

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

dataset, gan_stats = gan_load(hparams)
gan = GAN(gan_hparams, gan_stats)

block = tf.Variable(0)
step = tf.Variable(0)
pitch_start = 0
pitch_end = hparams['pitches']
step_size = 1
seed_pitches = tf.range(pitch_start, pitch_start+pitch_end, step_size)
seed = tf.Variable(tf.random.normal([seed_pitches.shape[0], hparams['latent_dim']]))
seed_pitches = tf.one_hot(seed_pitches, hparams['pitches'], axis=1)

ckpt = tf.train.Checkpoint(
    gan=gan,
    seed=seed,
    generator_optimizer=gan.generator_optimizer,
    discriminator_optimizer=gan.discriminator_optimizer,
    block=block,
    step=step,
)

manager = tf.train.CheckpointManager(ckpt,
                                        os.path.join(hparams['save_dir'], 'ckpts', hparams['name']),
                                        max_to_keep=3)

ckpt.restore(manager.latest_checkpoint)
if manager.latest_checkpoint:
    print("Restored from {} block {}".format(manager.latest_checkpoint, block.numpy()))
else:
    print("Initializing from scratch.")

_, upscaler_stats = upscaler_load()
upscaler = Upscaler(hparams, upscaler_stats)

checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

try:
    upscaler.model.load_weights(checkpoint_path)
except:
    print("Initializing from scratch.")

def generate_tones(pitches):
    seed = tf.random.normal((len(pitches), gan_hparams['latent_size']))
    pitches = tf.one_hot(pitches, gan_hparams['cond_vector_size'], axis=1)

    samples = gan.generator([seed, pitches], training=False)
    phases = upscaler(samples, training=False)
    samples = tf.reshape(samples, [-1, 128, 256])
    audio = invert((samples, phases))
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
