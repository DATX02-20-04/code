from dataclasses import dataclass
from fractions import Fraction

import struct
import matplotlib.pyplot as plt

@dataclass
class Midi:
    @dataclass
    class Event:
        time: Fraction

        def __lt__(a, b): return a.time < b.time
        def __le__(a, b): return a.time <= b.time
        def __ge__(a, b): return a.time >= b.time
        def __gt__(a, b): return a.time > b.time

    @dataclass
    class MetaEvent(Event):
        type: int
        data: bytes

    @dataclass
    class ChannelEvent(Event):
        channel: int

    @dataclass
    class ControlEvent(ChannelEvent):
        type: int
        value: int

    @dataclass
    class ProgramEvent(ChannelEvent):
        program: int

    @dataclass
    class BaseNoteEvent(ChannelEvent):
        pitch: int
        velocity: int

    @dataclass
    class NoteEvent(BaseNoteEvent):
        pass

    @dataclass
    class NoteUpEvent(BaseNoteEvent):
        pass

    type: int
    rate: int
    tracks: [[Event]]

    def flatten(self):
        return sorted((event for track in self.tracks for event in track))

#
# = Parsing
# http://www.personal.kent.edu/~sbirch/Music_Production/MP-II/MIDI/an_introduction_to_midi_contents.htm
#

def read_midi_track(rate, data):
    data = list(data)[::-1]
    def varint():
        x = 0
        next = True
        while next:
            b = data.pop()
            b, next = b & 0x7F, b & 0x80
            x = (x << 7) | b
        return x

    t = 0
    command = None
    while data:
        t += Fraction(varint(), rate)

        if data[-1] & 0x80:
            command = data.pop()

        if command == 0xFF:
            yield Midi.MetaEvent(t, data.pop(), bytes(data.pop() for _ in range(varint())))
        else:
            type, chan = command & 0xF0, command & 0xF
            if type in (0x80, 0x90): # Key up and down, respectively
                pitch, vel = data.pop(), data.pop()
                if type == 0x80:  e = Midi.NoteUpEvent(t, chan, pitch, vel)
                elif vel == 0:    e = Midi.NoteUpEvent(t, chan, pitch, 0x40)
                else:             e = Midi.NoteEvent  (t, chan, pitch, vel)
                yield e
            # Ax: aftertouch (pitch, pressure)
            elif type == 0xB0:
                yield Midi.ControlEvent(t, chan, data.pop(), data.pop())
            elif type == 0xC0:
                yield Midi.ProgramEvent(t, chan, data.pop())
            # Dx: global aftertouch (pressure)
            # Ex: pitch bend (lsb, msb)
            else:
                raise ValueError(type)

def read_midi(f):
    def chunk(name):
        name2, size = struct.unpack(">4sL", f.read(8))
        assert name2 == name.encode("ascii")
        return f.read(size)

    type, ntracks, rate = struct.unpack(">HHH", chunk("MThd"))
    assert not rate & 0x8000, "SMPTE not supported"
    tracks = [list(read_midi_track(rate, chunk("MTrk"))) for _ in range(ntracks)]
    assert not f.read(1)
    return Midi(type, rate, tracks)


#
# = Serializing
# Only a left inverse of read_midi
#

def write_midi_track(rate, track, validate=True):
    data = bytearray()

    def varint(i):
        d = []
        while i >= 0x80:
            d.append(i & 0x7F)
            i = i >> 7
        d.append(i)
        d = [0x80 | x for x in d]
        d[0] &= 0x7F
        data.extend(d[::-1])

    t = 0
    for e in track:
        if validate:
            assert e.time >= t, f"{e}: out of order"
            assert not e.time % Fraction(1, rate), f"{e}: time is not a multiple of {Fraction(1, rate)}"
            if isinstance(e, Midi.ChannelEvent):
                assert 0 <= e.channel <= 15, f"{e}: invalid channel"
            if isinstance(e, Midi.BaseNoteEvent):
                assert 0 <= e.pitch <= 127, f"{e}: invalid pitch"
                assert 0 <= e.velocity <= 127, f"{e}: invalid velocity"

        varint(int((e.time - t) * rate))
        t = e.time
        if isinstance(e, Midi.MetaEvent):
            data.extend([0xFF, e.type])
            varint(len(e.data))
            data.extend(e.data)
        elif isinstance(e, Midi.NoteUpEvent):  data.extend([0x80 | e.channel, e.pitch, e.velocity])
        elif isinstance(e, Midi.NoteEvent):    data.extend([0x90 | e.channel, e.pitch, e.velocity])
        elif isinstance(e, Midi.ControlEvent): data.extend([0xB0 | e.channel, e.type, e.value])
        elif isinstance(e, Midi.ProgramEvent): data.extend([0xC0 | e.channel, e.program])
        else: raise ValueError(e)

    if validate:
        assert isinstance(e, Midi.MetaEvent) and e.type == 47 and not e.data, \
            f"{e}: invalid end event (should be 47)"

    return data

def write_midi(f, midi, validate=True):
    def chunk(name, data):
        f.write(struct.pack(">4sL", name.encode("ascii"), len(data)))
        f.write(data)
    chunk("MThd", struct.pack(">HHH", midi.type, len(midi.tracks), midi.rate))
    for track in midi.tracks:
        chunk("MTrk", write_midi_track(midi.rate, track, validate=validate))

#
# = Rendering
#

def display_midi(midi, axis=None, **kwargs):
    if axis is None: axis = plt.gca()
    notes = pairNotes(midi.flatten())

    axis.hlines(
        [s.pitch for s, e in notes],
        [s.time for s, e in notes],
        [e.time for s, e in notes],
        **kwargs
    )

#
# = Utils
#

def pairNotes(track):
    notes = []
    currentNotes = {}
    for e in track:
        if isinstance(e, Midi.BaseNoteEvent) and (e.channel, e.pitch) in currentNotes:
            notes.append((currentNotes.pop((e.channel, e.pitch)), e))
        if isinstance(e, Midi.NoteEvent):
            currentNotes[(e.channel, e.pitch)] = e

    for e2 in currentNotes.values():
        notes.append((e2, Midi.NoteUpEvent(e.time, e2.channel, e2.pitch, 0x40)))
    return sorted(notes)

import heapq
def limitLength(midi, l):
    for track in midi.tracks:
        track[:] = heapq.merge(track, [
            Midi.NoteUpEvent(s.time+l, s.channel, s.pitch, 0x40)
            for s, e in pairNotes(track)
            if s.time < e.time - l
        ])


if __name__ == "__main__":
    with open("test.midi", "rb") as f:
        midi = read_midi(f)
    display_midi(midi)
    plt.show()
