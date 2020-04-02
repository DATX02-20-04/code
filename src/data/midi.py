from dataclasses import dataclass
from fractions import Fraction

import subprocess
import base64
import struct
# from yattag import Doc
# import IPython.display as display

@dataclass
class Midi:
    @dataclass
    class Event:
        time: Fraction

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

def write_midi_track(rate, track):
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

    return data

def write_midi(f, midi):
    def chunk(name, data):
        f.write(struct.pack(">4sL", name.encode("ascii"), len(data)))
        f.write(data)
    chunk("MThd", struct.pack(">HHH", midi.type, len(midi.tracks), midi.rate))
    for track in midi.tracks:
        chunk("MTrk", write_midi_track(midi.rate, track))

"""

#
# = Rendering to html (really svg)
#

def display_midi_track(doc, track):
    for e in track:
        if isinstance(e, Midi.NoteEvent):
            if e.velocity >  64: color = (255, 255-(e.velocity-64)*4, 0)
            if e.velocity == 64: color = (255, 255, 0)
            if e.velocity <  64: color = (e.velocity*4, 255, 0)
            doc.stag(
                "rect",
                x=float(e.time),
                y=127-e.pitch,
                width=float(e.end.time - e.time)+.1,
                height=1,
                fill="#%02X%02X%02X" % color,
                mask="url(#fade)",
                )

def display_midi(midi, xscale=16, yscale=2):
    r = getattr(display_midi, "r", 0) # So that it can be used multiple times in one block
    display_midi.r = r+1

    assert midi.type in (0, 1)
    w = max((e.time for track in midi.tracks for e in track), default=0)

    doc = Doc()
    for n, track in enumerate(midi.tracks):
        doc.stag("input", type="checkbox", checked="checked", id=f"c{r}-{n}")
        with doc.tag("style"):
            doc.asis(f"#c{r}-{n}:not(:checked) ~ * #g{r}-{n} {{ display: none }}")
    with doc.tag("div", style="overflow-x: scroll"):
        with doc.tag("svg", preserveAspectRatio="none", width=float(w*xscale), height=128*yscale, viewBox=f"0 0 {float(w)} 128"):
            with doc.tag("defs"):
                with doc.tag("linearGradient", id="fadeGrad", x2=1, y2=0):
                    doc.stag("stop", offset=0, **{"stop-color": "white", "stop-opacity": 1})
                    doc.stag("stop", offset=1, **{"stop-color": "white", "stop-opacity": 0})
                with doc.tag("mask", id="fade", maskContentUnits="objectBoundingBox"):
                    doc.stag("rect", width=1, height=1, fill="url(#fadeGrad)")

            for n, track in enumerate(midi.tracks):
                with doc.tag("g", id=f"g{r}-{n}"):
                    display_midi_track(doc, track)
    return display.HTML(doc.getvalue())


#
# = Rendering to audio
# Requires timidity to be installed: `!apt install timidity`
#

def play_midi(midi):
    proc = subprocess.Popen(["timidity", "-", "-Ow", "-o-"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    write_midi(proc.stdin, midi)
    data = proc.stdout.read()
    proc.wait()
    b64 = base64.b64encode(data).decode("ascii")
    return display.HTML(f'<audio controls><source src="data:audio/wav;base64,{b64}" /></audio>')
"""
