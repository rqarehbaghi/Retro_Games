#!/usr/bin/env python3
"""Generate a short two-host audition through a progression TTS backend."""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tools.progression import speech  # noqa: E402


HOST_ONE = "I trained these agents, so I know exactly where the difficult parts are."
HOST_TWO = "That sounds confident, but let us see which checkpoint actually survives."


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--voice-model", default=speech.DEFAULT)
    parser.add_argument("--voice-name", required=True,
                        help="host1.wav,host2.wav")
    parser.add_argument("--host1", default=HOST_ONE, help="host one's test line")
    parser.add_argument("--host2", default=HOST_TWO, help="host two's test line")
    parser.add_argument("--out", default=os.path.join(
        ROOT, "progression_out", "cosyvoice3_podcast_audition.wav"))
    args = parser.parse_args()

    speech.validate_voice(args.voice_model, args.voice_name, podcast=True)
    voices = [part.strip() for part in args.voice_name.split(",") if part.strip()]
    output = os.path.abspath(args.out)
    block_dir = os.path.splitext(output)[0] + "_blocks"
    clips = speech.speak([args.host1, args.host2], block_dir,
                         backend=args.voice_model, voice=voices)

    import numpy as np
    import soundfile as sf
    audio = []
    rate = None
    for path, _seconds in clips:
        samples, current_rate = sf.read(path, dtype="float32", always_2d=False)
        if samples.ndim > 1:
            samples = samples.mean(axis=1)
        if rate is not None and current_rate != rate:
            raise RuntimeError("voice blocks used different sample rates")
        rate = current_rate
        audio.append(samples)
        audio.append(np.zeros(int(current_rate * 0.35), dtype="float32"))
    os.makedirs(os.path.dirname(output), exist_ok=True)
    sf.write(output, np.concatenate(audio[:-1]), rate)
    print(output)


if __name__ == "__main__":
    main()
