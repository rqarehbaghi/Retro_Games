#!/usr/bin/env python3
"""Isolated CosyVoice 3 worker used by speech.py.

Run this with CosyVoice's Python 3.10, never the RetroGames trainer venv.
The parent process writes one JSON job so the 9 GB model is loaded once for a
whole batch of podcast turns.
"""
import json
import os
import sys


def _transcript(reference):
    """Return an optional exact transcript stored beside a reference WAV."""
    path = os.path.splitext(reference)[0] + ".txt"
    if not os.path.isfile(path):
        return ""
    with open(path, encoding="utf-8") as handle:
        return handle.read().strip()


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: cosyvoice_runner.py JOB.json")
    with open(sys.argv[1], encoding="utf-8") as handle:
        job = json.load(handle)

    home = os.getcwd()
    matcha = os.path.join(home, "third_party", "Matcha-TTS")
    if home not in sys.path:
        sys.path.insert(0, home)
    if matcha not in sys.path:
        sys.path.append(matcha)

    import torch
    import torchaudio
    from cosyvoice.cli.cosyvoice import AutoModel

    if not torch.cuda.is_available():
        raise RuntimeError("CosyVoice 3 requires CUDA for this pipeline")
    model = AutoModel(model_dir=job["model_dir"])
    style = job["style"]
    blocks = job.get("blocks") or []
    for index, block in enumerate(blocks):
        text = block["text"].strip()
        reference = os.path.abspath(block["reference"])
        transcript = _transcript(reference)
        if transcript:
            prompt = "You are a helpful assistant.<|endofprompt|>" + transcript
            generated = model.inference_zero_shot(
                text, prompt, reference, stream=False)
            mode = "zero-shot transcript"
        else:
            generated = model.inference_instruct2(
                text, style, reference, stream=False)
            mode = "podcast instruction"
        pieces = []
        for result in generated:
            samples = result["tts_speech"]
            pieces.append(samples.unsqueeze(0) if samples.ndim == 1 else samples)
        if not pieces:
            raise RuntimeError("CosyVoice generated no audio for block %d" % index)
        audio = torch.cat(pieces, dim=1).detach().cpu()
        output = os.path.abspath(block["output"])
        os.makedirs(os.path.dirname(output), exist_ok=True)
        torchaudio.save(output, audio, model.sample_rate)
        print("    cosyvoice [%d/%d] %s %.2fs"
              % (index + 1, len(blocks), mode,
                 audio.shape[-1] / float(model.sample_rate)), flush=True)


if __name__ == "__main__":
    main()
