#!/usr/bin/env python3
"""Isolated CosyVoice 3 worker used by speech.py.

Run this with CosyVoice's Python 3.10, never the RetroGames trainer venv.
The parent process writes one JSON job so the 9 GB model is loaded once for a
whole batch of podcast turns.
"""
import json
import os
import sys


def _prepare_imports():
    # Direct execution adds tools/progression to sys.path. Its select.py then
    # shadows Python's socket-select module, breaking requests/urllib3 downloads.
    # Import our helpers through the package, never through this script folder.
    worker_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path[:] = [p for p in sys.path
                   if os.path.abspath(p or os.getcwd()) != worker_dir]
    root = os.path.dirname(os.path.dirname(worker_dir))
    if root not in sys.path:
        sys.path.insert(0, root)


def _transcript(reference):
    """Return an optional exact transcript stored beside a reference WAV."""
    path = os.path.splitext(reference)[0] + ".txt"
    if not os.path.isfile(path):
        return ""
    with open(path, encoding="utf-8") as handle:
        return handle.read().strip()


def main():
    _prepare_imports()
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
    from tools.progression.speech_quality import Recognizer, assess

    if not torch.cuda.is_available():
        raise RuntimeError("CosyVoice 3 requires CUDA for this pipeline")
    recognizer = Recognizer()
    model = AutoModel(model_dir=job["model_dir"])
    default_style = job["style"]
    default_speed = float(job.get("speed", 1.0))
    blocks = job.get("blocks") or []
    for index, block in enumerate(blocks):
        text = block["text"].strip()
        reference = os.path.abspath(block["reference"])
        style = block.get("style") or default_style
        speed = float(block.get("speed", default_speed))
        transcript = _transcript(reference)
        output = os.path.abspath(block["output"])
        os.makedirs(os.path.dirname(output), exist_ok=True)
        attempts = []
        for attempt in range(3):
            if transcript:
                prompt = "You are a helpful assistant.<|endofprompt|>" + transcript
                generated = model.inference_zero_shot(
                    text, prompt, reference, stream=False, speed=speed)
                mode = "zero-shot transcript"
            else:
                instruction = (style if attempt == 0 else
                               'You are a helpful assistant. Speak clearly.<|endofprompt|>')
                generated = model.inference_instruct2(
                    text, instruction, reference, stream=False, speed=speed)
                mode = "podcast instruction"
            pieces = []
            for result in generated:
                samples = result['tts_speech']
                pieces.append(samples.unsqueeze(0) if samples.ndim == 1 else samples)
            if not pieces:
                raise RuntimeError('CosyVoice generated no audio for block %d' % index)
            audio = torch.cat(pieces, dim=1).detach().cpu()
            report = assess(text, recognizer.transcribe(audio, model.sample_rate),
                            audio.shape[-1] / float(model.sample_rate))
            attempts.append(report)
            with open(output + '.quality.json', 'w', encoding='utf-8') as handle:
                json.dump(attempts, handle, indent=2)
            if report['accepted']:
                break
            print('Speech check rejected block %d attempt %d: %s' %
                  (index, attempt + 1, report['heard']), flush=True)
        else:
            raise RuntimeError('Speech validation failed: %s.quality.json; '
                               'no finished narration will be published' % output)
        torchaudio.save(output, audio, model.sample_rate)
        print("    cosyvoice [%d/%d] %s %.2fs"
              % (index + 1, len(blocks), mode,
                 audio.shape[-1] / float(model.sample_rate)), flush=True)


if __name__ == "__main__":
    main()
