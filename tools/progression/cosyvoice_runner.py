#!/usr/bin/env python3
"""Isolated CosyVoice 3 worker used by speech.py.

Run this with CosyVoice's Python 3.10, never the RetroGames trainer venv.
The parent process writes one JSON job so the 9 GB model is loaded once for a
whole batch of podcast turns.
"""
import json
import os
import sys
from contextlib import contextmanager
import importlib
import logging
import re


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


@contextmanager
def _cached_normalizer():
    """WeText's default downloader checks remote revisions even with local FSTs.

    Prefer its supported offline lookup, but only accept a complete TN cache.
    Patch this one lookup during model construction; never edit site-packages.
    """
    try:
        module = importlib.import_module('wetext.wetext')
    except ImportError:
        yield
        return
    original = module.snapshot_download
    public = importlib.import_module('wetext')
    original_normalizer = public.Normalizer

    def local_normalizer(*args, **kwargs):
        # Older installations cached English FSTs at the snapshot root, while
        # current WeText expects en/tn/*.fst AND Chinese files for lang=auto.
        # Supply the supported explicit paths instead of redownloading assets.
        if not args and not kwargs.get('tagger_path') and kwargs.get('lang','auto') in ('auto','en'):
            try:
                folder = original('pengzhendong/wetext',local_files_only=True)
                pair = [os.path.join(folder,'en_tn_'+name+'.fst')
                        for name in ('tagger','verbalizer')]
                if kwargs.get('operator','tn')=='tn' and all(os.path.isfile(p) for p in pair):
                    english = original_normalizer(tagger_path=pair[0],verbalizer_path=pair[1],lang='en')
                    class LocalEnglish:
                        other = None
                        def normalize(self, text, **options):
                            if re.search(r'[\u4e00-\u9fff]',text):
                                # Never run Chinese through English rules.
                                if self.other is None:
                                    self.other = original_normalizer(*args,**kwargs)
                                return self.other.normalize(text,**options)
                            return english.normalize(text,**options)
                    print('    text normalizer: using cached English FSTs',flush=True)
                    return LocalEnglish()
            except (OSError, ValueError):
                pass
        return original_normalizer(*args, **kwargs)

    def cached_first(repo, *args, **kwargs):
        if repo == 'pengzhendong/wetext' and not args and 'local_files_only' not in kwargs:
            try:
                folder = original(repo, local_files_only=True, **kwargs)
                required = [os.path.join(folder,lang,'tn',name+'.fst')
                            for lang in ('en','zh') for name in ('tagger','verbalizer')]
                if all(os.path.isfile(path) for path in required):
                    print('    text normalizer: using cached local assets', flush=True)
                    return folder
            except (OSError, ValueError):
                pass
        return original(repo, *args, **kwargs)

    module.snapshot_download = cached_first
    public.Normalizer = local_normalizer
    try:
        yield
    finally:
        module.snapshot_download = original
        public.Normalizer = original_normalizer


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
    # Optional Hugging Face file probes are not failures; don't dump HTTP DEBUG
    # traffic from CosyVoice's global logging setup into the owner's terminal.
    logging.getLogger('urllib3').setLevel(logging.WARNING)

    if not torch.cuda.is_available():
        raise RuntimeError("CosyVoice 3 requires CUDA for this pipeline")
    print('    loading CPU speech checker ...', flush=True)
    recognizer = Recognizer()
    print('    loading CosyVoice ...', flush=True)
    with _cached_normalizer():
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
            print('    speech [%d/%d] generating, attempt %d ...' %
                  (index+1, len(blocks), attempt+1), flush=True)
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
            print('    speech [%d/%d] checking words ...' % (index+1,len(blocks)), flush=True)
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
