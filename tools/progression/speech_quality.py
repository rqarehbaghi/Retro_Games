"""Local transcript checks; no voice or script is sent to an API."""
import re
import unicodedata


def words(text):
    text = unicodedata.normalize('NFKC', text).lower().replace('’', "'")
    for short, full in {"i'm": "i am", "you're": "you are", "it's": "it is",
                        "don't": "do not", "doesn't": "does not",
                        "i'll": "i will", "that's": "that is"}.items():
        text = text.replace(short, full)
    return re.findall(r"[a-z0-9]+", text)


def assess(expected, heard, seconds):
    """Levenshtein word error rate plus broad duration guard, not an ASR oracle."""
    a, b = words(expected), words(heard)
    row = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        next_row = [i]
        for j, y in enumerate(b, 1):
            next_row.append(min(row[j] + 1, next_row[-1] + 1,
                                row[j-1] + (x != y)))
        row = next_row
    errors = row[-1]
    rate = errors / max(1, len(a))
    duration_ok = 0.25 <= seconds <= max(5.0, len(a) * 1.1 + 2)
    accepted = bool(a and b) and duration_ok and errors <= max(1, int(len(a)*0.18))
    return dict(accepted=accepted, word_error_rate=rate, errors=errors,
                seconds=seconds, expected=expected, heard=heard)


class Recognizer:
    def __init__(self):
        from transformers import WhisperProcessor, WhisperForConditionalGeneration
        self.processor = WhisperProcessor.from_pretrained('openai/whisper-base.en')
        self.model = WhisperForConditionalGeneration.from_pretrained(
            'openai/whisper-base.en').eval()  # CPU: no second GPU model

    def transcribe(self, audio, rate):
        from scipy.signal import resample_poly
        from math import gcd
        import torch
        samples = audio.squeeze().numpy()
        common = gcd(rate, 16000)
        samples = resample_poly(samples, 16000 // common, rate // common)
        # Long turns are split into <=25s chunks rather than silently truncated.
        text = []
        for offset in range(0, len(samples), 400000):
            inputs = self.processor(samples[offset:offset+400000], sampling_rate=16000,
                                    return_tensors='pt', return_attention_mask=True)
            with torch.inference_mode():
                result = self.model.generate(**inputs, max_new_tokens=256)
            text.append(self.processor.batch_decode(result, skip_special_tokens=True)[0])
        return ' '.join(text)
