#!/usr/bin/env python3
"""
Write the captions, commentary and descriptions with a local LLM.

Everything studio.py says is otherwise drawn from hardcoded pools, which is
fine for one video and obvious by the fifth: the same eight jokes in rotation,
the same description under every upload. This hands the writing to a model
running on your own machine instead, so each run is written fresh against what
actually happened in THAT run.

    ollama serve                     # in one terminal
    python studio.py --game <id> --writer ollama

NOTHING IS REQUIRED. With no --writer, or with Ollama not running, or if the
model returns something unparseable, studio.py uses its tables exactly as
before. A model that is merely unavailable must never cost you a recording you
have already played, so every entry point here returns None on any failure and
the caller falls back.

WHY OLLAMA. It is one install, it serves an HTTP API on localhost, it needs no
Python dependency here (urllib is enough), and it supports GRAMMAR-CONSTRAINED
decoding: passing a JSON schema as `format` restricts the sampler to tokens
that can legally continue a valid document, so a small model cannot wander off
and produce prose where a list was wanted.

    sudo apt-get install -y zstd     # the installer unpacks with it and
                                     # stops with an error if it is missing
    curl -fsSL https://ollama.com/install.sh | sh
    ollama pull qwen3:30b-a3b        # see MODEL_NOTES below; ~18GB download

On WSL2, systemd is often not running, so the installer's service never
starts and nothing is listening. Run `ollama serve` in its own terminal.
`ollama ps` then says whether a loaded model is on the GPU or has fallen back
to CPU -- on CPU a 30B model is far too slow to be worth waiting for, and
studio.py will simply appear to hang rather than fail.

The model gets the real event timeline -- the frame and kind of every death,
power-up, coin and level clear -- so it writes about what happened rather than
inventing a run.
"""
import json
import os
import urllib.error
import urllib.request

DEFAULT_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
DEFAULT_MODEL = "qwen3:30b-a3b"
TIMEOUT = 180

MODEL_NOTES = """\
Picking a model for a 24GB card (RTX 4090):

  qwen3:30b-a3b     Mixture-of-experts, ~30B total but only ~3.3B active per
                    token, so it loads like a 30B (~17GB at Q4) and generates
                    like a small model. The best speed/quality trade here.
  qwen3.6:27b       Dense, ~17GB at Q4, 256K context. Slower per token than
                    the MoE above, generally a bit sharper.
  qwen3:14b         ~9GB. Leaves plenty of headroom; noticeably blunter jokes.
  qwen3:8b          ~5GB. Fast, and it shows.

Qwen is Apache-2.0 and needs no account. Any Ollama model works -- pass
--writer-model. Bigger is funnier up to a point; the limiting factor for this
task is instruction-following on the length limits, not world knowledge.
"""

# The voice, stated once. Everything below asks for a different artifact in it.
VOICE = """\
You are writing for a retro gaming channel. The footage is one unbroken take of
a human playing a classic console game, mistakes included.

Your voice: a commentator who has watched a great deal of this and is not
easily impressed, but is FAIR. You take mistakes apart with relish. When the
player does something genuinely well you say so, as a backhanded compliment
rather than withholding it -- commentary that only ever sneers stops being
funny immediately, because nothing is at stake in the praise.

Be specific and be funny. Dry, observational, occasionally savage. Never
generic hype, never "epic", never emoji. Refer to the player in the third
person. The audience is adults who played this game as children, so nostalgia
lands and condescension does not."""


def available(host=DEFAULT_HOST, timeout=3):
    """Is an Ollama server actually there? Checked before anything slow."""
    try:
        with urllib.request.urlopen(host.rstrip("/") + "/api/tags", timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


def installed_models(host=DEFAULT_HOST):
    try:
        with urllib.request.urlopen(host.rstrip("/") + "/api/tags", timeout=5) as r:
            return [m["name"] for m in json.load(r).get("models", [])]
    except Exception:
        return []


def generate(prompt, schema, model=DEFAULT_MODEL, host=DEFAULT_HOST,
             timeout=TIMEOUT, temperature=0.9, verbose=True):
    """One constrained generation. Returns the parsed object, or None.

    `schema` is a JSON schema passed as `format`, which makes Ollama restrict
    decoding to tokens that keep the output valid against it -- the difference
    between parsing reliably and hoping."""
    body = json.dumps({
        "model": model,
        "prompt": prompt,
        "system": VOICE,
        "format": schema,
        "stream": False,
        # High enough that two runs of the same footage do not read alike,
        # which is the entire point of doing this.
        "options": {"temperature": temperature, "top_p": 0.95},
    }).encode()
    req = urllib.request.Request(host.rstrip("/") + "/api/generate", data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            payload = json.load(r)
        return json.loads(payload["response"])
    except urllib.error.URLError as exc:
        if verbose:
            print(f"  (writer: cannot reach Ollama at {host}: {exc.reason})")
    except (KeyError, ValueError) as exc:
        if verbose:
            print(f"  (writer: model returned nothing usable: {exc})")
    except Exception as exc:                                  # noqa: BLE001
        if verbose:
            print(f"  (writer: {exc.__class__.__name__}: {exc})")
    return None


def _timeline(events, fps):
    """The run, as something a model can read."""
    label = {
        "death": "died", "shrink": "shrank to small", "powerdown": "lost a power tier",
        "powerup": "collected a power-up", "1up": "got an extra life",
        "coin": "collected a coin", "clear": "finished the level",
        "pipe": "went down a pipe", "score": "scored points",
    }
    out = []
    for frame, kind, detail in events:
        secs = frame / fps
        out.append("  %d:%05.2f  %s (%s)" % (secs // 60, secs % 60,
                                             label.get(kind, kind), detail))
    return "\n".join(out) or "  (nothing notable happened)"


def _context(game, level, duration_s, players, events, fps):
    return (
        "GAME: %s\nSECTION: %s\nLENGTH: %.0f seconds\nPLAYERS: %s\n\n"
        "WHAT HAPPENED, in order:\n%s\n"
        % (game, level or "unspecified", duration_s,
           "one human" if players == 1 else "one human and one AI",
           _timeline(events, fps)))


# --------------------------------------------------------------- captions --
CAPTION_SCHEMA = {
    "type": "object",
    "properties": {
        "captions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "at": {"type": "number"},
                    "text": {"type": "string"},
                },
                "required": ["at", "text"],
            },
        }
    },
    "required": ["captions"],
}


def captions(game, level, duration_s, players, events, fps, max_chars=70,
             **kw):
    """Timed on-screen captions. Returns [{at, text}] or None."""
    prompt = (
        _context(game, level, duration_s, players, events, fps) +
        "\nWrite on-screen captions for this run.\n\n"
        "RULES:\n"
        "- One caption per interesting moment above. Skip the dull ones.\n"
        "- At most %d characters each. This is a hard limit: longer lines are\n"
        "  shrunk until they fit and become unreadable.\n"
        "- Open with a caption at 0.6 seconds setting up the run.\n"
        "- 'at' is the timestamp in SECONDS as a number.\n"
        "- Leave at least 6 seconds between captions.\n"
        "- Every caption must be different. No repeated jokes.\n"
        "- Praise the good moments as well as mocking the bad ones.\n"
        "Return JSON: {\"captions\": [{\"at\": 0.6, \"text\": \"...\"}]}"
        % max_chars)
    data = generate(prompt, CAPTION_SCHEMA, **kw)
    if not data:
        return None
    out = []
    for item in data.get("captions", []):
        text = str(item.get("text", "")).strip()
        # Trust nothing about length: the limit is what keeps the text legible,
        # and models treat character counts as a suggestion.
        if text and 0 <= float(item.get("at", -1)) < duration_s:
            out.append({"at": round(float(item["at"]), 2), "text": text[:max_chars]})
    out.sort(key=lambda c: c["at"])
    return out or None


# -------------------------------------------------------------- narration --
NARRATION_SCHEMA = {
    "type": "object",
    "properties": {
        "lines": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "at": {"type": "number"},
                    "text": {"type": "string"},
                },
                "required": ["at", "text"],
            },
        }
    },
    "required": ["lines"],
}


def narration(game, level, duration_s, players, events, fps, wpm=150, **kw):
    """A spoken commentary script. Returns [{at, text}] or None."""
    words = int(duration_s / 60.0 * wpm)
    prompt = (
        _context(game, level, duration_s, players, events, fps) +
        "\nWrite a spoken commentary track covering the WHOLE run.\n\n"
        "RULES:\n"
        "- About %d words total. It is read aloud at roughly %d words per\n"
        "  minute and must fill %.0f seconds, so do not stop after the\n"
        "  highlights -- keep talking between them.\n"
        "- Between events, talk about the game itself: how old it is, what it\n"
        "  meant to the people watching, what the player should be doing.\n"
        "- Open by naming the game and section. Close with a sign-off that\n"
        "  mentions how the run went.\n"
        "- 'at' is when the line is SPOKEN, in seconds, increasing, starting\n"
        "  near 0. Leave room for each line to be read before the next.\n"
        "- Full sentences. This is spoken, not captions.\n"
        "Return JSON: {\"lines\": [{\"at\": 0.4, \"text\": \"...\"}]}"
        % (words, wpm, duration_s))
    data = generate(prompt, NARRATION_SCHEMA, **kw)
    if not data:
        return None
    out = []
    for item in data.get("lines", []):
        text = str(item.get("text", "")).strip()
        if text and float(item.get("at", -1)) >= 0:
            out.append({"at": round(float(item["at"]), 2), "text": text})
    out.sort(key=lambda l: l["at"])
    return out or None


# ------------------------------------------------------------------- copy --
COPY_SCHEMA = {
    "type": "object",
    "properties": {
        "description": {"type": "string"},
        "tiktok": {"type": "string"},
        "instagram": {"type": "string"},
        "tags": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["description", "tiktok", "instagram", "tags"],
}


def copy(game, level, duration_s, players, events, fps, watermark="", **kw):
    """Platform copy. Returns {description, tiktok, instagram, tags} or None."""
    prompt = (
        _context(game, level, duration_s, players, events, fps) +
        "\nWrite the upload copy for this video.\n\n"
        "The pitch is NOSTALGIA. The audience played this game as children, or\n"
        "watched a sibling play it. Lead with that feeling before anything\n"
        "else, and be specific about the era rather than vaguely wistful.\n\n"
        "RULES:\n"
        "- description: 800 to 1500 characters for YouTube. Open with a\n"
        "  nostalgic hook, say how it was played (one take, no save states),\n"
        "  then a CHAPTERS list using the timestamps above in M:SS form\n"
        "  starting at 0:00, then a question inviting comments.\n"
        "- tiktok and instagram: one short caption each, under 200 characters,\n"
        "  different from each other, ending with hashtags.\n"
        "- tags: 10 to 15 YouTube search terms, lowercase, no # symbol.\n"
        "%s"
        "Return JSON with keys: description, tiktok, instagram, tags."
        % ("- Sign off with %s.\n" % watermark if watermark else ""))
    data = generate(prompt, COPY_SCHEMA, **kw)
    if not data:
        return None
    if not str(data.get("description", "")).strip():
        return None
    data["tags"] = [str(t).lstrip("#").strip() for t in data.get("tags", []) if str(t).strip()]
    return data
