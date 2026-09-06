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

BACKENDS = ("claude", "ollama")
DEFAULT_BACKEND = "claude"

DEFAULT_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
DEFAULT_MODEL = "qwen3:30b-a3b"
CLAUDE_MODEL = "claude-opus-5"
TIMEOUT = 600      # a 14B writing a full commentary track is not quick

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


THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"


def _strip_thinking(text):
    """Drop a reasoning block a thinking model emitted before its answer.

    Qwen3 and its relatives reason out loud first. With `think` honoured this
    never fires, but older Ollama builds ignore the flag and then every single
    call fails to parse -- which looked exactly like the model being bad at
    JSON rather than the request being wrong."""
    if THINK_CLOSE in text:
        text = text.rsplit(THINK_CLOSE, 1)[1]
    start = text.find("{")
    end = text.rfind("}")
    return text[start:end + 1] if 0 <= start < end else text


def generate(prompt, schema, model=DEFAULT_MODEL, host=DEFAULT_HOST,
             timeout=TIMEOUT, temperature=1.0, verbose=True, seed=None):
    """One constrained generation. Returns the parsed object, or None.

    `schema` is a JSON schema passed as `format`, which makes Ollama restrict
    decoding to tokens that keep the output valid against it -- the difference
    between parsing reliably and hoping."""
    options = {"temperature": temperature, "top_p": 0.95}
    if seed is not None:
        # An explicit, per-run random seed. Ollama seeds from the clock by
        # default, but being explicit is what guarantees that replaying the
        # same footage twice cannot produce the same script twice.
        options["seed"] = seed
    body = json.dumps({
        "model": model,
        "prompt": prompt,
        "system": VOICE,
        "format": schema,
        "stream": False,
        # Thinking models put their reasoning in the response before the
        # answer, which is not valid JSON no matter how well the schema
        # constrains the rest of it.
        "think": False,
        "options": options,
    }).encode()
    req = urllib.request.Request(host.rstrip("/") + "/api/generate", data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            payload = json.load(r)
        return json.loads(_strip_thinking(payload["response"]))
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


def claude_available():
    """Is the anthropic SDK importable? Credentials are resolved by the SDK."""
    try:
        import anthropic                                       # noqa: F401
        return True
    except ImportError:
        return False


def generate_claude(prompt, schema, model=CLAUDE_MODEL, verbose=True, **_kw):
    """One structured generation through the Anthropic API. Parsed object or None.

    output_config.format is the API's structured-output mode: it constrains the
    response to the schema, so the first text block is valid JSON against it.
    The same guarantee Ollama's `format` gives locally.

    Credentials come from the SDK's own resolution -- ANTHROPIC_API_KEY, or a
    profile from `ant auth login` -- so nothing here handles a key."""
    try:
        import anthropic
    except ImportError:
        if verbose:
            print("  (writer: the anthropic SDK is not installed -- pip install anthropic)")
        return None

    strict = dict(schema)
    strict.setdefault("additionalProperties", False)
    try:
        client = anthropic.Anthropic()
        response = client.messages.create(
            model=model,
            max_tokens=16000,
            system=VOICE,
            messages=[{"role": "user", "content": prompt}],
            output_config={"format": {"type": "json_schema", "schema": strict}},
        )
        if response.stop_reason == "refusal":
            if verbose:
                print("  (writer: the request was declined)")
            return None
        text = next(b.text for b in response.content if b.type == "text")
        return json.loads(text)
    except Exception as exc:                                   # noqa: BLE001
        if verbose:
            print(f"  (writer: {exc.__class__.__name__}: {exc})")
    return None


def write(prompt, schema, backend=DEFAULT_BACKEND, **kw):
    """Dispatch to whichever backend was asked for."""
    if backend == "claude":
        return generate_claude(prompt, schema,
                               model=kw.get("claude_model", CLAUDE_MODEL),
                               verbose=kw.get("verbose", True))
    return generate(prompt, schema,
                    **{k: v for k, v in kw.items() if k != "claude_model"})


def _timeline(events, fps):
    """The run, as something a model can read."""
    label = {
        "death": "died", "shrink": "shrank to small", "powerdown": "lost a power tier",
        "powerup": "collected a power-up", "1up": "got an extra life",
        "coin": "collected a coin", "clear": "finished the level",
        "pipe": "went down a pipe", "score": "scored points",
    }
    out = []
    for i, (frame, kind, detail) in enumerate(events):
        secs = frame / fps
        out.append("  [%d] %d:%05.2f  %s (%s)" % (i, secs // 60, secs % 60,
                                                  label.get(kind, kind), detail))
    return "\n".join(out) or "  (nothing notable happened)"


# How far BEFORE the logged frame each kind of moment starts on screen. Values
# are logged when the underlying RAM value changes, and some change long after
# the thing worth watching: SMB3's `lives` only decrements at the END of the
# death sequence, roughly three seconds after the hit that caused it.
EVENT_LEAD = {"death": 3.0, "clear": 3.0, "powerdown": 1.5, "shrink": 1.5,
              "1up": 1.0, "powerup": 0.8, "pipe": 1.0, "coin": 0.5}


def event_time(events, index, fps, duration_s):
    """Screen time of an event, or None if the index is not a real one.

    Times come from HERE, never from the model. Asking a model for timestamps
    and trusting them is what put captions on the wrong moments: it has no way
    to know when anything happened beyond the numbers in the prompt, and it
    approximates them. The frame numbers are exact, so the model writes the
    words and the timeline places them."""
    try:
        index = int(index)
    except (TypeError, ValueError):
        return None
    if not 0 <= index < len(events):
        return None
    frame, kind, _detail = events[index]
    at = frame / fps - EVENT_LEAD.get(kind, 1.0)
    return max(0.0, min(at, max(0.0, duration_s - 1.0)))


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
        "opening": {"type": "string"},
        "captions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "event": {"type": "integer"},
                    "text": {"type": "string"},
                },
                "required": ["event", "text"],
            },
        },
    },
    "required": ["opening", "captions"],
}

CAPTION_GAP = 6.0        # clear seconds between one caption and the next


def captions(game, level, duration_s, players, events, fps, max_chars=40,
             **kw):
    """Timed on-screen captions. Returns [{at, text}] or None.

    The model chooses WHICH moments to caption and what to say; the timeline
    decides WHEN each lands."""
    prompt = (
        _context(game, level, duration_s, players, events, fps) +
        "\nWrite on-screen captions for this run.\n\n"
        "These are READ IN PASSING while the game is playing, so they are the\n"
        "short form -- a punchline, not a paragraph. The long commentary goes\n"
        "in the narration track, not here.\n\n"
        "RULES:\n"
        "- At most %d CHARACTERS each. Aim for five to eight words. A caption\n"
        "  that runs long gets shrunk until it fits and stops being readable.\n"
        "- 'event' is the [N] NUMBER of the moment the caption is about, from\n"
        "  the list above. Do NOT write timestamps -- they are worked out from\n"
        "  the event you name.\n"
        "- Caption the interesting moments only. Skip the dull ones, and skip\n"
        "  most coins.\n"
        "- 'opening' is one caption shown at the very start, setting up the run.\n"
        "- Every caption must be different. No repeated jokes.\n"
        "- Praise the good moments as well as mocking the bad ones.\n\n"
        "Good: \"He walked into it. Fully aware.\"\n"
        "Good: \"A mushroom. Do not get attached.\"\n"
        "Too long: \"That enemy has stood there since 1988 waiting for this.\"\n\n"
        "Return JSON: {\"opening\": \"...\", \"captions\": "
        "[{\"event\": 3, \"text\": \"...\"}]}"
        % max_chars)
    data = write(prompt, CAPTION_SCHEMA, **kw)
    if not data:
        return None

    out = []
    opening = str(data.get("opening", "")).strip()
    if opening:
        out.append({"at": 0.6, "text": opening[:max_chars]})
    for item in data.get("captions", []):
        text = str(item.get("text", "")).strip()
        at = event_time(events, item.get("event"), fps, duration_s)
        if text and at is not None:
            out.append({"at": round(at, 2), "text": text[:max_chars]})
    out.sort(key=lambda c: c["at"])

    # Two captions on top of each other are unreadable, and a model asked for
    # "the interesting moments" will happily pick three in a row.
    spaced = []
    for cap in out:
        if spaced and cap["at"] - spaced[-1]["at"] < CAPTION_GAP:
            continue
        spaced.append(cap)
    return spaced or None


# -------------------------------------------------------------- narration --
NARRATION_SCHEMA = {
    "type": "object",
    "properties": {
        "opening": {"type": "string"},
        "events": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "event": {"type": "integer"},
                    "text": {"type": "string"},
                },
                "required": ["event", "text"],
            },
        },
        "filler": {"type": "array", "items": {"type": "string"}},
        "closing": {"type": "string"},
    },
    "required": ["opening", "events", "filler", "closing"],
}

NARRATION_GAP = 5.0      # fill any silence longer than this


def uncovered(filled, duration_s, spoken, limit=15.0):
    """True when some stretch longer than `limit` has nothing spoken over it.

    Worth saying out loud: a short filler pool leaves the back half of the run
    silent, and silence in a commentary track looks like a bug in the renderer
    rather than a model that wrote too little."""
    ordered = sorted(filled)
    for i, (at, text) in enumerate(ordered):
        end = ordered[i + 1][0] if i + 1 < len(ordered) else duration_s
        if end - (at + spoken(text)) > limit:
            return True
    return False


def narration(game, level, duration_s, players, events, fps, wpm=150, **kw):
    """A spoken commentary script. Returns [{at, text}] or None.

    Split deliberately: the model writes lines ABOUT events, plus loose filler
    with no timing at all. Event lines land on their real timestamps and the
    filler goes into whatever silence is left, so the track both syncs and
    covers the whole run -- neither of which survives asking a model to place
    its own lines across ninety seconds."""
    words = int(duration_s / 60.0 * wpm)
    prompt = (
        _context(game, level, duration_s, players, events, fps) +
        "\nWrite a spoken commentary track covering the WHOLE run.\n\n"
        "RULES:\n"
        "- 'opening' names the game and section and sets the run up.\n"
        "- 'events' are lines about specific moments. 'event' is the [N] NUMBER\n"
        "  from the list above. Do NOT write timestamps -- each line is placed\n"
        "  on the moment it names.\n"
        "- 'filler' is AT LEAST %d loose lines with no particular moment\n"
        "  attached, used to fill the silences between events. Too few and the\n"
        "  track runs out partway, leaving the rest of the video silent.\n"
        "  Talk about the game: how old it is, what it meant to the people\n"
        "  watching, what the player should be doing, what is coming up. Each\n"
        "  must stand alone and make sense in any order.\n"
        "- 'closing' signs off and says how the run went.\n"
        "- About %d words in total, read aloud at %d words per minute to fill\n"
        "  %.0f seconds. Full sentences -- this is spoken, not captions.\n"
        "Return JSON with keys: opening, events, filler, closing."
        % (max(8, int(duration_s / 5)), words, wpm, duration_s))
    data = write(prompt, NARRATION_SCHEMA, **kw)
    if not data:
        return None

    def spoken(text):
        return len(text.split()) / wpm * 60.0

    script = []
    opening = str(data.get("opening", "")).strip()
    if opening:
        script.append((0.4, opening))
    for item in data.get("events", []):
        text = str(item.get("text", "")).strip()
        at = event_time(events, item.get("event"), fps, duration_s)
        if text and at is not None:
            script.append((at, text))
    script.sort()

    filler = [str(f).strip() for f in data.get("filler", []) if str(f).strip()]
    closing = str(data.get("closing", "")).strip()

    filled, pool = [], list(filler)
    for i, (at, text) in enumerate(script):
        filled.append((at, text))
        cursor = at + spoken(text)
        end = script[i + 1][0] if i + 1 < len(script) else duration_s
        while pool and end - cursor > NARRATION_GAP:
            line = pool.pop(0)
            filled.append((cursor + 0.8, line))
            cursor += 0.8 + spoken(line)
    if uncovered(filled, duration_s, spoken):
        print("  (writer: too little filler came back -- part of the run has "
              "no commentary over it)")
    if closing:
        last = max((a + spoken(t) for a, t in filled), default=0.0)
        filled.append((min(max(last + 0.5, duration_s - spoken(closing) - 0.5),
                           max(0.0, duration_s - 0.5)), closing))
    filled.sort()
    return [{"at": round(a, 2), "text": t} for a, t in filled] or None


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
    data = write(prompt, COPY_SCHEMA, **kw)
    if not data:
        return None
    if not str(data.get("description", "")).strip():
        return None
    data["tags"] = [str(t).lstrip("#").strip() for t in data.get("tags", []) if str(t).strip()]
    return data
