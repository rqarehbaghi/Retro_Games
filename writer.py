#!/usr/bin/env python3
"""
Write the captions, commentary and descriptions with a local LLM.

Everything studio.py says is otherwise drawn from hardcoded pools, which is
fine for one video and obvious by the fifth: the same eight jokes in rotation,
the same description under every upload. This hands the writing to a model
running on your own machine instead, so each run is written fresh against what
actually happened in THAT run.

    python studio.py --game <id>                     # claude-code, the default
    python studio.py --game <id> --writer claude     # the Messages API
    python studio.py --game <id> --writer ollama     # a model on this machine

THREE BACKENDS, and the difference is what they bill against:

    claude-code   shells out to `claude -p`, which runs against a Claude Pro or
                  Max SUBSCRIPTION. Nothing extra to install or pay for if you
                  already have Claude Code. No schema-constrained decoding, so
                  the JSON is asked for in the prompt and dug back out.
    claude        the Messages API, billed by prepaid CREDITS -- a separate
                  product from the subscription. Constrained decoding.
    ollama        a model on this machine. Free, offline, and noticeably
                  blunter than either of the above.

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
import re
import shutil
import urllib.error
import urllib.request

BACKENDS = ("claude-code", "claude", "ollama")
DEFAULT_BACKEND = "claude-code"

DEFAULT_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
DEFAULT_MODEL = "qwen3:30b-a3b"
CLAUDE_MODEL = "claude-opus-4-8"

# Effort sets how much the model spends thinking and answering. The API
# default is "high"; "medium" is a deliberate step down, because writing
# captions is not an intelligence-sensitive task and the ceiling here is the
# voice rather than the reasoning. low | medium | high | xhigh | max.
#
# On Opus 4.8 specifically, NOT passing a `thinking` parameter means no
# thinking at all, which is the cheapest this gets -- unlike Opus 5, where
# thinking is on unless disabled.
CLAUDE_EFFORT = "medium"
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
             timeout=TIMEOUT, temperature=1.0, verbose=True, seed=None,
             think=True, **_kw):
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
        # Reasoning is ON. It was disabled to stop Qwen3 putting its thinking
        # in front of the JSON and breaking the parse -- but _strip_thinking
        # handles that now, and Ollama returns reasoning in its own field
        # anyway. Left off, a 30B was being judged with its reasoning
        # switched off, which is not a fair test of the model.
        "think": think,
        "options": options,
    }).encode()
    req = urllib.request.Request(host.rstrip("/") + "/api/generate", data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            payload = json.load(r)
        # With reasoning on, some builds return it separately and leave the
        # answer in `response`; others prepend it. Take whichever holds JSON.
        answer = payload.get("response") or ""
        if not answer.strip():
            answer = payload.get("thinking") or ""
        return json.loads(_strip_thinking(answer))
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


DEFAULT_CLI = "claude"

# Where the native installer puts it, and the usual reason it is not found: on
# Ubuntu and WSL ~/.local/bin is frequently missing from PATH.
CLI_HINTS = (
    "~/.local/bin/claude",
    "~/.claude/local/claude",
    "/usr/local/bin/claude",
)


def find_cli(name=DEFAULT_CLI):
    """The Claude Code binary, by PATH lookup or at a known install location."""
    found = shutil.which(name)
    if found:
        return found
    if os.sep in name or "/" in name:
        expanded = os.path.expanduser(name)
        return expanded if os.path.exists(expanded) else None
    for hint in CLI_HINTS:
        expanded = os.path.expanduser(hint)
        if os.path.exists(expanded):
            return expanded
    return None


def claude_code_available(name=DEFAULT_CLI):
    """Is the Claude Code CLI reachable?"""
    return find_cli(name) is not None


def generate_claude_code(prompt, schema, verbose=True, timeout=300,
                         cli=DEFAULT_CLI, model=None, **_kw):
    """One generation through the Claude Code CLI. Parsed object, or None.

    This is the path that costs nothing extra: `claude -p` runs against a
    Claude Pro or Max SUBSCRIPTION, while the Messages API is a separate
    product billed by prepaid credits. A subscription does not include API
    credits and an API balance does not include a subscription -- so with Pro
    and no credits, this is the way to reach a frontier model.

    No schema-constrained decoding here, unlike the API and Ollama paths, so
    the schema goes in the prompt and the answer is dug out of whatever comes
    back. _strip_thinking already handles prose either side of the JSON."""
    import subprocess
    binary = find_cli(cli)
    if binary is None:
        if verbose:
            print(f"  (writer: no Claude Code CLI found as {cli!r})")
        return None
    ask = (VOICE + "\n\n" + prompt +
           "\n\nRespond with ONLY the JSON object described above. No preamble,"
           "\nno explanation, no markdown fence. It is parsed by a program.")
    try:
        cmd = [binary, "-p", ask, "--output-format", "json"]
        if model:
            cmd += ["--model", model]
        result = subprocess.run(
            cmd,
            capture_output=True, text=True, timeout=timeout)
        if result.returncode != 0:
            if verbose:
                print("  (writer: claude CLI exited %d: %s)"
                      % (result.returncode, (result.stderr or "").strip()[:200]))
            return None
        # --output-format json wraps the answer in a metadata envelope; the
        # model's own output is the `result` field.
        envelope = json.loads(result.stdout)
        return json.loads(_strip_thinking(envelope.get("result", "")))
    except subprocess.TimeoutExpired:
        if verbose:
            print(f"  (writer: claude CLI did not answer within {timeout}s)")
    except (KeyError, ValueError) as exc:
        if verbose:
            print(f"  (writer: claude CLI returned nothing usable: {exc})")
    except Exception as exc:                                   # noqa: BLE001
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


def generate_claude(prompt, schema, model=CLAUDE_MODEL, effort=CLAUDE_EFFORT,
                    verbose=True, **_kw):
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
            output_config={"effort": effort,
                           "format": {"type": "json_schema", "schema": strict}},
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
    """Dispatch to whichever backend was asked for.

    Each backend is handed the arguments it actually takes, named explicitly.
    An earlier version passed everything through minus a blacklist of one key,
    so every option added afterwards -- claude_effort, cli -- leaked into the
    wrong backend and raised TypeError at the call. A blacklist has to be
    updated every time anything is added; naming what each one takes does not."""
    verbose = kw.get("verbose", True)
    if backend == "claude-code":
        return generate_claude_code(prompt, schema, verbose=verbose,
                                    cli=kw.get("cli", DEFAULT_CLI),
                                    model=kw.get("claude_model"))
    if backend == "claude":
        return generate_claude(prompt, schema,
                               model=kw.get("claude_model", CLAUDE_MODEL),
                               effort=kw.get("claude_effort", CLAUDE_EFFORT),
                               verbose=verbose)
    return generate(prompt, schema,
                    model=kw.get("model", DEFAULT_MODEL),
                    host=kw.get("host", DEFAULT_HOST),
                    seed=kw.get("seed"),
                    think=kw.get("think", True),
                    verbose=verbose)


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


# NO LEAD. Every value is logged on the frame the thing happens -- including
# `lives`, which decrements as Mario dies, not at the end of the death
# animation. An earlier version subtracted 0.5 to 3 seconds per kind on the
# theory that some values lagged the picture; that theory came from a note
# about a DIFFERENT byte (0x0749) and was never checked against video, and the
# result was every caption arriving about a second before the thing it was
# about. A caption is a reaction: it belongs on the frame, not ahead of it.
#
# caption_offset in studio.json is the remaining knob, and it is 0 unless set.


def event_time(events, index, fps, duration_s):
    """Screen time of an event, or None if the index is not a real one.

    The time IS the event's own frame -- no adjustment. Times come from HERE,
    never from the model. Asking a model for timestamps
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
    frame, _kind, _detail = events[index]
    return max(0.0, min(frame / fps, max(0.0, duration_s - 1.0)))


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
        "closing": {"type": "string"},
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
    "required": ["opening", "closing", "captions"],
}

def trim_words(text, limit):
    """Shorten to at most `limit` characters WITHOUT cutting a word in half.

    A hard slice produced "Which level should he try to actually fi" on screen.
    The renderer already wraps to two lines and shrinks to fit, so a little
    overshoot costs a couple of points of size and nothing else -- only a wild
    overshoot needs cutting, and then at a space."""
    text = text.strip()
    if len(text) <= limit:
        return text
    cut = text[:limit + 1]
    space = cut.rfind(" ")
    out = (cut[:space] if space > limit * 0.5 else text[:limit]).rstrip()
    return out.rstrip(",;:-").rstrip()


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
        "- 'closing' is one caption shown at the END, asking the viewer something\n"
        "  they can actually answer in a comment -- which level next, which game\n"
        "  next, whether to play one to the finish. A QUESTION, not a demand:\n"
        "  never 'like and subscribe', never 'smash that button'. Under %d\n"
        "  characters like the rest, and in the same voice.\n"
        "- Every caption must be different. No repeated jokes.\n"
        "- Praise the good moments as well as mocking the bad ones.\n\n"
        "Good: \"He walked into it. Fully aware.\"\n"
        "Good: \"A mushroom. Do not get attached.\"\n"
        "Too long: \"That enemy has stood there since 1988 waiting for this.\"\n\n"
        "Return JSON: {\"opening\": \"...\", \"closing\": \"...\", \"captions\": "
        "[{\"event\": 3, \"text\": \"...\"}]}"
        % (max_chars, max_chars))
    data = write(prompt, CAPTION_SCHEMA, **kw)
    if not data:
        return None

    out = []
    opening = str(data.get("opening", "")).strip()
    if opening:
        out.append({"at": 0.6, "text": trim_words(opening, max_chars * 3 // 2),
                    "event": None})
    for item in data.get("captions", []):
        text = str(item.get("text", "")).strip()
        index = item.get("event")
        at = event_time(events, index, fps, duration_s)
        if text and at is not None:
            # The event is kept so the caller can SHOW what each line was
            # pinned to. Two very different faults look identical on screen --
            # a line landing at the wrong second, and a line landing correctly
            # on an event it is not actually about -- and only the mapping
            # tells them apart.
            out.append({"at": round(at, 2),
                        "text": trim_words(text, max_chars * 3 // 2),
                        "event": int(index), "kind": events[int(index)][1]})
    out.sort(key=lambda c: c["at"])

    # The ask goes LAST, after the run has earned it. Asking up front reads as
    # a demand; the same question after ninety seconds of watching reads as a
    # conversation, and a question someone can answer gets replies where
    # "comment below" does not.
    closing = str(data.get("closing", "")).strip()
    if closing and duration_s > 8:
        out.append({"at": round(max(0.0, duration_s - 5.0), 2),
                    "text": trim_words(closing, max_chars * 3 // 2),
                    "event": None, "closing": True})

    # Two captions on top of each other are unreadable, and a model asked for
    # "the interesting moments" will happily pick three in a row.
    spaced = []
    for cap in out:
        if spaced and cap["at"] - spaced[-1]["at"] < CAPTION_GAP:
            # Never thin out the closing ask -- push whatever crowds it aside
            # instead. It is the one caption with a job beyond being funny.
            if not cap.get("closing"):
                continue
            spaced.pop()
        spaced.append(cap)
    return spaced or None


# -------------------------------------------------------------- narration --
NARRATION_SCHEMA = {
    "type": "object",
    "properties": {
        "script": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "event": {"type": "integer"},
                },
                "required": ["text"],
            },
        },
        "closing": {"type": "string"},
    },
    "required": ["script", "closing"],
}

# Built from chr() so no quote character sits inside a pattern literal.
_Q = "[" + chr(34) + chr(39) + "]"
# Field names a backend without constrained decoding sometimes hands back
# alongside the value it was asked for.
_KEYS = "(text|tone|line|script|closing|narration)"
_LEAD = "^" + _Q + "?" + _KEYS + _Q + "?" + r"\s*[:=]\s*" + _Q + "?"
_TRAIL = (_Q + "?[,;]" + r"\s*" + _Q + "?" + _KEYS + _Q + "?" +
          r"\s*[:=]\s*" + _Q + r"?[\w -]*" + _Q + "?$")
_OPEN = r"^[\s{\[" + chr(34) + chr(39) + "]+"
_CLOSE = r"[\s}\]" + chr(34) + chr(39) + "]+$"


def clean_spoken(text):
    """Strip anything structural that leaked into a line.

    A backend with no schema enforcement occasionally returns a field name
    along with its value, and the result was the voice reading "text" and
    "tone amused" out loud. Nothing shaped like JSON should reach the
    synthesiser, so it is stripped here rather than trusted upstream."""
    out = re.sub(_CLOSE, "", re.sub(_OPEN, "", str(text).strip()))
    # Repeatedly, because a whole object handed back leaves {"text": "..." and
    # each pass peels one layer.
    for _ in range(3):
        stripped = re.sub(_LEAD, "", out, flags=re.I)
        if stripped == out:
            break
        out = stripped
    out = re.sub(_TRAIL, "", out, flags=re.I)
    return out.strip().strip(chr(34)).strip(chr(39)).strip()


def narration(game, level, duration_s, players, events, fps, wpm=125, **kw):
    """A continuous spoken script. Returns [{at, text, closing}] or None.

    ONE MONOLOGUE, not lines pinned to moments. Anchoring commentary to events
    produced disconnected sentences dropped into gaps, which does not sound
    like a person talking -- it sounds like captions read aloud. A podcaster
    talking over footage is continuous, so the script is written as continuous
    speech and laid down end to end; the events are context for WHAT to say,
    never instructions for WHEN to say it.

    Timing comes from the speech itself, in tts.space_clips, which knows how
    long each line actually rendered to."""
    words = int(duration_s / 60.0 * wpm)
    prompt = (
        _context(game, level, duration_s, players, events, fps) +
        "\nWrite what a podcaster says over this footage, start to finish.\n\n"
        "It has to sound like ONE PERSON TALKING CONTINUOUSLY, not a list of\n"
        "remarks. Each entry in 'script' is the next sentence or two of the\n"
        "same monologue, read straight through with no gap between them, so\n"
        "they must flow into each other -- use the connective tissue real\n"
        "speech has: 'and honestly', 'which is mad when you think about it',\n"
        "'anyway', 'now watch this bit'.\n\n"
        "MOSTLY FACTS AND JOKES ABOUT THE GAME, not description of the screen.\n"
        "The viewer can see the screen. What they cannot see is how this game\n"
        "was made, what got cut, what this level is famous for, what everyone\n"
        "got stuck on as a child, how old it all is now. Weave the run in\n"
        "where it fits -- a death is worth a laugh -- but the trivia carries\n"
        "it. Say only what you are confident is TRUE; a wrong fact about a\n"
        "game this audience grew up with is worse than no fact at all.\n\n"
        "RULES:\n"
        "- About %d words TOTAL. Synthesised speech runs slower than you\n"
        "  expect, and anything past %.0f seconds is cut, so going over loses\n"
        "  the end rather than making a longer video.\n"
        "- Plain spoken prose. No stage directions, no speaker labels, no\n"
        "  field names, no markdown, no emoji, no timestamps. Every string is\n"
        "  read aloud EXACTLY as written.\n"
        "- When an entry is ABOUT one of the numbered moments above, give it\n"
        "  that [N] as 'event'. The speech is held back so the line lands while\n"
        "  that moment is on screen, so the entries must be in the same order\n"
        "  as the moments they name, with enough said in between to fill the\n"
        "  time. Leave 'event' out of the general trivia, which is most of it.\n"
        "- 'closing' is the last thing said: ask which level or which game to\n"
        "  play next. A question, never 'like and subscribe'.\n"
        "Return JSON: {\"script\": [{\"text\": \"...\"}, "
        "{\"text\": \"...\", \"event\": 2}], \"closing\": \"...\"}"
        % (words, duration_s))
    data = write(prompt, NARRATION_SCHEMA, **kw)
    if not data:
        return None

    lines = []
    for item in data.get("script", []):
        # Tolerate a bare string: a backend without schema enforcement may
        # ignore the object shape entirely.
        raw = item if isinstance(item, str) else item.get("text", "")
        text = clean_spoken(raw)
        if not text:
            continue
        anchor = None
        if isinstance(item, dict) and item.get("event") is not None:
            anchor = event_time(events, item.get("event"), fps, duration_s)
        lines.append({"text": text, "anchor": anchor, "closing": False})
    closing = clean_spoken(data.get("closing", ""))
    if closing:
        lines.append({"text": closing, "anchor": None, "closing": True})
    return lines or None


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
