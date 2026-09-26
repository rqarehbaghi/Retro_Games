#!/usr/bin/env python3
"""
Write the captions, commentary and descriptions with a model.

Everything studio.py says is otherwise drawn from hardcoded pools, which is
fine for one video and obvious by the fifth: the same eight jokes in rotation,
the same description under every upload. This hands the writing to a model
instead, so each run is written fresh against what actually happened in THAT
run.

    python studio.py --game <id>                    # ChatGPT through Codex (default)
    python studio.py --game <id> --writer auto      # ChatGPT, then Ollama
    python studio.py --game <id> --writer ollama    # a model on this machine

TWO BACKENDS, AND NEITHER CAN BILL AN API.

    chatgpt   shells out to the Codex CLI, which runs against an existing
              ChatGPT SUBSCRIPTION. Nothing metered, nothing prepaid.
    ollama    a model on this machine. Free, offline, and blunter.

That is the whole list on purpose. Every metered path -- the OpenAI Responses
API, the Anthropic Messages API, Gemini -- was removed rather than left
configurable, because a writer that CAN quietly start billing eventually does.
"auto" is chatgpt then ollama and nothing else; a named backend is pinned and
never falls through, so the provider cannot change underneath a run.

SETTING UP CHATGPT, once, inside WSL:

    curl -fsSL https://chatgpt.com/codex/install.sh | sh
    codex login --device-auth
    codex login status            # must say: Logged in using ChatGPT

It is called in an EMPTY temporary directory with a read-only sandbox, so the
agent has no repository to read and nothing it could modify. --output-schema
constrains the final message and -o writes only that message to a file, which
is what gets parsed; the conversation itself is discarded. stdin is closed:
with a pipe attached and never closed, "codex exec" waits on stdin forever
and the call dies on its timeout instead of running.

ONE WRITER MUST BE AVAILABLE before Studio records. The preflight checks that
up front, so a missing login or local server cannot cost a gameplay recording.
Every entry point still returns None on a per-call failure so callers can keep
the already-rendered media and report that its writing needs to be retried.

WHY OLLAMA for the local tier. One install, an HTTP API on localhost, no
Python dependency here (urllib is enough), and GRAMMAR-CONSTRAINED decoding:
passing a JSON schema as "format" restricts the sampler to tokens that can
legally continue a valid document, so a small model cannot wander off and
produce prose where a list was wanted.

    sudo apt-get install -y zstd     # the installer unpacks with it and
                                     # stops with an error if it is missing
    curl -fsSL https://ollama.com/install.sh | sh
    ollama pull qwen3:30b-a3b        # see MODEL_NOTES below; ~18GB download

On WSL2, systemd is often not running, so the installer's service never
starts and nothing is listening. Run ``ollama serve`` in its own terminal.
``ollama ps`` then says whether a loaded model is on the GPU or has fallen back
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
import subprocess
import tempfile
import urllib.error
import urllib.request

BACKENDS = ("chatgpt", "ollama", "auto")
DEFAULT_BACKEND = "chatgpt"
AUTO_BACKENDS = ("chatgpt", "ollama")

DEFAULT_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
DEFAULT_MODEL = "qwen3:30b-a3b"

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


CODEX_CLI = "codex"
CODEX_TIMEOUT = 300


def find_cli(name=CODEX_CLI):
    """Where the Codex CLI is, including the place its installer puts it.

    A login shell would have ~/.local/bin on PATH; a subprocess spawned from a
    render pipeline often does not, and the backend would look uninstalled."""
    found = shutil.which(name)
    if found:
        return found
    candidate = os.path.expanduser(os.path.join("~", ".local", "bin", name))
    return candidate if os.path.exists(candidate) else None


def _chatgpt_env():
    """Environment that cannot select an API/service credential by accident."""
    env = os.environ.copy()
    for name in ("OPENAI_API_KEY", "CODEX_API_KEY", "CODEX_ACCESS_TOKEN",
                 "OPENAI_FEDERATION_RULE_ID", "OPENAI_IDENTITY_TOKEN_FILE"):
        env.pop(name, None)
    return env


def chatgpt_available(name=CODEX_CLI, timeout=20):
    """Installed AND logged in.

    Both halves matter: the binary being present says nothing about whether a
    ChatGPT session is attached, and a logged-out CLI fails per call, slowly,
    after a recording has already been made."""
    cli = find_cli(name)
    if not cli:
        return False
    try:
        done = subprocess.run([cli, "login", "status"], capture_output=True,
                              text=True, timeout=timeout,
                              stdin=subprocess.DEVNULL, env=_chatgpt_env())
    except Exception:                                             # noqa: BLE001
        return False
    status = (done.stdout or "") + "\n" + (done.stderr or "")
    return done.returncode == 0 and "Logged in using ChatGPT" in status


def generate_chatgpt(prompt, schema, verbose=True, timeout=CODEX_TIMEOUT,
                     cli=CODEX_CLI, model=None, **_kw):
    """Ask the Codex CLI, on the ChatGPT subscription. None on any failure.

    Run in an EMPTY temporary directory with a read-only sandbox: the agent
    gets no repository to read, nothing it could modify, and no session files
    left behind. --output-schema constrains the final message and -o writes
    only that message, so the answer is read from a file rather than scraped
    out of a transcript.

    stdin is closed deliberately. With a pipe attached and never closed,
    "codex exec" prints "Reading additional input from stdin..." and waits
    there until the timeout kills it -- measured, not guessed."""
    path = find_cli(cli)
    if not path:
        if verbose:
            print("  (writer: the Codex CLI is not installed)")
        return None
    work = tempfile.mkdtemp(prefix="codex-work-")
    box = tempfile.mkdtemp(prefix="codex-out-")
    schema_file = os.path.join(box, "schema.json")
    answer_file = os.path.join(box, "last.json")
    try:
        with open(schema_file, "w", encoding="utf-8") as fh:
            json.dump(schema, fh)
        cmd = [path, "exec", "--skip-git-repo-check", "--ephemeral",
               "--ignore-user-config", "--color", "never",
               "-c", 'forced_login_method="chatgpt"',
               "-s", "read-only", "-C", work,
               "--output-schema", schema_file, "-o", answer_file]
        if model:
            cmd += ["-m", model]
        cmd.append(VOICE + "\n\n" + prompt)
        done = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=timeout, stdin=subprocess.DEVNULL,
                              cwd=work, env=_chatgpt_env())
        if done.returncode != 0:
            if verbose:
                print("  (writer: codex exec exited %d: %s)"
                      % (done.returncode, (done.stderr or "").strip()[:200]))
            return None
        if not os.path.exists(answer_file):
            if verbose:
                print("  (writer: codex exec wrote no final message)")
            return None
        with open(answer_file, encoding="utf-8") as fh:
            text = fh.read().strip()
        if not text:
            return None
        return json.loads(text)
    except subprocess.TimeoutExpired:
        if verbose:
            print("  (writer: codex exec timed out after %ds)" % timeout)
        return None
    except Exception as exc:                                      # noqa: BLE001
        if verbose:
            print("  (writer: codex exec failed: %s)" % exc)
        return None
    finally:
        for folder in (work, box):
            shutil.rmtree(folder, ignore_errors=True)


def cascade_order(value=None):
    """The order "auto" tries, validated.

    A list, so the order is stated in one place rather than implied by the
    shape of an if-chain -- but a SHORT list, and every name in it has to be a
    backend that cannot bill."""
    if value is None:
        names = list(AUTO_BACKENDS)
    elif isinstance(value, str):
        names = [part.strip() for part in value.split(",") if part.strip()]
    else:
        names = [str(part).strip() for part in value if str(part).strip()]
    invalid = [name for name in names if name not in BACKENDS or name == "auto"]
    if invalid:
        raise ValueError("unknown writer backend in cascade: %s" % ", ".join(invalid))
    if len(set(names)) != len(names):
        raise ValueError("writer cascade contains a duplicate backend")
    if not names:
        raise ValueError("writer cascade cannot be empty")
    return names


def backend_available(name, **kw):
    """Cheap preflight, run before a recording starts rather than after."""
    if name == "chatgpt":
        return chatgpt_available(kw.get("cli", CODEX_CLI))
    if name == "ollama":
        return available(kw.get("host", DEFAULT_HOST))
    return False


def _write_one(name, prompt, schema, **kw):
    if name == "chatgpt":
        return generate_chatgpt(prompt, schema, verbose=False,
                                cli=kw.get("cli", CODEX_CLI),
                                model=kw.get("chatgpt_model"),
                                timeout=kw.get("chatgpt_timeout", CODEX_TIMEOUT))
    if name == "ollama":
        return generate(prompt, schema, model=kw.get("model", DEFAULT_MODEL),
                        host=kw.get("host", DEFAULT_HOST), seed=kw.get("seed"),
                        think=kw.get("think", True), verbose=False)
    return None


def write(prompt, schema, backend=DEFAULT_BACKEND, **kw):
    """Dispatch to one pinned backend, or to the two-step automatic cascade."""
    verbose = kw.get("verbose", True)
    if backend not in BACKENDS:
        raise ValueError("unknown writer backend: %s" % backend)
    if backend == "auto":
        order = cascade_order(kw.get("cascade_order"))
    else:
        # A named backend is PINNED and never falls through. Silently changing
        # which model wrote something is not a detail: it changes the voice of
        # the video and, with any metered provider, who pays for it.
        order = [backend]

    labels = {"chatgpt": "ChatGPT (Codex CLI)", "ollama": "offline LLM (Ollama)"}
    for name in order:
        if not backend_available(name, **kw):
            if verbose:
                print("  [writer] %s is not available." % labels[name])
            continue
        result = _write_one(name, prompt, schema, **kw)
        if result is not None:
            if verbose:
                print("  [writer] generated successfully using %s" % labels[name])
            return result
        if verbose:
            print("  [writer] %s failed or returned empty output." % labels[name])
    if verbose:
        print("  [writer] All requested LLM backends failed or are unavailable.")
    return None


# A run collects a great many coins and almost nothing else in bulk, so a raw
# timeline is mostly noise: one real playthrough produced 97 events of which
# nearly all were coins. Long prompts of near-identical lines cost quality on
# every call and made the description call return nothing at all.
COIN_KINDS = ("coin", "score")
MAX_TIMELINE = 40


def _timeline(events, fps):
    """The run as something a model can read, with bulk collectibles collapsed.

    INDICES ARE PRESERVED. Captions and narration anchor to them, so a
    collapsed run still prints the real index of the events either side of it
    and the summary in between names the range it stands for."""
    label = {
        "death": "died", "shrink": "shrank to small", "powerdown": "lost a power tier",
        "powerup": "collected a power-up", "1up": "got an extra life",
        "coin": "collected a coin", "clear": "finished the level",
        "pipe": "went down a pipe", "score": "scored points",
    }

    def stamp_at(frame):
        secs = frame / fps
        return "%d:%05.2f" % (secs // 60, secs % 60)

    out, i, n = [], 0, len(events)
    while i < n:
        frame, kind, detail = events[i]
        if kind in COIN_KINDS:
            run = i
            while run + 1 < n and events[run + 1][1] == kind:
                run += 1
            if run - i >= 2:
                out.append("  [%d] %s  %s" % (i, stamp_at(frame), label.get(kind, kind)))
                out.append("      ... [%d]-[%d], %d more, through to %s ..."
                           % (i + 1, run, run - i, stamp_at(events[run][0])))
                i = run + 1
                continue
        out.append("  [%d] %s  %s (%s)" % (i, stamp_at(frame),
                                           label.get(kind, kind), detail))
        i += 1

    if len(out) > MAX_TIMELINE:
        kept = out[:MAX_TIMELINE - 1]
        kept.append("      ... and %d more lines, mostly collectibles ..."
                    % (len(out) - MAX_TIMELINE + 1))
        out = kept
    return "\n".join(out) or "  (nothing notable happened)"


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


def _context(game, level, duration_s, players, events, fps, two_human=False):
    if two_human or players == "two_human":
        p_str = "two human players"
    elif players == 1:
        p_str = "one human"
    else:
        p_str = "one human and one AI"
    return (
        "GAME: %s\nSECTION: %s\nLENGTH: %.0f seconds\nPLAYERS: %s\n\n"
        "WHAT HAPPENED, in order:\n%s\n"
        % (game, level or "unspecified", duration_s,
           p_str,
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
# The ask stays up longer than an ordinary caption -- it is asking for
# something, so it has to survive being read twice.
CLOSING_BONUS = 2.5
# ...and never starts with less than this much tape left, or the question
# leaves the screen before it can be answered.
CLOSING_ROOM = 8.0


def closing_time(events, fps, duration_s):
    """When to put the ask on screen.

    ON the moment the course is cleared, when the run has just paid off and
    the viewer is still looking. The old placement -- five seconds before the
    tape stops -- put it in the score tally or on the world map, seconds after
    anything interesting had finished happening. Falls back to the end of the
    tape when nothing was cleared, and is pulled earlier if the clear itself
    lands too near the end to be read."""
    cleared = [frame / fps for frame, kind, _d in events if kind == "clear"]
    fallback = max(0.0, duration_s - CLOSING_ROOM)
    return min(cleared[-1], fallback) if cleared else fallback


def captions(game, level, duration_s, players, events, fps, max_chars=32,
             two_human=False, **kw):
    """Timed on-screen captions. Returns [{at, text}] or None.

    The model chooses WHICH moments to caption and what to say; the timeline
    decides WHEN each lands."""
    prompt = (
        _context(game, level, duration_s, players, events, fps, two_human=two_human) +
        "\nWrite on-screen captions for this run.\n\n"
        "These are READ IN PASSING while the game is playing, so they are the\n"
        "short form -- a punchline, not a paragraph. The long commentary goes\n"
        "in the narration track, not here.\n\n"
        "RULES:\n"
        "- At most %d CHARACTERS each, and shorter is better. Four to six\n"
        "  words. The viewer is reading this WHILE watching a game, so a\n"
        "  caption they have to study is a caption they miss -- cut every word\n"
        "  that is not carrying the joke.\n"
        "- 'event' is the [N] NUMBER of the moment the caption is about, from\n"
        "  the list above. Do NOT write timestamps -- they are worked out from\n"
        "  the event you name.\n"
        "- Caption the interesting moments only. Skip the dull ones, and skip\n"
        "  most coins.\n"
        "- 'opening' is one caption shown at the very start, setting up the run.\n"
        "- 'closing' is the one caption that has a job. It goes up the moment\n"
        "  the course is cleared, and it asks the viewer something they can\n"
        "  actually answer in a comment -- which level next, which game next,\n"
        "  whether to play one all the way to the finish.\n"
        "  It must be PLAIN and it must be a QUESTION. Ordinary words, one\n"
        "  sentence, ending in a question mark, understandable at a glance by\n"
        "  someone who has read nothing else on screen. No wordplay, no\n"
        "  callback to earlier captions, no in-joke -- those make a funny line\n"
        "  and an unanswerable one. Never 'like and subscribe', never 'smash\n"
        "  that button'. It gets more room than the others: up to %d\n"
        "  characters.\n"
        "  Good: \"Which level should he try next?\"\n"
        "  Good: \"Worth playing this one to the end? Say so.\"\n"
        "  Bad: \"Raccoon or bust?\"  (cute, but what is the question?)\n"
        "- Every caption must be different. No repeated jokes.\n"
        "- Praise the good moments as well as mocking the bad ones.\n\n"
        "Good: \"Walked straight into it.\"\n"
        "Good: \"A mushroom. Do not get attached.\"\n"
        "Good: \"Thirty years of practice.\"\n"
        "Too long: \"He walked into that one completely aware of what it was.\"\n\n"
        "Return JSON: {\"opening\": \"...\", \"closing\": \"...\", \"captions\": "
        "[{\"event\": 3, \"text\": \"...\"}]}"
        % (max_chars, max_chars * 2))
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

    # The ask goes on the clear, after the run has earned it. Asking up front
    # reads as a demand; the same question the moment a course is finished
    # reads as a conversation, and a question someone can answer gets replies
    # where "comment below" does not. It is deliberately allowed to be longer
    # and to sit on screen longer than the jokes.
    closing = str(data.get("closing", "")).strip()
    if closing and duration_s > 8:
        out.append({"at": round(closing_time(events, fps, duration_s), 2),
                    "text": trim_words(closing, max_chars * 2),
                    "event": None, "closing": True,
                    "hold_bonus": CLOSING_BONUS})
    # Sorted AFTER the ask is added, because it no longer goes at the end.
    out.sort(key=lambda c: c["at"])

    # Two captions on top of each other are unreadable, and a model asked for
    # "the interesting moments" will happily pick three in a row.
    spaced = []
    for cap in out:
        # The ask holds longer than the rest, so it needs more clear air after
        # it than an ordinary caption does.
        gap = CAPTION_GAP + (CLOSING_BONUS if spaced and spaced[-1].get("closing")
                             else 0.0)
        if spaced and cap["at"] - spaced[-1]["at"] < gap:
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


def narration(game, level, duration_s, players, events, fps, wpm=125,
              two_human=False, **kw):
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
        _context(game, level, duration_s, players, events, fps, two_human=two_human) +
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


def copy(game, level, duration_s, players, events, fps, watermark="",
         two_human=False, **kw):
    """Platform copy. Returns {description, tiktok, instagram, tags} or None."""
    prompt = (
        _context(game, level, duration_s, players, events, fps, two_human=two_human) +
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
