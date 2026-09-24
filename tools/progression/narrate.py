"""Write and speak a narration for a progression video.

NOT play-by-play. The picture already shows the games; what it cannot show is
where the game came from, what the thing playing it actually is, and what the
numbers on the closing card mean. So the script is a short talk over footage:
the game's history, how the agent was trained, what the card says, and jokes.

Two rules carried over from the studio pipeline, both learned the hard way:

- The speaker must stay FIXED. Qwen3-TTS's VoiceDesign model invents a new
  voice per call, so a line-by-line render made every sentence a different
  person. `tts.speak_lines` uses a named CustomVoice preset for this reason.
- Speech is MEASURED, never estimated, and a line that cannot finish before
  the video ends is dropped rather than pushed past it (`tts.space_clips`).

And one that is specific to this video: the model is given the measured
numbers and told they are the only numbers it may say. A narration that
invents a figure about the training would be worse than one that says nothing
about it, because it sounds equally confident.
"""
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

VOICE = """\
You are writing narration for a short video about a neural network learning to
play a classic console game. The viewer watches several copies of the same game
at once, each played by a different training checkpoint, and the video ends on
a card of measured results.

Your voice: someone who knows both halves -- the machine learning and the
console -- and treats them as equally serious and equally absurd. Dry,
specific, unimpressed by jargon. You explain a thing once, plainly, and move
on. You are funny the way a good documentary narrator is funny: by saying the
true thing with perfect timing, never by announcing that something is amazing.

Never hype, never "epic", never emoji, never a rhetorical question you answer
yourself. The audience is adults who played this game and can follow a real
explanation."""

SCHEMA = {
    "type": "object",
    "properties": {
        "script": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"text": {"type": "string"},
                               "chart": {"type": "boolean"}},
                "required": ["text"],
            },
        },
        "closing": {"type": "string"},
    },
    "required": ["script", "closing"],
}


def duration(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                          "-of", "csv=p=0", path], capture_output=True, text=True, check=True)
    return float(out.stdout.strip())


def _facts(game, card, states, cap):
    """Everything the model is allowed to state as fact about THIS run.

    Written out as plain sentences rather than a JSON blob: backends without
    constrained decoding echo structure they are shown, and a narration that
    reads out a field name has happened before."""
    lines = ["The game is %s." % game,
             "Each panel is one complete game, played by one saved checkpoint of "
             "the same network, from a fixed start.",
             "%d different start positions were used, and every game was stopped "
             "after %d moves if it had not ended by then." % (len(states), cap)]
    if not card:
        return "\n".join(lines)
    for name in card["order"]:
        s = card["stats"][name]
        sd = "no spread to report from one game" if s["sd"] is None else \
             "standard deviation %.0f" % s["sd"]
        lines.append(
            "Checkpoint '%s': mean %.0f, %s, worst %.0f, best %.0f, over %d game%s, "
            "averaging %.0f moves per game; %d game%s ended on their own and %d were "
            "still going when the cap stopped them."
            % (name, s["mean"], sd, s["min"], s["max"], s["n"], "" if s["n"] == 1 else "s",
               s["decisions"], s["finished"], "" if s["finished"] == 1 else "s",
               s["capped"]))
    return "\n".join(lines)


def script(game, seconds, card, states, cap, algorithm="", wpm=125, **kw):
    """Ask a model for the talk. Returns [{text, anchor, closing}] or None."""
    import writer

    words = int(seconds / 60.0 * wpm)
    how = {"afterstate": (
        "The agent is a value network. For every legal placement of the current "
        "piece it simulates the board that placement would leave behind, scores "
        "that resulting board, and plays the best one. It was trained by "
        "temporal-difference learning: the score it gives a board is dragged "
        "towards what actually happened next, over and over."),
        "ppo": (
        "The agent is a policy network trained with PPO. It sees the screen as "
        "pixels and picks which buttons to hold, one decision per frame, and "
        "training nudges it towards the button presses that earned more reward "
        "without letting it change too fast in one step.")}.get(algorithm, "")

    prompt = (
        VOICE + "\n\n"
        "THE VIDEO\n"
        "It runs %.0f seconds. Several games play side by side, sped up, each "
        "labelled with which checkpoint is playing it. They end at different "
        "times and freeze. The last few seconds are a card of results.\n\n"
        "WHAT IS ACTUALLY HAPPENING\n%s\n%s\n\n"
        "WRITE\n"
        "One person talking continuously over this, start to finish. Each entry "
        "in 'script' is the next sentence or two of the SAME monologue, read "
        "straight through with no gaps, so they have to flow into each other.\n\n"
        "The subject is NOT what is happening on screen -- the viewer can see "
        "that, and describing it wastes the only thing you have. Talk about:\n"
        "- the game itself: when it came out, who made it, what it did that was "
        "  new, what it sold, what it is remembered for, how the people who are "
        "  frighteningly good at it play it\n"
        "- what this machine is and how it learned, in plain words\n"
        "- what the results mean, and what they do NOT mean at this sample size\n"
        "- jokes. Real ones, from the material.\n\n"
        "RULES\n"
        "- About %d words TOTAL. Synthesised speech is slower than you expect "
        "  and anything that cannot finish before the video ends is cut.\n"
        "- The ONLY numbers you may state about the training or the results are "
        "  the ones listed above. Numbers about the GAME's own history are "
        "  yours to bring, but say only what you are confident is TRUE -- a "
        "  wrong date about a game this audience grew up with is worse than no "
        "  date at all. If you are unsure, say it without the number.\n"
        "- Do not claim the agent is good or bad in general. It played %d games.\n"
        "- Plain spoken prose. No stage directions, no speaker labels, no field "
        "  names, no markdown, no emoji, no timestamps. Every string is read "
        "  aloud EXACTLY as written.\n"
        "- Mark 'chart': true on the entries that discuss the measured results. "
        "  Those are held back so they land while the card is on screen. Most "
        "  entries are not marked.\n"
        "- 'closing' is the last thing said, over the card. One sentence.\n"
        "Return JSON: {\"script\": [{\"text\": \"...\"}, {\"text\": \"...\", "
        "\"chart\": true}], \"closing\": \"...\"}"
        % (seconds, _facts(game, card, states, cap), how, words,
           sum(card["stats"][n]["n"] for n in card["order"]) if card else len(states)))

    data = writer.write(prompt, SCHEMA, **kw)
    if not data:
        return None
    lines = []
    for item in data.get("script", []):
        raw = item if isinstance(item, str) else item.get("text", "")
        text = writer.clean_spoken(raw)
        if not text:
            continue
        lines.append({"text": text,
                      "chart": bool(isinstance(item, dict) and item.get("chart")),
                      "closing": False})
    closing = writer.clean_spoken(data.get("closing", ""))
    if closing:
        lines.append({"text": closing, "chart": True, "closing": True})
    return lines or None


MIN_GAP = 0.35          # breath between one line ending and the next starting


def spread(items, start, end, min_gap=MIN_GAP):
    """Lay lines across a WINDOW, not end to end from the front.

    Read back to back, a script that is shorter than the footage finishes
    early and leaves the last stretch of video silent. Spacing them by the
    slack instead means the talk covers the whole thing: the gap between lines
    is whatever is left over, shared equally.

    Returns [(start, end, length, item)] and the items that did not fit -- if
    the speech is LONGER than the window there is nothing to share out, and
    the overflow is dropped rather than pushed past the end of the video."""
    lengths = [it["seconds"] for it in items]
    if not items:
        return [], []
    slack = (end - start) - sum(lengths)
    gap = max(min_gap, slack / len(items)) if len(items) else min_gap
    placed, dropped, cursor = [], [], start
    for item, length in zip(items, lengths):
        if cursor + length > end:
            dropped.append(item)
            continue
        placed.append((cursor, cursor + length, length, item))
        cursor += length + gap
    return placed, dropped


def plan(lines, seconds, chart_at):
    """When each sentence starts and stops, for the whole video.

    Two windows, because the closing card is the one moment the narration has
    to be in step with the picture: everything general is spread across the
    footage, and the lines about the numbers are spread across the card."""
    body = [l for l in lines if not l["chart"] and not l["closing"]]
    onchart = [l for l in lines if l["chart"] and not l["closing"]]
    closing = [l for l in lines if l["closing"]]
    reserve = (closing[0]["seconds"] + MIN_GAP) if closing else 0.0

    # With no card at the end there is no second window, and the body runs to
    # the start of the closing instead of stopping short of a card that is
    # not there.
    body_end = (chart_at - MIN_GAP) if onchart else (seconds - reserve)
    placed, dropped = spread(body, 0.6, max(0.6, body_end))
    tail, tail_dropped = spread(onchart, chart_at, max(chart_at, seconds - reserve))
    placed += tail
    dropped += tail_dropped
    if closing:
        at = max(0.0, seconds - closing[0]["seconds"] - 0.3)
        placed.append((at, at + closing[0]["seconds"], closing[0]["seconds"], closing[0]))
    return placed, dropped


def clock(t):
    return "%d:%05.2f" % (int(t // 60), t % 60)


def mux_silent(video, wav, out_path):
    """The grid video has no audio track at all, so there is nothing to duck.

    `tts.mux` mixes narration UNDER game audio and reads [0:a]; on a silent
    video that fails outright rather than degrading, so this is its own path."""
    subprocess.run(
        ["ffmpeg", "-nostdin", "-y", "-v", "error", "-i", video, "-i", wav,
         "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
         "-shortest", "-movflags", "+faststart", out_path],
        check=True)
    return out_path


def add(video, game, card, out_dir, states, cap, chart_seconds=12.0,
        backend="auto", verbose=True, **kw):
    """Write, speak and lay a narration over `video`. Returns the new file."""
    import tts
    from tools.progression import runners

    seconds = duration(video)
    try:
        algorithm = runners.spec_for(game).algorithm
    except Exception:                                             # noqa: BLE001
        algorithm = ""
    if verbose:
        print("  [voice] asking %s for a script ..." % backend)
    lines = script(game, seconds, card, states, cap, algorithm=algorithm,
                   backend=backend, **kw)
    if not lines:
        print("  (no script: the writer returned nothing)")
        return None
    if not card:
        # Nothing to hold a line back FOR: without the card there is no moment
        # in this video that the narration has to be in step with.
        for item in lines:
            item["chart"] = bool(item["closing"])

    chart_at = max(0.0, seconds - chart_seconds)
    txt = os.path.join(out_dir, "narration.txt")
    if not tts.available():
        with open(txt, "w") as fh:
            fh.write("\n".join(l["text"] for l in lines))
        print("  (script written to %s, but the TTS model is not installed, so "
              "there are no timings and nothing spoken)" % txt)
        return None

    model = None
    try:
        model = tts.load()
    except Exception as exc:                                      # noqa: BLE001
        print("  [voice] GPU load failed (%s), falling back to CPU" % exc)
        model = tts.load(device="cpu")
    clips = tts.speak_lines(lines, os.path.join(out_dir, "narration_lines"),
                            model=model, verbose=verbose)
    if not clips:
        return None

    # MEASURED, never estimated: how long a line takes is a property of the
    # rendered wav, and synthesised speech is reliably slower than any
    # words-per-minute arithmetic. The schedule is computed from the files.
    for item, clip in zip([l for l in lines if l["text"].strip()], clips):
        item["path"] = clip[1]
        item["seconds"] = tts.wav_seconds(clip[1])
    spoken = [l for l in lines if l.get("path")]
    placed, dropped = plan(spoken, seconds, chart_at)
    if dropped:
        print("  (voice: %d line%s dropped -- the script was longer than the "
              "video)" % (len(dropped), "" if len(dropped) == 1 else "s"))

    # Every line, with the window it occupies. This file is the schedule: what
    # is said, from when to when, and how much of the video is silence.
    talk = sum(length for _s, _e, length, _i in placed)
    with open(txt, "w") as fh:
        fh.write("%s -- %s of speech over %s of video (%.0f%% covered)\n\n"
                 % (os.path.basename(video), clock(talk), clock(seconds),
                    100.0 * talk / seconds if seconds else 0))
        for start, end, _length, item in placed:
            fh.write("[%s - %s]%s %s\n" % (clock(start), clock(end),
                                           "  (on the card)" if item["chart"] else "",
                                           item["text"]))
    with open(os.path.join(out_dir, "narration.json"), "w") as fh:
        json.dump({"video": os.path.basename(video), "seconds": round(seconds, 2),
                   "chart_at": round(chart_at, 2), "spoken_seconds": round(talk, 2),
                   "lines": [{"start": round(s, 2), "end": round(e, 2),
                              "on_chart": bool(i["chart"]), "text": i["text"],
                              "wav": os.path.basename(i["path"])}
                             for s, e, _l, i in placed]}, fh, indent=2)

    # build_track honours an anchor that is later than where it had got to,
    # so handing it the planned start of every line reproduces the schedule
    # above exactly rather than re-deriving a different one.
    track, _ = tts.build_track([(start, item["path"], item["closing"])
                                for start, _e, _l, item in placed], seconds,
                               os.path.join(out_dir, "narration.wav"))
    if not track:
        return None
    out_path = os.path.splitext(video)[0] + "_narrated.mp4"
    mux_silent(video, track, out_path)
    print("  [voice] %s of speech across %s of video, schedule in %s"
          % (clock(talk), clock(seconds), os.path.basename(txt)))
    return out_path
