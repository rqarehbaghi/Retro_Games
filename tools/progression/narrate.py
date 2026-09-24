"""Write and speak the narration for a progression video.

NOT play-by-play. The picture already shows the games. What it cannot show is
where the game came from, what the thing playing it actually is, why this
experiment was run, and what the closing card means.

Three rules this file exists to enforce, each of them learned from a bad take:

- THE NUMBERS ARE ONLY SPOKEN OVER THE CARD. Talking about a mean while a game
  is still playing asks the viewer to look at something that is not on screen
  yet. So the script is written in two parts, the card part is spoken only
  once the card is up, and the card is held for exactly as long as that part
  takes -- the video is cut to the narration, not the narration to the video.

- A BLOCK IS A PARAGRAPH, not a sentence. Sentence-at-a-time synthesis resets
  the voice at every full stop: each one arrives with its own intonation and
  the gaps land on silence instead of breath. See `speech.py`.

- EVERY FIGURE IS NAMED WITH ITS UNIT. Handed "mean 48" and "averaging 155",
  a model called a lines-cleared average "the average placement count" -- in a
  finished video, about a number printed on the card beside it.

The measured numbers are supplied to the model and it is told they are the
only figures it may state about the training. The game's own history it brings
itself, with the usual instruction to leave out anything it is unsure of.
"""
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tools.progression import speech                              # noqa: E402

BLOCK_GAP = 0.45        # breath between paragraphs, not a pause for effect
CARD_LEAD = 0.4         # beat after the card appears before the voice returns
WPM = 140               # measured, not assumed: see script()

VOICE = """\
You are writing the narration for a short film about a neural network learning
to play a classic console game. It is published on a channel called %s, whose
whole premise is that player two is a machine: the machine plays, the results
get posted, and the running question is whether it is yet good enough to be
worth a human sitting down opposite it.

Your voice: someone who knows both halves -- the machine learning and the
console -- and finds them equally serious and equally absurd. Dry, exact,
unimpressed by jargon. You explain a thing once, in plain words, and move on.
You are funny the way a good documentary narrator is funny: by saying the true
thing with perfect timing, never by announcing that something is remarkable.

Write for the EAR. Short sentences next to long ones. No sentence that needs
re-reading. No lists read aloud, no "firstly", no rhetorical question you then
answer yourself, no "welcome back", no hype, no emoji. The audience are adults
who played this game and can follow a real explanation."""

SCHEMA = {
    "type": "object",
    "properties": {
        "body": {"type": "array", "items": {"type": "string"}},
        "card": {"type": "array", "items": {"type": "string"}},
        "closing": {"type": "string"},
    },
    "required": ["body", "card", "closing"],
}


def duration(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                          "-of", "csv=p=0", path], capture_output=True, text=True, check=True)
    return float(out.stdout.strip())


def clock(t):
    return "%d:%05.2f" % (int(t // 60), t % 60)


def facts(game, card, states, cap):
    """Everything the model may state as fact about THIS experiment.

    Plain sentences rather than a JSON blob: a backend without constrained
    decoding echoes structure it is shown, and a narration that reads out a
    field name has happened here before. Every number carries its unit."""
    lines = ["The game is %s." % game,
             "Each panel is one complete game, played by one saved checkpoint of "
             "the same network, from a fixed starting position.",
             "%d different starting positions were used. Every game was stopped "
             "after %d moves if it had not ended on its own by then."
             % (len(states), cap)]
    if not card:
        return "\n".join(lines)
    metric = card.get("metric_label") or "points"
    work = card.get("work_label") or "moves"
    lines.append("The headline number for each game is %s. The other number is how "
                 "many %s it took." % (metric, work))
    for name in card["order"]:
        s = card["stats"][name]
        sd = ("no spread to report from a single game" if s["sd"] is None
              else "standard deviation %.0f %s" % (s["sd"], metric))
        lines.append(
            "Checkpoint '%s': mean %.0f %s per game, %s, worst game %.0f %s, best "
            "game %.0f %s, from %d game%s, averaging %.0f %s per game; %d ended on "
            "their own and %d were still going when the cap stopped them."
            % (name, s["mean"], metric, sd, s["min"], metric, s["max"], metric,
               s["n"], "" if s["n"] == 1 else "s", s["decisions"], work,
               s["finished"], s["capped"]))
    return "\n".join(lines)


def write(game, body_seconds, card, states, cap, algorithm="", channel="",
          wpm=WPM, **kw):
    """Ask a model for the script. Returns {"body": [...], "card": [...],
    "closing": str} or None.

    `body_seconds` is how long the GAMEPLAY runs. The card's length is not
    given, because the card is later held for as long as the card part takes
    to say -- which is the only way the numbers are never spoken over a game
    still in progress."""
    import writer

    words = int(body_seconds / 60.0 * wpm)
    how = {"afterstate": (
        "The agent is a value network. For every legal placement of the current "
        "piece it simulates the board that placement would leave behind, scores "
        "that resulting board with a single number, and plays the best one. It "
        "was trained by temporal-difference learning: the score it gives a board "
        "is dragged towards what actually happened next, over and over. Nobody "
        "told it that holes are bad; it had to find that in the consequences."),
        "ppo": (
        "The agent is a policy network trained with PPO. It sees the screen as "
        "pixels and chooses which buttons to hold, one decision per frame. "
        "Training nudges it towards the presses that earned more reward, while "
        "stopping it changing too much in any one step.")}.get(algorithm, "")

    prompt = (
        (VOICE % (channel or "this channel")) + "\n\n"
        "THE FILM\n"
        "Several games play side by side, sped up, each labelled with which "
        "training checkpoint is playing it. They finish at different times and "
        "freeze. Then a card of measured results appears and stays until the "
        "end.\n\n"
        "WHY IT EXISTS\n"
        "This is an experiment, not a highlight reel. The same network was "
        "saved at different points in its training and every copy was made to "
        "play the same positions, to see whether it is actually getting better "
        "-- and how close it is to being worth challenging a human.\n\n"
        "WHAT IS ACTUALLY HAPPENING\n%s\n%s\n\n"
        "WRITE TWO PARTS.\n\n"
        "'body' -- spoken over the GAMES. About %d words, in %d to %d "
        "paragraphs of two to four sentences. Each paragraph is read as one "
        "continuous take, and they run one after another, so they must follow "
        "on. This part is the game's own story and the machine's: when the game "
        "was made and by whom, what it did that was new, what it is remembered "
        "for, how the people who are frighteningly good at it play; then what "
        "this machine is, how it learned, and what it does not know. Jokes "
        "belong here. NO NUMBERS FROM THE RESULTS IN THIS PART -- they are not "
        "on screen yet.\n\n"
        "'card' -- spoken ONLY over the results card, which is held for exactly "
        "as long as this takes to say, so write what is worth saying and no "
        "more: two to four paragraphs. This is where the measured numbers go. "
        "Say what changed between checkpoints, name the units, and be honest "
        "about what %d games cannot show. If a later checkpoint is worse than "
        "an earlier one, say so plainly.\n\n"
        "'closing' -- the last thing said, over the card. One or two sentences "
        "that land the premise of %s: the machine is player two, this is how it "
        "is coming along, and whether it is ready to be sat down opposite a "
        "human. A real ending, not 'like and subscribe'.\n\n"
        "RULES\n"
        "- The ONLY numbers you may state about the training or the results are "
        "the ones listed above, with the units they are given with. Numbers "
        "about the GAME's own history are yours to bring, but say only what you "
        "are confident is TRUE: a wrong date about a game this audience grew up "
        "with is worse than no date.\n"
        "- Do not describe what is on screen. Do not say 'as you can see'.\n"
        "- Do not claim the agent is good or bad in general. It played %d games.\n"
        "- Plain spoken prose, read aloud EXACTLY as written. No stage "
        "directions, no speaker labels, no field names, no markdown, no "
        "headings, no emoji, no timestamps.\n"
        "Return JSON: {\"body\": [\"...\", \"...\"], \"card\": [\"...\"], "
        "\"closing\": \"...\"}"
        % (facts(game, card, states, cap), how, words,
           max(3, words // 90), max(4, words // 60),
           sum(card["stats"][n]["n"] for n in card["order"]) if card else len(states),
           channel or "the channel",
           sum(card["stats"][n]["n"] for n in card["order"]) if card else len(states)))

    data = writer.write(prompt, SCHEMA, **kw)
    if not data:
        return None
    clean = writer.clean_spoken
    out = {"body": [clean(t) for t in data.get("body", []) if clean(t)],
           "card": [clean(t) for t in data.get("card", []) if clean(t)],
           "closing": clean(data.get("closing", ""))}
    return out if (out["body"] or out["card"]) else None


def schedule(body, card, grid_seconds, lead=CARD_LEAD, gap=BLOCK_GAP):
    """When each block is spoken.

    The body runs end to end from the start; the card part begins once the
    card is up. Blocks are laid consecutively with one short breath between
    them -- NOT spread to fill, which was the previous design and produced
    fifteen second holes between sentences. Filling the video is the job of
    writing enough words and, if asked, of `--speed auto` setting the gameplay
    to the length of the speech.

    `body` and `card` are [(path, seconds)]. Returns [(start, end, path,
    on_card)]."""
    placed, t = [], 0.6
    for path, length in body:
        placed.append((t, t + length, path, False))
        t += length + gap
    t = max(t, grid_seconds + lead)
    for path, length in card:
        placed.append((t, t + length, path, True))
        t += length + gap
    return placed


def card_seconds(card, lead=CARD_LEAD, gap=BLOCK_GAP, tail=1.2):
    """How long the closing card must stay up to hold everything said over it."""
    if not card:
        return 0.0
    return lead + sum(length for _p, length in card) + gap * (len(card) - 1) + tail


def build_track(placed, total, out_path, sample_rate=speech.SAMPLE_RATE):
    """One wav the length of the film, each block starting at its time.

    Built on a silent bed of the right length so the track lines up with the
    picture on its own, without relying on the mux to position anything."""
    if not placed:
        return None
    inputs, chains, labels = [], [], []
    for i, (start, _end, path, _on_card) in enumerate(placed):
        inputs += ["-i", path]
        chains.append("[%d:a]aresample=%d,adelay=%d|%d[d%d]"
                      % (i + 1, sample_rate, int(start * 1000), int(start * 1000), i))
        labels.append("[d%d]" % i)
    labels.insert(0, "[0:a]")
    graph = ";".join(chains) + ";" + "".join(labels) + \
        "amix=inputs=%d:normalize=0:dropout_transition=0[out]" % len(labels)
    subprocess.run(
        ["ffmpeg", "-nostdin", "-y", "-v", "error",
         "-f", "lavfi", "-t", "%.3f" % total,
         "-i", "anullsrc=channel_layout=mono:sample_rate=%d" % sample_rate,
         *inputs, "-filter_complex", graph, "-map", "[out]",
         "-t", "%.3f" % total, out_path], check=True)
    return out_path


def mux(video, wav, out_path):
    """The grid has no audio track at all, so there is nothing to duck."""
    subprocess.run(
        ["ffmpeg", "-nostdin", "-y", "-v", "error", "-i", video, "-i", wav,
         "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
         "-shortest", "-movflags", "+faststart", out_path], check=True)
    return out_path


def save_schedule(placed, texts, out_dir, video_name, total, grid_seconds, take=""):
    """narration.txt is the readable schedule; narration.json is the same
    thing for anything that has to read it back (--respeak does)."""
    talk = sum(end - start for start, end, _p, _c in placed)
    txt = os.path.join(out_dir, "narration%s.txt" % take)
    with open(txt, "w") as fh:
        fh.write("%s\n%s of speech over %s of film (%.0f%% covered)\n"
                 "the results card is up from %s\n\n"
                 % (video_name, clock(talk), clock(total),
                    100.0 * talk / total if total else 0.0, clock(grid_seconds)))
        for (start, end, _path, on_card), text in zip(placed, texts):
            fh.write("[%s - %s]%s\n%s\n\n"
                     % (clock(start), clock(end), "  ON THE CARD" if on_card else "", text))
    with open(os.path.join(out_dir, "narration%s.json" % take), "w") as fh:
        json.dump({"video": video_name, "seconds": round(total, 2),
                   "card_at": round(grid_seconds, 2),
                   "spoken_seconds": round(talk, 2),
                   "blocks": [{"start": round(s, 2), "end": round(e, 2),
                               "on_card": bool(c), "text": t,
                               "wav": os.path.basename(p)}
                              for (s, e, p, c), t in zip(placed, texts)]},
                  fh, indent=2)
    return txt
