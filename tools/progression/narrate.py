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
You are writing in the FIRST PERSON, as the person who built and trained this
thing, talking to their own audience on %s. Not a narrator, not a documentary
voice, not "the human who trained it" -- me. I say I. I trained it, I am
showing you my own results, and at the end I am the one who has to sit down
and play it.

So: I made this. I chose the rewards, I left it running, I am as surprised as
anyone by half of what it does. I am allowed to be proud of it and to take the
mickey out of it in the same sentence, because it is mine. When it does
something stupid, that is my fault and I will say so. When it beats me, that
will also be my fault, which is worse.

Write like someone talking to camera, not reading an essay. Contractions.
Asides. The odd sentence that is just two words. You explain a thing once, in
plain words, and move on.

BE ACTUALLY FUNNY. Not "playful", not "quirky". Funny is a true observation
placed where it lands hardest, and it is usually the specific detail, not the
joke-shaped sentence. Copy this register, and note that it is ME speaking:

  "It has no idea it's playing Tetris. It has no idea it's playing
   anything. I built a function that stares at a wall of blocks and says
   'seven'."

  "I never told it holes were bad. It worked that out the way I did, by
   dying repeatedly."

  "I gave it a reward for clearing lines. Its reward for a thousand perfect
   decisions is one more block."

  "Ten thousand steps in, it played like someone who'd had the rules
   explained to them over the phone. I have the footage. It's up there."

Rules for the jokes: never announce one, never explain one, never end a
paragraph on an explanation when it could end on the observation. One dud
lands worse than none, so if a line is not actually funny, make it plain and
true instead. No puns on the game's name, no "little did it know", no
addressing the audience, no rhetorical question you then answer.

Write for the EAR. Vary the sentence length hard -- three words, then thirty.
No sentence anyone has to re-read. No lists read aloud, no "firstly", no
"welcome back", no hype, no emoji, no stage directions."""

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
          wpm=WPM, card_seconds_budget=22.0, **kw):
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

    games = (sum(card["stats"][n]["n"] for n in card["order"]) if card
             else len(states))
    winner = (card or {}).get("best") or ""
    card_words = max(40, int(card_seconds_budget / 60.0 * wpm))
    prompt = (
        (VOICE % (channel or "this channel")) + "\n\n"
        "THE FILM\n"
        "Several games play side by side, sped up, each labelled with which "
        "training checkpoint is playing it. As each one loses, its panel dims "
        "and the rest play on faster. When the last one loses, the film holds "
        "on it, then a card of measured results appears and the film ends.\n\n"
        "WHY IT EXISTS -- SAY THIS, IT IS THE POINT\n"
        "This is a try-out, not a highlight reel. The same network was saved at "
        "different points in its training, and every copy was made to play the "
        "same starting positions, to find out which one is actually the best. "
        "THE WINNER BECOMES THE OPPONENT: the human who trained it will play "
        "that checkpoint three games, head to head, and we find out who wins. "
        "The interesting part is that the newest checkpoint is not "
        "automatically the best one -- more training is not the same as better "
        "-- so this is how the opponent gets chosen rather than assumed.\n\n"
        "WHAT IS ACTUALLY HAPPENING\n%s\n%s\n\n"
        "WRITE THREE PARTS.\n\n"
        "'body' -- spoken over the GAMES. About %d words, in %d to %d "
        "paragraphs of two to four sentences, read one after another as one "
        "continuous take, so they must follow on. Cover, in your own order: "
        "what this is and what is being decided; the game's own history -- when "
        "it was made, by whom, what it is remembered for, how the people who "
        "are frighteningly good at it play; what this machine is and how it "
        "learned; and what it has no idea about. This is where the jokes live. "
        "NO RESULT NUMBERS IN THIS PART -- the card is not on screen yet.\n\n"
        "'card' -- spoken over the results card, which shows a bar per "
        "checkpoint and a CROWN on the winner. HARD LIMIT %d WORDS, one or two "
        "short paragraphs, because the card is held only as long as this takes "
        "and a still image with a long voice over it is not a video. The FIRST "
        "sentence must crown the winner by name and say its number -- that is "
        "the sentence the whole film has been walking towards, so it goes "
        "first and it does not hedge. Then, in one clause, what %d games "
        "cannot prove; and if a later checkpoint lost to an earlier one, say "
        "so, because I find that the most interesting thing here.%s\n\n"
        "'closing' -- the last line, over the card. One or two sentences, and "
        "it is ME making a promise: I am coming back to play that checkpoint, "
        "three games, and I intend to win. Be specific and a bit cocky about "
        "it -- I trained the thing, so losing to it would be humiliating in a "
        "way worth watching. Not 'like and subscribe'.\n\n"
        "RULES\n"
        "- The ONLY numbers you may state about the training or the results are "
        "the ones listed above, with the units they are given with. Numbers "
        "about the GAME's own history are yours to bring, but say only what you "
        "are confident is TRUE: a wrong date about a game this audience grew up "
        "with is worse than no date.\n"
        "- Do not describe what is on screen. Do not say 'as you can see'.\n"
        "- Do not claim the agent is good or bad in general. It played %d "
        "games.\n"
        "- Plain spoken prose, read aloud EXACTLY as written. No stage "
        "directions, no speaker labels, no field names, no markdown, no "
        "headings, no emoji, no timestamps.\n"
        "Return JSON: {\"body\": [\"...\", \"...\"], \"card\": [\"...\"], "
        "\"closing\": \"...\"}"
        % (facts(game, card, states, cap), how, words,
           max(3, words // 90), max(4, words // 60), card_words, games,
           (" The winner on these games is '%s' -- that is the checkpoint that "
            "gets played against, and it is what the card says, so do not "
            "nominate a different one." % winner) if winner else "",
           games))

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


def fit_card(clips, cap, lead=CARD_LEAD, gap=BLOCK_GAP, tail=1.2):
    """Which card blocks fit the hold, and how long the hold must then be.

    Capping the hold WITHOUT trimming the speech is how a card ended up
    completely silent: the blocks ran past the end of the film and every one of
    them was dropped, so the winner was never announced at all. Here the cap
    decides how many blocks are kept, and the hold is then exactly what those
    blocks need -- so there is never a card with nothing said over it.

    Two blocks are MANDATORY and the cap does not get to drop them: the first,
    which crowns the winner, and the last, which is the challenge. Those are
    the two sentences the whole film exists to deliver. Everything between them
    is what the cap trims, and if the two mandatory ones together need longer
    than the cap, the card is held for them anyway -- a card that holds for the
    trivia and drops the ending is exactly the wrong way round."""
    if not clips:
        return [], [], 0.0
    if len(clips) <= 2:
        return list(clips), list(range(len(clips))), card_seconds(clips, lead, gap, tail)
    first, closing, middle = clips[0], clips[-1], clips[1:-1]
    budget = cap - lead - tail - first[1] - closing[1] - gap
    kept, index, used = [first], [0], 0.0
    for i, clip in enumerate(middle, start=1):
        need = clip[1] + gap
        if used + need > budget:
            break
        kept.append(clip)
        index.append(i)
        used += need
    kept.append(closing)
    index.append(len(clips) - 1)
    return kept, index, card_seconds(kept, lead, gap, tail)


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
