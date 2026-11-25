# Three local writers

`tools/narrate_ensemble.py` creates two independent drafts and then supplies both
and the ORIGINAL reviewed evidence to a third model. The master is instructed to
reject unsupported draft claims rather than treat agreement as truth.

Default roles: Gemma 4 12B drafts a connected story; Qwen3.8 27B drafts an alternate
gamer voice; Gemma 4 31B writes the master. They run sequentially, unload after
generation, and use no cloud APIs or speech credits. All three must remain installed.

The current prompt treats this as entertaining Let's Play performance: a playful
opening, changing pressure, a change of pace, and a payoff. Factual grounding applies
to gameplay claims, while jokes and immediate reactions may be authored. The master
must rewrite a coherent story rather than stitch together draft sentences. Examples
span genres and are style demonstrations, not facts. No game-specific Python rules
are introduced. Each complete run is still reviewed for unsupported implications.

```bash
python tools/narrate_ensemble.py --evidence reviewed_evidence.json \
  --out-dir progression_out/narration_ensemble_01
```

Input JSON requires game/narrator context and a `segments` list. Each segment has
an `id`, `start`, `end` (source video seconds), `max_words`, and `observations`
(each with `id` and `text`). Supply reviewed evidence, not unchecked vision output.
The code and style prompt contain no game-specific rules. First-person gamer
reactions are authored voiceover, not a record of the owner's actual thoughts.

Outputs: `draft_a.txt`, `draft_b.txt`, `master.txt`, structured scripts with evidence
references, exact prompts/requests, raw responses, model digests and timings.
Limits and evidence IDs are validated, with one bounded regeneration on failure.
Semantic truth still requires review: evidence IDs alone cannot prove a sentence.
Time slots are editorial targets; fitting synthesized speech is a separate stage.
This command does NOT render audio or modify studio.py's production narrator.

Repeat with `--resume` to retain completed stages after interruption. Evidence,
prompt rules, model identities and ordering must match. Otherwise use a new output
folder. Keep original recordings and failed attempts for comparison.
