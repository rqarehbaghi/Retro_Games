# Local gameplay visual evidence

Replay integration extraction (run with the normal stable-retro venv):

```bash
python tools/extract_replay_evidence.py --replay /path/to/recording.bk2 \
  --out progression_out/replay_evidence.json
```

This exports the integration's original data.json, all declared variable changes,
frame numbers and emulator-derived timestamps. It continues through integration
done flags to match the recorded tape. Values remain in integration units: do not
assume they equal displayed scores, or that numeric suffixes identify players.
Initialization changes have a null before-value and are not gameplay achievements.
Pair only with the source video rendered from the same replay; edited videos need
an explicit time mapping. The original recording is never modified.

TMNT first live test: replay extraction covered 9,417 frames (156.689 seconds),
55 change records, and all six declared variables. Three visual windows covering
15–30 seconds were generated and one frame manually inspected. The HUD displayed
100 where the integration score was 10. Model-reported high confidence also
accompanied unsupported HUD/action claims. These are unverified observations,
not production-ready narration evidence. Numeric reconciliation and broader
visual validation are still required; there is no automatic fusion stage yet.

Dependencies: existing FFmpeg/ffprobe, Ollama and an installed vision model.
No Python packages, paid service, or training environment changes are required.
`ollama pull gemma4:12b` installs the default model if absent. Check support with
`ollama show gemma4:12b`. This machine already has it installed.

```bash
python tools/analyze_gameplay.py \
  --video /path/to/your_recording.mp4 \
  --game 'Your game title' \
  --player-context 'One human player, player 1' \
  --seconds 60 \
  --out-dir progression_out/visual_review_01
```

This is an evidence tool, not yet automatic studio narration integration.
It samples three chronological frames per five-second window. `--seconds`
sets the analyzed duration; `--start` selects a source offset. Use
`--window-seconds 2` for denser sampling of fast action, at greater inference cost.
This is sparse sampling, not continuous video understanding, audio transcription,
or a dedicated OCR engine. It cannot establish every hit, death, or intent.

All game-specific context is input data, not Python rules. Two-human recordings
should identify which on-screen player is the narrator in `--player-context`.
The resulting `evidence.json` includes an explicit completion flag, source hash,
model digest, timestamped observations, model-reported confidence and uncertainty.
Review the saved frame PNGs alongside observations before supplying evidence to
narration writers. Low confidence and unsupported claims must not become facts.
Raw model replies and exact text prompts are preserved for audit. No imagery
leaves the local Ollama endpoint; cloud models are refused.

Repeat the exact command with `--resume` after interruption. Completed windows
are reused only when source, settings and model digest match. No existing
recording or other export is overwritten. Run sequentially with other GPU jobs.
