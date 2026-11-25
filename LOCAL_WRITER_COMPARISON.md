# Compare local podcast writers

Run inside WSL with Ollama running. No TTS, paid API, or gameplay is invoked.
Keep GPU training and TTS idle during comparison. Models run sequentially and
are unloaded after each completed generation.

```bash
python tools/compare_local_writers.py \
  --facts tools/local_writer_sample.txt \
  --out-dir progression_out/writer_comparison_01
```

By default this exports every installed local text model. Cloud entries and
non-text models are skipped. `--models name1,name2` selects a subset.
Use a new output folder for every comparison; existing results are never overwritten.
Replace the facts file with your own UTF-8 gameplay evidence for a real production
comparison. The included sample is explicitly a showcase, not a controlled test.

Each model gets exactly the same prompt, temperature, seed, context limit, and
output token budget, with thinking disabled where supported. Matching seeds do
not make different model families equivalent or guarantee reproducibility.
The exports include source facts, exact prompt, per-model spoken script, raw
request/response, model digest, timing, errors and truncation status. Thinking
text is never substituted for missing spoken output. Errors do not discard
other models' exports. A timeout may leave Ollama finishing its current request.

Compare factual accuracy, conversational flow, clarity, forced humor and the
closing invitation. Do not infer voice quality from these text-only exports.

For a single human narrator describing their own gameplay, add `--style first-person`.
Use `--facts tools/local_writer_human_sample.txt` for the clearly labelled synthetic
two-human example, or supply facts from a real recording. All models receive the
same scenario. No models are removed by this tool. Delimited reasoning is excluded
from scripts and word counts, while raw responses remain available for debugging.

Installation sizing: this machine was measured with a 16 GB RTX 4090 Laptop GPU
and 15 GB WSL RAM. Large 18–20 GB models require partial CPU offload; a modest
context is deliberate. GLM-5.3 and DeepSeek-V4.1-Flash are not suitable downloads
for this machine and the owner's approximately 200 GB free-space budget.
