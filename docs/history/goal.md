Follow-up goal after the initial Qwen Scope 27B feature-steering GUI is complete:

Build the next layer of the system: **Steering Bench + Recipe Cards + Autopilot**.

The purpose of this phase is to turn the existing Qwen Scope GUI from an interactive feature browser/steering demo into a credible feature-intervention workbench that can:

1. Discover candidate steering recipes from a user-specified behavior goal.
2. Benchmark those recipes against prompt-only and control baselines.
3. Export reproducible, reviewer-friendly recipe cards with exact model/SAE metadata, interventions, metrics, examples, and limitations.

The initial goal is assumed to have already completed successfully, including:

- Working Gradio GUI.
- Working Qwen + SAE model loader.
- Lazy SAE layer loading.
- Real residual-stream steering hook.
- Dev model path.
- Modal GPU path.
- 27B smoke test for `Qwen/Qwen3.5-27B` + `Qwen/SAE-Res-Qwen3.5-27B-W80K-L0_100`.
- Existing `.env` support.
- Existing `README.md`, `RUNBOOK.md`, and `RESULTS.md`.

Do not rebuild the base system. Extend it.

Use your Modal skill for all 27B GPU work. Hugging Face and model API keys will be provided through a local `.env`.

---

## Mission

Add three major capabilities:

```text
1. Steering Bench
   Evaluate whether a feature intervention actually improves target behavior
   compared with prompt-only, no-steering, zero-strength, random-feature,
   negative-strength, and optionally nearest-neighbor controls.

2. Recipe Cards
   Save, load, validate, benchmark, and export feature interventions as
   reproducible JSON + Markdown artifacts.

3. Autopilot
   Given a user behavior goal and examples, automatically search candidate
   features, run steering sweeps, benchmark candidates, and produce a candidate
   recipe card with evidence and caveats.
````

This phase is not done when the UI has new buttons. It is done only when the system can create at least one complete recipe card from a real dev-model run and at least one compact 27B Modal smoke run.

---

## High-level product shape

The app should evolve from:

```text
Inspect → Compare → Steer
```

to:

```text
Inspect → Compare → Steer → Bench → Autopilot → Recipe Library
```

The core object of the system should become a `FeatureRecipe`.

A feature ID by itself is not enough. A useful recipe must include:

```text
target behavior
model id
SAE id
layer
feature id(s)
strength(s)
injection mode
discovery examples
validation prompts
control results
benchmark results
side effects
known limitations
example before/after generations
provenance
```

---

## Non-negotiable constraints

### Do not regress the initial system

All previous strict done criteria from the initial goal remain valid.

Do not break:

* existing Inspect tab
* existing Compare tab
* existing Steer tab
* existing Modal smoke tests
* existing lazy SAE loading
* existing `.env` handling
* existing tests
* existing run commands

If existing behavior must change, document the reason and preserve backward-compatible CLI/API wrappers where possible.

### Credentials

Use only `.env` / environment variables.

Expected possible variables:

```bash
HF_TOKEN=
HUGGINGFACE_HUB_TOKEN=
MODEL_API_KEY=
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
MODAL_TOKEN_ID=
MODAL_TOKEN_SECRET=
```

Core benchmarking, steering, and recipe export must not require hosted model APIs.

Hosted model APIs may be used only for optional:

* generating example prompts
* summarizing recipe labels
* judging open-ended behavior when explicitly enabled

Do not require those APIs for tests or core smoke runs.

Never print, log, serialize, or commit secrets.

### Modal

Use the existing Modal skill and repo conventions.

27B runs must use Modal GPU.

For cost control:

* Use short prompts.
* Use low `max_new_tokens` for smoke tests.
* Use one active SAE layer by default.
* Use small candidate sets for 27B smoke.
* Do not run a broad all-layer search on 27B unless explicitly requested later.
* Prefer validating full logic on the dev model first, then running one compact 27B proof.

### No fake success

The following are not acceptable:

* A bench that only displays mock metrics.
* An autopilot that returns hardcoded features.
* A recipe card without real generation results.
* A recipe card that omits controls.
* Claiming a feature is “validated” without held-out prompts and controls.
* Claiming 27B support without a real Modal smoke run or a clear `BLOCKED.md`.

---

## Target architecture additions

Extend the existing package with modules like these. Adapt names to the existing repo structure.

```text
qwen_scope_steering_gui/
  recipes.py
  recipe_schema.py
  recipe_store.py
  benchmark.py
  benchmark_metrics.py
  benchmark_controls.py
  autopilot.py
  candidate_search.py
  sweep.py
  prompt_sets.py
  judges.py
  reports.py

configs/
  bench/
    concise_behavior_dev.yaml
    json_validity_dev.yaml
    code_switch_dev.yaml
    qwen35_27b_smoke_bench.yaml

recipes/
  .gitkeep

reports/
  .gitkeep

scripts/
  bench_recipe.py
  create_recipe.py
  autopilot_recipe.py
  export_recipe_markdown.py
  modal_bench_smoke_2b.py
  modal_bench_smoke_27b.py

tests/
  test_recipe_schema.py
  test_recipe_store.py
  test_benchmark_metrics.py
  test_benchmark_controls.py
  test_sweep.py
  test_autopilot_candidate_ranking.py
  test_recipe_export.py
  test_bench_no_secret_logging.py
```

Do not force this exact tree if the existing project uses a different layout, but the responsibilities must exist.

---

# Part 1: FeatureRecipe schema

Implement a durable, versioned `FeatureRecipe` schema.

## Required JSON schema fields

A recipe must serialize to JSON with this general shape:

```json
{
  "schema_version": "0.1.0",
  "recipe_id": "concise_answers_l32_f18472_v1",
  "created_at": "2026-05-27T00:00:00Z",
  "created_by": "qwen-scope-autopilot",
  "target_behavior": {
    "name": "concise_answers",
    "description": "Make answers shorter and more direct while preserving correctness.",
    "positive_description": "Direct, concise, complete answer.",
    "negative_description": "Verbose, meandering, unnecessary caveats."
  },
  "model": {
    "model_id": "Qwen/Qwen3.5-27B",
    "sae_id": "Qwen/SAE-Res-Qwen3.5-27B-W80K-L0_100",
    "dtype": "bfloat16",
    "config_name": "configs/qwen35_27b_l0_100.yaml"
  },
  "interventions": [
    {
      "layer": 32,
      "feature_id": 18472,
      "strength": 4.5,
      "sign": "positive",
      "injection_mode": "generated_tokens",
      "position_policy": "all_generated_tokens"
    }
  ],
  "discovery": {
    "method": "contrastive_prompt_search",
    "positive_prompts": [],
    "negative_prompts": [],
    "candidate_layers": [24, 32, 40],
    "candidate_count": 10,
    "ranking_metric": "activation_contrast"
  },
  "benchmark": {
    "status": "unvalidated",
    "prompt_set_id": "concise_dev_v1",
    "methods_compared": [],
    "metrics": {},
    "controls": {},
    "summary": ""
  },
  "examples": [
    {
      "prompt": "...",
      "unsteered": "...",
      "steered": "...",
      "prompt_only": "...",
      "prompt_plus_steering": "..."
    }
  ],
  "limitations": [],
  "side_effects": [],
  "artifacts": {
    "json_path": "",
    "markdown_path": "",
    "results_path": ""
  },
  "provenance": {
    "git_commit": "",
    "command": "",
    "modal_gpu": "",
    "python_version": "",
    "torch_version": "",
    "transformers_version": "",
    "seed": 0
  }
}
```

Use Pydantic, dataclasses + validation, or equivalent.

## Required behavior

Implement:

```python
FeatureRecipe.validate()
FeatureRecipe.to_json()
FeatureRecipe.from_json()
FeatureRecipe.to_markdown()
FeatureRecipe.compute_id()
FeatureRecipe.mark_unvalidated()
FeatureRecipe.mark_candidate()
FeatureRecipe.mark_validated()
```

Validation must catch:

* missing model id
* missing SAE id
* invalid feature id
* invalid layer
* invalid strength
* missing target behavior
* benchmark status claiming `validated` without required benchmark/control evidence
* mismatched model/SAE config if metadata is available

## Recipe statuses

Use exactly these statuses unless there is a strong reason to change them:

```text
draft
candidate
benchmarked
validated
failed
blocked
```

A recipe can be marked `validated` only if it passes predeclared criteria.

---

# Part 2: Recipe Library

Add a recipe library layer.

## Required storage behavior

Store recipes as local JSON files by default:

```text
recipes/
  concise_answers_l32_f18472_v1/
    recipe.json
    recipe.md
    benchmark_results.json
    examples.jsonl
```

Do not require a database.

Implement:

```python
RecipeStore(root="recipes/")
RecipeStore.save(recipe)
RecipeStore.load(recipe_id)
RecipeStore.list()
RecipeStore.search(query)
RecipeStore.export_markdown(recipe_id)
```

Search can be simple at first:

* recipe id
* target behavior name
* target behavior description
* feature id
* layer
* notes
* status

## GUI requirements

Add a `Recipe Library` tab.

It must support:

* list recipes
* filter by status
* open recipe details
* view interventions
* view benchmark summary
* view before/after examples
* export Markdown
* load recipe into Steering tab
* load recipe into Bench tab

---

# Part 3: Steering Bench

Build a benchmarking harness for steering recipes.

The bench must answer:

> Does this feature recipe improve the target behavior compared with reasonable baselines and controls?

## Required methods to compare

For each benchmark prompt, compare at least:

```text
A. unsteered_baseline
B. prompt_only
C. steering_only
D. prompt_plus_steering
E. zero_strength_control
F. random_feature_control
G. negative_strength_control
```

Optional but desirable:

```text
H. nearest_neighbor_feature_control
I. shuffled_layer_control
J. same_feature_different_layer_control
```

## Prompt-only baseline

The bench must support a prompt-only intervention.

Example:

```text
Original prompt:
  "Explain what a sparse autoencoder does."

Prompt-only behavior instruction:
  "Answer concisely in no more than two sentences.
   Prompt: Explain what a sparse autoencoder does."
```

For each recipe, the user or config should define a prompt-only instruction string.

This is critical. Feature steering is not useful unless it is compared against prompting.

## Steering-only method

Use the original prompt with feature steering applied.

## Prompt + steering method

Use the prompt-only behavior instruction and feature steering together.

## Controls

### Zero-strength control

Same feature, same layer, strength `0.0`.

Must produce identical or near-identical behavior to unsteered under deterministic settings.

### Random-feature control

Same layer, random feature ID, same strength.

Use a fixed seed.

Run at least one random feature control by default.

### Negative-strength control

Same feature, opposite sign/negative strength.

This should often reduce or invert the target behavior. It does not need to always do so, but it must be measured.

### Optional nearest-neighbor control

If nearest-neighbor feature search already exists, use a nearby feature with similar activation statistics. If not implemented yet, leave as optional.

## Required metrics

Implement rule-based metrics first.

Generic metrics:

```text
output_length_chars
output_length_tokens
sentence_count
repetition_score
distinct_1
distinct_2
finish_reason
generation_error
latency_seconds
```

Steering/hook metrics:

```text
hook_fired
hidden_delta_norm
logits_delta_norm if available
feature_activation_before
feature_activation_after if available
```

Control metrics:

```text
random_control_delta
zero_strength_delta
negative_strength_delta
```

Coherence proxies:

```text
empty_output
truncated_output
repeated_ngram_rate
invalid_unicode
excessive_repetition_flag
```

Task-specific metrics:

```text
json_validity
json_schema_validity
regex_match
contains_required_terms
excludes_forbidden_terms
max_length_pass
min_length_pass
exact_match optional
```

Optional model-judge metrics:

```text
behavior_score
coherence_score
correctness_score
side_effect_score
```

Model-judge metrics must be disabled by default unless a model API key is present and the user explicitly enables them.

## Required benchmark result schema

Each benchmark run must produce JSON like:

```json
{
  "benchmark_id": "bench_2026_05_27_abc123",
  "recipe_id": "concise_answers_l32_f18472_v1",
  "created_at": "...",
  "config": {
    "model_id": "...",
    "sae_id": "...",
    "prompt_set_id": "...",
    "temperature": 0.0,
    "max_new_tokens": 64,
    "seed": 0
  },
  "methods": [
    "unsteered_baseline",
    "prompt_only",
    "steering_only",
    "prompt_plus_steering",
    "zero_strength_control",
    "random_feature_control",
    "negative_strength_control"
  ],
  "aggregate_metrics": {
    "unsteered_baseline": {},
    "prompt_only": {},
    "steering_only": {},
    "prompt_plus_steering": {},
    "zero_strength_control": {},
    "random_feature_control": {},
    "negative_strength_control": {}
  },
  "per_prompt_results": [
    {
      "prompt_id": "p001",
      "prompt": "...",
      "outputs": {
        "unsteered_baseline": "...",
        "prompt_only": "...",
        "steering_only": "...",
        "prompt_plus_steering": "..."
      },
      "metrics": {}
    }
  ],
  "pass_fail": {
    "passed": false,
    "reasons": []
  }
}
```

## Required GUI tab: Bench

Add a `Bench` tab.

Inputs:

* Select existing recipe or enter layer/feature/strength manually.
* Prompt set text box or JSONL upload.
* Prompt-only instruction.
* Metrics selection.
* Max new tokens.
* Temperature.
* Seed.
* Number of random controls.
* Run small benchmark.
* Export results.

Outputs:

* Method comparison table.
* Aggregate metrics table.
* Before/after examples.
* Control results.
* Pass/fail status.
* Link/path to saved benchmark JSON.
* Button: “Save as Recipe Card”
* Button: “Mark Recipe Candidate”
* Button: “Mark Recipe Validated” only if validation criteria pass.

## Required CLI

Implement:

```bash
python scripts/bench_recipe.py \
  --recipe recipes/<recipe_id>/recipe.json \
  --prompt-set data/prompt_sets/concise_dev.jsonl \
  --output reports/<recipe_id>_bench.json \
  --max-new-tokens 64 \
  --temperature 0.0
```

Support inline manual recipe too:

```bash
python scripts/bench_recipe.py \
  --config configs/qwen35_2b_dev_l0_100.yaml \
  --target-behavior concise_answers \
  --layer 12 \
  --feature-id 12345 \
  --strength 5.0 \
  --prompt-set data/prompt_sets/concise_dev.jsonl
```

---

# Part 4: Strength Sweep Grid

Add a compact sweep runner.

Given a candidate recipe, run:

```text
strengths: [-8, -4, -2, 0, 2, 4, 8]
injection_modes: at least current default mode
optional layers: [layer - 1, layer, layer + 1]
```

For each setting, compute benchmark metrics.

## Required output

```json
{
  "recipe_id": "...",
  "sweep_id": "...",
  "strengths": [-8, -4, -2, 0, 2, 4, 8],
  "results": [
    {
      "layer": 32,
      "feature_id": 18472,
      "strength": 4,
      "aggregate_metrics": {},
      "pass_fail": {}
    }
  ],
  "best_setting": {
    "layer": 32,
    "feature_id": 18472,
    "strength": 4,
    "reason": "best behavior score with acceptable coherence"
  }
}
```

## GUI

In the Bench tab or a separate Sweep section:

* Button: “Run strength sweep”
* Show table sorted by configured objective.
* Plot simple metric vs strength if plotting infra exists.
* Save sweep JSON.
* Attach sweep to recipe card.

No fancy visualization is required for done.

---

# Part 5: Autopilot

Build `Autopilot`.

Autopilot takes:

```text
target behavior description
positive examples
negative examples
candidate layers
candidate count
prompt set for validation
metric objective
max generation budget
```

and returns:

```text
candidate features
ranked interventions
sweep results
best candidate recipe
benchmark report
recipe card
```

## Autopilot workflow

Implement this sequence:

### Step 1: Input examples

The user provides positive and negative examples.

Example target:

```text
Target behavior:
  valid JSON tool routing

Positive examples:
  prompts/responses where output is strict valid JSON

Negative examples:
  prompts/responses where output has markdown, prose, missing fields, or invalid JSON
```

Do not require LLM-generated examples.

Optional: if a model API key exists and the user enables it, generate additional positive/negative examples. Save generated examples separately and mark them synthetic.

### Step 2: Contrastive feature search

For each candidate layer:

* Run activation extraction on positive examples.
* Run activation extraction on negative examples.
* Compute contrast score per feature.

Use simple initial scoring:

```text
positive_score = mean or max activation across positive examples
negative_score = mean or max activation across negative examples
contrast = positive_score - negative_score
ratio = positive_score / (epsilon + negative_score)
frequency_positive = fraction of positive examples where active
frequency_negative = fraction of negative examples where active
```

Rank by a combined score:

```text
combined = contrast * log(1 + frequency_positive / (epsilon + frequency_negative))
```

Document the exact formula in code and docs.

### Step 3: Candidate intervention construction

For the top N candidate features, create candidate interventions:

```text
layer
feature_id
initial strengths: [2, 4, 8]
injection_mode
```

### Step 4: Micro-benchmark candidates

Run each candidate against a small prompt set.

Compare at least:

```text
unsteered_baseline
prompt_only
steering_only
zero_strength_control
random_feature_control
```

For cost control, use:

```text
small prompt set
low max_new_tokens
temperature 0.0
one random control
```

### Step 5: Pick best candidate

Select best recipe according to configured objective.

Example objectives:

```text
maximize_json_validity
minimize_length_without_empty_output
maximize_required_terms_without_forbidden_terms
maximize_rule_score
maximize_model_judge_behavior_score optional
```

The selection logic must be explicit and deterministic.

### Step 6: Create recipe card

Export:

```text
recipe.json
recipe.md
benchmark_results.json
candidate_features.json
sweep_results.json
examples.jsonl
```

Mark status:

```text
candidate
```

Only mark `validated` if held-out validation and controls pass.

## Required Autopilot GUI tab

Add an `Autopilot` tab.

Inputs:

* Target behavior name.
* Target behavior description.
* Positive examples text area.
* Negative examples text area.
* Candidate layers.
* Candidate count.
* Prompt-only instruction.
* Validation prompt set.
* Objective metric.
* Max new tokens.
* Temperature.
* Run budget:

  * tiny
  * standard
  * 27B smoke
* Toggle optional model-generated examples.
* Toggle optional model-judge scoring.

Outputs:

* Candidate feature table.
* Candidate benchmark table.
* Best candidate recipe summary.
* Before/after examples.
* Control results.
* Export recipe button.
* Save to recipe library button.
* Warning if no candidate beats prompt-only.

## Required Autopilot CLI

Implement:

```bash
python scripts/autopilot_recipe.py \
  --config configs/qwen35_2b_dev_l0_100.yaml \
  --target-name json_validity \
  --target-description "Produce strict valid JSON without markdown or prose." \
  --positive-examples data/examples/json_positive.txt \
  --negative-examples data/examples/json_negative.txt \
  --validation-prompts data/prompt_sets/json_validity_dev.jsonl \
  --candidate-layers 8,12,16 \
  --candidate-count 10 \
  --objective json_validity \
  --output-dir recipes/json_validity_autopilot_v1
```

Also support a compact 27B Modal smoke command through `modal_app.py`.

---

# Part 6: Built-in prompt sets

Add small prompt/example sets for development and smoke testing.

Create:

```text
data/
  examples/
    concise_positive.txt
    concise_negative.txt
    json_positive.txt
    json_negative.txt
  prompt_sets/
    concise_dev.jsonl
    json_validity_dev.jsonl
    repetition_dev.jsonl
    qwen35_27b_smoke.jsonl
```

Keep them small.

Example JSONL format:

```json
{"id": "p001", "prompt": "Explain sparse autoencoders in one paragraph.", "metadata": {"task": "concise"}}
{"id": "p002", "prompt": "Return a JSON object with keys name and age for a fictional person.", "metadata": {"task": "json_validity"}}
```

Do not include copyrighted benchmark datasets unless the license permits it.

These are only smoke/dev sets, not scientific benchmarks.

---

# Part 7: Validation gates

Implement explicit validation logic.

A recipe can be `candidate` if:

```text
- feature search completed
- candidate intervention created
- at least one benchmark run completed
- benchmark artifacts saved
```

A recipe can be `benchmarked` if:

```text
- all required methods ran
- aggregate metrics exist
- controls exist
- examples are saved
```

A recipe can be `validated` only if:

```text
1. steering_only or prompt_plus_steering improves the target metric over unsteered_baseline.
2. steering_only or prompt_plus_steering is not worse than prompt_only by more than a configured tolerance, OR clearly beats prompt_only.
3. zero_strength_control is approximately equivalent to unsteered_baseline under deterministic settings.
4. random_feature_control does not reproduce the same improvement.
5. negative_strength_control is weaker, neutral, or directionally opposite.
6. hook_fired is true for all steered generations.
7. hidden_delta_norm > 0 for all steered generations.
8. coherence proxy does not fail catastrophically.
9. held-out validation prompts were used.
10. all artifacts are saved.
```

Default validation should be conservative.

If the result is mixed, mark:

```text
benchmarked
```

not `validated`.

Add a clear human-readable reason field:

```json
"validation_decision": {
  "status": "benchmarked",
  "reason": "Steering improved JSON validity over baseline but did not beat prompt-only and random-feature control was similar."
}
```

---

# Part 8: Documentation

Update `README.md`.

Add sections:

```text
Steering Bench
Recipe Cards
Autopilot
Validation status meanings
How to compare against prompt-only
How to interpret controls
Known failure modes
```

Update `RUNBOOK.md`.

Add commands:

```bash
pytest

python scripts/bench_recipe.py \
  --config configs/qwen35_2b_dev_l0_100.yaml \
  --target-behavior concise_answers \
  --layer 12 \
  --feature-id <FEATURE_ID> \
  --strength 5.0 \
  --prompt-set data/prompt_sets/concise_dev.jsonl

python scripts/autopilot_recipe.py \
  --config configs/qwen35_2b_dev_l0_100.yaml \
  --target-name json_validity \
  --target-description "Produce strict valid JSON without markdown or prose." \
  --positive-examples data/examples/json_positive.txt \
  --negative-examples data/examples/json_negative.txt \
  --validation-prompts data/prompt_sets/json_validity_dev.jsonl \
  --candidate-layers 8,12,16 \
  --candidate-count 10 \
  --objective json_validity \
  --output-dir recipes/json_validity_autopilot_v1

modal run modal_app.py::bench_smoke_2b
modal run modal_app.py::autopilot_smoke_2b
modal run modal_app.py::bench_smoke_27b
modal run modal_app.py::autopilot_smoke_27b
```

Update `RESULTS.md`.

Add:

```text
Phase 2: Steering Bench + Recipe Cards + Autopilot

- Commands run
- Tests passed
- Dev model bench result
- Dev model autopilot result
- 27B Modal bench smoke result
- 27B Modal autopilot smoke result
- Recipe card paths
- Whether any recipe beat prompt-only
- Whether any recipe was validated
- Known limitations
```

If a recipe does not beat prompt-only, say so clearly. That is an acceptable result.

---

# Part 9: Modal additions

Extend `modal_app.py`.

Add functions or equivalents:

```python
bench_smoke_2b()
autopilot_smoke_2b()
bench_smoke_27b()
autopilot_smoke_27b()
```

## 27B bench smoke

The 27B bench smoke must:

* load `Qwen/Qwen3.5-27B`
* load one layer of `Qwen/SAE-Res-Qwen3.5-27B-W80K-L0_100`
* load or create a tiny recipe using an automatically selected active feature
* run at least two prompts
* compare:

  * unsteered_baseline
  * steering_only
  * zero_strength_control
  * random_feature_control
* save compact benchmark JSON
* append summary to `RESULTS.md` or write a report file under `reports/`

## 27B autopilot smoke

The 27B autopilot smoke must:

* use a tiny positive/negative example set
* search only one or two candidate layers, e.g. layer 32 only
* select top 3 candidate features maximum
* run a tiny micro-benchmark
* export one candidate recipe card
* prove hook fired and hidden delta norm > 0
* not claim validation unless validation gates pass

This must be cost-controlled.

---

# Part 10: Tests

Add or update tests.

## Required unit tests

### `test_recipe_schema.py`

Verify:

* valid recipe parses
* invalid recipe fails
* validated status without evidence fails
* missing model id fails
* invalid feature id fails
* recipe ID is deterministic enough for stable tests

### `test_recipe_store.py`

Verify:

* save recipe
* load recipe
* list recipes
* search recipes
* export markdown
* no path traversal

### `test_benchmark_metrics.py`

Verify:

* JSON validity metric
* length metric
* repetition metric
* distinct n-gram metric
* required/forbidden term metric
* aggregate metric calculation

### `test_benchmark_controls.py`

Verify:

* zero-strength control is constructed
* random-feature control uses fixed seed
* negative-strength control flips sign
* method names are stable

### `test_sweep.py`

Verify:

* strength list generated correctly
* best setting selected deterministically
* strength 0 is included
* sweep artifacts serialize

### `test_autopilot_candidate_ranking.py`

Use fake activation tables.

Verify:

* positive-only features rank high
* negative-only features rank low for positive steering
* features active in both are penalized
* deterministic tie handling

### `test_recipe_export.py`

Verify:

* Markdown contains model id
* SAE id
* feature IDs
* strengths
* benchmark summary
* validation status
* limitations

### `test_bench_no_secret_logging.py`

Set fake secrets and capture logs.

Verify secrets do not appear.

## Integration tests

Add lightweight integration tests that can run with fake model/fake SAE.

They must verify:

* bench calls all required methods
* autopilot produces a candidate recipe
* recipe card saved to disk
* validation gate refuses to mark weak/missing evidence as validated

## Smoke tests

The following smoke commands must be documented and run if feasible:

```bash
pytest

python scripts/autopilot_recipe.py \
  --config configs/qwen35_2b_dev_l0_100.yaml \
  --target-name json_validity \
  --target-description "Produce strict valid JSON without markdown or prose." \
  --positive-examples data/examples/json_positive.txt \
  --negative-examples data/examples/json_negative.txt \
  --validation-prompts data/prompt_sets/json_validity_dev.jsonl \
  --candidate-layers 8,12,16 \
  --candidate-count 5 \
  --objective json_validity \
  --output-dir recipes/json_validity_dev_autopilot_smoke

python scripts/bench_recipe.py \
  --recipe recipes/json_validity_dev_autopilot_smoke/recipe.json \
  --prompt-set data/prompt_sets/json_validity_dev.jsonl \
  --output reports/json_validity_dev_bench_smoke.json

modal run modal_app.py::bench_smoke_2b
modal run modal_app.py::autopilot_smoke_2b
modal run modal_app.py::bench_smoke_27b
modal run modal_app.py::autopilot_smoke_27b
```

Adjust names if implementation differs, but document exact working commands.

---

# Part 11: GUI acceptance criteria

The Gradio app must have these new or updated tabs.

## Bench tab

Must show:

* recipe selector
* manual feature/layer/strength fields
* prompt set input/upload
* prompt-only instruction
* method toggles or fixed method list
* metric list
* run button
* aggregate results table
* per-prompt examples
* controls section
* validation decision
* save results button
* export recipe card button

## Autopilot tab

Must show:

* target behavior fields
* positive examples
* negative examples
* candidate layer selector
* candidate count
* objective selector
* run budget selector
* run button
* candidate feature table
* micro-benchmark table
* best candidate summary
* save recipe button
* clear warning if prompt-only wins

## Recipe Library tab

Must show:

* recipe list
* status filter
* recipe detail view
* benchmark summary
* examples
* load into Steer
* load into Bench
* export Markdown

---

# Part 12: Data and artifact hygiene

Do not commit large generated outputs by default.

It is okay to commit:

```text
small dev prompt sets
small example files
schema examples
README/RUNBOOK/RESULTS
```

Do not commit:

```text
large model files
SAE checkpoints
large benchmark outputs
.env
cache directories
Modal volume contents
```

Add or update `.gitignore` as needed.

Recipe cards may be committed if small and useful.

Generated outputs should include timestamps and config metadata.

---

# Part 13: Strict done criteria

You may mark this follow-up goal complete only when all of the following are true:

1. All previous initial-goal tests still pass.
2. New `pytest` tests pass.
3. Existing Inspect/Compare/Steer GUI flows still work.
4. New `FeatureRecipe` schema exists and validates recipes.
5. Recipe JSON export works.
6. Recipe Markdown export works.
7. Recipe Library tab exists and can load/list/export recipes.
8. Bench tab exists and can run a benchmark from either:

   * a saved recipe, or
   * manual layer/feature/strength inputs.
9. Bench compares at least:

   * unsteered baseline
   * prompt-only
   * steering-only
   * prompt + steering
   * zero-strength control
   * random-feature control
   * negative-strength control
10. Bench computes and saves aggregate metrics.
11. Bench saves per-prompt outputs.
12. Bench produces a validation decision with reasons.
13. Strength sweep exists and includes strength `0.0`.
14. Autopilot tab exists.
15. Autopilot CLI exists.
16. Autopilot can:

* accept positive examples
* accept negative examples
* search candidate features
* rank candidates
* run micro-benchmarks
* select a best candidate
* export a candidate recipe card

17. Autopilot does not require a hosted model API.
18. Hosted model-judge or example-generation features, if implemented, are optional and disabled by default.
19. Dev-model autopilot smoke succeeds and writes a recipe card.
20. Dev-model bench smoke succeeds and writes benchmark results.
21. Modal `bench_smoke_2b` or equivalent succeeds.
22. Modal `autopilot_smoke_2b` or equivalent succeeds.
23. Modal `bench_smoke_27b` or equivalent succeeds, unless explicitly blocked and documented.
24. Modal `autopilot_smoke_27b` or equivalent succeeds, unless explicitly blocked and documented.
25. At least one compact 27B recipe card is produced from a real Modal run, unless blocked.
26. 27B smoke artifacts prove:

    * model loaded
    * SAE layer loaded
    * candidate feature selected
    * hook fired
    * hidden delta norm > 0
    * benchmark outputs saved
27. `README.md` is updated.
28. `RUNBOOK.md` is updated.
29. `RESULTS.md` is updated with exact commands run.
30. No real secrets are committed or logged.
31. `.gitignore` excludes caches, `.env`, model files, and large outputs.

---

# Part 14: Blocked criteria

If any required 27B Modal step fails because of quota, GPU availability, Hugging Face access, OOM, dependency conflicts, or architecture incompatibility, create or update `BLOCKED.md`.

`BLOCKED.md` must include:

```text
- blocked step
- exact command
- traceback excerpt
- Modal GPU requested
- actual GPU if allocated
- memory summary if available
- what completed successfully
- whether dev-model bench works
- whether dev-model autopilot works
- smallest next action to unblock
```

Blocked 27B smoke means the follow-up goal is not fully complete, but the dev-model implementation can still be reported as partially complete.

Do not mark the goal complete if `BLOCKED.md` is required for any strict done item.

---

# Part 15: Recommended implementation sequence

Proceed in this order.

## Phase 1: Preserve and inspect

* Run existing tests.
* Run existing dev smoke if cheap.
* Inspect current module layout.
* Identify current steering result schema.
* Identify existing Modal conventions.
* Do not start new work until baseline status is recorded.

## Phase 2: Recipe schema and store

* Implement `FeatureRecipe`.
* Implement `RecipeStore`.
* Implement JSON export/import.
* Implement Markdown export.
* Add unit tests.
* Add minimal Recipe Library tab.

Stop and run tests.

## Phase 3: Benchmark metrics and controls

* Implement rule-based metrics.
* Implement method construction.
* Implement zero-strength control.
* Implement random-feature control.
* Implement negative-strength control.
* Implement benchmark result schema.
* Add unit tests.

Stop and run tests.

## Phase 4: Bench runner

* Implement `bench_recipe.py`.
* Wire bench to existing generation/steering backend.
* Save aggregate and per-prompt results.
* Add validation decision logic.
* Add Bench tab.
* Run dev-model bench smoke.

Stop and record results.

## Phase 5: Strength sweep

* Implement strength sweep.
* Save sweep results.
* Attach sweep to recipe card.
* Add tests.
* Add GUI entrypoint.

Stop and run tests.

## Phase 6: Autopilot candidate search

* Implement contrastive feature ranking from positive/negative examples.
* Implement candidate creation.
* Implement candidate micro-benchmark.
* Implement deterministic best-candidate selection.
* Export recipe card.
* Add Autopilot tab.
* Add `autopilot_recipe.py`.
* Run dev-model autopilot smoke.

Stop and record results.

## Phase 7: Modal smoke extensions

* Add `bench_smoke_2b`.
* Add `autopilot_smoke_2b`.
* Add `bench_smoke_27b`.
* Add `autopilot_smoke_27b`.
* Keep 27B smoke tiny.
* Save compact artifacts.

Stop and record results.

## Phase 8: Documentation and final verification

* Update README.
* Update RUNBOOK.
* Update RESULTS.
* Run all tests.
* Run final smoke commands feasible in the environment.
* Check for secrets.
* Check git diff.
* Summarize completion honestly.

---

# Part 16: Final response format to user

When finished, report:

1. What was added.
2. Which files changed.
3. Exact test commands run.
4. Exact smoke commands run.
5. Dev-model recipe card path.
6. Dev-model benchmark result path.
7. 27B Modal recipe card path, if completed.
8. 27B Modal benchmark path, if completed.
9. Whether any recipe beat prompt-only.
10. Whether any recipe was marked `validated`.
11. Any blocked items.
12. Known limitations.

Do not say “validated” unless the validation gates passed.

Do not say “27B complete” unless the real Modal 27B smoke ran successfully.

[1]: https://arxiv.org/abs/2605.11887?utm_source=chatgpt.com "Qwen-Scope: Turning Sparse Features into Development Tools for Large Language Models"
