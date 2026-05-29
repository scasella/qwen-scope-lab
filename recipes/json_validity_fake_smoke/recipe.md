# Recipe: json_validity_fake_smoke

Status: `candidate`

## Target Behavior

**json_validity**: Produce strict valid JSON without markdown or prose.

Positive: {"name":"Ada","age":31}; {"route":"search","arguments":{"query":"sparse autoencoder"}}

Negative: Here is the JSON you asked for:
```json
{"name":"Ada","age":31}
```; The answer is name Ada and age thirty one.

## Model and SAE

- Model: `fake/qwen-scope-test`
- SAE: `fake/sae-test`
- Dtype: `float32`
- Config: `configs/fake_test.yaml`

## Interventions

- Layer 0, feature 61, strength 4.0, mode `all_positions`

## Benchmark Summary

- Prompt set: `autopilot_validation`
- Methods: unsteered_baseline, prompt_only, steering_only, prompt_plus_steering, zero_strength_control, random_feature_control, negative_strength_control
- Summary: steering did not improve over unsteered baseline; random-feature control was similar to steering
- Validation status: `benchmarked`
- Reason: steering did not improve over unsteered baseline; random-feature control was similar to steering

## Examples

### Example 1

Prompt: `Return a JSON object with keys name and age for a fictional person.`

Unsteered:

```text
{"name":"Ada","age":31}
```

Steered:

```text
{"name":"Ada","age":31}
```
### Example 2

Prompt: `Return a JSON object with keys route and arguments for a search action.`

Unsteered:

```text
{"name":"Ada","age":31}
```

Steered:

```text
{"name":"Ada","age":31}
```

## Limitations

- None recorded.

## Side Effects

- None recorded.

## Provenance

```json
{
  "command": "",
  "git_commit": "",
  "modal_gpu": "",
  "python_version": "",
  "seed": 0,
  "torch_version": "",
  "transformers_version": ""
}
```
