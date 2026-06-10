# Recipe: concise_autopilot

Status: `candidate`

## Target Behavior

**concise**: Steer toward concise, brief answers.

Positive: A sparse autoencoder learns a compact set of features that reconstruct model activations.; It maps activations into sparse feature activations and decodes them back into the residual stream.

Negative: Sparse autoencoders are a broad family of machine learning systems, and before explaining them it can be helpful to discuss representation learning, dictionary learning, neural networks, interpretability history, and many caveats.; A complete answer might include long background, multiple analogies, several exceptions, and repeated qualifiers before reaching the main point.

## Model and SAE

- Model: `mlx-community/Qwen3.5-2B-bf16`
- SAE: `Qwen/SAE-Res-Qwen3.5-2B-Base-W32K-L0_100`
- Dtype: `float16`
- Config: `mlx://mlx-community/Qwen3.5-2B-bf16`

## Interventions

- Layer 12, feature 29073, strength 4.0, mode `all_positions`

## Benchmark Summary

- Prompt set: `autopilot_validation`
- Methods: unsteered_baseline, prompt_only, steering_only, prompt_plus_steering, zero_strength_control, random_feature_control, negative_strength_control
- Summary: steering was worse than prompt-only beyond tolerance; coherence proxy failed for one or more steered generations
- Validation status: `benchmarked`
- Reason: steering was worse than prompt-only beyond tolerance; coherence proxy failed for one or more steered generations

## Examples

### Example 1

Prompt: `Explain what a sparse autoencoder does.`

Unsteered:

```text


<think>

</think>

A **Sparse Autoencoder** is a type of neural network designed to learn a compressed representation of data by forcing the network to be **
```

Steered:

```text


A **model** is a system that predicts the output of a system. In the context of a ****, the model is a **** that predicts
```
### Example 2

Prompt: `Describe how the water cycle works.`

Unsteered:

```text


<think>

</think>

The **water cycle** (also known as the hydrologic cycle) is the continuous process by which water circulates through the Earth's
```

Steered:

```text


The water cycle is a continuous cycle of water movement that describes how water moves through the Earth. Water is constantly moving through the Earth, and it is constantly
```
### Example 3

Prompt: `What is photosynthesis?`

Unsteered:

```text


Photosynthesis is the process by which plants, algae, and some bacteria convert light energy into chemical energy, producing glucose and oxygen from carbon dioxide and water.
```

Steered:

```text


Model of photos

The model of photos is a system of equations that describes the relationship between the number of plants and the number of plants.

The model
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
