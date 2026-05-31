from __future__ import annotations

import argparse
from pathlib import Path
from functools import lru_cache
from typing import Any

import gradio as gr
import pandas as pd

from qwen_scope_lab_bench.autopilot import run_autopilot
from qwen_scope_lab_bench.benchmark import ServiceGenerationBackend, attach_benchmark_to_recipe, recipe_from_manual, run_benchmark, save_benchmark_result
from qwen_scope_lab_bench.prompt_sets import parse_prompt_text
from qwen_scope_lab_bench.recipe_schema import FeatureRecipe
from qwen_scope_lab_bench.recipe_store import RecipeStore
from qwen_scope_lab_bench.service import SteeringService


def _rows_to_frame(rows: list[dict[str, Any]], columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=columns)


def _flatten_inspection(result: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    token_rows = []
    feature_rows = []
    for row in result["top_features_by_token"]:
        features = row["features"]
        token_rows.append(
            {
                "token_index": row["token_index"],
                "token_text": row["token_text"],
                "top_features": ", ".join(f"{f['feature_id']}:{f['activation']:.3f}" for f in features[:8]),
            }
        )
        for feature in features:
            feature_rows.append(
                {
                    "token_index": row["token_index"],
                    "token_text": row["token_text"],
                    "feature_id": feature["feature_id"],
                    "activation": feature["activation"],
                }
            )
    return (
        _rows_to_frame(token_rows, ["token_index", "token_text", "top_features"]),
        _rows_to_frame(feature_rows, ["token_index", "token_text", "feature_id", "activation"]),
        result,
    )


def _aggregate_frame(result: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for method, metrics in result.get("aggregate_metrics", {}).items():
        row = {"method": method}
        row.update(metrics)
        rows.append(row)
    return pd.DataFrame(rows)


def _examples_frame(result: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for item in result.get("per_prompt_results", []):
        row = {"prompt_id": item.get("prompt_id"), "prompt": item.get("prompt")}
        row.update({f"{method}_output": output for method, output in item.get("outputs", {}).items()})
        rows.append(row)
    return pd.DataFrame(rows)


def _parse_layers(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


@lru_cache(maxsize=4)
def get_service(config_path: str) -> SteeringService:
    return SteeringService.from_config_path(config_path)


def build_demo(config_path: str) -> gr.Blocks:
    service = get_service(config_path)
    store = RecipeStore()

    def _recipe_choices() -> list[str]:
        return [row["recipe_id"] for row in store.list()]

    def inspect_cb(prompt: str, layer: int, top_k: int, max_seq_len: int):
        result = service.inspect_prompt(prompt, layer=layer, top_k=top_k, max_seq_len=max_seq_len or None)
        return _flatten_inspection(result)

    def compare_cb(positive_prompt: str, negative_prompt: str, layer: int, limit: int):
        result = service.compare_prompts(positive_prompt, negative_prompt, layer=layer, limit=limit)
        rows = result["positive_stronger"] + result["negative_stronger"]
        compare_columns = [
            "feature_id",
            "positive_max",
            "negative_max",
            "difference",
            "ratio",
            "positive_tokens",
            "negative_tokens",
        ]
        token_columns = ["token_index", "token_text", "max_activation", "mean_top_activation", "top_feature_ids"]
        return (
            _rows_to_frame(rows, compare_columns),
            _rows_to_frame(result["positive_token_summary"], token_columns),
            _rows_to_frame(result["negative_token_summary"], token_columns),
            result["method"],
            result,
        )

    def steer_cb(prompt: str, layer: int, feature_id: int, strength: float, max_new_tokens: int, temperature: float, mode: str):
        result = service.steer(prompt, layer, int(feature_id), strength, max_new_tokens, temperature, mode)
        return (
            result["unsteered_text"],
            result["steered_text"],
            result["hook_fired"],
            result["hidden_delta_norm"],
            result["logits_delta_norm"],
            result,
        )

    def save_note_cb(feature_id: int, layer: int, label: str, notes: str, example_prompts: str, observed_effects: str, failure_notes: str):
        entry = {
            "feature_id": int(feature_id),
            "layer": int(layer),
            "human_label": label,
            "notes": notes,
            "example_prompts": [line for line in example_prompts.splitlines() if line.strip()],
            "observed_effects": observed_effects,
            "failure_notes": failure_notes,
        }
        return service.save_notebook_entry(entry)

    def label_cb(layer: int, feature_id: int, top_tokens: str, positive_examples: str, negative_examples: str):
        payload = {
            "layer": int(layer),
            "feature_id": int(feature_id),
            "top_activating_tokens": [line for line in top_tokens.splitlines() if line.strip()],
            "positive_examples": [line for line in positive_examples.splitlines() if line.strip()],
            "negative_examples": [line for line in negative_examples.splitlines() if line.strip()],
        }
        return service.label_feature(payload)

    def bench_cb(
        recipe_id: str,
        target_behavior: str,
        target_description: str,
        manual_layer: int,
        manual_feature_id: int,
        manual_strength: float,
        prompt_set_text: str,
        prompt_only_instruction: str,
        bench_max_new_tokens: int,
        bench_temperature: float,
        bench_seed: int,
        objective: str,
    ):
        if recipe_id:
            recipe = store.load(recipe_id)
        else:
            recipe = recipe_from_manual(
                config=service.config,
                config_path=config_path,
                target_behavior=target_behavior or "manual_recipe",
                target_description=target_description,
                layer=int(manual_layer),
                feature_id=int(manual_feature_id),
                strength=float(manual_strength),
            )
        prompts = parse_prompt_text(prompt_set_text)
        result = run_benchmark(
            recipe,
            prompts,
            ServiceGenerationBackend(service),
            prompt_only_instruction=prompt_only_instruction,
            max_new_tokens=int(bench_max_new_tokens),
            temperature=float(bench_temperature),
            seed=int(bench_seed),
            objective=objective,
            prompt_set_id="gui_inline",
        )
        output_path = f"reports/{recipe.recipe_id}_{result['benchmark_id']}.json"
        save_benchmark_result(result, output_path)
        recipe = attach_benchmark_to_recipe(recipe, result)
        recipe.artifacts["results_path"] = output_path
        store.save(recipe, benchmark_results=result, examples=recipe.examples)
        can_validate = bool(result["validation_decision"].get("passed"))
        return (
            _aggregate_frame(result),
            _examples_frame(result),
            result["aggregate_metrics"].get("zero_strength_control", {}),
            result["validation_decision"],
            output_path,
            recipe.recipe_id,
            gr.update(choices=_recipe_choices(), value=recipe.recipe_id),
            gr.update(visible=can_validate),
        )

    def mark_candidate_cb(recipe_id: str):
        recipe = store.load(recipe_id)
        recipe.mark_candidate()
        store.save(recipe)
        return recipe.to_dict()

    def mark_validated_cb(recipe_id: str):
        recipe = store.load(recipe_id)
        recipe.mark_validated()
        store.save(recipe)
        return recipe.to_dict()

    def autopilot_cb(
        target_name: str,
        target_description: str,
        positive_examples: str,
        negative_examples: str,
        candidate_layers: str,
        candidate_count: int,
        validation_prompt_text: str,
        prompt_only_instruction: str,
        objective: str,
        auto_max_new_tokens: int,
        auto_temperature: float,
        run_budget: str,
        generate_examples: bool,
        model_judge: bool,
    ):
        layer_values = _parse_layers(candidate_layers)
        prompt_rows = parse_prompt_text(validation_prompt_text)
        output_dir = Path("recipes") / f"{target_name or 'autopilot'}_gui_candidate"
        result = run_autopilot(
            config=service.config,
            config_path=config_path,
            target_name=target_name or "autopilot",
            target_description=target_description,
            positive_examples=[line for line in positive_examples.splitlines() if line.strip()],
            negative_examples=[line for line in negative_examples.splitlines() if line.strip()],
            validation_prompts=prompt_rows,
            candidate_layers=layer_values,
            candidate_count=int(candidate_count),
            objective=objective,
            backend=ServiceGenerationBackend(service),
            service=service,
            output_dir=output_dir,
            prompt_only_instruction=prompt_only_instruction,
            max_new_tokens=min(int(auto_max_new_tokens), 8) if run_budget == "tiny" else int(auto_max_new_tokens),
            temperature=float(auto_temperature),
            seed=0,
        )
        benchmark = result["benchmark"]
        return (
            pd.DataFrame(result["candidate_features"]),
            pd.DataFrame(result["candidate_benchmarks"]),
            result["best_recipe"],
            _examples_frame(benchmark),
            {
                "controls": {
                    key: benchmark["aggregate_metrics"].get(key, {})
                    for key in ("zero_strength_control", "random_feature_control", "negative_strength_control")
                },
                "warning": result["warning"],
                "optional_model_generated_examples_enabled": bool(generate_examples),
                "optional_model_judge_enabled": bool(model_judge),
            },
            result["output_paths"],
            gr.update(choices=_recipe_choices(), value=Path(output_dir).name),
        )

    def refresh_library_cb(query: str, status: str):
        return pd.DataFrame(store.search(query, status=status))

    def recipe_detail_cb(recipe_id: str):
        recipe = store.load(recipe_id)
        return (
            recipe.to_dict(),
            recipe.to_markdown(),
            pd.DataFrame([item.__dict__ for item in recipe.interventions]),
            _aggregate_frame({"aggregate_metrics": recipe.benchmark.get("metrics", {})}),
            pd.DataFrame(recipe.examples),
        )

    def export_recipe_cb(recipe_id: str):
        return store.export_markdown(recipe_id)

    def load_recipe_values_cb(recipe_id: str):
        recipe = store.load(recipe_id)
        intervention = recipe.interventions[0]
        return intervention.layer, intervention.feature_id, intervention.strength, recipe.recipe_id

    with gr.Blocks(title="Qwen Scope Steering") as demo:
        gr.Markdown("# Qwen Scope Steering")
        with gr.Tabs():
            with gr.Tab("Inspect prompt features"):
                prompt = gr.Textbox(label="Prompt", lines=4, value="The capital of France is")
                with gr.Row():
                    layer = gr.Number(label="Layer", value=service.config.default_layer, precision=0)
                    top_k = gr.Number(label="Top features per token", value=min(10, service.config.top_k), precision=0)
                    max_seq_len = gr.Number(label="Max sequence length", value=128, precision=0)
                inspect_btn = gr.Button("Inspect", variant="primary")
                token_table = gr.Dataframe(label="Tokens", interactive=False)
                feature_table = gr.Dataframe(label="Features", interactive=False)
                inspect_json = gr.JSON(label="Inspection JSON")
                inspect_btn.click(inspect_cb, [prompt, layer, top_k, max_seq_len], [token_table, feature_table, inspect_json])

            with gr.Tab("Compare prompts"):
                pos_prompt = gr.Textbox(label="Positive prompt", lines=4, value="Write a concise factual answer.")
                neg_prompt = gr.Textbox(label="Negative prompt", lines=4, value="Write a long rambling story.")
                with gr.Row():
                    compare_layer = gr.Number(label="Layer", value=service.config.default_layer, precision=0)
                    compare_limit = gr.Number(label="Features", value=20, precision=0)
                compare_btn = gr.Button("Compare", variant="primary")
                compare_table = gr.Dataframe(label="Contrastive features", interactive=False)
                pos_token_table = gr.Dataframe(label="Positive token summary", interactive=False)
                neg_token_table = gr.Dataframe(label="Negative token summary", interactive=False)
                compare_method = gr.Textbox(label="Scoring method", interactive=False)
                compare_json = gr.JSON(label="Comparison JSON")
                compare_btn.click(
                    compare_cb,
                    [pos_prompt, neg_prompt, compare_layer, compare_limit],
                    [compare_table, pos_token_table, neg_token_table, compare_method, compare_json],
                )

            with gr.Tab("Steer generation"):
                steer_prompt = gr.Textbox(label="Prompt", lines=4, value="Write one sentence about Paris.")
                with gr.Row():
                    steer_layer = gr.Number(label="Layer", value=service.config.default_layer, precision=0)
                    feature_id = gr.Number(label="Feature id", value=0, precision=0)
                    strength = gr.Slider(label="Strength", minimum=-20.0, maximum=20.0, step=0.5, value=5.0)
                with gr.Row():
                    max_new_tokens = gr.Number(label="Max new tokens", value=service.config.default_max_new_tokens, precision=0)
                    temperature = gr.Slider(label="Temperature", minimum=0.0, maximum=2.0, step=0.05, value=0.7)
                    mode = gr.Dropdown(label="Mode", choices=["all_positions"], value="all_positions")
                steer_btn = gr.Button("Generate", variant="primary")
                unsteered = gr.Textbox(label="Original generation", lines=6)
                steered = gr.Textbox(label="Steered generation", lines=6)
                with gr.Row():
                    hook_fired = gr.Checkbox(label="Hook fired", interactive=False)
                    hidden_delta = gr.Number(label="Hidden delta norm", interactive=False)
                    logits_delta = gr.Number(label="Logits delta norm", interactive=False)
                steer_json = gr.JSON(label="Steering JSON")
                steer_btn.click(
                    steer_cb,
                    [steer_prompt, steer_layer, feature_id, strength, max_new_tokens, temperature, mode],
                    [unsteered, steered, hook_fired, hidden_delta, logits_delta, steer_json],
                )

            with gr.Tab("Bench"):
                bench_recipe_id = gr.Dropdown(label="Recipe", choices=_recipe_choices(), allow_custom_value=True)
                with gr.Row():
                    bench_target = gr.Textbox(label="Target behavior", value="concise_answers")
                    bench_layer = gr.Number(label="Layer", value=service.config.default_layer, precision=0)
                    bench_feature = gr.Number(label="Feature id", value=0, precision=0)
                    bench_strength = gr.Number(label="Strength", value=5.0)
                bench_target_description = gr.Textbox(label="Target description", lines=2, value="Make answers shorter and more direct while preserving correctness.")
                bench_prompt_set = gr.Textbox(
                    label="Prompt set",
                    lines=5,
                    value='{"id":"p001","prompt":"Explain sparse autoencoders in one paragraph."}\n{"id":"p002","prompt":"Describe why feature steering needs controls."}',
                )
                bench_prompt_instruction = gr.Textbox(label="Prompt-only instruction", lines=2, value="Answer concisely in no more than two sentences. Prompt: {prompt}")
                with gr.Row():
                    bench_tokens = gr.Number(label="Max new tokens", value=16, precision=0)
                    bench_temp = gr.Slider(label="Temperature", minimum=0.0, maximum=2.0, step=0.05, value=0.0)
                    bench_seed = gr.Number(label="Seed", value=0, precision=0)
                    bench_objective = gr.Dropdown(label="Objective", choices=["maximize_rule_score", "maximize_json_validity", "minimize_length_without_empty_output"], value="maximize_rule_score")
                bench_run = gr.Button("Run small benchmark", variant="primary")
                bench_aggregate = gr.Dataframe(label="Aggregate metrics", interactive=False)
                bench_examples = gr.Dataframe(label="Before/after examples", interactive=False)
                bench_controls = gr.JSON(label="Control results")
                bench_decision = gr.JSON(label="Validation decision")
                bench_path = gr.Textbox(label="Benchmark JSON path", interactive=False)
                bench_saved_recipe_id = gr.Textbox(label="Saved recipe id", interactive=False)
                with gr.Row():
                    mark_candidate = gr.Button("Mark Recipe Candidate")
                    mark_validated = gr.Button("Mark Recipe Validated", visible=False)
                bench_recipe_json = gr.JSON(label="Recipe status JSON")
                bench_run.click(
                    bench_cb,
                    [
                        bench_recipe_id,
                        bench_target,
                        bench_target_description,
                        bench_layer,
                        bench_feature,
                        bench_strength,
                        bench_prompt_set,
                        bench_prompt_instruction,
                        bench_tokens,
                        bench_temp,
                        bench_seed,
                        bench_objective,
                    ],
                    [bench_aggregate, bench_examples, bench_controls, bench_decision, bench_path, bench_saved_recipe_id, bench_recipe_id, mark_validated],
                )
                mark_candidate.click(mark_candidate_cb, bench_saved_recipe_id, bench_recipe_json)
                mark_validated.click(mark_validated_cb, bench_saved_recipe_id, bench_recipe_json)

            with gr.Tab("Autopilot"):
                auto_target_name = gr.Textbox(label="Target behavior name", value="json_validity")
                auto_target_description = gr.Textbox(label="Target behavior description", lines=2, value="Produce strict valid JSON without markdown or prose.")
                auto_positive = gr.Textbox(label="Positive examples", lines=4, value='{"name":"Ada","age":31}\n{"route":"search","arguments":{"query":"sparse autoencoder"}}')
                auto_negative = gr.Textbox(label="Negative examples", lines=4, value="Here is the JSON you asked for:\n```json\n{\"name\":\"Ada\",\"age\":31}\n```")
                with gr.Row():
                    auto_layers = gr.Textbox(label="Candidate layers", value=str(service.config.default_layer))
                    auto_count = gr.Number(label="Candidate count", value=3, precision=0)
                    auto_objective = gr.Dropdown(label="Objective metric", choices=["maximize_rule_score", "maximize_json_validity", "minimize_length_without_empty_output"], value="maximize_json_validity")
                    auto_budget = gr.Dropdown(label="Run budget", choices=["tiny", "standard", "27B smoke"], value="tiny")
                auto_validation_prompts = gr.Textbox(
                    label="Validation prompt set",
                    lines=4,
                    value='{"id":"json_001","prompt":"Return a JSON object with keys name and age for a fictional person."}\n{"id":"json_002","prompt":"Return a JSON object with keys route and arguments for a search action."}',
                )
                auto_prompt_instruction = gr.Textbox(label="Prompt-only instruction", lines=2, value="Return only strict valid JSON. Do not include markdown or prose. Prompt: {prompt}")
                with gr.Row():
                    auto_tokens = gr.Number(label="Max new tokens", value=8, precision=0)
                    auto_temp = gr.Slider(label="Temperature", minimum=0.0, maximum=2.0, step=0.05, value=0.0)
                    auto_generate_examples = gr.Checkbox(label="Optional model-generated examples", value=False)
                    auto_model_judge = gr.Checkbox(label="Optional model-judge scoring", value=False)
                auto_run = gr.Button("Run autopilot", variant="primary")
                auto_candidates = gr.Dataframe(label="Candidate features", interactive=False)
                auto_benchmarks = gr.Dataframe(label="Candidate benchmark table", interactive=False)
                auto_best = gr.JSON(label="Best candidate recipe summary")
                auto_examples = gr.Dataframe(label="Before/after examples", interactive=False)
                auto_controls = gr.JSON(label="Control results and warning")
                auto_paths = gr.JSON(label="Exported recipe artifacts")
                auto_run.click(
                    autopilot_cb,
                    [
                        auto_target_name,
                        auto_target_description,
                        auto_positive,
                        auto_negative,
                        auto_layers,
                        auto_count,
                        auto_validation_prompts,
                        auto_prompt_instruction,
                        auto_objective,
                        auto_tokens,
                        auto_temp,
                        auto_budget,
                        auto_generate_examples,
                        auto_model_judge,
                    ],
                    [auto_candidates, auto_benchmarks, auto_best, auto_examples, auto_controls, auto_paths, bench_recipe_id],
                )

            with gr.Tab("Recipe Library"):
                with gr.Row():
                    library_query = gr.Textbox(label="Search")
                    library_status = gr.Dropdown(label="Status", choices=["all", "draft", "candidate", "benchmarked", "validated", "failed", "blocked"], value="all")
                refresh_library = gr.Button("Refresh recipes", variant="primary")
                library_table = gr.Dataframe(label="Recipes", value=pd.DataFrame(store.list()), interactive=False)
                library_recipe_id = gr.Textbox(label="Recipe id")
                open_recipe = gr.Button("Open recipe")
                recipe_detail = gr.JSON(label="Recipe details")
                recipe_markdown = gr.Textbox(label="Recipe Markdown", lines=12)
                recipe_interventions = gr.Dataframe(label="Interventions", interactive=False)
                recipe_benchmark = gr.Dataframe(label="Benchmark summary", interactive=False)
                recipe_examples = gr.Dataframe(label="Examples", interactive=False)
                with gr.Row():
                    export_recipe = gr.Button("Export Markdown")
                    load_steer = gr.Button("Load into Steer")
                    load_bench = gr.Button("Load into Bench")
                exported_path = gr.Textbox(label="Markdown path", interactive=False)
                refresh_library.click(refresh_library_cb, [library_query, library_status], library_table)
                open_recipe.click(recipe_detail_cb, library_recipe_id, [recipe_detail, recipe_markdown, recipe_interventions, recipe_benchmark, recipe_examples])
                export_recipe.click(export_recipe_cb, library_recipe_id, exported_path)
                load_steer.click(load_recipe_values_cb, library_recipe_id, [steer_layer, feature_id, strength, bench_recipe_id])
                load_bench.click(load_recipe_values_cb, library_recipe_id, [bench_layer, bench_feature, bench_strength, bench_recipe_id])

            with gr.Tab("Feature notebook"):
                note_feature = gr.Number(label="Feature id", value=0, precision=0)
                note_layer = gr.Number(label="Layer", value=service.config.default_layer, precision=0)
                note_label = gr.Textbox(label="Human label")
                note_notes = gr.Textbox(label="Notes", lines=3)
                note_examples = gr.Textbox(label="Example prompts", lines=4)
                note_effects = gr.Textbox(label="Observed effects", lines=3)
                note_failures = gr.Textbox(label="Failure notes", lines=3)
                save_note = gr.Button("Save", variant="primary")
                notebook_json = gr.JSON(label="Notebook JSON", value=service.notebook())
                save_note.click(
                    save_note_cb,
                    [note_feature, note_layer, note_label, note_notes, note_examples, note_effects, note_failures],
                    notebook_json,
                )
                gr.Markdown("Speculative labels are optional and disabled unless a model API key is configured.")
                label_tokens = gr.Textbox(label="Top activating tokens", lines=3)
                label_positive = gr.Textbox(label="Positive examples", lines=3)
                label_negative = gr.Textbox(label="Negative examples", lines=3)
                label_btn = gr.Button("Suggest speculative label")
                label_json = gr.JSON(label="Speculative label JSON")
                label_btn.click(label_cb, [note_layer, note_feature, label_tokens, label_positive, label_negative], label_json)

            with gr.Tab("System/status"):
                status_btn = gr.Button("Refresh", variant="primary")
                status_json = gr.JSON(label="Status", value=service.status())
                status_btn.click(lambda: service.status(), outputs=status_json)

    return demo


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/qwen35_2b_dev_l0_100.yaml")
    parser.add_argument("--server-name", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()
    demo = build_demo(args.config)
    demo.queue().launch(server_name=args.server_name, server_port=args.server_port, share=args.share)


if __name__ == "__main__":
    main()
