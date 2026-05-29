from __future__ import annotations

import random
from dataclasses import dataclass

from .recipe_schema import Intervention


REQUIRED_METHODS = [
    "unsteered_baseline",
    "prompt_only",
    "steering_only",
    "prompt_plus_steering",
    "zero_strength_control",
    "random_feature_control",
    "negative_strength_control",
]


@dataclass(frozen=True)
class MethodSpec:
    name: str
    prompt_only: bool
    intervention: Intervention | None


def random_feature_id(d_sae: int, excluded_feature_id: int, seed: int) -> int:
    if d_sae <= 1:
        return 0
    rng = random.Random(seed)
    candidate = rng.randrange(d_sae)
    if candidate == excluded_feature_id:
        candidate = (candidate + 1) % d_sae
    return candidate


def build_method_specs(
    intervention: Intervention,
    *,
    d_sae: int,
    seed: int = 0,
    include_prompt_plus: bool = True,
) -> list[MethodSpec]:
    random_id = random_feature_id(d_sae, intervention.feature_id, seed)
    methods = [
        MethodSpec("unsteered_baseline", False, None),
        MethodSpec("prompt_only", True, None),
        MethodSpec("steering_only", False, intervention),
    ]
    if include_prompt_plus:
        methods.append(MethodSpec("prompt_plus_steering", True, intervention))
    methods.extend(
        [
            MethodSpec(
                "zero_strength_control",
                False,
                Intervention(
                    layer=intervention.layer,
                    feature_id=intervention.feature_id,
                    strength=0.0,
                    sign="zero",
                    injection_mode=intervention.injection_mode,
                    position_policy=intervention.position_policy,
                ),
            ),
            MethodSpec(
                "random_feature_control",
                False,
                Intervention(
                    layer=intervention.layer,
                    feature_id=random_id,
                    strength=intervention.strength,
                    sign="control",
                    injection_mode=intervention.injection_mode,
                    position_policy=intervention.position_policy,
                ),
            ),
            MethodSpec(
                "negative_strength_control",
                False,
                Intervention(
                    layer=intervention.layer,
                    feature_id=intervention.feature_id,
                    strength=-abs(intervention.strength),
                    sign="negative",
                    injection_mode=intervention.injection_mode,
                    position_policy=intervention.position_policy,
                ),
            ),
        ]
    )
    return methods


def stable_method_names() -> list[str]:
    return list(REQUIRED_METHODS)
