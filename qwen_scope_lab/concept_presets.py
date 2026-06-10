"""Continuous-concept presets for manifold fitting + steering.

Each concept is an ordered set of values (a 1-D manifold: ordinal -> open curve,
cyclic -> closed loop). ``templates`` are fixed-prefix carrier sentences (item at the
end) used to build per-value activation centroids; ``steer_prompt`` is a sentence whose
continuation reveals the value, so replacing the item's residual with another value's
manifold point shifts the generated continuation. Shared by serve_web and modal_app so
the local and real-model paths use identical definitions.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Concept:
    name: str
    label: str
    kind: str  # "ordinal" (open curve) | "cyclic" (closed loop)
    items: tuple[str, ...]
    templates: tuple[str, ...]
    steer_prompt: str  # must contain "{item}"
    best_layer: int | None = None  # atlas-derived layer where this concept's manifold is cleanest (real 2B); None = diffuse, fall back to config default


_INTEGER_TEMPLATES = ("The number is {item}", "I counted {item}", "There were {item}",
                      "It costs {item} dollars", "Page {item}", "Chapter {item}")
_SIZE_TEMPLATES = ("The box was {item}", "It felt {item}", "A {item} thing",
                   "The size was {item}", "Quite {item}", "Remarkably {item}")
_DAY_TEMPLATES = ("Today is {item}", "The meeting is on {item}", "See you {item}",
                  "It happened last {item}", "Every {item}", "By {item}")
_MONTH_TEMPLATES = ("The month is {item}", "We met in {item}", "It was a cold {item}",
                    "Born in {item}", "Since last {item}", "By {item}")
_LETTER_TEMPLATES = ("The letter is {item}", "Section {item}", "Grade {item}",
                     "Point {item}", "Option {item}", "Part {item}")


CONCEPTS: dict[str, Concept] = {
    "days_of_week": Concept(
        "days_of_week", "Days of the week", "cyclic",
        ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"),
        _DAY_TEMPLATES, "Today is {item}. Tomorrow is", best_layer=14),
    "months": Concept(
        "months", "Months", "cyclic",
        ("January", "February", "March", "April", "May", "June",
         "July", "August", "September", "October", "November", "December"),
        _MONTH_TEMPLATES, "This month is {item}. Next month is"),
    "integers_0_20": Concept(
        "integers_0_20", "Integers 0–20", "ordinal",
        tuple(str(i) for i in range(21)),
        _INTEGER_TEMPLATES, "The number is {item}. The next number is", best_layer=8),
    "size": Concept(
        "size", "Size", "ordinal",
        ("tiny", "small", "medium", "large", "huge", "enormous"),
        _SIZE_TEMPLATES, "The object is {item}. An even bigger one is", best_layer=16),
    "letters": Concept(
        "letters", "Letters A–Z", "ordinal",
        tuple(chr(c) for c in range(ord("A"), ord("Z") + 1)),
        _LETTER_TEMPLATES, "The letter is {item}. The next letter is", best_layer=16),
}


_GENERIC = ("The level is {item}", "It was {item}", "Rated {item}", "Quite {item}", "Marked {item}")

# Extra candidate continuous concepts for the manifold ATLAS census (not steering presets).
_ATLAS_EXTRA: dict[str, Concept] = {
    "ordinals": Concept("ordinals", "Ordinals", "ordinal",
        ("first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth"),
        ("The {item} one", "In {item} place", "Ranked {item}", "The {item} time", "Came in {item}"), "It finished {item}", best_layer=8),
    "temperature": Concept("temperature", "Temperature", "ordinal",
        ("freezing", "cold", "cool", "mild", "warm", "hot", "scorching"),
        ("The weather was {item}", "It felt {item}", "A {item} day", "Quite {item}", "Remarkably {item}"), "The weather is {item}", best_layer=6),
    "brightness": Concept("brightness", "Brightness", "ordinal",
        ("pitch-black", "dark", "dim", "lit", "bright", "dazzling"),
        ("The room was {item}", "It looked {item}", "A {item} space", "Quite {item}", "Remarkably {item}"), "The room is {item}", best_layer=6),
    "loudness": Concept("loudness", "Loudness", "ordinal",
        ("silent", "quiet", "soft", "moderate", "loud", "deafening"),
        ("The sound was {item}", "It seemed {item}", "A {item} noise", "Quite {item}", "Remarkably {item}"), "The sound is {item}", best_layer=6),
    "speed": Concept("speed", "Speed", "ordinal",
        ("motionless", "slow", "steady", "brisk", "fast", "blistering"),
        ("The pace was {item}", "It moved {item}", "A {item} pace", "Quite {item}", "Remarkably {item}"), "The pace is {item}", best_layer=12),
    "valence": Concept("valence", "Emotion valence", "ordinal",
        ("miserable", "sad", "gloomy", "neutral", "content", "happy", "ecstatic"),
        ("She felt {item}", "He seemed {item}", "A {item} mood", "Quite {item}", "Remarkably {item}"), "She feels {item}", best_layer=16),
    "agreement": Concept("agreement", "Agreement (Likert)", "ordinal",
        ("strongly disagree", "disagree", "neutral", "agree", "strongly agree"),
        ("I {item}", "They {item}", "We {item}", "She would {item}", "Most {item}"), "My answer is {item}", best_layer=8),
    "education": Concept("education", "Education level", "ordinal",
        ("kindergarten", "elementary", "middle school", "high school", "college", "graduate", "doctorate"),
        ("They are in {item}", "Completed {item}", "Studying at {item}", "A {item} student", "Through {item}"), "They reached {item}", best_layer=8),
    "frequency": Concept("frequency", "Frequency", "ordinal",
        ("never", "rarely", "sometimes", "often", "usually", "always"),
        ("It happens {item}", "She {item} does it", "They {item} arrive", "We {item} agree", "He {item} calls"), "It {item} happens", best_layer=6),
    "compass": Concept("compass", "Compass directions", "cyclic",
        ("North", "Northeast", "East", "Southeast", "South", "Southwest", "West", "Northwest"),
        ("Head {item}", "Facing {item}", "It lies to the {item}", "Travel {item}", "Winds from the {item}"), "We headed {item}"),
    "decades": Concept("decades", "Decades", "ordinal",
        ("1950s", "1960s", "1970s", "1980s", "1990s", "2000s", "2010s", "2020s"),
        ("Back in the {item}", "During the {item}", "A {item} song", "The {item} era", "Set in the {item}"), "It was the {item}", best_layer=6),
    "rank": Concept("rank", "Military rank", "ordinal",
        ("private", "corporal", "sergeant", "lieutenant", "captain", "major", "colonel", "general"),
        ("Promoted to {item}", "The rank of {item}", "A {item} led them", "Serving as {item}", "Addressed as {item}"), "Promoted to {item}", best_layer=20),
    # Emotion lines (Goodfire-style manifold steering through emotion spaces, à la Anthropic's
    # emotion-vectors work). See docs/experiments/EMOTION_MANIFOLD.md: all three form clean residual
    # manifolds (isometry r 0.997-0.999), but manifold-vs-linear routing only wins on the monotone
    # arousal axis — valence lines can't cross the affect sign flip; fear is steer-resistant at 2B.
    "emotion_valence_intensity": Concept("emotion_valence_intensity", "Emotion (anger->joy)", "ordinal",
        ("furious", "angry", "annoyed", "calm", "content", "delighted", "euphoric"),
        ("She felt {item}", "He seemed {item}", "I am {item}", "A {item} mood", "They looked {item}", "Feeling {item}"),
        "Right now she feels {item}. Honestly, she is", best_layer=8),
    "emotion_fear": Concept("emotion_fear", "Fear gradation", "ordinal",
        ("terrified", "afraid", "anxious", "uneasy", "calm"),
        ("She felt {item}", "He seemed {item}", "I am {item}", "A {item} mood", "They looked {item}", "Feeling {item}"),
        "Right now he feels {item}. Honestly, he is", best_layer=8),
    "emotion_arousal": Concept("emotion_arousal", "Arousal (low->high)", "ordinal",
        ("numb", "bored", "calm", "alert", "excited", "frantic"),
        ("She felt {item}", "He seemed {item}", "I am {item}", "A {item} mood", "They looked {item}", "Feeling {item}"),
        "Right now they feel {item}. Honestly, they are", best_layer=12),
}

ATLAS_CONCEPTS: list[Concept] = list(CONCEPTS.values()) + list(_ATLAS_EXTRA.values())


_ALL: dict[str, Concept] = {**CONCEPTS, **_ATLAS_EXTRA}


def get_concept(name: str) -> Concept:
    concept = _ALL.get(name)
    if concept is None:
        raise ValueError(f"unknown concept {name!r}; choose one of: {', '.join(_ALL)}")
    return concept


def preset_summaries() -> list[dict]:
    return [{"name": c.name, "label": c.label, "kind": c.kind, "n_items": len(c.items),
             "best_layer": c.best_layer}
            for c in ATLAS_CONCEPTS]
