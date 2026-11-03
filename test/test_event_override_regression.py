"""
Regression tests for the event-Pokemon override gate.

Bug #1 - Bagon -> "Autoromantic Flag Vivillon" misprediction:
Root cause: the event-Pokemon override reused the classifier's raw logits as a
similarity "embedding". Same-species vs. different-species cosine similarity
only differs by ~1.5 points (e.g. real bagon vs shiny bagon ~99.3% vs. bagon
vs charmander/pikachu ~97.6-98.5%), which sits inside the min_sim thresholds
that were configured for most event labels - so confidently-classified real
Pokemon spuriously "matched" unrelated costumed event Pokemon.

Fix: `PokemonRecognizer._base_species_matches()` gates any event label that
has a `base_species` configured (costumed variants of a real Pokemon, e.g.
autoromantic_flag_vivillon -> vivillon) on the ONNX classifier's own top guess
already being that species, before embedding similarity is even considered.
This closes the collision without disabling event detection entirely - event
overrides (PokemonRecognizer.EVENT_OVERRIDE_ENABLED) stay on so original-design
event Pokemon with no real base species (peakychu, mosslax, ...) keep working,
protected instead by their own low onnx_ceiling / high min_sim thresholds in
event_label_config.json.

This test locks in that the known-bad case still resolves to the correct
species even with overrides enabled, and that the base-species gate is what's
doing the blocking (not a blanket disable).

Bug #2 - Prof Tangrowth never detected:
The base_species gate from bug #1 has a false-negative side effect: for a
costume different enough from its base species (white lab-coat wool instead
of green vines), the ONNX classifier's top guess isn't the base species at
all (observed: "nihilego" at ~47% confidence, vs. 99.83% embedding similarity
to prof_tangrowth) - so the gate blocked a *correct* override instead of a
spurious one. Same root issue already hit dj_rotom (see its _note in
event_label_config.json). Fix: drop `base_species` for prof_tangrowth and
protect it instead with a low onnx_ceiling (0.6) + high min_sim (0.96), same
pattern as dj_rotom - the override only fires when ONNX itself is quite
unsure, and only accepts a near-exact embedding match.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

from recognition import PokemonRecognizer

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "bagon_spawn.jpg")
PROF_TANGROWTH_FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "prof_tangrowth_spawn.jpg")


def test_bagon_is_not_misidentified_as_event_vivillon():
    recognizer = PokemonRecognizer()
    with open(FIXTURE, "rb") as f:
        image_bytes = f.read()

    pokemon_name, confidence = recognizer.predict(image_bytes)

    assert pokemon_name == "bagon", (
        f"Expected 'bagon', got '{pokemon_name}' ({confidence:.4f}) - "
        f"the event-embedding override may have fired again."
    )
    assert confidence > 0.9


def test_bagon_is_blocked_by_base_species_gate_even_when_onnx_is_uncertain():
    recognizer = PokemonRecognizer()
    assert recognizer.EVENT_OVERRIDE_ENABLED is True, (
        "Event-Pokemon overrides are expected to be enabled - original-design "
        "event Pokemon (peakychu, mosslax, ...) have no other way to be "
        "detected. If you're disabling this, custom-character predictions "
        "will stop working again."
    )

    with open(FIXTURE, "rb") as f:
        image_bytes = f.read()
    # Even with a confidence low enough to normally pass every configured
    # onnx_ceiling, the override must stay blocked because the ONNX top guess
    # ("bagon") doesn't match autoromantic_flag_vivillon's base_species
    # ("vivillon") - that's the gate that fixes the original bug.
    result = recognizer.predict_with_embeddings(image_bytes, onnx_confidence=0.0, onnx_label="bagon")
    assert result is None


def test_prof_tangrowth_is_correctly_detected():
    recognizer = PokemonRecognizer()
    with open(PROF_TANGROWTH_FIXTURE, "rb") as f:
        image_bytes = f.read()

    pokemon_name, confidence = recognizer.predict(image_bytes)

    assert pokemon_name == "prof_tangrowth", (
        f"Expected 'prof_tangrowth', got '{pokemon_name}' ({confidence:.4f}) - "
        f"the base_species gate may be blocking this costume again."
    )
    assert confidence > 0.96


if __name__ == "__main__":
    test_bagon_is_not_misidentified_as_event_vivillon()
    test_bagon_is_blocked_by_base_species_gate_even_when_onnx_is_uncertain()
    test_prof_tangrowth_is_correctly_detected()
    print("All regression tests passed.")
