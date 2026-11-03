"""
Test recognition of Halloween 2026 event Pokmon.

Verifies that all 14 Halloween 2026 event Pokmon fixtures predict
accurately with high confidence via the embedding override system.
"""
import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

from recognition import PokemonRecognizer

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures", "halloween_2026")

HALLOWEEN_POKEMON = [
    "imp_victini", "cherub_victini", "castle_palossand", "haunted_house_palossand",
    "cauldron_reuniclus", "elixir_reuniclus", "celestial_absol", "void_absol",
    "shrivelled_sunflora", "statue_simipour", "medusa_simipour",
    "witch_meowscarada", "oracle_meowscarada", "enchanted_applin"
]

@pytest.fixture(scope="module")
def recognizer():
    return PokemonRecognizer()

@pytest.mark.parametrize("pokemon_name", HALLOWEEN_POKEMON)
def test_halloween_pokemon_prediction(recognizer, pokemon_name):
    img_path = os.path.join(FIXTURES_DIR, f"{pokemon_name}.png")
    assert os.path.exists(img_path), f"Fixture not found: {img_path}"
    with open(img_path, "rb") as f:
        img_bytes = f.read()
    predicted_name, confidence = recognizer.predict(img_bytes)
    assert predicted_name == pokemon_name, (
        f"Expected '{pokemon_name}', got '{predicted_name}' with confidence {confidence:.4f}"
    )
    assert confidence >= 0.95, f"Confidence too low: {confidence:.4f}"


# Real 800x500 Poketwo spawns. ONNX guesses unrelated species on these
# (palkia-origin, nihilego), so they exercise the strong-match bypass.
SPAWNS_DIR = os.path.join(os.path.dirname(__file__), "fixtures", "halloween_2026_spawns")


@pytest.mark.parametrize("pokemon_name", ["witch_meowscarada", "haunted_house_palossand", "prof_tangrowth"])
def test_real_spawn_prediction(recognizer, pokemon_name):
    with open(os.path.join(SPAWNS_DIR, f"{pokemon_name}.jpg"), "rb") as f:
        predicted_name, confidence = recognizer.predict(f.read())
    assert predicted_name == pokemon_name, f"got '{predicted_name}' ({confidence:.4f})"
