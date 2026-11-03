"""Offline regression through the same recognizer used by Modal.

spawn.webp is the supplied real Discord spawn, never an enrollment input.
50308.png is the official sprite used to build the event index.
"""
import io
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from recognition import PokemonRecognizer

FIXTURES = ROOT / "test" / "fixtures" / "rotten_applin"


@pytest.fixture(scope="module")
def recognizer():
    return PokemonRecognizer(**{
        key: str(ROOT / "models" / filename) for key, filename in {
            "model_path": "pokemon_cnn_v2.onnx", "labels_path": "labels_v2.json",
            "embedding_index_path": "event_embedding_index.npz",
            "embedding_meta_path": "event_embedding_meta.json",
            "event_label_config_path": "event_label_config.json",
            "event_labels_path": "event_labels.json",
        }.items()
    })


def test_real_spawn_and_identity(recognizer):
    assert recognizer.predict((FIXTURES / "spawn.webp").read_bytes())[0] == "rotten_applin"
    assert recognizer.event_labels["rotten_applin"]["id"] == 50308
    assert "rotten_applin" in recognizer.embedding_labels


@pytest.mark.parametrize("width", [800, 640, 400, 200])
@pytest.mark.parametrize("quality", [95, 70, 40])
def test_real_spawn_resolution_and_compression(recognizer, width, quality):
    im = Image.open(FIXTURES / "spawn.webp").convert("RGB")
    im = im.resize((width, round(width * im.height / im.width)), Image.Resampling.LANCZOS)
    b = io.BytesIO()
    im.save(b, "JPEG", quality=quality)
    assert recognizer.predict(b.getvalue())[0] == "rotten_applin"


def test_real_spawn_mirrored_and_webp(recognizer):
    im = ImageOps.mirror(Image.open(FIXTURES / "spawn.webp").convert("RGB"))
    b = io.BytesIO()
    im.save(b, "WEBP", quality=65)
    assert recognizer.predict(b.getvalue())[0] == "rotten_applin"


def test_official_sprite(recognizer):
    assert recognizer.predict((FIXTURES / "50308.png").read_bytes())[0] == "rotten_applin"


@pytest.mark.parametrize("pid", [840, 841, 842, 1011, 50175, 50307])
def test_similar_species_and_event_forms_do_not_become_rotten(recognizer, pid):
    art = Image.open(FIXTURES / f"{pid}.png").convert("RGBA")
    art = art.crop(art.getbbox())
    for scale, flip, color in [(0.5, False, (60, 100, 60)), (0.7, True, (25, 40, 55)),
                               (0.85, False, (150, 130, 80))]:
        fg = ImageOps.mirror(art) if flip else art
        h = round(500 * scale)
        fg = fg.resize((round(fg.width * h / fg.height), h), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (800, 500), color)
        canvas.paste(fg, ((800 - fg.width) // 2, (500 - h) // 2), fg)
        b = io.BytesIO()
        canvas.save(b, "JPEG", quality=70)
        assert recognizer.predict(b.getvalue())[0] != "rotten_applin"


def test_one_inference_per_spawn(recognizer, monkeypatch):
    calls = []
    original = recognizer.session.run
    def run(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(recognizer.session, "run", run)
    recognizer.predict((FIXTURES / "spawn.webp").read_bytes())
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_modal_byte_json_and_url_endpoints(recognizer, monkeypatch):
    import base64
    import importlib.util
    from types import SimpleNamespace
    pytest.importorskip("modal")
    import httpx
    from supabase_client import supabase_manager
    spec = importlib.util.spec_from_file_location("modal_deployment", ROOT / "deployment/modal/app.py")
    deployment = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(deployment)
    deployment.recognizer = recognizer
    monkeypatch.setattr(supabase_manager, "client", None)
    data = (FIXTURES / "spawn.webp").read_bytes()
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=deployment.fastapi_app),
                                base_url="http://test") as client:
        for response in [await client.post("/predict", content=data),
                         await client.post("/predict", json={"image_bytes": base64.b64encode(data).decode()})]:
            assert response.status_code == 200, response.text
            assert response.json()["pokemon"] == "rotten_applin"
            assert response.json()["pokemon_id"] == 50308
        import requests
        monkeypatch.setattr(requests, "get", lambda *a, **k: SimpleNamespace(
            content=data, raise_for_status=lambda: None))
        result = await client.post("/predict/url", json={"url": "https://example.invalid/image"})
        assert result.status_code == 200, result.text
        assert result.json()["pokemon_id"] == 50308


def test_persistent_cache_is_versioned():
    import hashlib
    from supabase_client import supabase_manager
    from recognition import RECOGNITION_VERSION
    data = (FIXTURES / "spawn.webp").read_bytes()
    assert supabase_manager._generate_image_hash(data) != hashlib.sha256(data).hexdigest()
    assert supabase_manager._generate_image_hash(data) == hashlib.sha256(
        RECOGNITION_VERSION.encode() + b"\0" + data).hexdigest()


def test_existing_fixtures_have_identical_answers_without_new_label(recognizer):
    # Same logits, old index versus enriched index: no unrelated prediction changes.
    old_labels = recognizer.embedding_labels
    old_matrix = recognizer._normalized_embeddings
    keep = np.array([lb != "rotten_applin" for lb in old_labels])
    for path in sorted((ROOT / "test" / "fixtures").rglob("*")):
        if path.suffix.lower() not in (".jpg", ".png") or FIXTURES in path.parents:
            continue
        logits = recognizer.extract_embedding(path.read_bytes())
        probs = np.exp(logits - logits.max()); probs /= probs.sum()
        name = recognizer.labels[int(probs.argmax())]
        conf = float(probs.max())
        after = recognizer.predict_with_embeddings(b"", conf, name, logits) or (name, conf)
        try:
            recognizer.embedding_labels = [lb for lb in old_labels if lb != "rotten_applin"]
            recognizer._normalized_embeddings = old_matrix[keep]
            before = recognizer.predict_with_embeddings(b"", conf, name, logits) or (name, conf)
        finally:
            recognizer.embedding_labels = old_labels
            recognizer._normalized_embeddings = old_matrix
        assert after[0] == before[0], path.name
