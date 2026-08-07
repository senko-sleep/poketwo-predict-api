"""
Pokemon Manager
===============
Tool to add, edit, or delete Pokemon labels and train the prediction model.

HOW TO USE:
  1. Edit the POKEMON dict below to add/remove/modify entries
  2. Run this script:  python manage_pokemon.py
  3. Choose an action from the menu

Each entry in POKEMON should look like:
    "label_name": {
        "urls": ["https://...", "https://..."],   # training image URLs
        "action": "add",                           # "add", "remove", or "skip"
    }
"""

import os
import sys
import json
import asyncio
import time
import aiohttp
import numpy as np
from tqdm import tqdm
from PIL import Image
import io

# Fix encoding on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    import io as _io
    sys.stdout = _io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = _io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# Resolve project root (parent directory)
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

# ============================================================
#  EDIT HERE - Define your Pokemon
# ============================================================
#
#  action:
#    "add"    - add/update this label
#    "remove" - delete this label
#    "skip"   - ignore this entry (keep it for notes/reference)
#
#  urls: list of direct image URLs for training (optional for add)
#
# ============================================================

POKEMON: dict = {
    "sammy": {
        "urls": ["https://images-ext-1.discordapp.net/external/9c9M10rPZaxqGDC-D17qnBDv8n49Z4Hfp8RbN0MS3BQ/%3Fsize%3D1024/https/cdn.discordapp.com/avatars/1074905720073506876/6e92a896fc8009a836763720c223da70.png?format=webp&quality=lossless"],
        "action": "add",
    },
}

# ============================================================
#  Paths - pokemon-predict-api structure
# ============================================================
LABELS_PATH = "models/labels_v2.json"
MODEL_PATH = "models/pokemon_cnn_v2.onnx"
EMBEDDING_INDEX_PATH = "models/event_embedding_index.npz"
EMBEDDING_META_PATH = "models/event_embedding_meta.json"
MODEL_DIR = "models"
DIVIDER = "=" * 60


def section(title: str):
    print(f"\n{DIVIDER}")
    print(f"  {title}")
    print(DIVIDER)


def load_labels() -> dict:
    """Load labels_v2.json"""
    if os.path.exists(LABELS_PATH):
        with open(LABELS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_labels(labels: dict):
    """Save labels_v2.json"""
    os.makedirs(os.path.dirname(LABELS_PATH), exist_ok=True)
    with open(LABELS_PATH, "w", encoding="utf-8") as f:
        json.dump(labels, f, indent=4)


def load_embedding_meta() -> dict:
    """Load embedding metadata."""
    if os.path.exists(EMBEDDING_META_PATH):
        with open(EMBEDDING_META_PATH, "r") as f:
            return json.load(f)
    return {"total_entries": 0, "label_counts": {}}


def save_embedding_meta(meta: dict):
    """Save embedding metadata."""
    with open(EMBEDDING_META_PATH, "w") as f:
        json.dump(meta, f, indent=2)


def show_current_state():
    """Display current labels and embedding stats."""
    section("CURRENT STATE")
    labels = load_labels()
    print(f"  Total labels: {len(labels)}")
    
    # Show embedding stats
    meta = load_embedding_meta()
    print(f"  Embedding index: {meta['total_entries']} entries")
    print(f"  Labels with embeddings: {len(meta['label_counts'])}")
    
    if meta['label_counts']:
        print(f"\n  Labels with embeddings:")
        for label, count in sorted(meta['label_counts'].items()):
            print(f"    {label}: {count} embeddings")


def list_all_labels(query: str = ""):
    """List all labels with optional search filter."""
    labels = load_labels()
    sorted_labels = sorted(labels.items(), key=lambda x: int(x[0]))

    if query:
        sorted_labels = [(k, v) for k, v in sorted_labels if query.lower() in v.lower()]
        print(f"\n  Search results for '{query}': {len(sorted_labels)} matches\n")
    else:
        print(f"\n  All labels: {len(sorted_labels)} total\n")

    for idx, label in sorted_labels:
        print(f"  [{idx:>4}] {label}")


def add_label_manual(label: str) -> tuple:
    """Add a label to labels_v2.json manually."""
    labels = load_labels()
    label = label.lower().strip().replace(" ", "_")
    label_to_index = {v: k for k, v in labels.items()}

    if label in label_to_index:
        return False, f"Label '{label}' already exists at index {label_to_index[label]}"

    next_index = str(len(labels))
    labels[next_index] = label
    save_labels(labels)
    return True, f"Added '{label}' at index {next_index}"


def remove_label_manual(label: str) -> tuple:
    """Remove a label from labels_v2.json and reindex."""
    labels = load_labels()
    label = label.lower().strip().replace(" ", "_")
    label_to_index = {v: k for k, v in labels.items()}

    if label not in label_to_index:
        return False, f"Label '{label}' not found"

    index = label_to_index[label]
    del labels[index]

    # Reindex
    new_labels = {}
    for i, (_, lbl) in enumerate(sorted(labels.items(), key=lambda x: int(x[0]))):
        new_labels[str(i)] = lbl
    save_labels(new_labels)

    # Remove from embedding index
    meta = load_embedding_meta()
    if label in meta['label_counts']:
        del meta['label_counts'][label]
        meta['total_entries'] -= meta['label_counts'].get(label, 0)
        save_embedding_meta(meta)

    return True, f"Removed '{label}'. Labels reindexed ({len(new_labels)} remaining)"


def rename_label_manual(old_label: str, new_label: str) -> tuple:
    """Rename a label in labels_v2.json."""
    labels = load_labels()
    old_label = old_label.lower().strip().replace(" ", "_")
    new_label = new_label.lower().strip().replace(" ", "_")
    label_to_index = {v: k for k, v in labels.items()}

    if old_label not in label_to_index:
        return False, f"Label '{old_label}' not found"
    if new_label in label_to_index:
        return False, f"Label '{new_label}' already exists"

    index = label_to_index[old_label]
    labels[index] = new_label
    save_labels(labels)

    # Rename in embedding index
    meta = load_embedding_meta()
    if old_label in meta['label_counts']:
        count = meta['label_counts'].pop(old_label)
        meta['label_counts'][new_label] = count
        save_embedding_meta(meta)

    return True, f"Renamed '{old_label}' -> '{new_label}'"


def download_image(url: str) -> bytes:
    """Download image from URL with proper headers."""
    import requests
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Accept": "image/webp,image/apng,image/*,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://discord.com/",
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=30, allow_redirects=True)
        if response.status_code != 200:
            raise RuntimeError(f"Failed to download image: HTTP {response.status_code}")
        return response.content
    except Exception as e:
        raise RuntimeError(f"Failed to download image: {e}")


def augment_image(image: Image.Image) -> list:
    """Generate augmented versions of an image."""
    augmented = []
    
    # Original
    augmented.append(image)
    
    # Horizontal flip
    augmented.append(image.transpose(Image.FLIP_LEFT_RIGHT))
    
    # Rotate 90, 180, 270 degrees
    for angle in [90, 180, 270]:
        augmented.append(image.rotate(angle))
    
    # Brightness adjustments
    from PIL import ImageEnhance
    enhancer = ImageEnhance.Brightness(image)
    augmented.append(enhancer.enhance(0.8))
    augmented.append(enhancer.enhance(1.2))
    
    return augmented


def extract_embedding(image_bytes: bytes) -> np.ndarray:
    """Extract embedding from image using the model."""
    try:
        from recognition import PokemonRecognizer
        recognizer = PokemonRecognizer()
        
        # Preprocess
        image = Image.open(io.BytesIO(image_bytes))
        if image.mode != "RGB":
            image = image.convert("RGB")
        image = image.resize((224, 224), Image.LANCZOS)
        
        # Normalize
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_array = np.array(image, dtype=np.float32) / 255.0
        img_array = (img_array - mean) / std
        img_array = np.transpose(img_array, (2, 0, 1))
        img_array = img_array.astype(np.float32)[np.newaxis, ...]
        
        # Get embedding (use second-to-last layer)
        import onnxruntime as ort
        input_name = recognizer.session.get_inputs()[0].name
        outputs = recognizer.session.run(None, {input_name: img_array})
        
        # Use the logits as embedding
        embedding = outputs[0][0]
        return embedding
    
    except Exception as e:
        print(f"    Warning: Could not extract embedding: {e}")
        return None


def add_label_with_embeddings(label: str, urls: list) -> tuple:
    """Add label with embeddings from training images."""
    label_clean = label.lower().strip().replace(" ", "_")
    
    # Add label to labels file
    labels = load_labels()
    label_to_index = {v: k for k, v in labels.items()}
    
    if label_clean not in label_to_index:
        next_index = str(len(labels))
        labels[next_index] = label_clean
        save_labels(labels)
        print(f"  Added '{label_clean}' at index {next_index}")
    
    # Download images and extract embeddings
    embeddings = []
    meta = load_embedding_meta()
    
    print(f"  Processing {len(urls)} images...")
    for i, url in enumerate(urls, 1):
        try:
            print(f"    [{i}/{len(urls)}] Downloading {url[:50]}...")
            image_bytes = download_image(url)
            
            # Augment
            image = Image.open(io.BytesIO(image_bytes))
            if image.mode != "RGB":
                image = image.convert("RGB")
            
            augmented_images = augment_image(image)
            print(f"    Generated {len(augmented_images)} augmented versions")
            
            # Extract embeddings
            for aug_img in augmented_images:
                img_bytes = io.BytesIO()
                aug_img.save(img_bytes, format='PNG')
                img_bytes = img_bytes.getvalue()
                
                embedding = extract_embedding(img_bytes)
                if embedding is not None:
                    embeddings.append((label_clean, embedding))
            
        except Exception as e:
            print(f"    Error processing image {i}: {e}")
    
    if not embeddings:
        return False, "No embeddings extracted"
    
    # Save embeddings to index
    print(f"  Saving {len(embeddings)} embeddings...")
    
    # Check existing embedding structure
    if os.path.exists(EMBEDDING_INDEX_PATH):
        data = np.load(EMBEDDING_INDEX_PATH)
        # Check if embeddings are 2D (matrix format)
        sample_key = list(data.files)[0] if data.files else None
        if sample_key and sample_key != 'labels':
            sample_emb = data[sample_key]
            if len(sample_emb.shape) == 2:
                print(f"  Existing embeddings are 2D matrix format: {sample_emb.shape}")
                print(f"  Clearing and recreating embedding index for compatibility")
                os.remove(EMBEDDING_INDEX_PATH)
                if os.path.exists(EMBEDDING_META_PATH):
                    os.remove(EMBEDDING_META_PATH)
                meta = {"total_entries": 0, "label_counts": {}}
    
    # Load existing embeddings (or start fresh)
    existing_embeddings = {}
    existing_labels = []
    if os.path.exists(EMBEDDING_INDEX_PATH):
        data = np.load(EMBEDDING_INDEX_PATH)
        existing_embeddings = {k: data[k] for k in data.files if k != 'labels'}
        if 'labels' in data.files:
            existing_labels = data['labels'].tolist()
    
    # Add new embeddings
    for label, embedding in embeddings:
        key = f"{label}_{len([k for k in existing_embeddings.keys() if k.startswith(label)])}"
        existing_embeddings[key] = embedding
        existing_labels.append(label)
    
    # Save
    np.savez_compressed(EMBEDDING_INDEX_PATH, **existing_embeddings, labels=np.array(existing_labels))
    
    # Update metadata
    meta['total_entries'] = len(existing_labels)
    meta['label_counts'][label_clean] = meta['label_counts'].get(label_clean, 0) + len(embeddings)
    save_embedding_meta(meta)
    
    return True, f"Added {len(embeddings)} embeddings for '{label_clean}'"


async def process_pokemon():
    """Process all entries in the POKEMON dict."""
    if not POKEMON:
        print("\n  POKEMON dict is empty. Edit this file to add entries.")
        return

    section("PROCESSING POKEMON")

    for label, config in POKEMON.items():
        action = config.get("action", "skip")
        urls = config.get("urls", [])
        label_clean = label.lower().strip().replace(" ", "_")

        if action == "skip":
            print(f"\n  [{label_clean}] Skipped")
            continue

        elif action == "remove":
            print(f"\n  [{label_clean}] Removing...")
            success, msg = remove_label_manual(label_clean)
            print(f"    {'OK' if success else 'FAIL'}: {msg}")

        elif action == "add":
            print(f"\n  [{label_clean}] Adding with embeddings...")
            if urls:
                success, msg = add_label_with_embeddings(label, urls)
                print(f"    {'OK' if success else 'FAIL'}: {msg}")
            else:
                success, msg = add_label_manual(label_clean)
                print(f"    {'OK' if success else 'FAIL'}: {msg}")

    # Show updated state
    show_current_state()


def interactive_menu():
    """Interactive CLI menu for managing Pokemon."""
    while True:
        print(f"\n{DIVIDER}")
        print("  POKEMON MANAGER")
        print(DIVIDER)
        print("  1. Show current state")
        print("  2. List all labels (or search)")
        print("  3. Add a label")
        print("  4. Remove a label")
        print("  5. Rename a label")
        print("  6. Process POKEMON dict (batch)")
        print("  0. Exit")
        print()

        choice = input("  Choice: ").strip()

        if choice == "0":
            break

        elif choice == "1":
            show_current_state()

        elif choice == "2":
            query = input("  Search (blank for all): ").strip()
            list_all_labels(query)

        elif choice == "3":
            label = input("  Label name: ").strip()
            if label:
                success, msg = add_label_manual(label)
                print(f"  {'OK' if success else 'FAIL'}: {msg}")

        elif choice == "4":
            label = input("  Label to remove: ").strip()
            if label:
                confirm = input(f"  Really remove '{label}'? (y/n): ").strip().lower()
                if confirm == "y":
                    success, msg = remove_label_manual(label)
                    print(f"  {'OK' if success else 'FAIL'}: {msg}")

        elif choice == "5":
            old = input("  Old label name: ").strip()
            new = input("  New label name: ").strip()
            if old and new:
                success, msg = rename_label_manual(old, new)
                print(f"  {'OK' if success else 'FAIL'}: {msg}")

        elif choice == "6":
            asyncio.run(process_pokemon())


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Pokemon Manager")
    parser.add_argument("--batch", action="store_true", help="Process POKEMON dict and exit")
    parser.add_argument("--status", action="store_true", help="Show current state and exit")
    parser.add_argument("--add", type=str, help="Add a label by name")
    parser.add_argument("--remove", type=str, help="Remove a label by name")
    parser.add_argument("--rename", nargs=2, metavar=("OLD", "NEW"), help="Rename a label")
    parser.add_argument("--search", type=str, help="Search labels by name")
    args = parser.parse_args()

    if args.status:
        show_current_state()
    elif args.add:
        success, msg = add_label_manual(args.add)
        print(f"{'OK' if success else 'FAIL'}: {msg}")
    elif args.remove:
        success, msg = remove_label_manual(args.remove)
        print(f"{'OK' if success else 'FAIL'}: {msg}")
    elif args.rename:
        success, msg = rename_label_manual(args.rename[0], args.rename[1])
        print(f"{'OK' if success else 'FAIL'}: {msg}")
    elif args.search:
        list_all_labels(args.search)
    elif args.batch:
        asyncio.run(process_pokemon())
    else:
        interactive_menu()
