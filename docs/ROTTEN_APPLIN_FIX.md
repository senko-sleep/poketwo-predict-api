# Rotten Applin recognition verification — 2026-11-03

The supplied real Discord spawn previously returned `pidgeot` with classifier
confidence 0.2557254 on the production Modal endpoint (289.51 ms). The fixed
endpoint returns `rotten_applin`, `pokemon_id: 50308`, with event similarity
0.9976103. Event similarity is a match score, not a calibrated probability.

## Cause and changes

- Rotten Applin was missing from the API event embedding index, metadata and
  thresholds. Appending an event label to classifier labels cannot teach the
  1290-output ONNX classifier a new class. Its best guess was Pidgeot.
- The official sprite is mirrored in the actual spawn, scaled, compressed and
  composited onto a forest background. Enrollment uses 96 deterministic
  sprite-derived spawn composites (backgrounds, placement, size, flips,
  brightness and JPEG) plus the official sprite. The real spawn was reserved
  for verification and was never enrolled. The event has its own label and ID;
  it is not gated on the classifier predicting ordinary Applin.
- `models/event_embedding_index.npz`, `event_embedding_meta.json`,
  `event_labels.json`, and `event_label_config.json` register the event. Existing
  event thresholds and classifier weights are retained.
- `src/recognition.py` reuses the classifier logits for event matching, runs
  inference once per image, normalizes the index once at load time, closes the
  index archive, and limits ONNX thread oversubscription.
- `src/supabase_client/client.py` versions cache keys so an old Pidgeot answer
  cannot survive the recognition update.
- `deployment/modal/app.py` returns the event ID, sends inference, downloads
  and cache calls to the thread pool, and serializes cold recognizer creation.
- The parent bot repo adds `50308,rotten-applin` to `pokemon_names.csv`, accepts
  list-valued event base-species configs in the local recognizer, and tests the
  full slug/ID/catch-name path (`c Rotten Applin`).

## Results

- API tests: **45 passed** (real input, official sprite, resolution/compression,
  mirrored WebP, similar species and event forms, existing Halloween fixtures,
  Bagon/Prof Tangrowth regressions, one inference per request, raw/JSON/URL
  ASGI endpoints, cache invalidation).
- Bot recognition/form/catch tests: **43 passed, 1 skipped**. The run uses
  `SPAWN_LOCAL_RECOGNITION=on` because this workstation's available memory
  otherwise activates the existing memory safety budget. No production budget
  was changed. The skip is an existing optional runtime condition.
- **592** ordinary-species spawn composites: **572 correct before and after**,
  **zero changed predictions**, **zero false Rotten Applin matches**. These are
  synthetic spawns made from official art and existing spawn backgrounds, not
  592 independently captured Discord screenshots.
- Live Modal: original, 12 JPEG variants (widths 800/640/400/200; qualities
  95/70/40), mirrored WebP, four repeated originals, and the original URL all
  resolve to Rotten Applin / 50308. Seven ordinary/event controls retained
  their expected identities. Median reported inference across 18 byte requests
  was **47.67 ms**. Baseline and final request timing are individual production
  measurements, not a controlled throughput benchmark.
- Local centered-index recognizer also identifies the actual spawn as Rotten
  Applin (score 0.92958).

Full production results are in `rotten_applin_live_results.json`; the ordinary
spawn comparison is in `rotten_applin_regression_results.json`.

## Reproduce

From this API repo:

```powershell
python -m pytest test/test_rotten_applin.py test/test_halloween_2026.py test/test_event_override_regression.py -q
$env:PYTHONUTF8='1'
python -m modal deploy deployment/modal/app.py
```

Modal deploys `src/` and `models/`. The files under `test/fixtures/rotten_applin`
are offline test assets, including the original `spawn.webp` and official
sprites from `https://cdn.poketwo.net/images/{id}.png`. The source spawn is the
Discord attachment `1482577608775565397/1566761565888323676/pokemon.jpg`.
SHA-256: `f48c128f645259699f8a473b97b39d4eb2e23a52b83adb5b8165ffebf6ba66b4`.
The URL's temporary signature is unnecessary for offline regression tests.

The parent bot repo includes `scripts/enroll_api_event_sprite.py` to regenerate
sprite-derived enrollment using the existing background set. Run the tests
after enrollment and before deployment.

## Limits

Heavily compressed 120x75 thumbnails (JPEG quality 70/40) still return Elekid
or Sizzlipede. Quality 95 at that size succeeds. The verified resolution range
is 200–800 pixels wide. Existing errors on 20 of the 592 ordinary composites
remain unchanged. Supabase is disabled in the deployed environment, so cache
versioning is verified offline. The main bot process was not restarted here;
the Modal predictor is live, and the bot-side mapping is published on GitHub.
