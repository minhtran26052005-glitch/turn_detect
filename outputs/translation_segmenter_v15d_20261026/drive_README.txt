Turn_detect v15d: direct Colab -> user mounted Google Drive

Use latest_manifest.json to map logical paths to SHA256-named files in objects/. Keep destination_identity.json. previous_manifest.json retains the previous committed generation. The current manifest contains data/crawl state, source, model weights/tokenizers, optimizer/RNG checkpoints, reports, and frozen held-out assets. Each object is SHA checked at source and after Drive-mounted readback.

A separate recovery_verification.json records successful CPU deserialization of all three cycle-48 optimizer checkpoints and exact training fingerprint reconstruction from data cycle 50. It verifies backup provenance, not model quality or an executed training resume. More recent manifests have independent per-object hash/readback verification.

On a new Colab VM mount this same Drive, verify destination identity and committed manifest/object hashes, restore the data archive, source_bundle and matching teacher optimizer checkpoints/receipts. Do not start from base if recoverable checkpoints exist. Windows large downloads are disabled; only small reports/review samples go to the PC. No old local files were deleted.
