

## Embedding & pipeline notes

- Representation
    - Type: vector (frozen feature embedding extracted from labram / cbramod).
    - Current embedding dimension: 256.
    - Batch input shape used by the disentanglement pipeline: (B, 256).

- Current objective
    - For now we reconstruct the embeddings themselves (reconstruction target shape = (B, 256)).
    - Later: extend to reconstructing raw EEG or hand-crafted signal features — target shape will vary (e.g., (B, C, T) for raw signals or (B, F) for features).

- How embeddings flow into the disentanglement pipeline
    - Each factor (subject, task, device, …) has its own encoder that takes the embedding as input:
        - encoder_input_shape = (B, 256)
        - encoder outputs: z_subject ∈ (B, L_sub), z_task ∈ (B, L_task), z_device ∈ (B, L_dev), z_noise ∈ (B, L_noise)
    - Classifiers:
        - Each factor-specific classifier takes only its corresponding z_* as input (e.g., classifier_subject input shape = (B, L_sub)).
    - Reconstruction:
        - Concatenate all latents: z_cat ∈ (B, L_sub + L_task + L_dev + L_noise)
        - Decoder maps z_cat → reconstructed embedding ∈ (B, 256) (or to raw signal when extended).

- Parameters to choose / tune (placeholders)
    - embedding_dim = 256 (fixed by current extractors)
    - L_sub, L_task, L_dev, L_noise (latent sizes per encoder) — choose per-factor capacity needs
    - classifier hidden sizes / heads → shapes depend on L_*
    - decoder hidden sizes → final output size must match embedding_dim (256) or target signal shape
    - loss weights: reconstruction_weight, classification_weight, disentanglement_regularizers

- Integration with labram / cbramod
    - Both models produce saved .pt embeddings that serve as input to the disentanglement encoders.
    - Input to encoders: (B, 256). No change required to encoders when swapping between labram and cbramod as long as embedding_dim remains 256.
    TODO: this is atm for MI as the segments have that length, check this for other tasks as well, as e.g. sleep is just 240
    - If a model changes embedding_dim, update encoder_input_shape and decoder output size accordingly.

TODO
- Decide latent sizes L_*.




### To-do replacement: latent sizes, reconstruction targets, and labram/cbramod I/O

- Latent-size defaults and guidance
    - Defaults (tunable): L_sub = 64, L_task = 64, L_dev = 32, L_noise = 32
    - Rationale: allocate capacity proportional to expected information per factor; subject/task usually need more capacity than device/noise.
    - How to pick:
        - Start with defaults above and validate reconstruction + classification performances.
        - If a classifier for a factor underfits, increase that factor's L_*.
        - If decoder struggles to reconstruct, increase total latent budget (sum of L_*) or increase decoder capacity.
    - Hard constraints:
        - Decoder input size = sum(L_*) (must match encoder concatenation).
        - Classifier input size = corresponding L_*.

- Reconstruction target format for raw EEG (configurable)
    - General shape notation: (B, C, T)
        - B = batch size
        - C = number of channels / electrodes
        - T = time samples per segment
    - Recommended config entries (store centrally per dataset/task):
        - segment_length_samples (T)
        - n_channels (C)
        - sampling_rate_hz
        - sample_unit (e.g., "microvolts")
    - Examples (patterns, not prescriptive):
        - Short motor-imagery segments: (B, C, T) with T chosen to cover the MI window used by the extractor (e.g., T = segment_length_samples used by labram/cbramod).
        - Sleep / long-window tasks: use larger T; if T varies, either pad/trim or train a variable-length decoder.
    - Implementation notes:
        - Decoder final output size must equal C * T (or produce shape (B,C,T) directly).
        - If reconstructing features (hand-crafted), target shape = (B, F) where F = number of features.
        - Save dataset/task config so encoder/decoder can be automatically reconfigured when switching tasks.

- labram / cbramod — input parameters and I/O contract
    - Common behavior
        - Both extractors produce frozen embeddings saved as .pt files for downstream disentanglement.
        - Embedding dimension currently = embedding_dim (default 256). If this changes, update encoder_input_shape and decoder output dims.
        - Each saved embedding bundle should include metadata so the disentanglement pipeline can adapt automatically.
    - Expected saved .pt bundle structure (recommended)
        - dict with keys:
            - "embeddings": Tensor[N, embedding_dim] or Tensor[S, embedding_dim] (S = number of segments)
            - "sample_ids": list[string] or Tensor[N] — identifiers mapping embeddings to raw data
            - "params": dict containing:
                - "embedding_dim": int
                - "segment_length_samples": int (T) — if applicable
                - "n_channels": int (C) — raw data channels used to produce embedding
                - "sampling_rate_hz": float
                - "preprocessing": short string describing filters/normalization
            - optionally: "labels": dict of per-factor labels (e.g., subject_ids, task_ids, device_ids)
    - labram — recommended input parameters and outputs
        - Inputs (to labram extractor):
            - raw_data shape: (S, C, T) or dataset iterator yielding (C, T) segments
            - params: segment_length_samples, n_channels, sampling_rate_hz, preprocessing options (filter bank, normalization)
        - Outputs:
            - embeddings: Tensor[S, embedding_dim]
            - saved file: .pt containing the bundle structure above
    - cbramod — recommended input parameters and outputs
        - Inputs (to cbramod extractor):
            - raw_data shape: (S, C, T) or iterator
            - params: same set as labram (segment_length_samples, n_channels, sampling_rate_hz, preprocessing)
            - model-specific hyperparams: model_variant, checkpoint_path (if applicable)
        - Outputs:
            - embeddings: Tensor[S, embedding_dim]
            - saved file: .pt with bundle structure above

- Integration checklist (practical)
    - Ensure .pt bundle "params.embedding_dim" matches encoder_input_shape.
    - Ensure "params.segment_length_samples" and "params.n_channels" are available when enabling full-signal reconstruction so decoder output can be reshaped to (B, C, T).
    - Add unit tests that load a saved .pt bundle and assert shapes and presence of required metadata keys.
    - When changing embedding_dim or segment lengths, bump a version field in the .pt params to avoid silent mismatches.

Place these entries at $SELECTION_PLACEHOLDER$ to complete the README.
