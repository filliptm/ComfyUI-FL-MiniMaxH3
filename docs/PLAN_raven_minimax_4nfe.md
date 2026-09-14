# RAVEN × FL MiniMax H3 — integration plan (4-NFE consistency / streaming)

Status: **scoping**. Grounded in the released `mvp-ai-lab/RAVEN` repo (2026-08-19
MiniMax-H3 release) and ComfyUI's native H3 implementation. This is a research
test, not a target: the released preview LoRA is explicitly undertrained.

## 1. What we confirmed (facts, source-verified)

### The two H3 DiTs are the same model
RAVEN `projects/minimax_h3/configs/dit.yaml` vs ComfyUI
`comfy/ldm/minimax/model.py:MiniMaxH3Model`:

| knob | RAVEN | ComfyUI |
|---|---|---|
| hidden_size | 5376 | 5376 |
| num_layers | 50 | 50 |
| token_refiner layers | 2 | 2 |
| heads × dim | 56 × 128 | 56 × 128 |
| ffn_hidden_size | 14336 | 14336 |
| patch | [1,2,2] | (1,2,2) |
| text_dim | 5120 | 5120 |
| latent channels | 24 vid / 32 aud | 24 / 32 |

→ Same weights. The released MiniMax-H3-RAVEN-Streaming-LoRA-4NFE-Preview
(r=128, alpha=128) targets `qkv_proj, out_proj, fc1, fc2, linear,
condition_proj, video_patch_proj, audio_patch_proj, video_out, audio_out,
proj_in, proj_out` — every name exists in Comfy's DiTBlock/FinalLayer/MiniMaxH3Model
(`qkv_proj`, `out_proj`, `fc1`, `fc2`, `video_patch_proj`, `audio_patch_proj`,
`condition_proj`, `video_out`, `audio_out`). `linear/proj_in/proj_out` map onto
the AdalN time embedder/final AdalN heads — must be confirmed against Comfy's
naming when loading (see Open Questions). **The LoRA should load with ComfyUI's
standard LoRA path.**

### Where the two actually diverge (not the weights — the pipeline)
Everything portable checkpoints-side; the differences are at
sample-time. This is the real work:

1. **Prediction target / timestep convention.** ComfyUI's H3 predicts
   **velocity** (`ModelType.FLOW_AV`, `return [-video_out, -audio_out]`).
   RAVEN wraps the same model as an **x0-prediction** model via
   `MiniMaxH3X0Model`, with its own timestep convention `t_h3 = 1 - t_repo`,
   a per-modality "clean constant", and the velocity sign handled explicitly.
2. **No CFG.** RAVEN outputs are guidance-distilled: one positive pass, no
   negative prompt, no guidance (`cfg ≈ 1`).
3. **4-NFE consistency sampler.** Their sampling config:
   - `ConsistencySampler` (`x_t → x0(≈pred) → re-noise to next s` per step)
   - `lerp` (linear-interp VP) schedule, `T=1.0`, pred_type `x_0`
   - `trailing` timesteps: video `num_sampling_steps=4, shift=12.0`;
     audio same 4 steps, `shift=3.0`
   - video and audio share a step index but not a timestep (two shifted grids
     driven in lockstep), audio ≈ small fraction of packed rows.
4. **Causal streaming rollout** (the actual contribution, hardest part):
   - chunk-causal attention over **packed** token rows (video / text / audio
     tags), KV-cached, chunk-by-chunk rollout with `sink=2, window=2`
   - prefix (text) cache-fill as its own forward so text can't attend media
   - `NaiveCache` + sparse flex-attention BlockMask
   - this is what turns "one fixed clip" into **autoregressive extrapolation**

ComfyUI currently gives us: the model, both VAEs, the Qwen3-VL text encoder,
nested `[video, audio]` latent plumbing, flow-shift (`sigma_shift_video=12 /
sigma_shift_audio=3` defaults already match RAVEN's trial), and a
`KSampler`-compatible custom-sampler seam. FL pack already patches nested H3
latents and sampling via model patcher (motion-context hook, beat KSampler).

## 2. Decision

Split into two phases so we get a cheap, decisive signal before committing to
the expensive part. The cheap phase tests whether RAVEN's **sampling recipe**
and the **released LoRA** do anything useful in ComfyUI at all, using the
*bidirectional* model we already run. The streaming phase is gated on that.

- If LoRA + 4-NFE beats the 20-step `res_multistep` baseline on image quality
  at 4 steps → strong signal; proceed to Phase 2 (and flag to upstream that a
  bidirectional-4NFE checkpoint is worth releasing).
- If it's mush (expected) → we've validated the sampler against the baseline,
  and we know the LoRA is causal-only; then Phase 2's "real RAVEN" is the only
  way to test it, which is a much bigger commitment — user decision point.

Phase 1 is low risk, fully reversible, shares nothing with production FL nodes
until it passes review.

## 3. Phase 1 — bidirectional 4-NFE consistency sampler (harness)

Goal: reproduce RAVEN's sampling math on the ComfyUI H3 model, with and
without the released LoRA, head-to-head vs the user's current
`res_multistep/simple, 20 steps, cfg 1` baseline (the active workflow).

### New nodes (scoped to a new file, not touching existing FL H3 nodes)
1. **FL MiniMax H3 Consistency KSampler** — a `ComfyNode` that, given the H3
   `MODEL`, `positive` (), a latent, the num steps, and shift knobs, runs a
   4-NF consistency schedule:
   - register a custom KSAMPLER (`sample_consistency_x0`-style) in
     `comfy.samplers.KSAMPLER` via the FL pack's loader, **not** editing core
     ComfyUI.
   - at each of the `N` trailing timesteps: predict `x0` from the model
     (bridge velocity→x0 through the existing AV sampling pipeline, reusing the
     `audio_scale`/nested `[video,audio]` plumbing already used on H3), then
     re-noise to the next `s` (lerp/VP like `eps*std + …`), repeat.
   - video/audio run the same 4 steps on their own shifted grids, exactly like
     RAVEN's trailing twin-schedule.
   - no negative prompt / no CFG input path on this node (CFG-locked at 1 for H3).
2. **FL MiniMax H3 X0 / apply-timestep wrapper (model patcher, not a new
   checkpoint)** — set `pred_type=x_0` + `t_h3 = 1 - t_repo` + velocity-sign
   bridge to the sampler. Equivalent to what `MiniMaxH3X0Model` does server-side.
3. **Reference / audio conditioning reuse** — connect existing H3
   `MiniMaxH3ReferenceToVideo` conditioning + latent + audio crop so the test
   uses *your* actual workflow content (ref image + song), not a synthetic one.

Optional but cheap and worth it:
- **FL MiniMax H3 Basline A/B node** — profile pass: render 4-NFE and baseline
  20-NFE to the same decode + side-by-side mosaic (their validation style),
   which is exactly the `h3_context_loop` naming user already expects.

### Config
Mirror RAVEN's trial exactly on first run:
```
steps=4, scheduler=consistency(vpm lerp), video 4 steps, audio 4 steps
video shift 12.0 / audio shift 3.0, cfg=1, no negative
seed-parameterized, latent from FL pack
res 512 (or match user workflow), length 243
```

### Phase-1 acceptance
- Deterministic output at 4 steps (`fixed` seed).
- Side-by-side diff vs the 20-step baseline; report NFE-visible quality/artifacts.
- LoRA vs no-LoRA @ 4 steps (verifies the released adapter even loads + transfers).
- No regressions/behavior change to existing FL H3 nodes or the user's
  currently-saved workflow (nouveau nodes only, no ID reuse).

### Risks
- The released preview LoRA is self-described "undertrained, limited detail" →
  4-NFE likely mush; this is a harness, not a feature yet. That's the point.
- Custom sampler must reproduce the audio `audio_scale` carry + velocity-sign
  exactly; wrong → plausible-but-wrong audio/video. Test harness will decode
  AV and compare spectrally/visually to the baseline.

## 4. Phase 2 (gated) — causal/autoregressive streaming on H3

Only if Phase 1 warrants. This is where "real-time extrapolation" actually lives
and where the effort lives:

1. Causal packed layout reimplemented for ComfyUI H3 (video/text/audio token
   tags, `sink=2, window=2`), mirroring `projects/minimax_h3/modeling/packing.py`.
2. Causal attention forward on the DiT (they use manual flex-attention
   BlockMask + KV cache). Port to ComfyUI's patcher/hook seam (the FL pack
   already patches the model; reuse that seam instead of touching
   `comfy/ldm/minimax`).
3. Chunk-causal rollout: text-prefix cache-fill forward, then one `denoise →
   cache-fill → next` per chunk, streaming frames out with
   `MiniMaxH3X0` + consistency 4-NFE.
4. New nodes: `FL MiniMax H3 Streaming Rollout`, `FL MiniMax H3 Stream Continue`
   (feed previous tail as the causal prefix → keep generating), reuse the
   existing shot/motion-context/assembler where it's simpler.

Effort: multi-week; SM90 (Hopper) flex-attention assumptions may not hold on
user's consumer GPU for the flash path (fall back to chunked masked attn in
torch). Likely slower than 20-step bidirectional at same res since causal →
longer rollout, but *streaming-extendable*, which your beat-hard-cut workflow
doesn't need today.

## 5. Open questions / to confirm
- Exact ComfyUI module names for `linear` / `proj_in` / `proj_out` targets in
  the LoRA (may be absent → use Comfy LoRA apply with subsetting).
- User's GPU / VRAM (4-step strictly lighter than 20-step; only relevant for
  the Phase-2 flex path).
- License ordering: downloading the LoRA from mvp-lab + base under MiniMax-H3
  Community License — the model is already on the box, only the LoRA is new.

## 6. Suggested order of work
1. (Re)confirm LoRA loads no-error into user's H3 checkpoint with Comfy LoRA
   loader (offline, 10 min).
2. Ship 4-NFE consistency KSampler node + x0 bridge (Phase-1 milestone 1).
3. Baseline A/B on user's actual workflow content.
4. If good: open Phase 2. If bad: we have a documented sampler-and-transfer
   verdict either way.

## Notes
- This repo's boundary keeps ComfyUI core untouched: all work is in
  `ComfyUI-FL-MiniMaxH3` + standard model patcher hooks. No internet request
  paths, no core code changes.