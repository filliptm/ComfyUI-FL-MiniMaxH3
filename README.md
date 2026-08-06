# FL MiniMax H3

MiniMax H3 workflow nodes for ComfyUI. The pack adds strict prompt timelines, beat-aligned render planning, grouped and independent shot sampling, pixel-space latent refinement, and final shot assembly while preserving MiniMax H3's native nested video/audio latent format.

[![MiniMax H3](https://img.shields.io/badge/MiniMax-H3-7c3aed?style=for-the-badge)](https://github.com/Comfy-Org/ComfyUI)
[![Patreon](https://img.shields.io/badge/Patreon-Support%20Me-F96854?style=for-the-badge&logo=patreon&logoColor=white)](https://www.patreon.com/Machinedelusions)

## Features

- **Strict prompt timelines** - Schedule prompt conditioning over H3 video tokens in seconds, frames, or beats
- **Reference-aware conditioning** - Use H3 reference images, videos, video audio, and standalone audio
- **Beat shot planning** - Turn an `FL_PROMPT_SCHEDULE` into independently renderable or explicitly grouped shots
- **Nested latent sampling** - Sample MiniMax H3 video and audio latents without flattening their native structure
- **Pixel-space refinement** - Decode, resize, re-encode, and run a controlled low-denoise detail pass
- **Shot assembly** - Decode and concatenate planned renders in timeline order
- **Workflow compatibility** - Keeps the original Fill Nodes node IDs, sockets, custom types, and metadata

## Nodes

| Node | Description |
|------|-------------|
| **FL MiniMax H3 Prompt Timeline** | Builds native H3 video/audio latents and strict temporal conditioning from a manual or connected prompt schedule |
| **FL MiniMax H3 Apply Timeline** | Rebuilds the strict temporal conditioning after an H3 latent has been spatially resized |
| **FL MiniMax H3 Beat Shot Planner** | Converts a beat prompt schedule into grouped or independent H3 render plans with matching audio slices |
| **FL MiniMax H3 Beat KSampler** | Samples every planned H3 render and returns editable nested latents |
| **FL MiniMax H3 Beat Pixel Upscale KSampler** | Decodes each render to pixels, resizes proportionally, re-encodes, and refines at configurable denoise |
| **FL MiniMax H3 Shot Assembler** | Decodes and concatenates completed render latents into one image sequence |

## Installation

### ComfyUI Manager

Search for **FL MiniMax H3** and install the pack after it is available in the Comfy Registry.

### Manual

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/filliptm/ComfyUI-FL-MiniMaxH3.git
```

Restart ComfyUI after installation.

## Example workflow

`example_workflows/MiniMax H3 Beat Shot Music Video.json` contains the complete beat-scheduled planning, sampling, pixel-upscale, and assembly path used during development. Replace its image, audio, and model selections with files available in your ComfyUI installation.

## Required ComfyUI support

This pack uses the MiniMax H3 implementation included with current ComfyUI. It does not bundle or download model weights.

A normal H3 workflow uses:

| Component | Typical model |
|-----------|---------------|
| Diffusion model | A supported MiniMax H3 `ref2va` diffusion model |
| Text encoder | MiniMax-compatible Qwen3-VL text encoder |
| Video VAE | `minimax_h3_video_vae_fp16.safetensors` |
| Audio VAE | `minimax_h3_audio_vae_fp32.safetensors` |

Use ComfyUI's standard model loaders and supported H3 model formats. Model files remain in the normal ComfyUI model directories.

## Quick start: one continuous H3 render

1. Load the H3 text encoder, video VAE, and audio VAE.
2. Add **FL MiniMax H3 Prompt Timeline**.
3. Connect any reference images, videos, or audio.
4. Enter the global prompt and timeline sections.
5. Connect `scheduled` to the first sampler.
6. Decode the resulting nested video and audio latent with the standard H3 VAEs.

Manual frame syntax:

```text
[0 - 48]
Wide establishing shot. The subject turns toward camera.

[48 - 96]
Hard cut to a low tracking shot as the subject moves forward.
```

Frame ranges are zero-based and their end is exclusive.

## Beat-scheduled music-video workflow

The recommended audio-reactive path uses **FL Audio Beat Prompt Schedule** from [ComfyUI Fill Nodes](https://github.com/filliptm/ComfyUI_Fill-Nodes):

```text
FL Audio Beat Prompt Schedule
    -> FL MiniMax H3 Beat Shot Planner
    -> FL MiniMax H3 Beat KSampler
    -> optional FL MiniMax H3 Beat Pixel Upscale KSampler
    -> FL MiniMax H3 Shot Assembler
```

Fill Nodes is recommended for this beat-sequenced path but is not required for the manual Prompt Timeline workflow.

The scheduler owns waveform analysis, beat-grid editing, audio trimming, prompt blocks, crossfades, and render groups. The H3 planner consumes its `FL_PROMPT_SCHEDULE` output without importing or duplicating the audio implementation.

## Render groups

The scheduler can mark touching prompt sections as one render group:

- Grouped sections are encoded and rendered together so H3 can model internal cuts with shared context.
- Ungrouped sections become independent renders.
- The sampler always returns one editable nested latent per planned render.
- The assembler restores the planned timeline order after sampling or refinement.

This provides explicit control over which cuts share generative context without forcing the entire music video into one long render.

## Prompt Timeline outputs

- `scheduled` - strict temporal conditioning for the initial sampling pass
- `latent` - native nested H3 video/audio latent
- `semantic` - single semantic conditioning suitable for low-denoise refinement
- `timeline` - reusable encoded timeline for **Apply Timeline**

Use **Apply Timeline** when a spatial resize or VAE round trip changes the video latent width and height but preserves its temporal dimensions.

## Pixel upscale refinement

**FL MiniMax H3 Beat Pixel Upscale KSampler** performs this sequence for each planned render:

1. Decode the video latent through the H3 video VAE.
2. Resize pixels proportionally to the requested long side.
3. Re-encode the resized frames into H3 latent space.
4. Reapply the render's prompt timeline at the new spatial dimensions.
5. Sample with the configured low denoise.

The output remains a native nested H3 latent so it can continue through other latent-aware ComfyUI nodes before decoding.

## Key parameters

- **length** - Requested video frame count at H3's fixed 24 FPS
- **affect_audio** - Apply scheduled text masks to video only or to video and audio tokens
- **schedule_policy** - Reject, clamp, or fit schedule ranges that exceed the aligned H3 duration
- **ref_image_size** - Match the output canvas or allow H3's maximum reference area
- **target_long_side** - Pixel long side used by the upscale refinement pass; must be divisible by 32
- **denoise** - Refinement strength after the pixel resize and VAE round trip
- **seed_stride** - Deterministic seed offset between planned renders

## Migration from Fill Nodes

These nodes were originally distributed inside `ComfyUI_Fill-Nodes`. Existing workflows do not need node replacement because the node IDs and data contracts are unchanged.

To avoid duplicate registrations:

1. Update Fill Nodes to a version where the MiniMax nodes have been removed.
2. Install this pack.
3. Restart ComfyUI.

Do not combine this pack with an older Fill Nodes release that still registers the same six MiniMax node IDs.

## Requirements

- A current ComfyUI installation with MiniMax H3 support
- MiniMax H3 model, text encoder, video VAE, and audio VAE files
- Sufficient VRAM for the selected model, resolution, frame count, and reference media
- ComfyUI Fill Nodes only when using its audio beat scheduler or audio-reactive envelope ecosystem

No additional Python dependencies are installed by this pack.

## License

[Apache-2.0](LICENSE)
