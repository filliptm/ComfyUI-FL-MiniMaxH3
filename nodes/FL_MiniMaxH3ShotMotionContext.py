import logging
import math
import re

from comfy_api.latest import io
from comfy_extras import nodes_minimax_h3 as minimax_h3

from ._motion_context import VIDEO_CONTEXT_FRAMES, VIDEO_CONTEXT_STEPS
from .FL_MiniMaxH3PromptTimeline import (
    _align_h3_sections,
    _apply_timeline,
    _envelope_at_time,
    _h3_tensors,
)


H3ShotPlan = io.Custom("FL_H3_SHOT_PLAN")
_TIME_FIELDS = (
    "start",
    "end",
    "fade_in_end",
    "fade_out_start",
    "crossfade_start",
    "crossfade_end",
)
_OVERRIDE = re.compile(r"^(\d+)\s*:\s*(0|1|5|22|39)\s*,\s*(\d+)\s*$")


def _validate_plan(plan):
    if not isinstance(plan, dict) or plan.get("type") != "minimax_h3_beat_shot_plan":
        raise TypeError("FL MiniMax H3 Shot Motion Context received an invalid shot plan.")
    if plan.get("version") != 1:
        raise ValueError("FL MiniMax H3 Shot Motion Context supports shot plan version 1.")
    shots = plan.get("shots")
    if not isinstance(shots, list) or not shots:
        raise ValueError("FL MiniMax H3 Shot Motion Context requires at least one planned render.")
    return shots


def _parse_overrides(value, shot_count):
    overrides = {}
    for line_number, original in enumerate((value or "").splitlines(), 1):
        line = original.partition("#")[0].strip()
        if not line:
            continue
        match = _OVERRIDE.fullmatch(line)
        if match is None:
            raise ValueError(
                "FL MiniMax H3 Shot Motion Context override line "
                f"{line_number} must use 'render: video_frames, audio_frames'."
            )
        render, video_frames, audio_frames = map(int, match.groups())
        if render < 2 or render > shot_count:
            raise ValueError(
                "FL MiniMax H3 Shot Motion Context override line "
                f"{line_number} names render {render}; valid destination renders are 2-{shot_count}."
            )
        if audio_frames > 240:
            raise ValueError(
                "FL MiniMax H3 Shot Motion Context override line "
                f"{line_number} audio context must be between 0 and 240 frames."
            )
        if render in overrides:
            raise ValueError(
                f"FL MiniMax H3 Shot Motion Context repeats render {render} in its overrides."
            )
        overrides[render] = (video_frames, audio_frames)
    return overrides


def _shift_section(section, seconds):
    resolved = dict(section)
    for field in _TIME_FIELDS:
        if field in resolved:
            resolved[field] += seconds
    return resolved


def _context_section(first, video_frames):
    end = video_frames / minimax_h3.FPS
    return {
        **first,
        "start": 0.0,
        "end": end,
        "fade_in_end": 0.0,
        "fade_out_start": end,
        "crossfade_start": 0.0,
        "crossfade_end": 0.0,
        "motion_context": True,
    }


def _shift_prompt_envelopes(envelopes, offset_frames, render_frames):
    offset = offset_frames / minimax_h3.FPS
    duration = render_frames / minimax_h3.FPS
    resolved = []
    for envelope in envelopes:
        count = max(1, math.ceil(duration * envelope["fps"]))
        weights = []
        for position in range(count):
            source_time = (position + 0.5) / envelope["fps"] - offset
            weights.append(0.0 if source_time < 0.0 else _envelope_at_time(envelope, source_time))
        resolved.append({
            **envelope,
            "weights": weights,
            "duration": duration,
        })
    return resolved


def _shift_conditioning_groups(groups):
    resolved = []
    for group in groups:
        indices = [index + 1 for index in group["section_indices"]]
        if 0 in group["section_indices"]:
            indices.insert(0, 0)
        resolved.append({**group, "section_indices": indices})
    return resolved


def _extend_shot(shot, width, height, video_frames):
    authored_frames = shot["authored_frames"]
    latent, render_frames = minimax_h3._empty_av_latent(
        width,
        height,
        authored_frames + video_frames,
    )
    video, audio = _h3_tensors(latent)
    source_timeline = shot.get("timeline")
    if not isinstance(source_timeline, dict):
        raise ValueError(
            "FL MiniMax H3 Shot Motion Context requires a plan from the current Beat Shot Planner."
        )
    authored_sections = source_timeline.get("authored_sections")
    if not isinstance(authored_sections, list) or not authored_sections:
        raise ValueError("FL MiniMax H3 Shot Motion Context received an incomplete prompt timeline.")

    offset = video_frames / minimax_h3.FPS
    shifted_authored = [_shift_section(section, offset) for section in authored_sections]
    sections = [_context_section(authored_sections[0], video_frames), *shifted_authored]
    authored_with_context = authored_frames + video_frames
    sections, adjustments, extended_final = _align_h3_sections(
        sections,
        authored_with_context,
        render_frames,
        video.shape[2],
        "hard",
    )
    timeline = {
        **source_timeline,
        "frame_count": render_frames,
        "authored_frame_count": authored_with_context,
        "padding_frames": render_frames - authored_with_context,
        "video_t": video.shape[2],
        "audio_t": audio.shape[-1],
        "duration": render_frames / minimax_h3.FPS,
        "authored_sections": [
            _context_section(authored_sections[0], video_frames),
            *shifted_authored,
        ],
        "sections": sections,
        "boundary_adjustments": adjustments,
        "extended_final_section": extended_final,
        "conditioning_groups": _shift_conditioning_groups(
            source_timeline["conditioning_groups"]
        ),
        "prompt_envelopes": _shift_prompt_envelopes(
            source_timeline.get("prompt_envelopes", []),
            video_frames,
            render_frames,
        ),
    }
    return {
        **shot,
        "render_frames": render_frames,
        "padding_frames": render_frames - authored_with_context,
        "timeline": timeline,
        "conditioning": _apply_timeline(timeline, latent),
        "latent": latent,
    }, adjustments


class FL_MiniMaxH3ShotMotionContext(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="FL_MiniMaxH3ShotMotionContext",
            display_name="FL MiniMax H3 Shot Motion Context",
            category="FL/MiniMax H3/Prompting",
            description=(
                "Carries a configurable tail of each completed render into the next render while "
                "keeping the authored result as a hard cut."
            ),
            inputs=[
                H3ShotPlan.Input(
                    "shot_plan",
                    tooltip="Shot plan from FL MiniMax H3 Beat Shot Planner.",
                ),
                io.Combo.Input(
                    "video_context_frames",
                    options=[str(value) for value in VIDEO_CONTEXT_FRAMES],
                    default="5",
                    tooltip=(
                        "Prior authored video frames conditioned into each next render. Values match "
                        "native H3 temporal windows."
                    ),
                ),
                io.Int.Input(
                    "audio_context_frames",
                    default=22,
                    min=0,
                    max=240,
                    tooltip="Prior authored audio duration measured on the 24 fps video timeline.",
                ),
                io.String.Input(
                    "per_render_overrides",
                    default="",
                    multiline=True,
                    tooltip=(
                        "Optional 1-based destination overrides, one per line: 3: 22, 39. "
                        "The first number is render, followed by video and audio context frames."
                    ),
                ),
            ],
            outputs=[
                H3ShotPlan.Output(
                    display_name="shot_plan",
                    tooltip="Motion-context shot plan for the Beat KSampler and all upscale passes.",
                )
            ],
        )

    @classmethod
    def execute(
        cls,
        shot_plan,
        video_context_frames,
        audio_context_frames,
        per_render_overrides="",
    ):
        shots = _validate_plan(shot_plan)
        video_context_frames = int(video_context_frames)
        if video_context_frames not in VIDEO_CONTEXT_FRAMES:
            raise ValueError(
                "FL MiniMax H3 Shot Motion Context video context must be 0, 1, 5, 22, or 39 frames."
            )
        if (
            isinstance(audio_context_frames, bool)
            or not isinstance(audio_context_frames, int)
            or not 0 <= audio_context_frames <= 240
        ):
            raise ValueError(
                "FL MiniMax H3 Shot Motion Context audio context must be between 0 and 240 frames."
            )
        overrides = _parse_overrides(per_render_overrides, len(shots))
        if not overrides and video_context_frames == 0 and audio_context_frames == 0:
            return io.NodeOutput(shot_plan)

        resolved_shots = []
        total_render_frames = 0
        adjustment_count = 0
        for index, source in enumerate(shots):
            if index == 0:
                video_frames, audio_frames = 0, 0
            else:
                video_frames, audio_frames = overrides.get(
                    index + 1,
                    (video_context_frames, audio_context_frames),
                )
                previous_length = shots[index - 1].get("authored_frames")
                if not isinstance(previous_length, int) or previous_length <= 0:
                    raise ValueError(
                        f"FL MiniMax H3 Shot Motion Context source render {index} has no authored length."
                    )
                if video_frames > previous_length:
                    smaller = max(value for value in VIDEO_CONTEXT_FRAMES if value <= previous_length)
                    raise ValueError(
                        "FL MiniMax H3 Shot Motion Context render "
                        f"{index + 1} requests {video_frames} video frames from source render {index}, "
                        f"which has {previous_length}; use {smaller} or less."
                    )
                if audio_frames > previous_length:
                    raise ValueError(
                        "FL MiniMax H3 Shot Motion Context render "
                        f"{index + 1} requests {audio_frames} audio frames from source render {index}, "
                        f"which has {previous_length}; use {previous_length} or less."
                    )

            shot = dict(source)
            adjustments = []
            if video_frames:
                shot, adjustments = _extend_shot(
                    shot,
                    shot_plan["width"],
                    shot_plan["height"],
                    video_frames,
                )
            shot["motion_context"] = {
                "version": 1,
                "source_index": index - 1 if index else None,
                "video_frames": video_frames,
                "video_steps": VIDEO_CONTEXT_STEPS[video_frames],
                "audio_frames": audio_frames,
                "trim_frames": video_frames,
            }
            resolved_shots.append(shot)
            total_render_frames += shot["render_frames"]
            adjustment_count += len(adjustments)

        plan = {
            **shot_plan,
            "total_render_frames": total_render_frames,
            "shots": resolved_shots,
            "motion_context": {
                "version": 1,
                "video_context_frames": video_context_frames,
                "audio_context_frames": audio_context_frames,
                "overrides": overrides,
            },
        }
        logging.info(
            "FL MiniMax H3 shot motion context: %d renders, %d render frames, "
            "%d shifted prompt boundaries.",
            len(resolved_shots),
            total_render_frames,
            adjustment_count,
        )
        return io.NodeOutput(plan)
