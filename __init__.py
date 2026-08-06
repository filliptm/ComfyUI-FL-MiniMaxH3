from .nodes import (
    FL_MiniMaxH3ApplyTimeline,
    FL_MiniMaxH3BeatKSampler,
    FL_MiniMaxH3BeatShotPlanner,
    FL_MiniMaxH3BeatUpscaleKSampler,
    FL_MiniMaxH3PromptTimeline,
    FL_MiniMaxH3ShotAssembler,
)


NODE_CLASS_MAPPINGS = {
    "FL_MiniMaxH3PromptTimeline": FL_MiniMaxH3PromptTimeline,
    "FL_MiniMaxH3ApplyTimeline": FL_MiniMaxH3ApplyTimeline,
    "FL_MiniMaxH3BeatShotPlanner": FL_MiniMaxH3BeatShotPlanner,
    "FL_MiniMaxH3BeatKSampler": FL_MiniMaxH3BeatKSampler,
    "FL_MiniMaxH3BeatUpscaleKSampler": FL_MiniMaxH3BeatUpscaleKSampler,
    "FL_MiniMaxH3ShotAssembler": FL_MiniMaxH3ShotAssembler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FL_MiniMaxH3PromptTimeline": "FL MiniMax H3 Prompt Timeline",
    "FL_MiniMaxH3ApplyTimeline": "FL MiniMax H3 Apply Timeline",
    "FL_MiniMaxH3BeatShotPlanner": "FL MiniMax H3 Beat Shot Planner",
    "FL_MiniMaxH3BeatKSampler": "FL MiniMax H3 Beat KSampler",
    "FL_MiniMaxH3BeatUpscaleKSampler": "FL MiniMax H3 Beat Pixel Upscale KSampler",
    "FL_MiniMaxH3ShotAssembler": "FL MiniMax H3 Shot Assembler",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
