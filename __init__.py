from .nodes import (
    FL_MiniMaxH3ApplyTimeline,
    FL_MiniMaxH3BeatKSampler,
    FL_MiniMaxH3BeatShotPlanner,
    FL_MiniMaxH3BeatUpscaleKSampler,
    FL_MiniMaxH3LatentUpscale,
    FL_MiniMaxH3LatentUpscaleModelLoader,
    FL_MiniMaxH3NeuralLatentUpscale2D,
    FL_MiniMaxH3NeuralLatentUpscale3D,
    FL_MiniMaxH3PromptTimeline,
    FL_MiniMaxH3ShotAssembler,
    FL_MiniMaxH3ShotMotionContext,
    FL_MiniMaxH3TemporalReshotAssembler,
    FL_MiniMaxH3TemporalReshotPlanner,
    FL_MiniMaxH3TransitionAssembler,
    FL_MiniMaxH3TransitionPrep,
    FL_MiniMaxH3VDN,
)

from . import routes as routes


NODE_CLASS_MAPPINGS = {
    "FL_MiniMaxH3PromptTimeline": FL_MiniMaxH3PromptTimeline,
    "FL_MiniMaxH3ApplyTimeline": FL_MiniMaxH3ApplyTimeline,
    "FL_MiniMaxH3BeatShotPlanner": FL_MiniMaxH3BeatShotPlanner,
    "FL_MiniMaxH3ShotMotionContext": FL_MiniMaxH3ShotMotionContext,
    "FL_MiniMaxH3BeatKSampler": FL_MiniMaxH3BeatKSampler,
    "FL_MiniMaxH3BeatUpscaleKSampler": FL_MiniMaxH3BeatUpscaleKSampler,
    "FL_MiniMaxH3LatentUpscale": FL_MiniMaxH3LatentUpscale,
    "FL_MiniMaxH3LatentUpscaleModelLoader": FL_MiniMaxH3LatentUpscaleModelLoader,
    "FL_MiniMaxH3NeuralLatentUpscale2D": FL_MiniMaxH3NeuralLatentUpscale2D,
    "FL_MiniMaxH3NeuralLatentUpscale3D": FL_MiniMaxH3NeuralLatentUpscale3D,
    "FL_MiniMaxH3ShotAssembler": FL_MiniMaxH3ShotAssembler,
    "FL_MiniMaxH3TemporalReshotPlanner": FL_MiniMaxH3TemporalReshotPlanner,
    "FL_MiniMaxH3TemporalReshotAssembler": FL_MiniMaxH3TemporalReshotAssembler,
    "FL_MiniMaxH3TransitionPrep": FL_MiniMaxH3TransitionPrep,
    "FL_MiniMaxH3TransitionAssembler": FL_MiniMaxH3TransitionAssembler,
    "FL_MiniMaxH3VDN": FL_MiniMaxH3VDN,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FL_MiniMaxH3PromptTimeline": "FL MiniMax H3 Prompt Timeline",
    "FL_MiniMaxH3ApplyTimeline": "FL MiniMax H3 Apply Timeline",
    "FL_MiniMaxH3BeatShotPlanner": "FL MiniMax H3 Beat Shot Planner",
    "FL_MiniMaxH3ShotMotionContext": "FL MiniMax H3 Shot Motion Context",
    "FL_MiniMaxH3BeatKSampler": "FL MiniMax H3 Beat KSampler",
    "FL_MiniMaxH3BeatUpscaleKSampler": "FL MiniMax H3 Beat Pixel Upscale KSampler",
    "FL_MiniMaxH3LatentUpscale": "FL MiniMax H3 Latent Upscale",
    "FL_MiniMaxH3LatentUpscaleModelLoader": "FL MiniMax H3 Load Latent Upscaler",
    "FL_MiniMaxH3NeuralLatentUpscale2D": "FL MiniMax H3 Neural Latent Upscale 2D",
    "FL_MiniMaxH3NeuralLatentUpscale3D": "FL MiniMax H3 Neural Latent Upscale 3D",
    "FL_MiniMaxH3ShotAssembler": "FL MiniMax H3 Shot Assembler",
    "FL_MiniMaxH3TemporalReshotPlanner": "FL MiniMax H3 Temporal Reshot Planner",
    "FL_MiniMaxH3TemporalReshotAssembler": "FL MiniMax H3 Temporal Reshot Assembler",
    "FL_MiniMaxH3TransitionPrep": "FL MiniMax H3 Transition Prep",
    "FL_MiniMaxH3TransitionAssembler": "FL MiniMax H3 Transition Assembler",
    "FL_MiniMaxH3VDN": "FL MiniMax H3 VDN",
}

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
