from comfy_api.latest import io


class FL_MiniMaxH3ParallaxPrompts(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="FL_MiniMaxH3ParallaxPrompts", display_name="FL MiniMax H3 Parallax Prompts", category="FL/MiniMax H3/Prompting",
            description="Four locked-camera plate prompts from shared role templates. The compositor supplies camera movement.",
            inputs=[io.String.Input("scene", multiline=True, default="A peaceful Japanese lakeside village at golden hour, blue mountains, warm wooden houses and a small shrine by the water, flowering cherry branches and reeds. No people.")],
            outputs=[io.String.Output(display_name=name) for name in ("background", "midground", "near_objects", "foreground_frame")])

    @classmethod
    def execute(cls, scene):
        if not scene.strip():
            raise ValueError("Describe the scene for the parallax plates.")
        shared = (
            "\n\nHand-painted anime illustration, confident ink contours, rich painted detail and restrained cel shading. "
            "FIXED CAMERA. Stable framing and object positions, only very gentle local motion. No pan, zoom, orbit, cuts or camera movement. "
            "No text, captions, logos or contact sheets.\n\n"
            f"Scene context for selecting the objects, palette and lighting ONLY: {scene.strip()} "
            "Do not render context objects outside this plate's assigned role."
        )
        roles = (
            "DISTANT BACKGROUND PLATE: Fill the entire frame with the distant environment and atmosphere appropriate to this scene. "
            "Show the far horizon, distant terrain or architectural backdrop, sky or distant interior wall, and a continuous lower supporting surface. "
            "Leave the central view open. Exclude nearby objects, framing foliage, large foreground buildings and people. "
            "This plate must be opaque and complete edge to edge; never use a white studio backdrop for this plate.",
            "MIDGROUND CUTOUT PLATE: Choose one coherent cluster of the scene's principal mid-distance landmarks, such as its buildings, "
            "shrine, rocks or larger environmental structures. Draw only that cluster against a perfectly uniform pure white studio background. "
            "Keep the cluster in the lower-middle part of the canvas, with its base near eighty-five percent of image height. "
            "Use about sixty percent of canvas width and fifty percent of canvas height. Leave generous white space above and on both sides. "
            "No sky, horizon, full ground plane, scenery behind the objects, drop shadows on white or vignette. Preserve all object edges inside the frame.",
            "NEAR OBJECT CUTOUT PLATE: Choose a few near-camera environmental details appropriate to this world: vegetation, rocks, railings, "
            "furniture or other scene props. Place two asymmetrical groups near the lower left and lower right, separated by a broad empty center. "
            "Use the lower forty percent of the canvas. Isolate these details on perfectly uniform pure white. "
            "No background landscape, buildings behind them, horizon, full ground plane, people or shadow on the white backdrop. "
            "Do not redraw the scene's central landmark. Retain generous white margins around the groups.",
            "EXTREME FOREGROUND CUTOUT PLATE: Choose one or two large decorative framing elements from this environment, such as an overhanging "
            "branch, leaves, an awning edge or hanging details. Draw them near the upper corners, curving inward slightly. "
            "Keep at least seventy percent of the canvas empty white, especially the center and bottom. "
            "Perfectly uniform pure white backdrop; no complete scene, distant structures, ground, people or shadows on white. "
            "Keep crisp, detailed silhouettes with all tips visible inside the frame. Only very gentle local swaying."
        )
        return io.NodeOutput(*("integrated_multimodal_description: [Shot 1] " + role + shared +
                     ("\n\nOnly the isolated cutout objects on a pure white background. The white background occupies most of the image." if i else "") +
                     "\n\noverall_soundscape: Quiet natural ambience. No speech.\n\nnon_diegetic_music: N/A" for i, role in enumerate(roles)))
