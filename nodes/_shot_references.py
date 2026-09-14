import math


def selection(section):
    reference = section.get("references") or {"mode": "defaults", "asset_ids": []}
    mode = reference.get("mode", "defaults")
    ids = reference.get("asset_ids", [])
    if mode not in {"defaults", "custom", "none"} or not isinstance(ids, list) or any(not isinstance(value, str) for value in ids):
        raise ValueError("Invalid shot reference selection.")
    if len(ids) != len(set(ids)) or (mode != "custom" and ids):
        raise ValueError("Invalid shot reference asset IDs.")
    return mode, tuple(ids)


def resolve_shot_references(sections, library, images, videos, video_audios, audios, cache=None):
    mode, ids = selection(sections[0])
    if any(selection(section) != (mode, ids) for section in sections[1:]):
        raise ValueError("Grouped render sections must use the same ordered references. Ungroup them or match their selections.")
    if mode == "defaults":
        return images, videos, video_audios, audios, []
    if mode == "none" or not ids:
        return None, None, None, None, []
    if not library or library.get("version") != 1:
        raise ValueError("Connect FL Prompt Reference Library to use custom section references.")
    assets = library["assets"]
    selected_images, selected_videos, selected_video_audios, selected_audios = {}, {}, {}, {}
    for asset_id in ids:
        if asset_id not in assets:
            raise ValueError(f"Missing shot reference {asset_id}. Connect the library for this schedule.")
        asset = assets[asset_id]
        kind = asset["kind"]
        if kind == "image":
            selected_images[f"ref_image_{len(selected_images)}"] = asset["value"]
        elif kind == "video":
            index = len(selected_videos)
            frames = asset["value"]
            fps = asset.get("fps", 24.0)
            if not isinstance(fps, (int, float)) or not math.isfinite(fps) or fps <= 0:
                raise ValueError("Reference video has an invalid frame rate.")
            if abs(fps - 24.0) > 1e-6:
                key = ("reference_video_fps", id(frames), fps)
                converted = cache.get(key) if cache is not None else None
                if converted is None:
                    count = max(1, round(frames.shape[0] * 24 / fps))
                    converted = frames[[min(frames.shape[0]-1, round(frame * fps / 24)) for frame in range(count)]]
                    if cache is not None:
                        cache[key] = converted
                frames = converted
            selected_videos[f"ref_video_{index}"] = frames
            if asset.get("audio") is not None:
                selected_video_audios[f"ref_video_audio_{index}"] = asset["audio"]
        elif kind == "audio":
            selected_audios[f"ref_audio_{len(selected_audios)}"] = asset["value"]
        else:
            raise ValueError(f"Unsupported shot reference kind: {kind}")
    return selected_images, selected_videos, selected_video_audios, selected_audios, list(ids)
