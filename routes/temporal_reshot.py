import asyncio

import av
from aiohttp import web
from server import PromptServer

from ..nodes.FL_MiniMaxH3TemporalReshot import (
    probe_source_video,
    resolve_video_path,
    video_library_entries,
)


async def temporal_reshot_info(request):
    try:
        path = resolve_video_path(request.query.get("filename", ""))
        info = await asyncio.to_thread(probe_source_video, path)
    except (OSError, ValueError, av.error.FFmpegError) as error:
        return web.json_response({"error": str(error)}, status=400)
    return web.json_response(info)


async def temporal_reshot_files(request):
    try:
        files = await asyncio.to_thread(video_library_entries)
    except (OSError, ValueError, av.error.FFmpegError) as error:
        return web.json_response({"error": str(error)}, status=400)
    return web.json_response({"files": files})


if getattr(PromptServer, "instance", None) is not None:
    PromptServer.instance.routes.get("/fl/minimax-h3/temporal-reshot/info")(
        temporal_reshot_info
    )
    PromptServer.instance.routes.get("/fl/minimax-h3/temporal-reshot/files")(
        temporal_reshot_files
    )
