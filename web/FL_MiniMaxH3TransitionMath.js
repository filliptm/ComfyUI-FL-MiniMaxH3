export const H3_FRAME_STEP = 17;
export const H3_FRAME_OFFSET = 5;
const H3_TOKEN_FRAME_PATTERN = [1, 4, 4, 4, 4];

function h3TokenIndex(frame) {
  let edge = 0;
  let token = 0;
  while (edge < frame) {
    edge += H3_TOKEN_FRAME_PATTERN[token % H3_TOKEN_FRAME_PATTERN.length];
    token += 1;
  }
  return edge === frame ? token : -1;
}

export function alignH3Frames(value) {
  let frames = Math.max(22, Math.round(Number(value) || 22));
  while (frames % H3_FRAME_STEP !== H3_FRAME_OFFSET) frames += 1;
  return frames;
}

export function referenceFrameOptions(length) {
  const frames = alignH3Frames(length);
  const options = [];
  for (let value = H3_FRAME_OFFSET; value * 2 < frames; value += H3_FRAME_STEP) {
    options.push(value);
  }
  return options;
}

export function normalizeTransitionSettings(length, referenceFrames) {
  const frames = alignH3Frames(length);
  const options = referenceFrameOptions(frames);
  let reference = Number(referenceFrames);
  if (!options.includes(reference)) {
    reference = options.includes(22) ? 22 : options[options.length - 1];
  }
  const generatedFrames = frames - 2 * reference;
  const sourceAEditFrames = Math.floor(generatedFrames / 2);
  const tokenStart = h3TokenIndex(reference);
  const tokenEnd = h3TokenIndex(frames - reference);
  return {
    length: frames,
    referenceFrames: reference,
    generatedFrames,
    sourceAEditFrames,
    sourceBEditFrames: generatedFrames - sourceAEditFrames,
    maxMaskFeatherTokens: Math.max(0, Math.floor((tokenEnd - tokenStart - 1) / 2)),
    duration: frames / 24,
    options,
  };
}
