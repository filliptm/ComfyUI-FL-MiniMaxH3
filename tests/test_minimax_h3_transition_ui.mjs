import assert from "node:assert/strict";
import test from "node:test";

import {
  alignH3Frames,
  normalizeTransitionSettings,
  referenceFrameOptions,
} from "../web/FL_MiniMaxH3TransitionMath.js";


test("transition length snaps to the H3 frame grid", () => {
  assert.equal(alignH3Frames(89), 90);
  assert.equal(alignH3Frames(90), 90);
  assert.equal(alignH3Frames(91), 107);
});

test("reference choices always leave a generated middle", () => {
  assert.deepEqual(referenceFrameOptions(90), [5, 22, 39]);
  assert.deepEqual(referenceFrameOptions(22), [5]);
});

test("timeline summary derives overlap without frame indices", () => {
  const settings = normalizeTransitionSettings(90, 22);
  assert.equal(settings.referenceFrames, 22);
  assert.equal(settings.generatedFrames, 46);
  assert.equal(settings.sourceAEditFrames, 23);
  assert.equal(settings.sourceBEditFrames, 23);
  assert.equal(settings.maxMaskFeatherTokens, 6);
  assert.equal(settings.duration, 3.75);
});

test("short seam repairs clamp feathering to leave a full center token", () => {
  assert.equal(normalizeTransitionSettings(56, 22).maxMaskFeatherTokens, 1);
});
