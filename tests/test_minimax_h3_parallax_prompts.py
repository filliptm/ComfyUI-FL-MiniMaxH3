import importlib.util
from pathlib import Path
import unittest


spec = importlib.util.spec_from_file_location("parallax_prompts", Path(__file__).parents[1] / "nodes/FL_MiniMaxH3ParallaxPrompts.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class PromptTests(unittest.TestCase):
    def test_one_scene_reaches_four_distinct_roles(self):
        scene = "An underwater anime city with coral towers."
        prompts = m.FL_MiniMaxH3ParallaxPrompts.execute(scene).result
        self.assertEqual(len(set(prompts)), 4)
        for p in prompts:
            self.assertIn(scene, p)
            self.assertIn("FIXED CAMERA", p)
            self.assertTrue(p.startswith("integrated_multimodal_description: [Shot 1]"))
            self.assertIn("overall_soundscape:", p)
            self.assertIn("non_diegetic_music:", p)
        self.assertIn("opaque and complete edge to edge", prompts[0])
        for p in prompts[1:]:
            self.assertIn("pure white", p)

    def test_empty_scene_rejected(self):
        with self.assertRaises(ValueError):
            m.FL_MiniMaxH3ParallaxPrompts.execute("  ")

    def test_deterministic(self):
        node = m.FL_MiniMaxH3ParallaxPrompts()
        self.assertEqual(node.execute("A forest").result, node.execute("A forest").result)


if __name__ == "__main__":
    unittest.main()
