import importlib.util
import pathlib
import sys
import unittest


ROOT = pathlib.Path(__file__).parents[1]
PACKAGE_NAME = "fl_minimax_h3_registration_tests"
SPEC = importlib.util.spec_from_file_location(
    PACKAGE_NAME,
    ROOT / "__init__.py",
    submodule_search_locations=[str(ROOT)],
)
package = importlib.util.module_from_spec(SPEC)
sys.modules[PACKAGE_NAME] = package
SPEC.loader.exec_module(package)


EXPECTED_NODES = {
    "FL_MiniMaxH3PromptTimeline",
    "FL_MiniMaxH3ApplyTimeline",
    "FL_MiniMaxH3BeatShotPlanner",
    "FL_MiniMaxH3ShotMotionContext",
    "FL_MiniMaxH3BeatKSampler",
    "FL_MiniMaxH3BeatUpscaleKSampler",
    "FL_MiniMaxH3LatentUpscale",
    "FL_MiniMaxH3LatentUpscaleModelLoader",
    "FL_MiniMaxH3NeuralLatentUpscale2D",
    "FL_MiniMaxH3NeuralLatentUpscale3D",
    "FL_MiniMaxH3ShotAssembler",
    "FL_MiniMaxH3TemporalReshotPlanner",
    "FL_MiniMaxH3TemporalReshotAssembler",
    "FL_MiniMaxH3TransitionPrep",
    "FL_MiniMaxH3TransitionAssembler",
    "FL_MiniMaxH3VDN",
}


class RegistrationTests(unittest.TestCase):
    def test_pack_registers_only_the_minimax_nodes(self):
        self.assertEqual(set(package.NODE_CLASS_MAPPINGS), EXPECTED_NODES)
        self.assertEqual(set(package.NODE_DISPLAY_NAME_MAPPINGS), EXPECTED_NODES)

    def test_node_ids_and_categories_are_pack_owned(self):
        for node_id, node_class in package.NODE_CLASS_MAPPINGS.items():
            schema = node_class.define_schema()
            self.assertEqual(schema.node_id, node_id)
            self.assertTrue(schema.category.startswith("FL/MiniMax H3/"))

    def test_pack_serves_the_web_extensions(self):
        self.assertEqual(package.WEB_DIRECTORY, "./web")
        self.assertTrue((ROOT / "web" / "FL_MiniMaxH3BeatSamplerPreview.js").is_file())
        self.assertTrue((ROOT / "web" / "FL_MiniMaxH3TemporalReshot.js").is_file())
        self.assertTrue((ROOT / "web" / "FL_MiniMaxH3TemporalReshotMath.js").is_file())
        self.assertTrue((ROOT / "web" / "FL_MiniMaxH3TemporalReshotModal.js").is_file())
        self.assertTrue((ROOT / "web" / "FL_MiniMaxH3TemporalReshotEditor.js").is_file())
        self.assertTrue((ROOT / "web" / "FL_MiniMaxH3TemporalReshotState.js").is_file())
        self.assertTrue((ROOT / "web" / "FL_MiniMaxH3TemporalReshotStyles.js").is_file())
        self.assertTrue((ROOT / "web" / "FL_MiniMaxH3Transition.js").is_file())
        self.assertTrue((ROOT / "web" / "FL_MiniMaxH3TransitionMath.js").is_file())


if __name__ == "__main__":
    unittest.main()
