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
    "FL_MiniMaxH3ShotAssembler",
}


class RegistrationTests(unittest.TestCase):
    def test_pack_registers_only_the_eight_minimax_nodes(self):
        self.assertEqual(set(package.NODE_CLASS_MAPPINGS), EXPECTED_NODES)
        self.assertEqual(set(package.NODE_DISPLAY_NAME_MAPPINGS), EXPECTED_NODES)

    def test_node_ids_and_categories_are_pack_owned(self):
        for node_id, node_class in package.NODE_CLASS_MAPPINGS.items():
            schema = node_class.define_schema()
            self.assertEqual(schema.node_id, node_id)
            self.assertTrue(schema.category.startswith("FL/MiniMax H3/"))

    def test_pack_serves_the_sampler_preview_extension(self):
        self.assertEqual(package.WEB_DIRECTORY, "./web")
        self.assertTrue((ROOT / "web" / "FL_MiniMaxH3BeatSamplerPreview.js").is_file())


if __name__ == "__main__":
    unittest.main()
