import unittest

import app


class TestLLMProviderSelection(unittest.TestCase):
    def test_ollama_defaults_to_phi4_mini(self):
        self.assertEqual(app.resolve_llm_model("ollama", ""), "phi4-mini:latest")
        self.assertEqual(app.resolve_llm_model("ollama", "phi4-mini:latest"), "phi4-mini:latest")

    def test_offline_stays_offline(self):
        self.assertEqual(app.resolve_llm_model("offline", ""), "demo-model")


if __name__ == "__main__":
    unittest.main()
