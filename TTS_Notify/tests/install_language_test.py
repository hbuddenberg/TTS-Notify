import unittest
from unittest.mock import MagicMock, patch, AsyncMock
import sys
import asyncio
from pathlib import Path

# Add src to path to import modules
sys.path.append(str(Path(__file__).parent.parent / "src"))

from tts_notify.ui.cli.main import TTSNotifyCLI
from tts_notify.core.coqui_engine import CoquiTTSEngine

class TestLanguageInstallation(unittest.TestCase):
    def setUp(self):
        self.cli = TTSNotifyCLI()
        
    @patch('tts_notify.ui.cli.main.coqui_installer')
    @patch('tts_notify.ui.cli.main.engine_registry')
    @patch('builtins.input')
    @patch('builtins.print')
    def test_interactive_language_selection(self, mock_print, mock_input, mock_registry, mock_installer):
        # Setup mocks
        mock_installer.install_coqui_tts = AsyncMock(return_value=MagicMock(success=True))
        
        # Mock Coqui Engine
        mock_engine = MagicMock(spec=CoquiTTSEngine)
        mock_engine.get_single_language_models.return_value = {
            "es": ["tts_models/es/css10/vits", "tts_models/esu/fairseq/vits"],
            "fr": ["tts_models/fr/css10/vits"]
        }
        mock_engine.download_model = AsyncMock(return_value=True)
        mock_registry.get.return_value = mock_engine
        
        # Simulate user input: 
        # 1. "2" to select "Specific Language"
        # 2. "1" to select "es" (sorted keys: es, fr)
        mock_input.side_effect = ["2", "1"]
        
        # Run installation
        asyncio.run(self.cli.install_coqui_tts())
        
        # Verify download_model was called with the correct model
        # The code selects the first model in the list for the language
        expected_model = "tts_models/es/css10/vits"
        mock_engine.download_model.assert_called_with(model_name=expected_model, force=True)
        
    @patch('tts_notify.ui.cli.main.coqui_installer')
    @patch('tts_notify.ui.cli.main.engine_registry')
    @patch('builtins.input')
    @patch('builtins.print')
    def test_interactive_language_selection_fallback(self, mock_print, mock_input, mock_registry, mock_installer):
        # Setup mocks for fallback scenario
        mock_installer.install_coqui_tts = AsyncMock(return_value=MagicMock(success=True))
        
        mock_engine = MagicMock(spec=CoquiTTSEngine)
        mock_engine.get_single_language_models.return_value = {
            "es": ["tts_models/es/css10/vits", "tts_models/esu/fairseq/vits"]
        }
        
        # First download fails, second succeeds
        mock_engine.download_model = AsyncMock(side_effect=[False, True])
        mock_registry.get.return_value = mock_engine
        
        mock_input.side_effect = ["2", "1"]
        
        asyncio.run(self.cli.install_coqui_tts())
        
        # Verify download_model was called twice
        self.assertEqual(mock_engine.download_model.call_count, 2)
        # First call with first model
        mock_engine.download_model.assert_any_call(model_name="tts_models/es/css10/vits", force=True)
        # Second call with second model
        mock_engine.download_model.assert_any_call(model_name="tts_models/esu/fairseq/vits", force=True)

if __name__ == '__main__':
    unittest.main()
