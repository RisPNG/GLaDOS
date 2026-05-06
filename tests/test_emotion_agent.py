from collections import deque
import importlib.util
from pathlib import Path
import sys
import threading
import types

from glados.autonomy.config import EmotionConfig
from glados.autonomy.emotion_state import EmotionState
from glados.autonomy.llm_client import LLMConfig

_src_path = Path(__file__).parent.parent / "src"
_agents_package = types.ModuleType("glados.autonomy.agents")
_agents_package.__path__ = [str(_src_path / "glados" / "autonomy" / "agents")]
sys.modules.setdefault("glados.autonomy.agents", _agents_package)
_spec = importlib.util.spec_from_file_location(
    "glados.autonomy.agents.emotion_agent",
    _src_path / "glados" / "autonomy" / "agents" / "emotion_agent.py",
)
_emotion_agent_module = importlib.util.module_from_spec(_spec)
sys.modules["glados.autonomy.agents.emotion_agent"] = _emotion_agent_module
_spec.loader.exec_module(_emotion_agent_module)
EmotionAgent = _emotion_agent_module.EmotionAgent


def test_emotion_agent_does_not_call_llm_when_idle(mocker) -> None:
    agent = EmotionAgent.__new__(EmotionAgent)
    agent._llm_config = LLMConfig(url="http://localhost:12330/v1/chat/completions", model="test")
    agent._emotion_config = EmotionConfig()
    agent._state = EmotionState(pleasure=0.1, arousal=-0.1, dominance=0.6)
    agent._events = deque()
    agent._events_lock = threading.Lock()
    agent._save_state = lambda: None
    llm_call = mocker.patch("glados.autonomy.agents.emotion_agent.llm_call")

    output = agent.tick()

    llm_call.assert_not_called()
    assert output is not None
    assert output.status == "idle"
    assert output.summary.startswith("[emotion]")


def test_emotion_agent_extracts_json_from_messy_response() -> None:
    response = """
Here is the update:
```json
{"pleasure": 0.2, "arousal": -0.1, "dominance": 0.7, "mood_pleasure": 0.1, "mood_arousal": -0.1, "mood_dominance": 0.6}
```
"""

    data = EmotionAgent._extract_state_json(response)

    assert data is not None
    assert data["pleasure"] == 0.2
    assert data["dominance"] == 0.7


def test_emotion_agent_ignores_non_json_response() -> None:
    assert EmotionAgent._extract_state_json("I feel emotionally adequate.") is None
