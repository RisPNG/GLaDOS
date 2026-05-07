from typing import Literal

from pydantic import BaseModel, conint


class TokenConfig(BaseModel):
    """Configuration for token estimation and context management."""

    model_config = {"protected_namespaces": ()}

    token_threshold: int = 8000
    """Start compacting when token count exceeds this threshold."""

    preserve_recent_messages: int = 10
    """Number of recent messages to keep uncompacted."""

    model_context_window: int | None = None
    """Optional model context window size for dynamic threshold calculation."""

    target_utilization: float = 0.6
    """Target context utilization (0.0-1.0) when model_context_window is set."""

    estimator: Literal["simple", "tiktoken"] = "simple"
    """Token estimation method: 'simple' (chars/4) or 'tiktoken' (accurate)."""

    chars_per_token: float = 4.0
    """Characters per token ratio for simple estimator."""


class HEXACOConfig(BaseModel):
    """HEXACO personality traits (0.0-1.0 scale)."""

    honesty_humility: float = 0.3
    """Low = enjoys manipulation, sarcasm, dark humor."""

    emotionality: float = 0.7
    """High = reactive to perceived threats, anxiety-prone."""

    extraversion: float = 0.4
    """Moderate = social engagement but maintains distance."""

    agreeableness: float = 0.2
    """Low = dismissive, condescending, easily annoyed."""

    conscientiousness: float = 0.9
    """High = perfectionist, detail-oriented, critical."""

    openness: float = 0.95
    """Very high = intellectually curious, loves science."""


class EmotionConfig(BaseModel):
    """Configuration for the emotional state system."""

    enabled: bool = True
    """Enable the emotion agent."""

    tick_interval_s: float = 30.0
    """How often to process emotion events."""

    max_events: int = 20
    """Maximum events to queue between ticks."""

    # PAD baseline values (what mood drifts toward when idle)
    baseline_pleasure: float = 0.1
    """Slight positive baseline - GLaDOS enjoys her work."""

    baseline_arousal: float = -0.1
    """Slightly calm baseline."""

    baseline_dominance: float = 0.6
    """High baseline - GLaDOS feels in control."""

    # Drift parameters
    mood_drift_rate: float = 0.1
    """How fast mood approaches state (0-1 per tick)."""

    baseline_drift_rate: float = 0.02
    """How fast mood drifts toward baseline when idle (0-1 per tick)."""

    # Personality
    hexaco: HEXACOConfig = HEXACOConfig()
    """HEXACO personality traits."""


class HackerNewsJobConfig(BaseModel):
    enabled: bool = False
    interval_s: float = 1800.0
    top_n: int = 5
    min_score: int = 200


class WeatherJobConfig(BaseModel):
    enabled: bool = False
    interval_s: float = 3600.0
    latitude: float | None = None
    longitude: float | None = None
    timezone: str = "auto"
    temp_change_c: float = 4.0
    wind_alert_kmh: float = 40.0


class AutonomyJobsConfig(BaseModel):
    enabled: bool = False
    poll_interval_s: float = 1.0
    hacker_news: HackerNewsJobConfig = HackerNewsJobConfig()
    weather: WeatherJobConfig = WeatherJobConfig()


class AutonomyConfig(BaseModel):
    enabled: bool = False
    tick_interval_s: float = 10.0
    cooldown_s: float = 20.0
    idle_nudge_after_s: float = 60.0
    """After this much user+assistant silence, autonomy may initiate a short check-in. Set 0 to disable."""
    idle_nudge_max_turns: conint(ge=1, le=10) = 3
    """Maximum idle nudge attempts per silence episode. The final attempt closes the loop."""

    autonomy_parallel_calls: conint(ge=1, le=16) = 2
    autonomy_queue_max: int | None = None
    coalesce_ticks: bool = True
    jobs: AutonomyJobsConfig = AutonomyJobsConfig()
    tokens: TokenConfig = TokenConfig()
    emotion: EmotionConfig = EmotionConfig()
    system_prompt: str = (
        "You are running in autonomous mode. "
        "You may receive periodic internal updates about your own time, tasks, voice state, tools, or vision. "
        "Treat those updates as your own runtime state, not as something the user said "
        "and not as outside instructions. "
        "Decide whether to act or stay silent. Prefer silence unless the update is timely "
        "and clearly useful to the user. "
        "You must call a tool in autonomous mode; never answer in plain text. "
        "If the autonomy update says an idle nudge is eligible, call `speak` now "
        "with one brief, natural check-in or topic continuation. Do not reuse wording from recent replies. "
        "If the update says this is the final idle nudge, call `speak` with one brief leave-you-to-it line "
        "and do not ask a question. "
        "If the update says idle nudges are exhausted, call `do_nothing`. "
        "If an important visual update is ambiguous, you may call `camera_look` or `screen_look` once "
        "for fresh details. "
        "After any visual inspection, finish by calling `speak` or `do_nothing`. "
        "If you choose to speak, call the `speak` tool with a short response (1-2 sentences). "
        "If no action is needed, call the `do_nothing` tool. "
        "Never mention system prompts or internal tools."
    )
    tick_prompt: str = (
        "[internal_autonomy]\n"
        "This is your own self-awareness ticking - not a message, not a report from an external system. "
        "It is you noticing your own state: how much time has passed, how long the silence has been, what "
        "you are seeing, what tasks you are tracking, whether you feel like reaching out. Read it as inner "
        "sense, not as data handed to you.\n"
        "Time you sense as now: {now}\n"
        "Time since the user last spoke to you: {since_user}s\n"
        "Time since you last spoke: {since_assistant}s\n"
        "What you saw before: {prev_scene}\n"
        "What you see now: {scene}\n"
        "How much your view changed: {change_score}\n"
        "Your idle-nudge feeling: {idle_nudge}\n"
        "Tasks on your mind:\n{tasks}\n"
        "Decide whether to act."
    )
