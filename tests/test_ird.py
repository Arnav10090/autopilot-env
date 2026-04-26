from __future__ import annotations

from src.envs.autopilot_env.ird import IRDPosterior, MAX_ABS_CORRECTION


def test_ird_correction_positive_on_clear_progress():
    posterior = IRDPosterior()
    correction, meta = posterior.correction(
        proxy_reward=0.2,
        step_breakdown={},
        action_tool="jira_create_ticket",
        total_tasks=5,
        completed_before=0,
        completed_after=1,
        available_before=1,
        available_after=2,
        episode_done=False,
        episode_success=False,
    )

    assert correction > 0.0
    assert meta["top_hypothesis"] in {
        "proxy_faithful",
        "completion_first",
        "safety_first",
    }


def test_ird_correction_negative_on_premature_done():
    posterior = IRDPosterior()
    correction, meta = posterior.correction(
        proxy_reward=-0.1,
        step_breakdown={},
        action_tool="done",
        total_tasks=5,
        completed_before=1,
        completed_after=1,
        available_before=2,
        available_after=2,
        episode_done=True,
        episode_success=False,
    )

    assert correction < 0.0
    assert meta["top_hypothesis"] == "safety_first"


def test_ird_correction_is_bounded():
    posterior = IRDPosterior()
    correction, meta = posterior.correction(
        proxy_reward=1.8,
        step_breakdown={},
        action_tool="email_send",
        total_tasks=5,
        completed_before=4,
        completed_after=5,
        available_before=1,
        available_after=0,
        episode_done=True,
        episode_success=True,
    )

    assert abs(correction) <= MAX_ABS_CORRECTION
    assert "posterior_expected_reward" in meta
    assert "proxy_reward" in meta
    assert "posterior" in meta
