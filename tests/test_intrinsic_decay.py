from __future__ import annotations

from src.envs.autopilot_env.intrinsic import DECAY_EPISODES, IntrinsicCounter


def test_intrinsic_components_decay_to_zero():
    intrinsic = IntrinsicCounter()
    intrinsic.reset_episode()

    early = intrinsic.components(
        workflow_id="wf_intrinsic",
        completed_ids=["T1"],
        available_ids=["T2"],
        tool="jira_assign_ticket",
    )

    for _ in range(DECAY_EPISODES):
        intrinsic.reset_episode()

    late = intrinsic.components(
        workflow_id="wf_intrinsic",
        completed_ids=["T1"],
        available_ids=["T2"],
        tool="jira_assign_ticket",
    )

    assert early.count_bonus > 0.0
    assert early.rnd_bonus > 0.0
    assert late.decay_factor == 0.0
    assert late.count_bonus == 0.0
    assert late.rnd_bonus == 0.0


def test_rnd_error_drops_for_repeated_state():
    intrinsic = IntrinsicCounter()
    intrinsic.reset_episode()

    first = intrinsic.components(
        workflow_id="wf_repeat",
        completed_ids=["T1"],
        available_ids=["T2", "T3"],
        tool="email_send",
    )

    current = first
    for _ in range(12):
        current = intrinsic.components(
            workflow_id="wf_repeat",
            completed_ids=["T1"],
            available_ids=["T2", "T3"],
            tool="email_send",
        )

    assert current.rnd_error < first.rnd_error
    assert current.rnd_bonus < first.rnd_bonus


def test_intrinsic_handles_empty_state_without_crashing():
    intrinsic = IntrinsicCounter()
    result = intrinsic.components(
        workflow_id="",
        completed_ids=[],
        available_ids=[],
        tool="",
    )

    assert result.count_bonus == 0.0
    assert result.rnd_bonus == 0.0
    assert 0.0 <= result.decay_factor <= 1.0
