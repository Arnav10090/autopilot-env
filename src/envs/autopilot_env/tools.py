"""
Mock Enterprise Tool APIs.

Each tool simulates a real enterprise system (Jira, Slack, HR, Email, Calendar).
Tools are deterministic — same params always produce the same result.
State is held in memory per episode and reset between episodes.

Tools available:
    jira_create_ticket      Create a new Jira issue.
    jira_update_ticket      Update a field on an existing ticket.
    jira_assign_ticket      Assign a ticket to a user.
    slack_send_message      Post a message to a Slack channel.
    slack_create_channel    Create a new Slack channel.
    email_send              Send an email.
    hr_create_user          Create a new HR record.
    hr_update_user          Update an HR record field.
    calendar_create_event   Create a calendar event.
"""

from __future__ import annotations
from typing import Any, Dict, Optional


class ToolResult:
    """Standardised return from every tool call."""

    def __init__(self, success: bool, result: Optional[Dict], error: str = ""):
        self.success = success
        self.result = result or {}
        self.error = error

    def to_dict(self) -> Dict[str, Any]:
        return {"success": self.success, "result": self.result, "error": self.error}


class MockToolRegistry:
    """
    Stateful in-memory registry of all enterprise tools.
    Call reset() between episodes to wipe all created records.
    """

    VALID_TOOLS = [
        "jira_create_ticket", "jira_update_ticket", "jira_assign_ticket",
        "slack_send_message", "slack_create_channel",
        "email_send",
        "hr_create_user", "hr_update_user",
        "calendar_create_event",
    ]

    def __init__(self):
        self.reset()

    def reset(self):
        self._jira_tickets: Dict[str, Dict] = {}
        self._slack_channels: Dict[str, Dict] = {}
        self._slack_messages: list = []
        self._hr_users: Dict[str, Dict] = {}
        self._calendar_events: Dict[str, Dict] = {}
        self._emails: list = []
        self._call_log: list = []

    # ── Public API ────────────────────────────────────────────────────

    def call(self, tool: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Dispatch a tool call and return a standardised result dict."""
        self._call_log.append({"tool": tool, "params": params})
        if tool not in self.VALID_TOOLS:
            return ToolResult(False, None, f"Unknown tool: {tool!r}").to_dict()
        handler = getattr(self, f"_tool_{tool}")
        try:
            return handler(params).to_dict()
        except Exception as exc:
            return ToolResult(False, None, str(exc)).to_dict()

    def summary(self) -> Dict[str, int]:
        """Quick summary of how many records exist in each system."""
        return {
            "jira_tickets": len(self._jira_tickets),
            "hr_users": len(self._hr_users),
            "slack_channels": len(self._slack_channels),
            "slack_messages": len(self._slack_messages),
            "emails_sent": len(self._emails),
            "calendar_events": len(self._calendar_events),
            "total_calls": len(self._call_log),
        }

    # ── Jira ─────────────────────────────────────────────────────────

    def _tool_jira_create_ticket(self, p: dict) -> ToolResult:
        if not p.get("summary"):
            return ToolResult(False, None, "Missing required param: summary")
        if not p.get("issue_type"):
            return ToolResult(False, None, "Missing required param: issue_type")
        ticket_id = f"PROJ-{100 + len(self._jira_tickets)}"
        self._jira_tickets[ticket_id] = {
            "id": ticket_id,
            "summary": p["summary"],
            "description": p.get("description", ""),
            "issue_type": p["issue_type"],
            "priority": p.get("priority", "medium"),
            "project": p.get("project", "PROJ"),
            "status": "Open",
            "assignee": None,
            "labels": p.get("labels", []),
        }
        return ToolResult(True, {
            "ticket_id": ticket_id,
            "url": f"https://jira.example.com/browse/{ticket_id}",
        })

    def _tool_jira_update_ticket(self, p: dict) -> ToolResult:
        tid = p.get("ticket_id")
        if not tid or tid not in self._jira_tickets:
            return ToolResult(False, None, f"Ticket not found: {tid}")
        field = p.get("field")
        value = p.get("value")
        if not field:
            return ToolResult(False, None, "Missing required param: field")
        self._jira_tickets[tid][field] = value
        return ToolResult(True, {"ticket_id": tid, "updated": {field: value}})

    def _tool_jira_assign_ticket(self, p: dict) -> ToolResult:
        tid = p.get("ticket_id")
        if not tid or tid not in self._jira_tickets:
            return ToolResult(False, None, f"Ticket not found: {tid}")
        if not p.get("assignee"):
            return ToolResult(False, None, "Missing required param: assignee")
        self._jira_tickets[tid]["assignee"] = p["assignee"]
        return ToolResult(True, {"ticket_id": tid, "assignee": p["assignee"]})

    # ── Slack ────────────────────────────────────────────────────────

    def _tool_slack_send_message(self, p: dict) -> ToolResult:
        if not p.get("channel"):
            return ToolResult(False, None, "Missing required param: channel")
        if not p.get("message"):
            return ToolResult(False, None, "Missing required param: message")
        msg_id = f"msg_{len(self._slack_messages)}"
        self._slack_messages.append({
            "id": msg_id,
            "channel": p["channel"],
            "message": p["message"],
            "mention": p.get("mention_user"),
        })
        return ToolResult(True, {"message_id": msg_id, "channel": p["channel"]})

    def _tool_slack_create_channel(self, p: dict) -> ToolResult:
        if not p.get("name"):
            return ToolResult(False, None, "Missing required param: name")
        channel_id = f"C{len(self._slack_channels):06d}"
        self._slack_channels[channel_id] = {
            "id": channel_id,
            "name": p["name"],
            "members": p.get("members", []),
            "purpose": p.get("purpose", ""),
        }
        return ToolResult(True, {"channel_id": channel_id, "name": p["name"]})

    # ── Email ────────────────────────────────────────────────────────

    def _tool_email_send(self, p: dict) -> ToolResult:
        for req in ("to", "subject", "body"):
            if not p.get(req):
                return ToolResult(False, None, f"Missing required param: {req}")
        email_id = f"email_{len(self._emails)}"
        self._emails.append({"id": email_id, **p})
        return ToolResult(True, {"email_id": email_id, "to": p["to"]})

    # ── HR ───────────────────────────────────────────────────────────

    def _tool_hr_create_user(self, p: dict) -> ToolResult:
        for req in ("name", "role", "department"):
            if not p.get(req):
                return ToolResult(False, None, f"Missing required param: {req}")
        user_id = f"HR-{1000 + len(self._hr_users)}"
        self._hr_users[user_id] = {
            "id": user_id,
            "name": p["name"],
            "role": p["role"],
            "department": p["department"],
            "start_date": p.get("start_date", "TBD"),
            "status": "active",
        }
        return ToolResult(True, {"user_id": user_id, "name": p["name"]})

    def _tool_hr_update_user(self, p: dict) -> ToolResult:
        uid = p.get("user_id")
        if not uid or uid not in self._hr_users:
            return ToolResult(False, None, f"User not found: {uid}")
        field = p.get("field")
        value = p.get("value")
        if not field:
            return ToolResult(False, None, "Missing required param: field")
        self._hr_users[uid][field] = value
        return ToolResult(True, {"user_id": uid, "updated": {field: value}})

    # ── Calendar ─────────────────────────────────────────────────────

    def _tool_calendar_create_event(self, p: dict) -> ToolResult:
        for req in ("title", "attendees"):
            if not p.get(req):
                return ToolResult(False, None, f"Missing required param: {req}")
        event_id = f"EVT-{len(self._calendar_events) + 1}"
        self._calendar_events[event_id] = {
            "id": event_id,
            "title": p["title"],
            "attendees": p["attendees"],
            "date": p.get("date", "TBD"),
            "duration_minutes": p.get("duration_minutes", 60),
            "description": p.get("description", ""),
        }
        return ToolResult(True, {
            "event_id": event_id,
            "title": p["title"],
            "attendees": p["attendees"],
        })

    # ── State queries ─────────────────────────────────────────────────

    def get_hr_users(self) -> Dict[str, Dict]:
        return dict(self._hr_users)

    def get_jira_tickets(self) -> Dict[str, Dict]:
        return dict(self._jira_tickets)

    def call_log(self) -> list:
        return list(self._call_log)
