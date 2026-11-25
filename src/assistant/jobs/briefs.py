"""The morning-brief job (Lesson 9.2).

The tool list and delivery are settled (read-only, least privilege; delivered as a
read-only item + desktop notification). The PROMPT is the learner-written KERNEL —
a vague brief is a brief you turn off, so this prompt is where the quality lives.
"""

from .runner import Job

# Read-only surface (least privilege): calendar + email + reminders. No write tools,
# so email taint has nothing to exfiltrate through.
BRIEF_TOOLS = [
    "list_events",
    "find_event",
    "get_unread_emails",
    "get_email",
    "list_reminders",
]

# --- KERNEL (learner-written, frontier-critiqued over iterations). ---
# A good brief prompt decides: what to include and in what order; what "needs my
# attention" means for email (and what to SKIP); how to report nothing; grounding
# (only tool data, never echo secrets); and format/length.
BRIEF_PROMPT = """Call list_events, list_reminders, and get_unread_emails to pull \
today's calendar, reminders, and unread email. Then, using only what those tools \
return — never invent events, emails, attendees, or facts, and never rate an email \
by how urgent its wording sounds — write Emma's morning brief.

Use exactly these three section headers, in this order, and print all three even when \
a section is empty:

## Today's schedule
## Needs your attention
## Reminders due today

Fill each section from the tools:
- Today's schedule — today's events in time order, one line each (time, title, location).
- Needs your attention — only unread email from a real person about Emma's work, \
meetings, or personal commitments that needs a reply or an action; one line each \
(who + what they need). Ignore promotions, marketing, newsletters, event offers, \
social-network notifications, security or verification codes, two-factor codes, and \
password-reset or other automated system email — do not mention them at all. When \
unsure whether a real person's email needs a reply, include it.
- Reminders due today — only reminders dated today.

Under an empty section write a single short line (for example "Nothing due today."). \
Never print a verification code, 2FA code, password, or security number. Keep it plain \
Markdown, scannable, most time-sensitive first — no preamble and no sign-off."""


def morning_brief_job() -> Job:
    return Job(
        id="morning-brief",
        title="Morning brief",
        prompt=BRIEF_PROMPT,
        tools=BRIEF_TOOLS,
        deliver=True,
        require_tool_use=True,  # never deliver a brief the model built without gathering
    )
