"""Load sample tickets through the API so the dashboard has data to show.

Usage:  python seed.py            (server must be running on :8000)
        python seed.py http://host:port
"""

import sys

import httpx

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"

SAMPLES = [
    ("dana@acme.io", "Charged twice this month",
     "Hi, I see two charges of $49 on my card for the March subscription. Can I get a refund for the duplicate?"),
    ("ops@bigcorp.com", "Dashboard down for our whole team",
     "Since 9am nobody on our team can load the dashboard. Everyone gets a 500 error. This is a full outage for all users."),
    ("li.wei@example.com", "Locked out after password reset",
     "I reset my password but now the login page says my account is locked. I can't sign in and have a deadline today."),
    ("sam@startup.dev", "Webhook integration timing out",
     "Our API integration started timing out on the /events webhook. Requests take 30s+ then fail. Started after your last release."),
    ("jordan@example.com", "Dark mode?",
     "Feature suggestion: it would be nice to have a dark mode in the web app. Any plans?"),
    ("pat@example.com", "Export button crashes the page",
     "Clicking Export on the reports page crashes the tab in Chrome. Happens every time."),
    ("alex@example.com", "Question about invoices",
     "How do I download past invoices for our accounting team? Just wondering where they live."),
    ("security@partner.org", "Possible account compromise",
     "We noticed logins from an unknown country on an admin account. Possible security breach, please advise."),
]

with httpx.Client(base_url=BASE, timeout=60) as client:
    for email, subject, body in SAMPLES:
        t = client.post("/tickets", json={"customer_email": email, "subject": subject, "body": body}).json()
        src = t["triage_results"][-1]["source"]
        print(f"#{t['id']:<3} {t['priority']:<7} {t['category']:<16} [{src}] {subject}")
    print(f"\nOpen {BASE} to view the dashboard.")
