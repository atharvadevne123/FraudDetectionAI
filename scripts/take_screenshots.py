"""
Automated screenshot capture for README documentation.
Takes dashboard screenshots (all tiers + scorer results) and API response snapshots.
"""

import json
import time
import http.server
import threading
import os
from pathlib import Path
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).parent.parent
SS = ROOT / "screenshots"
SS.mkdir(exist_ok=True)
UI_FILE = ROOT / "ui" / "index.html"
API_URL = "http://localhost:8001"

# Serve the HTML dashboard over HTTP (avoids file:// CORS issues)
class SilentHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *a): pass

def start_server(port=9090):
    os.chdir(ROOT / "ui")
    srv = http.server.HTTPServer(("", port), SilentHandler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    return srv

TIER_PAYLOADS = {
    "CRITICAL": {
        "transaction_id": "TXN-CRIT-01",
        "user_id": 12345,
        "amount": 4999.99,
        "merchant_category": "crypto",
        "payment_method": "wire",
        "device_type": "mobile",
        "channel": "online",
        "account_age_days": 12,
        "credit_utilization": 0.92,
        "prior_fraud_count": 1,
        "explain": False,
    },
    "HIGH": {
        "transaction_id": "TXN-HIGH-01",
        "user_id": 55678,
        "amount": 2800.00,
        "merchant_category": "gambling",
        "payment_method": "credit",
        "device_type": "mobile",
        "channel": "online",
        "account_age_days": 65,
        "credit_utilization": 0.78,
        "prior_fraud_count": 1,
        "explain": False,
    },
    "MEDIUM": {
        "transaction_id": "TXN-MED-01",
        "user_id": 34901,
        "amount": 890.00,
        "merchant_category": "retail",
        "payment_method": "credit",
        "device_type": "desktop",
        "channel": "online",
        "account_age_days": 240,
        "credit_utilization": 0.62,
        "prior_fraud_count": 0,
        "explain": False,
    },
    "LOW": {
        "transaction_id": "TXN-LOW-01",
        "user_id": 78234,
        "amount": 145.50,
        "merchant_category": "grocery",
        "payment_method": "debit",
        "device_type": "desktop",
        "channel": "pos",
        "account_age_days": 820,
        "credit_utilization": 0.38,
        "prior_fraud_count": 0,
        "explain": False,
    },
    "CLEAN": {
        "transaction_id": "TXN-CLEAN-01",
        "user_id": 90012,
        "amount": 12.40,
        "merchant_category": "restaurant",
        "payment_method": "debit",
        "device_type": "mobile",
        "channel": "mobile_app",
        "account_age_days": 2200,
        "credit_utilization": 0.08,
        "prior_fraud_count": 0,
        "explain": False,
    },
}

TIER_COLOR = {
    "CRITICAL": "#ef4444",
    "HIGH": "#f97316",
    "MEDIUM": "#f59e0b",
    "LOW": "#3b82f6",
    "CLEAN": "#22c55e",
}


def save_api_json(name: str, data: dict):
    """Save pretty-printed JSON as a text artifact (embedded in README as code block)."""
    path = SS / f"api_{name}.json"
    path.write_text(json.dumps(data, indent=2))
    print(f"  ✓ saved api_{name}.json")


def capture_api_screenshots(page):
    """Hit every API endpoint and save JSON + take a terminal-style visual screenshot."""
    import urllib.request, urllib.error

    def get(url):
        try:
            with urllib.request.urlopen(url, timeout=5) as r:
                return json.loads(r.read())
        except Exception as e:
            return {"error": str(e)}

    def post(url, payload):
        try:
            data = json.dumps(payload).encode()
            req = urllib.request.Request(url, data=data,
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=10) as r:
                return json.loads(r.read())
        except Exception as e:
            return {"error": str(e)}

    print("\n── API Snapshots ───────────────────────────────────────")

    # GET endpoints
    save_api_json("health", get(f"{API_URL}/health"))
    save_api_json("model_info", get(f"{API_URL}/model/info"))

    # POST /predict — one per tier
    for tier, payload in TIER_PAYLOADS.items():
        result = post(f"{API_URL}/predict", payload)
        result["_request_summary"] = {
            "amount": payload["amount"],
            "merchant_category": payload["merchant_category"],
            "payment_method": payload["payment_method"],
            "account_age_days": payload["account_age_days"],
            "credit_utilization": payload["credit_utilization"],
            "prior_fraud_count": payload["prior_fraud_count"],
        }
        save_api_json(f"predict_{tier.lower()}", result)

    # POST /predict/batch
    batch_payload = {"transactions": list(TIER_PAYLOADS.values())}
    batch_result = post(f"{API_URL}/predict/batch", batch_payload)
    save_api_json("predict_batch", batch_result)

    # Render a visual API terminal page for each tier
    for tier, payload in TIER_PAYLOADS.items():
        json_path = SS / f"api_predict_{tier.lower()}.json"
        data = json.loads(json_path.read_text()) if json_path.exists() else {}
        clr = TIER_COLOR[tier]
        html = build_terminal_html(tier, payload, data, clr)
        page.set_content(html)
        page.wait_for_timeout(400)
        page.screenshot(path=str(SS / f"api_predict_{tier.lower()}.png"), full_page=False)
        print(f"  ✓ screenshot: api_predict_{tier.lower()}.png")

    # Health + model info terminal screenshots
    for name in ("health", "model_info"):
        json_path = SS / f"api_{name}.json"
        data = json.loads(json_path.read_text()) if json_path.exists() else {}
        html = build_info_terminal_html(name.replace("_", " ").upper(), data)
        page.set_content(html)
        page.wait_for_timeout(300)
        page.screenshot(path=str(SS / f"api_{name}.png"), full_page=False)
        print(f"  ✓ screenshot: api_{name}.png")


def build_terminal_html(tier, payload, result, clr):
    fs = result.get("fraud_score", 0)
    as_ = result.get("anomaly_score", 0)
    rt = result.get("risk_tier", tier)
    lat = result.get("latency_ms", 0)
    req_json = json.dumps(payload, indent=4)
    res_json = json.dumps({
        "transaction_id": result.get("transaction_id", payload.get("transaction_id")),
        "fraud_score": fs,
        "anomaly_score": as_,
        "fraud_label": result.get("fraud_label", 0),
        "risk_tier": rt,
        "latency_ms": lat,
    }, indent=4)
    return f"""<!DOCTYPE html><html><head><meta charset="UTF-8">
<style>
*{{margin:0;padding:0;box-sizing:border-box;}}
body{{background:#07090f;font-family:'Courier New',monospace;font-size:13px;padding:0;width:860px;}}
.terminal{{background:#0b0f1c;border:1px solid #182236;border-radius:10px;overflow:hidden;margin:20px;}}
.titlebar{{background:#111827;padding:10px 14px;display:flex;align-items:center;gap:8px;border-bottom:1px solid #182236;}}
.dot{{width:12px;height:12px;border-radius:50%;}}
.d1{{background:#ef4444;}}.d2{{background:#f59e0b;}}.d3{{background:#22c55e;}}
.title{{color:#64748b;font-size:12px;margin-left:8px;}}
.body{{padding:16px 18px;}}
.row{{display:flex;gap:20px;}}
.col{{flex:1;}}
.label{{color:#475569;font-size:10px;letter-spacing:.8px;text-transform:uppercase;margin-bottom:6px;}}
pre{{color:#94a3b8;font-size:12px;line-height:1.6;white-space:pre-wrap;}}
.kw{{color:#7dd3fc;}}.str{{color:#86efac;}}.num{{color:#fde68a;}}.key{{color:{clr};}}
.result-box{{background:#0f1625;border:1px solid {clr}33;border-radius:8px;padding:12px;margin-top:14px;}}
.score-row{{display:flex;align-items:center;gap:12px;margin-bottom:8px;}}
.badge{{display:inline-block;padding:4px 12px;border-radius:4px;font-size:11px;font-weight:700;background:{clr}18;color:{clr};border:1px solid {clr}44;letter-spacing:.5px;}}
.bar-wrap{{flex:1;height:8px;background:#182236;border-radius:4px;overflow:hidden;}}
.bar-fill{{height:100%;border-radius:4px;background:{clr};width:{min(fs*100,100):.1f}%;}}
.score-val{{color:{clr};font-weight:700;min-width:60px;text-align:right;font-size:13px;}}
.meta{{display:flex;gap:16px;margin-top:8px;}}
.meta-item{{color:#64748b;font-size:11px;}}.meta-item span{{color:#94a3b8;}}
.endpoint{{color:#6366f1;font-size:12px;margin-bottom:10px;}}
</style></head><body>
<div class="terminal">
  <div class="titlebar"><div class="dot d1"></div><div class="dot d2"></div><div class="dot d3"></div>
    <span class="title">POST  {API_URL}/predict  →  {tier} TRANSACTION</span></div>
  <div class="body">
    <div class="row">
      <div class="col">
        <div class="label">Request Payload</div>
        <pre><span class="kw">{{</span>
  <span class="key">"user_id"</span>: <span class="num">{payload['user_id']}</span>,
  <span class="key">"amount"</span>: <span class="num">{payload['amount']}</span>,
  <span class="key">"merchant_category"</span>: <span class="str">"{payload['merchant_category']}"</span>,
  <span class="key">"payment_method"</span>: <span class="str">"{payload['payment_method']}"</span>,
  <span class="key">"account_age_days"</span>: <span class="num">{payload['account_age_days']}</span>,
  <span class="key">"credit_utilization"</span>: <span class="num">{payload['credit_utilization']}</span>,
  <span class="key">"prior_fraud_count"</span>: <span class="num">{payload['prior_fraud_count']}</span>
<span class="kw">}}</span></pre>
      </div>
      <div class="col">
        <div class="label">Response</div>
        <pre><span class="kw">{{</span>
  <span class="key">"transaction_id"</span>: <span class="str">"{result.get('transaction_id','')}"</span>,
  <span class="key">"fraud_score"</span>: <span class="num">{fs:.6f}</span>,
  <span class="key">"anomaly_score"</span>: <span class="num">{as_:.6f}</span>,
  <span class="key">"fraud_label"</span>: <span class="num">{result.get('fraud_label',0)}</span>,
  <span class="key">"risk_tier"</span>: <span class="str" style="color:{clr}">"{rt}"</span>,
  <span class="key">"latency_ms"</span>: <span class="num">{lat:.2f}</span>
<span class="kw">}}</span></pre>
      </div>
    </div>
    <div class="result-box">
      <div class="score-row">
        <div class="badge">{tier} RISK</div>
        <div class="bar-wrap"><div class="bar-fill"></div></div>
        <div class="score-val">{fs:.4f}</div>
      </div>
      <div class="meta">
        <div class="meta-item">Fraud Score <span>{fs:.6f}</span></div>
        <div class="meta-item">Anomaly Score <span>{as_:.6f}</span></div>
        <div class="meta-item">Latency <span>{lat:.1f}ms</span></div>
        <div class="meta-item">Label <span>{'FRAUD' if result.get('fraud_label') else 'LEGIT'}</span></div>
      </div>
    </div>
  </div>
</div></body></html>"""


def build_info_terminal_html(title, data):
    lines = []
    for k, v in data.items():
        val = f'"{v}"' if isinstance(v, str) else str(v).lower() if isinstance(v, bool) else str(v)
        clr = "#86efac" if isinstance(v, str) else "#fde68a" if isinstance(v, (int, float)) else "#7dd3fc"
        lines.append(f'  <span style="color:#6366f1">"{k}"</span>: <span style="color:{clr}">{val}</span>')
    body = ",\n".join(lines)
    return f"""<!DOCTYPE html><html><head><meta charset="UTF-8">
<style>
*{{margin:0;padding:0;box-sizing:border-box;}}
body{{background:#07090f;font-family:'Courier New',monospace;font-size:13px;width:600px;}}
.terminal{{background:#0b0f1c;border:1px solid #182236;border-radius:10px;overflow:hidden;margin:20px;}}
.titlebar{{background:#111827;padding:10px 14px;display:flex;align-items:center;gap:8px;border-bottom:1px solid #182236;}}
.dot{{width:12px;height:12px;border-radius:50%;}}
.d1{{background:#ef4444;}}.d2{{background:#f59e0b;}}.d3{{background:#22c55e;}}
.title{{color:#64748b;font-size:12px;margin-left:8px;}}
pre{{padding:16px 18px;color:#94a3b8;font-size:13px;line-height:1.8;}}
</style></head><body>
<div class="terminal">
  <div class="titlebar"><div class="dot d1"></div><div class="dot d2"></div><div class="dot d3"></div>
    <span class="title">GET  {API_URL}/{title.lower().replace(' ','/')}</span></div>
  <pre><span style="color:#7dd3fc">{{</span>
{body}
<span style="color:#7dd3fc">}}</span></pre>
</div></body></html>"""


def capture_dashboard_screenshots(page, server_url="http://localhost:9090"):
    print("\n── Dashboard Screenshots ───────────────────────────────")
    page.set_viewport_size({"width": 1440, "height": 860})

    # 1. Full dashboard — all tiers
    page.goto(server_url, wait_until="networkidle")
    page.wait_for_timeout(1800)
    page.screenshot(path=str(SS / "dashboard_overview.png"), full_page=False)
    print("  ✓ dashboard_overview.png")

    # 2. Each tier filtered
    for tier in ["CRITICAL", "HIGH", "MEDIUM", "LOW", "CLEAN"]:
        page.goto(server_url, wait_until="networkidle")
        page.wait_for_timeout(1200)
        page.select_option("#tfilter", tier)
        page.wait_for_timeout(600)
        page.screenshot(path=str(SS / f"dashboard_{tier.lower()}.png"), full_page=False)
        print(f"  ✓ dashboard_{tier.lower()}.png")

    # 3. Scorer panel with CRITICAL result (simulate by filling form)
    page.goto(server_url, wait_until="networkidle")
    page.wait_for_timeout(1200)
    # Click first CRITICAL row to populate form
    page.select_option("#tfilter", "CRITICAL")
    page.wait_for_timeout(500)
    first_row = page.locator("tr.drow").first
    first_row.click()
    page.wait_for_timeout(500)
    page.locator("#sbtn").click()
    page.wait_for_timeout(2200)
    page.screenshot(path=str(SS / "dashboard_scorer_critical.png"), full_page=False)
    print("  ✓ dashboard_scorer_critical.png")

    # 4. Scorer with CLEAN result
    page.goto(server_url, wait_until="networkidle")
    page.wait_for_timeout(1200)
    page.select_option("#tfilter", "CLEAN")
    page.wait_for_timeout(500)
    first_row = page.locator("tr.drow").first
    first_row.click()
    page.wait_for_timeout(500)
    page.locator("#sbtn").click()
    page.wait_for_timeout(2200)
    page.screenshot(path=str(SS / "dashboard_scorer_clean.png"), full_page=False)
    print("  ✓ dashboard_scorer_clean.png")


def main():
    print("Starting screenshot capture for SENTINELLA README...")
    srv = start_server(9090)
    time.sleep(0.5)

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        ctx = browser.new_context(viewport={"width": 1440, "height": 860})
        page = ctx.new_page()

        capture_api_screenshots(page)
        capture_dashboard_screenshots(page)

        browser.close()

    srv.shutdown()
    print(f"\n✓ All screenshots saved to {SS}")
    print(f"  Files: {sorted([f.name for f in SS.glob('*.png')])}")


if __name__ == "__main__":
    main()
