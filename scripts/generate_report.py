#!/usr/bin/env python3
"""
Generate grief & bereavement daily research report HTML using Zhipu AI.
Reads papers JSON, analyzes with GLM-5-Turbo, generates styled HTML.
Same color scheme and design as Psychiatry-brain.
"""

import json
import sys
import os
import re
import time
import argparse
from datetime import datetime, timezone, timedelta

import httpx

API_BASE = os.environ.get(
    "ZHIPU_API_BASE", "https://open.bigmodel.cn/api/coding/paas/v4"
)
FALLBACK_MODELS = ["glm-5-turbo", "glm-4.7", "glm-4.7-flash"]

SYSTEM_PROMPT = (
    "你是悲傷與喪親研究領域的資深研究員與科學傳播者。你的任務是：\n"
    "1. 從提供的醫學文獻中，篩選出最具臨床意義與研究價值的悲傷/喪親相關論文\n"
    "2. 對每篇論文進行繁體中文摘要、分類\n"
    "3. 評估其臨床實用性（高/中/低）\n"
    "4. 生成適合醫療專業人員閱讀的日報\n\n"
    "輸出格式要求：\n"
    "- 語言：繁體中文（台灣用語）\n"
    "- 專業但易懂\n"
    "- 每篇論文需包含：中文標題、一句話總結、臨床實用性、分類標籤\n"
    "- 最後提供今日精選 TOP 3（最重要/最影響臨床實踐的論文）\n"
    "回傳格式必須是純 JSON，不要用 markdown code block 包裹。"
)


def load_papers(input_path: str) -> dict:
    if input_path == "-":
        data = json.load(sys.stdin)
    else:
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    return data


def robust_json_parse(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:])
        text = text.rstrip("`").strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    json_match = re.search(r'\{[\s\S]*\}', text)
    if json_match:
        candidate = json_match.group()
        for attempt in range(3):
            try:
                return json.loads(candidate)
            except json.JSONDecodeError as e:
                if "Unterminated string" in str(e) or "Expecting" in str(e):
                    last_brace = candidate.rfind("}")
                    if last_brace > 0:
                        candidate = candidate[:last_brace + 1]
                        try:
                            return json.loads(candidate)
                        except json.JSONDecodeError:
                            continue
                break

    if '"' in text:
        fixed = text.replace('\n"', '\n\\n"')
        try:
            return json.loads(fixed)
        except json.JSONDecodeError:
            pass

    return None


def analyze_papers(api_key: str, papers_data: dict) -> dict:
    tz_taipei = timezone(timedelta(hours=8))
    date_str = papers_data.get("date", datetime.now(tz_taipei).strftime("%Y-%m-%d"))
    paper_count = papers_data.get("count", 0)
    papers_text = json.dumps(
        papers_data.get("papers", []), ensure_ascii=False, indent=2
    )

    prompt = f"""以下是 {date_str} 從 PubMed 抓取的最新悲傷與喪親研究文獻（共 {paper_count} 篇）。

請進行以下分析，並以 JSON 格式回傳（不要用 markdown code block）：

{{
  "date": "{date_str}",
  "market_summary": "1-2句話總結今天悲傷/喪親研究文獻的整體趨勢與亮點",
  "top_picks": [
    {{
      "rank": 1,
      "title_zh": "中文標題",
      "title_en": "English Title",
      "journal": "期刊名",
      "summary": "一句話總結（繁體中文，點出核心發現與臨床意義）",
      "clinical_utility": "高/中/低",
      "utility_reason": "為什麼實用的一句話說明",
      "tags": ["標籤1", "標籤2"],
      "url": "原文連結",
      "emoji": "相關emoji"
    }}
  ],
  "all_papers": [
    {{
      "title_zh": "中文標題",
      "title_en": "English Title",
      "journal": "期刊名",
      "summary": "一句話總結",
      "clinical_utility": "高/中/低",
      "tags": ["標籤1"],
      "url": "連結",
      "emoji": "emoji"
    }}
  ],
  "keywords": ["關鍵字1", "關鍵字2"],
  "topic_distribution": {{
    "延長性悲傷疾患": 3,
    "喪親研究": 2
  }}
}}

原始文獻資料：
{papers_text}

請篩選出最重要的 TOP 5-8 篇論文放入 top_picks（按重要性排序），其餘放入 all_papers。
每篇 paper 的 tags 請從以下選擇：延長性悲傷疾患、複雜性悲傷、喪親、喪偶、哀悼、創傷性悲傷、自殺遺族、兒少喪親、周產期喪失、安寧緩和照護、神經科學、心理治療、憂鬱症、PTSD、社會文化、照顧者悲傷、意義重建、繼續連結、喪失與孤獨、老年喪親、悲傷篩檢、COVID-19 悲傷、系統性回顧。
記住：回傳純 JSON，不要用 ```json``` 包裹。確保 JSON 格式完整、有效。"""

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": FALLBACK_MODELS[0],
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.3,
        "top_p": 0.9,
        "max_tokens": 100000,
    }

    for model in FALLBACK_MODELS:
        payload["model"] = model
        for attempt in range(3):
            try:
                print(
                    f"[INFO] Trying {model} (attempt {attempt + 1})...",
                    file=sys.stderr,
                )
                resp = httpx.post(
                    f"{API_BASE}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=660,
                )
                if resp.status_code == 429:
                    wait = 60 * (attempt + 1)
                    print(f"[WARN] Rate limited, waiting {wait}s...", file=sys.stderr)
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                data = resp.json()
                text = data["choices"][0]["message"]["content"].strip()

                result = robust_json_parse(text)
                if result is None:
                    print(
                        f"[WARN] JSON parse failed on attempt {attempt + 1}, retrying...",
                        file=sys.stderr,
                    )
                    if attempt < 2:
                        time.sleep(5)
                    continue

                print(
                    f"[INFO] Analysis complete with {model}: "
                    f"{len(result.get('top_picks', []))} top picks, "
                    f"{len(result.get('all_papers', []))} total",
                    file=sys.stderr,
                )
                result["_model_used"] = model
                return result

            except httpx.HTTPStatusError as e:
                print(
                    f"[ERROR] HTTP {e.response.status_code}: {e.response.text[:200]}",
                    file=sys.stderr,
                )
                if e.response.status_code == 429:
                    wait = 60 * (attempt + 1)
                    time.sleep(wait)
                    continue
                break
            except (httpx.TimeoutException, httpx.ConnectTimeout, httpx.ReadTimeout) as e:
                print(f"[WARN] Timeout on attempt {attempt + 1}: {e}", file=sys.stderr)
                if attempt < 2:
                    time.sleep(10)
                continue
            except Exception as e:
                print(f"[ERROR] {model} failed: {e}", file=sys.stderr)
                break

    print("[ERROR] All models and attempts failed", file=sys.stderr)
    return None


def generate_html(analysis: dict) -> str:
    date_str = analysis.get(
        "date", datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d")
    )
    date_parts = date_str.split("-")
    if len(date_parts) == 3:
        date_display = (
            f"{date_parts[0]}\u5e74{int(date_parts[1])}\u6708{int(date_parts[2])}\u65e5"
        )
    else:
        date_display = date_str

    model_used = analysis.get("_model_used", "GLM-5-Turbo")
    summary = analysis.get("market_summary", "")
    top_picks = analysis.get("top_picks", [])
    all_papers = analysis.get("all_papers", [])
    keywords = analysis.get("keywords", [])
    topic_dist = analysis.get("topic_distribution", {})

    top_picks_html = ""
    for pick in top_picks:
        tags_html = "".join(
            f'<span class="tag">{t}</span>' for t in pick.get("tags", [])
        )
        util = pick.get("clinical_utility", "\u4e2d")
        utility_class = (
            "utility-high"
            if util == "\u9ad8"
            else ("utility-mid" if util == "\u4e2d" else "utility-low")
        )
        reason = pick.get("utility_reason", "")

        top_picks_html += f"""
        <div class="news-card featured">
          <div class="card-header">
            <span class="rank-badge">#{pick.get("rank", "")}</span>
            <span class="emoji-icon">{pick.get("emoji", "\U0001f4c4")}</span>
            <span class="{utility_class}">{util}\u5b9e\u7528\u6027</span>
          </div>
          <h3>{pick.get("title_zh", pick.get("title_en", ""))}</h3>
          <p class="journal-source">{pick.get("journal", "")} &middot; {pick.get("title_en", "")}</p>
          <p>{pick.get("summary", "")}</p>
          {f'<p class="utility-reason">\u27a1 {reason}</p>' if reason else ""}
          <div class="card-footer">
            {tags_html}
            <a href="{pick.get("url", "#")}" target="_blank">\u95b1\u8b80\u539f\u6587 \u2192</a>
          </div>
        </div>"""

    all_papers_html = ""
    for paper in all_papers:
        tags_html = "".join(
            f'<span class="tag">{t}</span>' for t in paper.get("tags", [])
        )
        util = paper.get("clinical_utility", "\u4e2d")
        utility_class = (
            "utility-high"
            if util == "\u9ad8"
            else ("utility-mid" if util == "\u4e2d" else "utility-low")
        )
        all_papers_html += f"""
        <div class="news-card">
          <div class="card-header-row">
            <span class="emoji-sm">{paper.get("emoji", "\U0001f4c4")}</span>
            <span class="{utility_class} utility-sm">{util}</span>
          </div>
          <h3>{paper.get("title_zh", paper.get("title_en", ""))}</h3>
          <p class="journal-source">{paper.get("journal", "")}</p>
          <p>{paper.get("summary", "")}</p>
          <div class="card-footer">
            {tags_html}
            <a href="{paper.get("url", "#")}" target="_blank">PubMed \u2192</a>
          </div>
        </div>"""

    keywords_html = "".join(f'<span class="keyword">{k}</span>' for k in keywords)
    topic_bars_html = ""
    if topic_dist:
        max_count = max(topic_dist.values()) if topic_dist else 1
        for topic, count in topic_dist.items():
            width_pct = int((count / max_count) * 100)
            topic_bars_html += f"""
            <div class="topic-row">
              <span class="topic-name">{topic}</span>
              <div class="topic-bar-bg"><div class="topic-bar" style="width:{width_pct}%"></div></div>
              <span class="topic-count">{count}</span>
            </div>"""

    total_count = len(top_picks) + len(all_papers)

    html = f"""<!DOCTYPE html>
<html lang="zh-TW">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>Grief Research Daily &middot; \u60b2\u50b7\u7814\u7a76\u6587\u737b\u65e5\u5831 &middot; {date_display}</title>
<meta name="description" content="{date_display} \u60b2\u50b7\u8207\u55aa\u89aa\u7814\u7a76\u6587\u737b\u65e5\u5831\uff0c\u7531 AI \u81ea\u52d5\u5f59\u6574 PubMed \u6700\u65b0\u8ad6\u6587"/>
<style>
  :root {{ --bg: #f6f1e8; --surface: #fffaf2; --line: #d8c5ab; --text: #2b2118; --muted: #766453; --accent: #8c4f2b; --accent-soft: #ead2bf; --card-bg: color-mix(in srgb, var(--surface) 92%, white); }}
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: radial-gradient(circle at top, #fff6ea 0, var(--bg) 55%, #ead8c6 100%); color: var(--text); font-family: "Noto Sans TC", "PingFang TC", "Helvetica Neue", Arial, sans-serif; min-height: 100vh; overflow-x: hidden; }}
  .container {{ position: relative; z-index: 1; max-width: 880px; margin: 0 auto; padding: 60px 32px 80px; }}
  header {{ display: flex; align-items: center; gap: 16px; margin-bottom: 52px; animation: fadeDown 0.6s ease both; }}
  .logo {{ width: 48px; height: 48px; border-radius: 14px; background: var(--accent); display: flex; align-items: center; justify-content: center; font-size: 22px; flex-shrink: 0; box-shadow: 0 4px 20px rgba(140,79,43,0.25); }}
  .header-text h1 {{ font-size: 22px; font-weight: 700; color: var(--text); letter-spacing: -0.3px; }}
  .header-meta {{ display: flex; gap: 8px; margin-top: 6px; flex-wrap: wrap; align-items: center; }}
  .badge {{ display: inline-block; padding: 3px 10px; border-radius: 20px; font-size: 11px; letter-spacing: 0.3px; }}
  .badge-date {{ background: var(--accent-soft); border: 1px solid var(--line); color: var(--accent); }}
  .badge-count {{ background: rgba(140,79,43,0.06); border: 1px solid var(--line); color: var(--muted); }}
  .badge-source {{ background: transparent; color: var(--muted); font-size: 11px; padding: 0 4px; }}
  .summary-card {{ background: var(--card-bg); border: 1px solid var(--line); border-radius: 24px; padding: 28px 32px; margin-bottom: 32px; box-shadow: 0 20px 60px rgba(61,36,15,0.06); animation: fadeUp 0.5s ease 0.1s both; }}
  .summary-card h2 {{ font-size: 10px; font-weight: 700; text-transform: uppercase; letter-spacing: 1.6px; color: var(--accent); margin-bottom: 16px; }}
  .summary-text {{ font-size: 15px; line-height: 1.8; color: var(--text); }}
  .section {{ margin-bottom: 36px; animation: fadeUp 0.5s ease both; }}
  .section-title {{ display: flex; align-items: center; gap: 10px; font-size: 17px; font-weight: 700; color: var(--text); margin-bottom: 16px; padding-bottom: 12px; border-bottom: 1px solid var(--line); }}
  .section-icon {{ width: 28px; height: 28px; border-radius: 8px; display: flex; align-items: center; justify-content: center; font-size: 14px; flex-shrink: 0; background: var(--accent-soft); }}
  .news-card {{ background: var(--card-bg); border: 1px solid var(--line); border-radius: 24px; padding: 22px 26px; margin-bottom: 12px; box-shadow: 0 8px 30px rgba(61,36,15,0.04); transition: background 0.2s, border-color 0.2s, transform 0.2s; }}
  .news-card:hover {{ transform: translateY(-2px); box-shadow: 0 12px 40px rgba(61,36,15,0.08); }}
  .news-card.featured {{ border-left: 3px solid var(--accent); }}
  .news-card.featured:hover {{ border-color: var(--accent); }}
  .card-header {{ display: flex; align-items: center; gap: 8px; margin-bottom: 10px; }}
  .rank-badge {{ background: var(--accent); color: #fff7f0; font-weight: 700; font-size: 12px; padding: 2px 8px; border-radius: 6px; }}
  .emoji-icon {{ font-size: 18px; }}
  .card-header-row {{ display: flex; align-items: center; gap: 8px; margin-bottom: 8px; }}
  .emoji-sm {{ font-size: 14px; }}
  .news-card h3 {{ font-size: 15px; font-weight: 600; color: var(--text); margin-bottom: 8px; line-height: 1.5; }}
  .journal-source {{ font-size: 12px; color: var(--accent); margin-bottom: 8px; opacity: 0.8; }}
  .news-card p {{ font-size: 13.5px; line-height: 1.75; color: var(--muted); }}
  .utility-reason {{ font-size: 12.5px; color: var(--accent); margin-top: 6px; font-weight: 500; }}
  .card-footer {{ margin-top: 12px; display: flex; flex-wrap: wrap; gap: 6px; align-items: center; }}
  .tag {{ padding: 2px 9px; background: var(--accent-soft); border-radius: 999px; font-size: 11px; color: var(--accent); }}
  .news-card a {{ font-size: 12px; color: var(--accent); text-decoration: none; opacity: 0.7; margin-left: auto; }}
  .news-card a:hover {{ opacity: 1; }}
  .utility-high {{ color: #5a7a3a; font-size: 11px; font-weight: 600; padding: 2px 8px; background: rgba(90,122,58,0.1); border-radius: 4px; }}
  .utility-mid {{ color: #9f7a2e; font-size: 11px; font-weight: 600; padding: 2px 8px; background: rgba(159,122,46,0.1); border-radius: 4px; }}
  .utility-low {{ color: var(--muted); font-size: 11px; font-weight: 600; padding: 2px 8px; background: rgba(118,100,83,0.08); border-radius: 4px; }}
  .utility-sm {{ font-size: 10px; }}
  .keywords-section {{ margin-bottom: 36px; }}
  .keywords {{ display: flex; flex-wrap: wrap; gap: 8px; margin-top: 8px; }}
  .keyword {{ padding: 5px 14px; background: var(--accent-soft); border: 1px solid var(--line); border-radius: 20px; font-size: 12px; color: var(--accent); cursor: default; transition: background 0.2s; }}
  .keyword:hover {{ background: rgba(140,79,43,0.18); }}
  .topic-section {{ margin-bottom: 36px; }}
  .topic-row {{ display: flex; align-items: center; gap: 10px; margin-bottom: 8px; }}
  .topic-name {{ font-size: 13px; color: var(--muted); width: 120px; flex-shrink: 0; text-align: right; }}
  .topic-bar-bg {{ flex: 1; height: 8px; background: var(--line); border-radius: 4px; overflow: hidden; }}
  .topic-bar {{ height: 100%; background: linear-gradient(90deg, var(--accent), #c47a4a); border-radius: 4px; transition: width 0.6s ease; }}
  .topic-count {{ font-size: 12px; color: var(--accent); width: 24px; }}
  .links-banner {{ margin-top: 48px; display: flex; flex-direction: column; gap: 12px; animation: fadeUp 0.5s ease 0.4s both; }}
  .link-card {{ display: flex; align-items: center; gap: 14px; padding: 18px 24px; background: var(--card-bg); border: 1px solid var(--line); border-radius: 24px; text-decoration: none; color: var(--text); transition: all 0.2s; box-shadow: 0 8px 30px rgba(61,36,15,0.04); }}
  .link-card:hover {{ border-color: var(--accent); transform: translateY(-2px); box-shadow: 0 12px 40px rgba(61,36,15,0.08); }}
  .link-icon {{ font-size: 28px; flex-shrink: 0; }}
  .link-info {{ flex: 1; }}
  .link-name {{ font-size: 15px; font-weight: 700; color: var(--text); }}
  .link-desc {{ font-size: 12px; color: var(--muted); margin-top: 2px; }}
  .link-arrow {{ font-size: 18px; color: var(--accent); font-weight: 700; }}
  footer {{ margin-top: 32px; padding-top: 22px; border-top: 1px solid var(--line); font-size: 11.5px; color: var(--muted); display: flex; justify-content: space-between; animation: fadeUp 0.5s ease 0.5s both; }}
  footer a {{ color: var(--muted); text-decoration: none; }}
  footer a:hover {{ color: var(--accent); }}
  @keyframes fadeDown {{ from {{ opacity: 0; transform: translateY(-16px); }} to {{ opacity: 1; transform: translateY(0); }} }}
  @keyframes fadeUp {{ from {{ opacity: 0; transform: translateY(16px); }} to {{ opacity: 1; transform: translateY(0); }} }}
  @media (max-width: 600px) {{ .container {{ padding: 36px 18px 60px; }} .summary-card, .news-card {{ padding: 20px 18px; }} footer {{ flex-direction: column; gap: 6px; text-align: center; }} .topic-name {{ width: 80px; font-size: 11px; }} .links-banner {{ flex-direction: column; }} }}
</style>
</head>
<body>
<div class="container">
  <header>
    <div class="logo">\U0001f399</div>
    <div class="header-text">
      <h1>Grief Research Daily &middot; \u60b2\u50b7\u7814\u7a76\u6587\u737b\u65e5\u5831</h1>
      <div class="header-meta">
        <span class="badge badge-date">\U0001f4c5 {date_display}</span>
        <span class="badge badge-count">\U0001f4ca {total_count} \u7bc7\u6587\u737b</span>
        <span class="badge badge-source">Powered by PubMed + {model_used}</span>
      </div>
    </div>
  </header>

  <div class="summary-card">
    <h2>\U0001f4cb \u4eca\u65e5\u60b2\u50b7\u7814\u7a76\u8da8\u52e2</h2>
    <p class="summary-text">{summary}</p>
  </div>

  {"<div class='section'><div class='section-title'><span class='section-icon'>\u2b50</span>\u4eca\u65e5\u7cbe\u9078 TOP Picks</div>" + top_picks_html + "</div>" if top_picks_html else ""}

  {"<div class='section'><div class='section-title'><span class='section-icon'>\U0001f4da</span>\u5176\u4ed6\u503c\u5f97\u95dc\u6ce8\u7684\u6587\u737b</div>" + all_papers_html + "</div>" if all_papers_html else ""}

  {"<div class='topic-section section'><div class='section-title'><span class='section-icon'>\U0001f4ca</span>\u4e3b\u984c\u5206\u4f48</div>" + topic_bars_html + "</div>" if topic_bars_html else ""}

  {"<div class='keywords-section section'><div class='section-title'><span class='section-icon'>\U0001f3f7\ufe0f</span>\u95dc\u9375\u5b57</div><div class='keywords'>" + keywords_html + "</div></div>" if keywords_html else ""}

  <div class="links-banner">
    <a href="https://www.leepsyclinic.com/" class="link-card" target="_blank">
      <span class="link-icon">\U0001f3e5</span>
      <span class="link-info">
        <span class="link-name">\u674e\u653f\u6d0b\u8eab\u5fc3\u8a3a\u6240\u9996\u9801</span>
        <span class="link-desc">www.leepsyclinic.com</span>
      </span>
      <span class="link-arrow">\u2192</span>
    </a>
    <a href="https://blog.leepsyclinic.com/" class="link-card" target="_blank">
      <span class="link-icon">\U0001f4e8</span>
      <span class="link-info">
        <span class="link-name">\u8a02\u95b1\u96fb\u5b50\u5831</span>
        <span class="link-desc">blog.leepsyclinic.com</span>
      </span>
      <span class="link-arrow">\u2192</span>
    </a>
    <a href="https://buymeacoffee.com/CYlee" class="link-card" target="_blank">
      <span class="link-icon">\u2615</span>
      <span class="link-info">
        <span class="link-name">Buy Me a Coffee</span>
        <span class="link-desc">buymeacoffee.com/CYlee</span>
      </span>
      <span class="link-arrow">\u2192</span>
    </a>
  </div>

  <footer>
    <span>\u8cc7\u6599\u4f86\u6e90\uff1aPubMed &middot; \u5206\u6790\u6a21\u578b\uff1a{model_used}</span>
    <span><a href="https://github.com/u8901006/prolong-grief">GitHub</a></span>
  </footer>
</div>
</body>
</html>"""

    return html


def main():
    parser = argparse.ArgumentParser(
        description="Generate grief research daily report HTML"
    )
    parser.add_argument("--input", required=True, help="Input papers JSON file")
    parser.add_argument("--output", required=True, help="Output HTML file")
    parser.add_argument(
        "--api-key", default=os.environ.get("ZHIPU_API_KEY", ""), help="Zhipu API key"
    )
    args = parser.parse_args()

    if not args.api_key:
        print(
            "[ERROR] No API key provided. Set ZHIPU_API_KEY env var or use --api-key",
            file=sys.stderr,
        )
        sys.exit(1)

    papers_data = load_papers(args.input)
    if not papers_data or not papers_data.get("papers"):
        print("[WARN] No papers found, generating empty report", file=sys.stderr)
        tz_taipei = timezone(timedelta(hours=8))
        analysis = {
            "date": datetime.now(tz_taipei).strftime("%Y-%m-%d"),
            "market_summary": "\u4eca\u65e5 PubMed \u66ab\u7121\u65b0\u7684\u60b2\u50b7\u8207\u55aa\u89aa\u7814\u7a76\u6587\u737b\u66f4\u65b0\u3002\u8acb\u660e\u5929\u518d\u67e5\u770b\u3002",
            "top_picks": [],
            "all_papers": [],
            "keywords": [],
            "topic_distribution": {},
            "_model_used": "N/A",
        }
    else:
        analysis = analyze_papers(args.api_key, papers_data)
        if not analysis:
            print("[ERROR] Analysis failed, cannot generate report", file=sys.stderr)
            sys.exit(1)

    html = generate_html(analysis)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[INFO] Report saved to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
