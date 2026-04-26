#!/usr/bin/env python3
"""
Fetch latest grief & bereavement research papers from PubMed E-utilities API.
Targets grief-related journals and covers prolonged grief, bereavement, mourning,
perinatal loss, suicide bereavement, palliative care grief, and related topics.
Tracks processed PMIDs to avoid duplicate daily reports.
"""

import json
import sys
import argparse
import os
import xml.etree.ElementTree as ET
from datetime import datetime, timezone, timedelta
from urllib.request import urlopen, Request
from urllib.error import URLError
from urllib.parse import quote_plus

PUBMED_SEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
PUBMED_FETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

JOURNALS = [
    "The American Journal of Psychiatry",
    "JAMA Psychiatry",
    "The Lancet Psychiatry",
    "World Psychiatry",
    "Molecular Psychiatry",
    "Translational Psychiatry",
    "Psychological Medicine",
    "Journal of Affective Disorders",
    "Depression and Anxiety",
    "Acta Psychiatrica Scandinavica",
    "British Journal of Psychiatry",
    "Frontiers in Psychiatry",
    "BMC Psychiatry",
    "Biological Psychiatry",
    "Neuropsychopharmacology",
    "Brain Behavior and Immunity",
    "Psychoneuroendocrinology",
    "Neuroscience and Biobehavioral Reviews",
    "Social Cognitive and Affective Neuroscience",
    "Journal of Consulting and Clinical Psychology",
    "Clinical Psychological Science",
    "Clinical Psychology Review",
    "Behaviour Research and Therapy",
    "Journal of Anxiety Disorders",
    "Traumatology",
    "Journal of Traumatic Stress",
    "Death Studies",
    "Omega",
    "Journal of Loss and Trauma",
    "Journal of Pain and Symptom Management",
    "Palliative Medicine",
    "Palliative and Supportive Care",
    "Journal of Palliative Medicine",
    "BMC Palliative Care",
    "Psycho-Oncology",
    "Supportive Care in Cancer",
    "Suicide and Life-Threatening Behavior",
    "Crisis",
    "Archives of Suicide Research",
    "The Gerontologist",
    "Aging and Mental Health",
    "International Psychogeriatrics",
    "Journal of the American Academy of Child and Adolescent Psychiatry",
    "Journal of Child Psychology and Psychiatry",
    "Development and Psychopathology",
    "Birth",
    "BMC Pregnancy and Childbirth",
    "Archives of Women's Mental Health",
    "Social Science and Medicine",
    "PLOS ONE",
    "BMJ Open",
]

GRIEF_QUERY_CORE = (
    '("Bereavement"[Mesh] OR "Grief"[Mesh] OR "Prolonged Grief Disorder"[Mesh] '
    'OR "Depression"[Mesh] OR "Stress Disorders, Post-Traumatic"[Mesh] '
    'OR "Suicide"[Mesh] OR "Palliative Care"[Mesh] OR "Hospice Care"[Mesh] '
    'OR "Caregivers"[Mesh] OR "Loneliness"[Mesh] OR "Attachment Behavior"[Mesh])'
)

GRIEF_QUERY_TEXT = (
    '(bereavement[tiab] OR bereaved[tiab] OR grief[tiab] OR grieving[tiab] '
    'OR mourning[tiab] OR widowhood[tiab] OR "prolonged grief"[tiab] '
    'OR "prolonged grief disorder"[tiab] OR "complicated grief"[tiab] '
    'OR "complex grief"[tiab] OR "persistent complex bereavement disorder"[tiab] '
    'OR "pathological grief"[tiab] OR "traumatic grief"[tiab] OR "chronic grief"[tiab] '
    'OR "ambiguous loss"[tiab] OR "disenfranchised grief"[tiab] '
    'OR "anticipatory grief"[tiab] OR "suicide bereavement"[tiab] '
    'OR "traumatic bereavement"[tiab] OR "perinatal loss"[tiab] '
    'OR stillbirth[tiab] OR miscarriage[tiab] OR "pregnancy loss"[tiab] '
    'OR "child bereavement"[tiab] OR "adolescent grief"[tiab] '
    'OR "parental death"[tiab] OR "continuing bonds"[tiab] '
    'OR "meaning reconstruction"[tiab] OR "bereavement care"[tiab] '
    'OR "bereavement support"[tiab] OR "homicide bereavement"[tiab])'
)

HEADERS = {"User-Agent": "GriefResearchBot/1.0 (research aggregator)"}
PROCESSED_FILE = "data/processed_pmids.json"


def load_processed_pmids():
    if os.path.exists(PROCESSED_FILE):
        try:
            with open(PROCESSED_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                return set(data.get("pmids", []))
        except (json.JSONDecodeError, IOError):
            return set()
    return set()


def save_processed_pmids(pmids_set):
    os.makedirs(os.path.dirname(PROCESSED_FILE), exist_ok=True)
    pmids_list = sorted(pmids_set)
    cutoff = datetime.now(timezone.utc) - timedelta(days=90)
    with open(PROCESSED_FILE, "w", encoding="utf-8") as f:
        json.dump({"last_updated": datetime.now(timezone.utc).isoformat(),
                    "count": len(pmids_list), "pmids": pmids_list},
                   f, ensure_ascii=False, indent=2)


def build_query(days: int = 7, use_journals: bool = True) -> str:
    lookback = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y/%m/%d")
    date_part = f'"{lookback}"[Date - Publication] : "3000"[Date - Publication]'
    grief_block = f"({GRIEF_QUERY_CORE} OR {GRIEF_QUERY_TEXT})"

    if use_journals:
        journal_part = " OR ".join([f'"{j}"[Journal]' for j in JOURNALS[:25]])
        return f"({journal_part}) AND {grief_block} AND {date_part}"
    else:
        return f"{grief_block} AND {date_part}"


def search_papers(query: str, retmax: int = 60) -> list[str]:
    params = (
        f"?db=pubmed&term={quote_plus(query)}&retmax={retmax}"
        f"&sort=date&retmode=json"
    )
    url = PUBMED_SEARCH + params
    try:
        req = Request(url, headers=HEADERS)
        with urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())
        return data.get("esearchresult", {}).get("idlist", [])
    except Exception as e:
        print(f"[ERROR] PubMed search failed: {e}", file=sys.stderr)
        return []


def fetch_details(pmids: list[str]) -> list[dict]:
    if not pmids:
        return []
    all_papers = []
    batch_size = 50
    for i in range(0, len(pmids), batch_size):
        batch = pmids[i:i + batch_size]
        ids = ",".join(batch)
        params = f"?db=pubmed&id={ids}&retmode=xml"
        url = PUBMED_FETCH + params
        try:
            req = Request(url, headers=HEADERS)
            with urlopen(req, timeout=60) as resp:
                xml_data = resp.read().decode()
        except Exception as e:
            print(f"[ERROR] PubMed fetch failed: {e}", file=sys.stderr)
            continue

        try:
            root = ET.fromstring(xml_data)
            for article in root.findall(".//PubmedArticle"):
                medline = article.find(".//MedlineCitation")
                art = medline.find(".//Article") if medline else None
                if art is None:
                    continue

                title_el = art.find(".//ArticleTitle")
                title = ""
                if title_el is not None:
                    title = "".join(title_el.itertext()).strip()

                abstract_parts = []
                for abs_el in art.findall(".//Abstract/AbstractText"):
                    label = abs_el.get("Label", "")
                    text = "".join(abs_el.itertext()).strip()
                    if label and text:
                        abstract_parts.append(f"{label}: {text}")
                    elif text:
                        abstract_parts.append(text)
                abstract = " ".join(abstract_parts)[:3000]

                journal_el = art.find(".//Journal/Title")
                journal = ""
                if journal_el is not None and journal_el.text:
                    journal = journal_el.text.strip()

                pub_date = art.find(".//PubDate")
                date_str = ""
                if pub_date is not None:
                    year = pub_date.findtext("Year", "")
                    month = pub_date.findtext("Month", "")
                    day = pub_date.findtext("Day", "")
                    parts = [p for p in [year, month, day] if p]
                    date_str = " ".join(parts)

                pmid_el = medline.find(".//PMID")
                pmid = pmid_el.text if pmid_el is not None else ""
                link = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else ""

                keywords = []
                for kw in medline.findall(".//KeywordList/Keyword"):
                    if kw.text:
                        keywords.append(kw.text.strip())

                authors = []
                for author in art.findall(".//AuthorList/Author")[:6]:
                    ln = author.findtext("LastName", "")
                    ini = author.findtext("Initials", "")
                    if ln:
                        authors.append(f"{ln} {ini}")
                author_str = ", ".join(authors)
                if len(art.findall(".//AuthorList/Author")) > 6:
                    author_str += " et al."

                all_papers.append({
                    "pmid": pmid,
                    "title": title,
                    "authors": author_str,
                    "journal": journal,
                    "date": date_str,
                    "abstract": abstract,
                    "url": link,
                    "keywords": keywords,
                })
        except ET.ParseError as e:
            print(f"[ERROR] XML parse failed: {e}", file=sys.stderr)

    return all_papers


def main():
    parser = argparse.ArgumentParser(description="Fetch grief papers from PubMed")
    parser.add_argument("--days", type=int, default=7, help="Lookback days")
    parser.add_argument("--max-papers", type=int, default=50, help="Max papers to fetch")
    parser.add_argument("--output", default="-", help="Output file (- for stdout)")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    processed = load_processed_pmids()
    print(f"[INFO] Already processed PMIDs: {len(processed)}", file=sys.stderr)

    query = build_query(days=args.days, use_journals=True)
    print(f"[INFO] Searching PubMed for grief papers from last {args.days} days...",
          file=sys.stderr)

    pmids = search_papers(query, retmax=args.max_papers)
    print(f"[INFO] Found {len(pmids)} papers from journal-filtered search", file=sys.stderr)

    if len(pmids) < args.max_papers // 2:
        broad_query = build_query(days=args.days, use_journals=False)
        extra_pmids = search_papers(broad_query, retmax=args.max_papers - len(pmids))
        new_pmids = [p for p in extra_pmids if p not in set(pmids)]
        pmids.extend(new_pmids)
        print(f"[INFO] Broad search added {len(new_pmids)} more papers", file=sys.stderr)

    new_pmids = [p for p in pmids if p not in processed]
    print(f"[INFO] After dedup: {len(new_pmids)} new papers (skipped {len(pmids) - len(new_pmids)} already processed)",
          file=sys.stderr)

    if not new_pmids:
        print("[INFO] No new papers to process", file=sys.stderr)
        tz_taipei = timezone(timedelta(hours=8))
        output_data = {
            "date": datetime.now(tz_taipei).strftime("%Y-%m-%d"),
            "count": 0,
            "papers": [],
        }
    else:
        papers = fetch_details(new_pmids)
        print(f"[INFO] Fetched details for {len(papers)} papers", file=sys.stderr)

        new_pmids_set = set(p["pmid"] for p in papers)
        all_processed = processed | new_pmids_set
        save_processed_pmids(all_processed)

        tz_taipei = timezone(timedelta(hours=8))
        output_data = {
            "date": datetime.now(tz_taipei).strftime("%Y-%m-%d"),
            "count": len(papers),
            "papers": papers,
        }

    out_str = json.dumps(output_data, ensure_ascii=False, indent=2)

    if args.output == "-":
        print(out_str)
    else:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(out_str)
        print(f"[INFO] Saved to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
