#!/usr/bin/env python3
"""litcheck — literature occupancy check over arXiv + DBLP, through the local proxy.

Why this exists: the app's WebSearch returns empty for every query and WebFetch
cannot verify any domain from this machine. Plain curl works, and the local proxy
(127.0.0.1:7890) reaches even the domains that are blocked directly. This wraps
that path so novelty claims are checked the same way and the result is recorded.

Sources
  arxiv : preprints, full-text abstracts, newest first (cat: filter supported)
  cvf   : CVF open-access proceedings index (CVPR/ICCV/ECCV/WACV). Fetched once
          per venue-year (bulk HTML) and grepped locally -- no per-query API call,
          so no rate limit. DBLP is NOT used: it sits behind an Anubis anti-bot
          wall, and OpenAlex/Crossref return 429 on this shared proxy exit IP.

Usage
  python litcheck.py --q 'all:"open-vocabulary" AND all:"granularity"' --cat cs.CV --since 2025-01-01
  python litcheck.py --ids 2608.16103 --abstracts
  python litcheck.py --source cvf --venue CVPR2026,ICCV2025 --query "open-vocabulary gaussian"
  python litcheck.py --source both --q 'all:"conformal" AND all:segmentation' \
      --query "conformal 3D segmentation" --cat cs.CV --since 2025-01-01 --record NOVELTY_LOG.md

Exit codes: 0 ok, 2 no response (the claim must then be marked UNVERIFIED).
"""
import argparse, json, os, re, subprocess, sys, time
import urllib.parse

ARXIV = "https://export.arxiv.org/api/query"
CVF = "https://openaccess.thecvf.com/%s?day=all"   # e.g. CVPR2026, ICCV2025, ECCV2024
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache")
UA = "litcheck/1.1 (research novelty audit)"
DEFAULT_PROXY = "http://127.0.0.1:7890"

STOP = {"the", "and", "for", "with", "from", "that", "this", "into", "than",
        "using", "based", "via", "our", "can", "not", "but", "are", "was",
        "has", "have", "its", "their", "which", "when", "what", "does"}


def curl(url, proxy=None, tries=3):
    for k in range(tries):
        cmd = ["curl", "-sSL", "--max-time", "45", "-A", UA]
        if proxy:
            cmd += ["-x", proxy]
        p = subprocess.run(cmd + [url], capture_output=True)
        if p.returncode == 0 and p.stdout:
            return p.stdout.decode("utf-8", "ignore")
        time.sleep(2 + 2 * k)
    return ""


def build_query(text, terms):
    toks = ([t.strip() for t in re.split(r"[,;]", terms) if t.strip()] if terms else
            [t for t in re.findall(r"[A-Za-z][A-Za-z0-9\-]+", text)
             if len(t) > 2 and t.lower() not in STOP])
    return " AND ".join('all:"%s"' % t for t in toks)


def parse_arxiv(xml):
    out = []
    for e in re.findall(r"<entry>(.*?)</entry>", xml, re.S):
        def g(tag):
            m = re.search(r"<%s[^>]*>(.*?)</%s>" % (tag, tag), e, re.S)
            return re.sub(r"\s+", " ", m.group(1)).strip() if m else ""
        out.append(dict(source="arxiv", id=g("id").split("/")[-1], title=g("title"),
                        published=g("published")[:10], summary=g("summary"), venue="arXiv"))
    return out


def cvf_index(venue, proxy, refresh=False):
    """Bulk-download a CVF proceedings index once and cache it."""
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, "cvf_%s.html" % venue)
    fresh = os.path.exists(path) and (time.time() - os.path.getmtime(path)) < 7 * 86400
    if refresh or not fresh:
        txt = curl(CVF % venue, proxy)
        if txt:
            open(path, "w", encoding="utf-8").write(txt)
    if not os.path.exists(path):
        return []
    s = open(path, encoding="utf-8", errors="ignore").read()
    out = []
    for m in re.finditer(r'<dt class="ptitle">(.*?)</dt>', s, re.S):
        blk = m.group(1)
        a = re.search(r'<a href="([^"]+)"[^>]*>(.*?)</a>', blk, re.S)
        if not a:
            continue
        title = re.sub(r"<[^>]+>", "", a.group(2))
        title = re.sub(r"\s+", " ", title).strip()
        out.append(dict(source="cvf", id=a.group(1).split("/")[-1].replace(".html", ""),
                        title=title, published=venue[-4:], venue=venue, summary="",
                        url="https://openaccess.thecvf.com" + a.group(1)))
    return out


def cvf_match(items, terms):
    toks = [t.lower() for t in terms if t]
    out = []
    for it in items:
        tl = it["title"].lower()
        if all(t in tl for t in toks):
            out.append(it)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("text", nargs="*", help="free-text claim (arxiv)")
    ap.add_argument("--source", choices=["arxiv", "cvf", "both"], default="arxiv")
    ap.add_argument("--venue", default="CVPR2026", help="comma list, e.g. CVPR2026,ICCV2025")
    ap.add_argument("--refresh", action="store_true", help="re-download the CVF index")
    ap.add_argument("--q", help="raw arXiv query (overrides text)")
    ap.add_argument("--query", help="free-text query for DBLP")
    ap.add_argument("--terms", help="comma separated AND terms (overrides text)")
    ap.add_argument("--ids", help="comma separated arXiv ids -> abstracts")
    ap.add_argument("--cat", default="", help="arXiv category filter, e.g. cs.CV")
    ap.add_argument("--since", default="", help="YYYY-MM-DD lower bound")
    ap.add_argument("--max", type=int, default=25)
    ap.add_argument("--abstracts", action="store_true")
    ap.add_argument("--proxy", default=os.environ.get("LITCHECK_PROXY", DEFAULT_PROXY))
    ap.add_argument("--no-proxy", action="store_true")
    ap.add_argument("--json", help="write results to this json file")
    ap.add_argument("--record", help="append a markdown block to this file")
    a = ap.parse_args()
    proxy = None if a.no_proxy else (a.proxy or None)

    hits, queries = [], []
    if a.source in ("arxiv", "both"):
        if a.ids:
            q = "(by id)"
            url = "%s?id_list=%s&max_results=%d" % (ARXIV, a.ids, len(a.ids.split(",")))
        else:
            q = a.q or build_query(" ".join(a.text), a.terms)
            if not q:
                sys.exit("no arXiv query: pass text, --q or --terms")
            if a.cat:
                q = "cat:%s AND (%s)" % (a.cat, q)
            url = ("%s?search_query=%s&start=0&max_results=%d"
                   "&sortBy=submittedDate&sortOrder=descending"
                   % (ARXIV, urllib.parse.quote(q), a.max))
        xml = curl(url, proxy)
        if not xml:
            print("LITCHECK_FAILED: no arXiv response (claim must be UNVERIFIED)")
            sys.exit(2)
        h = parse_arxiv(xml)
        if a.since:
            h = [x for x in h if x["published"] >= a.since]
        hits += h
        queries.append("arxiv: " + q)

    if a.source in ("cvf", "both"):
        cq = a.query or " ".join(a.text) or a.q or ""
        toks = ([t.strip() for t in re.split(r"[,;]", a.terms) if t.strip()] if a.terms
                else [t for t in re.findall(r"[A-Za-z][A-Za-z0-9\-]+", cq)
                      if len(t) > 2 and t.lower() not in STOP])
        if not toks:
            sys.exit("no CVF query: pass --query or --terms")
        for ven in [v.strip() for v in a.venue.split(",") if v.strip()]:
            items = cvf_index(ven, proxy, a.refresh)
            if not items:
                print("LITCHECK_WARN: no CVF index for %s" % ven)
                continue
            h = cvf_match(items, toks)
            if a.since:
                h = [x for x in h if x["published"] >= a.since[:4]]
            h = h[: a.max]
            hits += h
            queries.append("cvf: %s  [%s]  (%d titles in index)" % (" ".join(toks), ven, len(items)))

    for q in queries:
        print("query: %s" % q)
    print("hits: %d%s" % (len(hits), (" (since %s)" % a.since) if a.since else ""))
    for h in hits:
        ven = (" [%s]" % h["venue"]) if h.get("venue") and h["source"] == "cvf" else ""
        print("  %s  %s%s  %s" % (h["published"], h["id"][:24], ven, h["title"][:100]))
        if a.abstracts and h["summary"]:
            print("      %s" % h["summary"][:400])

    if a.json:
        json.dump(dict(queries=queries, since=a.since, hits=hits), open(a.json, "w"), indent=1)
        print("wrote %s" % a.json)

    if a.record:
        block = ["", "## %s  %s" % (time.strftime("%Y-%m-%d %H:%M"), " | ".join(queries)), "",
                 "| date | id | venue | title |", "|---|---|---|---|"]
        for h in hits[:14]:
            block.append("| %s | %s | %s | %s |" % (h["published"], h["id"][:24],
                                                    h.get("venue", ""), h["title"][:90]))
        block += ["", "**Verdict:** OCCUPIED / ADJACENT / OPEN  (fill in)",
                  "**Delta we claim:** (fill in)", ""]
        with open(a.record, "a", encoding="utf-8") as f:
            f.write("\n".join(block) + "\n")
        print("appended to %s" % a.record)


if __name__ == "__main__":
    main()