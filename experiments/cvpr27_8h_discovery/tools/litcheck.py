#!/usr/bin/env python3
"""litcheck — literature occupancy check against the arXiv API.

The app's built-in WebSearch/WebFetch do not work in this environment (empty
results / domain verification failure), while plain `curl` to
https://export.arxiv.org works. This wraps that path so every novelty claim can
be checked the same way and the result recorded.

Usage
  python litcheck.py "training-free granularity alignment 3D Gaussian"
  python litcheck.py --q 'all:"open-vocabulary" AND all:"3D"' --since 2025-01-01
  python litcheck.py --ids 2608.16103,2610.16177
  python litcheck.py "conformal prediction 3D retrieval" --abstracts --record NOVELTY_LOG.md

Exit codes: 0 ok, 2 network/API failure (then treat the claim as UNVERIFIED).
"""
import argparse, json, os, re, subprocess, sys, time
import urllib.parse

API = "https://export.arxiv.org/api/query"
UA = "litcheck/1.0 (research novelty audit)"


def curl(url, tries=3):
    for k in range(tries):
        p = subprocess.run(
            ["curl", "-sSL", "--max-time", "45", "-A", UA, url],
            capture_output=True)
        if p.returncode == 0 and p.stdout:
            return p.stdout.decode("utf-8", "ignore")
        time.sleep(2 + 2 * k)
    return ""


def build_query(text, terms):
    if terms:
        toks = [t.strip() for t in re.split(r"[,;]", terms) if t.strip()]
    else:
        toks = [t for t in re.findall(r"[A-Za-z][A-Za-z0-9\-]+", text)
                if len(t) > 2 and t.lower() not in STOP]
    return " AND ".join('all:"%s"' % t for t in toks)


STOP = {"the", "and", "for", "with", "from", "that", "this", "into", "than",
        "using", "based", "via", "our", "can", "not", "but", "are", "was",
        "has", "have", "its", "their", "which", "when", "what", "does"}


def parse(xml):
    out = []
    for e in re.findall(r"<entry>(.*?)</entry>", xml, re.S):
        def g(tag):
            m = re.search(r"<%s[^>]*>(.*?)</%s>" % (tag, tag), e, re.S)
            return re.sub(r"\s+", " ", m.group(1)).strip() if m else ""
        out.append(dict(
            id=g("id").split("/")[-1],
            title=g("title"),
            published=g("published")[:10],
            summary=g("summary"),
            authors=", ".join(re.findall(r"<name>(.*?)</name>", e))[:80]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("text", nargs="*", help="free-text claim/description")
    ap.add_argument("--q", help="raw arXiv query (overrides text)")
    ap.add_argument("--terms", help="comma separated AND terms (overrides text)")
    ap.add_argument("--ids", help="comma separated arXiv ids -> abstracts")
    ap.add_argument("--cat", default="", help="restrict to an arXiv category, e.g. cs.CV")
    ap.add_argument("--since", default="", help="YYYY-MM-DD lower bound")
    ap.add_argument("--max", type=int, default=25)
    ap.add_argument("--abstracts", action="store_true")
    ap.add_argument("--json", help="write results to this json file")
    ap.add_argument("--record", help="append a markdown block to this file")
    a = ap.parse_args()

    if a.ids:
        url = "%s?id_list=%s&max_results=%d" % (API, a.ids, len(a.ids.split(",")))
        q = "(by id)"
    else:
        q = a.q or build_query(" ".join(a.text), a.terms)
        if a.cat and q:
            q = "cat:%s AND (%s)" % (a.cat, q)
        if not q:
            sys.exit("no query: pass text, --q or --terms")
        url = ("%s?search_query=%s&start=0&max_results=%d"
               "&sortBy=submittedDate&sortOrder=descending"
               % (API, urllib.parse.quote(q), a.max))

    xml = curl(url)
    if not xml:
        print("LITCHECK_FAILED: no response from arXiv (claim must be marked UNVERIFIED)")
        sys.exit(2)
    hits = parse(xml)
    if a.since:
        hits = [h for h in hits if h["published"] >= a.since]

    print("query: %s" % q)
    print("hits: %d%s" % (len(hits), (" (since %s)" % a.since) if a.since else ""))
    for h in hits:
        print("  %s  %s  %s" % (h["published"], h["id"], h["title"][:110]))
        if a.abstracts:
            print("      %s" % h["summary"][:400])

    if a.json:
        json.dump(dict(query=q, since=a.since, hits=hits), open(a.json, "w"), indent=1)
        print("wrote %s" % a.json)

    if a.record:
        block = ["", "## %s  %s" % (time.strftime("%Y-%m-%d %H:%M"), q), "",
                 "| date | id | title |", "|---|---|---|"]
        for h in hits[:12]:
            block.append("| %s | %s | %s |" % (h["published"], h["id"], h["title"][:100]))
        block += ["", "**Verdict:** OCCUPIED / ADJACENT / OPEN  (fill in)",
                  "**Delta we claim:** (fill in)", ""]
        with open(a.record, "a", encoding="utf-8") as f:
            f.write("\n".join(block) + "\n")
        print("appended to %s" % a.record)


if __name__ == "__main__":
    main()