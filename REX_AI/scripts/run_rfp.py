"""Run the Educational RFP & Tender Research Agent from the command line (owner: Member 2).

    python scripts/run_rfp.py                          # feeds + tender documents, Egypt and Saudi Arabia
    python scripts/run_rfp.py --country EG --days 45   # Egypt only, deadlines in the next 45 days
    python scripts/run_rfp.py --live                   # also read the public portals in data_sources.yaml
    python scripts/run_rfp.py --list                   # show the stored pipeline
    python scripts/run_rfp.py --ask "which tenders need cybersecurity training?"
    python scripts/run_rfp.py --export rfps.csv        # CSV for Excel / Google Sheets

Pin the date for the fixed sample data with REX_RFP_TODAY=2026-10-24.
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")                 # Arabic titles on the Windows console
    sys.stderr.reconfigure(encoding="utf-8")

from rex_ai.agents.rfp.agent import RFPResearchAgent  # noqa: E402
from rex_ai.core import db  # noqa: E402
from rex_ai.core.schemas import Country, RFPAskRequest, RFPCategory, RFPRunRequest  # noqa: E402


def show(records, priority_orgs=()) -> None:
    if not records:
        print("\nNo tenders match these filters.")
    for r in records:
        star = "  * priority organisation" if RFPResearchAgent.is_priority(r, list(priority_orgs)) else ""
        print(f"\n{r.rfp_id}  [{r.country.value}] {r.status.value:<15} fit={r.qualification.value:<9} "
              f"deadline={r.deadline or '-'}{star}")
        print(f"  {r.title}\n  {r.issuing_organization} | {r.training_category.value}")
        print(f"  {r.summary}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--country", choices=["EG", "SA"], action="append", help="repeatable; default both")
    ap.add_argument("--category", choices=[c.name for c in RFPCategory], action="append")
    ap.add_argument("--days", type=int, help="only deadlines within this many days")
    ap.add_argument("--keyword", action="append", default=[],
                    help="only tenders mentioning this topic (repeatable: any of them)")
    ap.add_argument("--priority-org", action="append", default=[],
                    help="organisation/sector to rank first and mark with * (repeatable)")
    ap.add_argument("--live", action="store_true", help="also fetch the public tender portals")
    ap.add_argument("--no-notify", action="store_true")
    ap.add_argument("--list", action="store_true", help="list stored RFPs (the filters above apply)")
    ap.add_argument("--ask", help="question about the stored tenders (RAG)")
    ap.add_argument("--export", help="write stored RFPs to this CSV file (the filters above apply)")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.ERROR, format="%(levelname)s %(message)s")

    db.init_db()
    agent = RFPResearchAgent()
    req = RFPRunRequest(
        countries=[Country(c) for c in args.country] if args.country else [Country.EG, Country.SA],
        categories=[RFPCategory[c] for c in args.category] if args.category else list(RFPCategory),
        deadline_to=agent.today + timedelta(days=args.days) if args.days else None,
        keywords=args.keyword, priority_organizations=args.priority_org,
        include_live_sources=args.live, notify=not args.no_notify)
    if args.ask:
        country = req.countries[0] if len(req.countries) == 1 else None
        res = agent.ask(RFPAskRequest(question=args.ask, country=country))
        print(res.answer)
        for s in res.sources:
            print(f"  - {s.source_id}: {s.title}")
        return
    if args.list or args.export:
        records = agent.filter_stored(agent.search(), req)
        if args.export:
            Path(args.export).write_text(agent.export_csv(records), encoding="utf-8")
            print(f"{len(records)} RFPs written to {args.export}")
        else:
            show(records, args.priority_org)
        return

    res = agent.run(req)
    print(f"Job {res.job_id}: found {res.found}, kept {res.kept} (new Egypt flagged: {len(res.new_flagged_egypt)}, "
          f"new Saudi monitoring: {len(res.new_monitoring_saudi)}, already known: {res.already_known})")
    print(f"Rejected - country: {res.rejected_country}, not training: {res.rejected_irrelevant}, "
          f"expired: {res.expired}, duplicates: {res.duplicates}, filters: {res.rejected_filters}. "
          f"Alerts: {res.alerts_sent}")
    for e in res.source_errors:
        print(f"  source error {e['source_id']}: {e['error']}")
    if args.verbose:
        for r in res.rejections:
            print(f"  x {r['rejected']:<20} {r['title'][:60]}  ({r['reason']})")
    show(res.records, args.priority_org)


if __name__ == "__main__":
    main()
