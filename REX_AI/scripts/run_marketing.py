"""Run the AI Marketing Agent from the command line (owner: Member 1).

    python scripts/run_marketing.py --channels                                       # the 5 channels and their limits
    python scripts/run_marketing.py --course DS-101                                  # all channels, 2 variations each
    python scripts/run_marketing.py --course BU-101 --audience "Project managers" --channel linkedin_post \
        --channel email --objective "webinar sign-ups" --budget 5000 --variations 3
    python scripts/run_marketing.py --course DS-101 --language ar --channel whatsapp --full
    python scripts/run_marketing.py --course SW-201 --curriculum CUR-1234ABCD        # copy from that curriculum
    python scripts/run_marketing.py --course AI-201 --document DOC-1234ABCD          # from an approved document
    python scripts/run_marketing.py --list
    python scripts/run_marketing.py --show CMP-1234ABCD --full
    python scripts/run_marketing.py --edit CMP-1234ABCD --reviewer Admin --remove whatsapp:2
    python scripts/run_marketing.py --edit CMP-1234ABCD --reviewer Admin --instructions "shorter WhatsApp text"  # LLM
    python scripts/run_marketing.py --approve CMP-1234ABCD --reviewer Admin
    python scripts/run_marketing.py --publish CMP-1234ABCD --reviewer Admin --variant linkedin_post:1 --to lead@example.com
    python scripts/run_marketing.py --handoffs                                      # approved curricula / documents
    python scripts/run_marketing.py --ingest-refs                                   # (re)index the brand guide etc.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")                 # Arabic copy on the Windows console
    sys.stderr.reconfigure(encoding="utf-8")

from rex_ai.agents.marketing.agent import MarketingAgent, NotFound  # noqa: E402
from rex_ai.agents.marketing.rules import ensure_refs_indexed  # noqa: E402
from rex_ai.core import db  # noqa: E402
from rex_ai.core.schemas import CampaignEdit, CampaignPublish, CampaignRequest, Channel, ReviewAction  # noqa: E402


def show(c, full: bool = False) -> None:
    m = c.metadata
    print(f"\n{c.campaign_id}  v{c.version}  [{c.status.value}]  {c.title}")
    print(f"  {c.course_id} · {c.language.value} · by {m.get('generated_by')} · audience: "
          f"{c.strategy_plan.get('audience', {}).get('persona_en')}")
    print(f"  Brief PDF: {m.get('file_path')}")
    for n in m.get("notes") or []:
        print(f"  note: {n}")
    for n in m.get("llm_rejected") or []:
        print(f"  rejected: {n}")
    print("\n" + "\n".join(f"  {line}" for line in c.strategy.splitlines()))
    for v in c.variants:
        print(f"\n  - {v.key} [{v.angle}] {v.headline!r} -> CTA {v.cta!r} ({v.char_count} chars)")
        for check in v.checks:
            print(f"      check: {check}")
        if full:
            print("\n".join(f"      | {line}" for line in v.body.splitlines()))
    print("\n  Sources: " + ", ".join(dict.fromkeys(s.source_id for s in c.sources)))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--course", help="catalogue course id to promote")
    ap.add_argument("--curriculum", help="curriculum id (default: the course's newest approved curriculum)")
    ap.add_argument("--document", help="approved document id handed over by the Document Agent")
    ap.add_argument("--audience", help="target audience (default: the course's audience)")
    ap.add_argument("--channel", action="append", choices=[c.value for c in Channel], help="repeatable; default all")
    ap.add_argument("--objective", default="registrations", help="registrations | awareness | leads | webinar sign-ups")
    ap.add_argument("--offer")
    ap.add_argument("--cta")
    ap.add_argument("--budget", type=float)
    ap.add_argument("--variations", type=int, default=2)
    ap.add_argument("--language", choices=["en", "ar"], default="en")
    ap.add_argument("--wishes", help="extra instructions for the LLM")
    ap.add_argument("--channels", action="store_true", help="list the channels and their limits")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--show")
    ap.add_argument("--full", action="store_true", help="print the copy of every variant")
    ap.add_argument("--edit", metavar="CAMPAIGN_ID")
    ap.add_argument("--instructions")
    ap.add_argument("--strategy", help="with --edit: new strategy text")
    ap.add_argument("--remove", action="append", default=[], help="with --edit: variant key (email:2) or channel")
    ap.add_argument("--approve", metavar="CAMPAIGN_ID")
    ap.add_argument("--reject", metavar="CAMPAIGN_ID")
    ap.add_argument("--publish", metavar="CAMPAIGN_ID")
    ap.add_argument("--variant", action="append", default=[], help="with --publish: variant key (default: all)")
    ap.add_argument("--to", action="append", default=[], help="with --publish: e-mail the e-mail variant")
    ap.add_argument("--reviewer", default="cli")
    ap.add_argument("--handoffs", action="store_true")
    ap.add_argument("--ingest-refs", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.ERROR, format="%(levelname)s %(message)s")

    db.init_db()
    agent = MarketingAgent()
    try:
        if args.ingest_refs:
            print(ensure_refs_indexed(force=True))
            return
        if args.channels:
            for c in agent.channels():
                print(f"{c.channel.value:<22} headline <= {c.headline_max:<4} body <= {c.body_max:<5} "
                      f"hashtags {c.hashtags}  {'paid' if c.paid else 'organic'}  {c.name} / {c.name_ar}")
            return
        if args.list:
            for c in agent.search(course_id=args.course and args.course.upper()):
                print(f"{c.campaign_id}  v{c.version} {c.status.value:<17} {c.language.value} {c.title}")
            return
        if args.handoffs:
            for h in agent.pending_handoffs():
                print(f"{h['kind']:<10} {h['id']:<14} {h['course_id']:<7} campaigns: {h['campaigns']}  {h['title']}")
            return
        if args.show:
            show(agent.get(args.show), full=args.full)
            return
        if args.edit:
            c = agent.edit(args.edit, CampaignEdit(reviewer=args.reviewer, instructions=args.instructions,
                                                   strategy=args.strategy, remove_variants=args.remove))
            print(c.metadata.get("last_edit"))
            show(c)
            return
        if args.approve or args.reject:
            action = "approved" if args.approve else "rejected"
            c = agent.review(args.approve or args.reject, ReviewAction(action=action, reviewer=args.reviewer))
            print(f"{c.campaign_id} {action} by {args.reviewer}. Brief: {c.metadata.get('file_path')}")
            return
        if args.publish:
            c = agent.publish(args.publish, CampaignPublish(publisher=args.reviewer, variant_keys=args.variant,
                                                            email_to=args.to))
            print(c.metadata["publications"][-1])
            return
        if not args.course:
            ap.error("give --course (or one of --channels / --list / --show / --handoffs)")
        req = CampaignRequest(course_id=args.course, curriculum_id=args.curriculum, document_id=args.document,
                              target_audience=args.audience, channels=args.channel or list(Channel),
                              objective=args.objective, offer=args.offer, cta=args.cta, budget=args.budget,
                              variations_per_channel=args.variations, language=args.language,
                              instructions=args.wishes)
        show(agent.generate(req), full=args.full)
    except NotFound as exc:
        sys.exit(f"Not found: {exc}")
    except (ValueError, LookupError) as exc:
        sys.exit(f"Error: {exc}")


if __name__ == "__main__":
    main()
