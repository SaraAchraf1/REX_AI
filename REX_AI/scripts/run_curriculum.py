"""Run the Curriculum Research & Accreditation Alignment Agent from the command line (owner: Member 2).

    python scripts/run_curriculum.py --title "Data Analysis for Healthcare Quality" --audience "Quality officers" \
        --level Beginner --hours 40 --industry Healthcare --outcome "Build a KPI dashboard" --pdf
    python scripts/run_curriculum.py --course DS-101 --framework "SFIA 9" --country SA --pdf
    python scripts/run_curriculum.py --import-course SW-201 --framework ISTQB      # compare an existing course
    python scripts/run_curriculum.py --research --title "Cloud Security" --audience "IT staff" --level Intermediate --hours 30
    python scripts/run_curriculum.py --list --department Software
    python scripts/run_curriculum.py --gap CUR-1234ABCD --framework "NAQAAE"
    python scripts/run_curriculum.py --revise CUR-1234ABCD --reviewer "Dr. Mona" --instructions "add module Power BI"
    python scripts/run_curriculum.py --approve CUR-1234ABCD --reviewer "Dean"
    python scripts/run_curriculum.py --ingest-references                            # (re)index data/curriculum/references
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")                 # Arabic titles on the Windows console
    sys.stderr.reconfigure(encoding="utf-8")

from rex_ai.agents.curriculum.agent import CurriculumAgent, NotFound  # noqa: E402
from rex_ai.agents.curriculum.research import ensure_references_indexed  # noqa: E402
from rex_ai.core import db  # noqa: E402
from rex_ai.core.schemas import (CurriculumImport, CurriculumRequest, CurriculumRevision,  # noqa: E402
                                 ReviewAction)


def show(cur, agent: CurriculumAgent) -> None:
    m = cur.metadata
    print(f"\n{cur.curriculum_id}  v{cur.version}  [{cur.status.value}]  {cur.course_title}")
    print(f"  {m.get('department')} · {m.get('skill_level')} · {cur.training_hours:g} h · {m.get('country')} · "
          f"framework: {m.get('accreditation_framework')} · by {m.get('generated_by')} from {', '.join(m.get('evidence', []))}")
    print("  Objectives:")
    for o in cur.objectives:
        print(f"    - {o}")
    print("  Learning outcomes:")
    for i, o in enumerate(cur.learning_outcomes, 1):
        print(f"    {i}. {o}")
    print("  Modules:")
    for i, mod in enumerate(cur.modules, 1):
        print(f"    {i}. {mod.title} ({mod.hours:g} h): {'; '.join(mod.topics)}")
        for e in mod.exercises:
            print(f"         > {e}")
    print("  Assessments: " + ", ".join(f"{a.name} {a.weight_pct:g}%" for a in cur.assessments))
    print(f"  Capstone: {cur.capstone_project}")
    for r in agent.gap_reports(cur.curriculum_id):
        print(f"  Alignment {r.framework_id}: {r.coverage_pct:g}%"
              + (f" | missing: {', '.join(r.missing_topics + r.missing_outcomes)}" if r.missing_topics or r.missing_outcomes else "")
              + (f" | partial: {', '.join(r.partial)}" if r.partial else ""))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--title")
    ap.add_argument("--audience")
    ap.add_argument("--level", choices=["Beginner", "Intermediate", "Advanced"])
    ap.add_argument("--hours", type=int)
    ap.add_argument("--outcome", action="append", default=[], help="expected learning outcome (repeatable)")
    ap.add_argument("--project", action="append", default=[], help="required project (repeatable; first = capstone)")
    ap.add_argument("--industry")
    ap.add_argument("--department")
    ap.add_argument("--framework", help="e.g. 'ISTQB Foundation', 'SFIA 9', NAQAAE, NQF-KSA")
    ap.add_argument("--country", choices=["EG", "SA"], default="EG")
    ap.add_argument("--language", choices=["en", "ar"], default="en")
    ap.add_argument("--course", help="existing catalogue course id to redesign (fills the other answers)")
    ap.add_argument("--research", action="store_true", help="only show the research, store nothing")
    ap.add_argument("--import-course", help="import a catalogue course and compare it with --framework")
    ap.add_argument("--pdf", action="store_true", help="write the PDF after generating")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--show")
    ap.add_argument("--gap", metavar="CURRICULUM_ID")
    ap.add_argument("--revise", metavar="CURRICULUM_ID")
    ap.add_argument("--instructions", help="modification request, lines separated by ';'")
    ap.add_argument("--approve", metavar="CURRICULUM_ID")
    ap.add_argument("--reviewer", default="cli")
    ap.add_argument("--export", metavar="CURRICULUM_ID", help="write the PDF of a stored curriculum")
    ap.add_argument("--ingest-references", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.ERROR, format="%(levelname)s %(message)s")

    db.init_db()
    agent = CurriculumAgent()
    try:
        if args.ingest_references:
            print(ensure_references_indexed(force=True))
            return
        if args.list:
            for c in agent.search(department=args.department, country=args.country if "--country" in sys.argv else None,
                                  framework=args.framework, q=args.title):
                cov = " · ".join(f"{a['framework_id']} {a['coverage_pct']:g}%" for a in c.metadata.get("alignment", []))
                print(f"{c.curriculum_id}  v{c.version} {c.status.value:<17} {c.course_title}  ({cov})")
            return
        if args.show:
            show(agent.get(args.show), agent)
            return
        if args.gap:
            r = agent.gap_report(args.gap, args.framework)
            print(f"{r.framework}: {r.coverage_pct:g}%")
            for i in r.items:
                print(f"  {i.status:<8} {i.competency}  ({i.evidence})")
            for rec in r.recommendations:
                print(f"  -> {rec}")
            print(r.disclaimer)
            return
        if args.revise:
            cur = agent.revise(args.revise, CurriculumRevision(reviewer=args.reviewer,
                                                               instructions=(args.instructions or "").replace(";", "\n")))
            print(cur.metadata.get("last_revision"))
            show(cur, agent)
            return
        if args.approve:
            cur = agent.review(args.approve, ReviewAction(action="approved", reviewer=args.reviewer))
            print(f"{cur.curriculum_id} approved by {args.reviewer}; handed to the Document Agent "
                  f"({cur.metadata.get('handoff_event')}). PDF: {agent.export_pdf(cur.curriculum_id)}")
            return
        if args.export:
            print(agent.export_pdf(args.export))
            return
        if args.import_course:
            cur = agent.import_curriculum(CurriculumImport(course_id=args.import_course, framework=args.framework,
                                                           country=args.country, language=args.language))
            show(cur, agent)
            return
        req = CurriculumRequest(course_id=args.course, course_title=args.title, target_audience=args.audience,
                                skill_level=args.level, total_hours=args.hours, expected_outcomes=args.outcome,
                                required_projects=args.project, industry=args.industry, department=args.department,
                                accreditation_framework=args.framework, country=args.country, language=args.language)
        if args.research:
            print(agent.research(req).model_dump_json(indent=2))
            return
        cur = agent.generate(req)
        show(cur, agent)
        if args.pdf:
            print(f"\nPDF: {agent.export_pdf(cur.curriculum_id)}")
    except NotFound as exc:
        sys.exit(f"Not found: {exc}")
    except ValueError as exc:
        sys.exit(f"Error: {exc}")


if __name__ == "__main__":
    main()
