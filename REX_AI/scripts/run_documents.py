"""Run the Educational Content & Document Generation Agent from the command line (owner: Member 3).

    python scripts/run_documents.py --types                                         # the 10 types and templates
    python scripts/run_documents.py --type participant_guide --curriculum CUR-1234ABCD
    python scripts/run_documents.py --type course_proposal --course DS-101 --audience "Ministry of Health"
    python scripts/run_documents.py --type question_bank --course DS-101 --language ar
    python scripts/run_documents.py --pack --curriculum CUR-1234ABCD                # description + guide + question bank
    python scripts/run_documents.py --all --course SW-201                           # all 10 types
    python scripts/run_documents.py --list
    python scripts/run_documents.py --show DOC-1234ABCD
    python scripts/run_documents.py --edit DOC-1234ABCD --reviewer "Ms Salma" --title "QA Participant Guide"
    python scripts/run_documents.py --edit DOC-1234ABCD --reviewer QA --instructions "use banking examples"   # LLM
    python scripts/run_documents.py --approve DOC-1234ABCD --reviewer Dean
    python scripts/run_documents.py --email DOC-1234ABCD --to coordinator@academy.example --reviewer Dean
    python scripts/run_documents.py --handoffs                                      # approved curricula waiting
    python scripts/run_documents.py --ingest-templates                              # (re)index the reference examples
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

from rex_ai.agents.documents.agent import DocumentAgent, NotFound  # noqa: E402
from rex_ai.agents.documents.templates import ensure_templates_indexed  # noqa: E402
from rex_ai.core import db  # noqa: E402
from rex_ai.core.schemas import (DocumentEdit, DocumentEmail, DocumentPackRequest, DocumentRequest,  # noqa: E402
                                 DocumentType, ReviewAction)


def show(doc, full: bool = False) -> None:
    m = doc.metadata
    print(f"\n{doc.document_id}  v{doc.version}  [{doc.status.value}]  {doc.title}")
    print(f"  {doc.doc_type.value} · {doc.language.value} · {doc.template_id} · by {m.get('generated_by')} from "
          f"{doc.curriculum_id or doc.course_id} ({m.get('origin')})")
    print(f"  PDF: {doc.file_path}")
    for n in m.get("notes") or []:
        print(f"  note: {n}")
    for s in doc.sections:
        indent = "  " if s.level == 1 else "      "
        print(f"{indent}- {s.heading}" + (" [locked]" if s.locked else ""))
        if full:
            for p in s.paragraphs:
                print(f"{indent}    {p}")
            for b in s.bullets:
                print(f"{indent}    * {b}")
            if s.table:
                print(f"{indent}    | " + " | ".join(s.table.columns))
                for r in s.table.rows:
                    print(f"{indent}    | " + " | ".join(r))
    print("  Sources: " + ", ".join(dict.fromkeys(s.source_id for s in doc.sources)))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--type", choices=[t.value for t in DocumentType])
    ap.add_argument("--curriculum", help="curriculum id (CUR-...)")
    ap.add_argument("--course", help="catalogue course id (uses its newest approved curriculum, else the catalogue)")
    ap.add_argument("--language", choices=["en", "ar"], help="default: the curriculum's language")
    ap.add_argument("--audience", help="who it is prepared for, e.g. the client of a proposal")
    ap.add_argument("--template", help="template id from --types (default: the type's default template)")
    ap.add_argument("--wishes", help="extra instructions for the LLM")
    ap.add_argument("--email-to", action="append", default=[], help="send after approval (repeatable)")
    ap.add_argument("--pack", action="store_true", help="course description + participant guide + question bank")
    ap.add_argument("--all", action="store_true", help="all 10 document types")
    ap.add_argument("--types", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--show")
    ap.add_argument("--full", action="store_true", help="with --show: print the content")
    ap.add_argument("--edit", metavar="DOCUMENT_ID")
    ap.add_argument("--title")
    ap.add_argument("--instructions")
    ap.add_argument("--remove", action="append", default=[], help="with --edit: section key to remove")
    ap.add_argument("--approve", metavar="DOCUMENT_ID")
    ap.add_argument("--submit", metavar="DOCUMENT_ID", help="submit for approval")
    ap.add_argument("--email", metavar="DOCUMENT_ID")
    ap.add_argument("--to", action="append", default=[])
    ap.add_argument("--reviewer", default="cli")
    ap.add_argument("--handoffs", action="store_true")
    ap.add_argument("--ingest-templates", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.ERROR, format="%(levelname)s %(message)s")

    db.init_db()
    agent = DocumentAgent()
    try:
        if args.ingest_templates:
            print(ensure_templates_indexed(force=True))
            return
        if args.types:
            for t in agent.types():
                print(f"{t.doc_type.value:<26} {t.template_id:<22} {t.title} / {t.title_ar} - {t.description}")
            return
        if args.list:
            for d in agent.search(doc_type=args.type, course_id=args.course and args.course.upper(),
                                  curriculum_id=args.curriculum):
                print(f"{d.document_id}  v{d.version} {d.status.value:<17} {d.language.value} {d.title}")
            return
        if args.handoffs:
            for h in agent.pending_handoffs():
                print(f"{h['curriculum_id']}  {h['curriculum_status']:<9} {h['course_title']}  "
                      f"documents: {h['documents']} {h['doc_types']}")
            return
        if args.show:
            show(agent.get(args.show), full=args.full)
            return
        if args.edit:
            doc = agent.edit(args.edit, DocumentEdit(reviewer=args.reviewer, title=args.title,
                                                     instructions=args.instructions, remove_sections=args.remove))
            print(doc.metadata.get("last_edit"))
            show(doc)
            return
        if args.submit:
            show(agent.review(args.submit, ReviewAction(action="pending_approval", reviewer=args.reviewer)))
            return
        if args.approve:
            doc = agent.review(args.approve, ReviewAction(action="approved", reviewer=args.reviewer))
            print(f"{doc.document_id} approved by {args.reviewer}; handed to the Marketing Agent "
                  f"({doc.metadata.get('handoff_event')}). PDF: {doc.file_path}")
            return
        if args.email:
            doc = agent.email(args.email, DocumentEmail(to=args.to, sender=args.reviewer))
            print(doc.metadata["emails"][-1])
            return
        if not (args.curriculum or args.course):
            ap.error("give --curriculum or --course (or one of --types / --list / --show / --handoffs)")
        if args.pack or args.all:
            types = list(DocumentType) if args.all else DocumentPackRequest.model_fields["doc_types"].default_factory()
            docs = agent.generate_pack(DocumentPackRequest(curriculum_id=args.curriculum, course_id=args.course,
                                                           doc_types=types, language=args.language,
                                                           audience=args.audience))
            for d in docs:
                print(f"{d.document_id}  {d.doc_type.value:<26} {d.file_path}")
            return
        if not args.type:
            ap.error("give --type (or --pack / --all)")
        payload = {"doc_type": args.type, "curriculum_id": args.curriculum, "course_id": args.course,
                   "audience": args.audience, "template_id": args.template, "instructions": args.wishes,
                   "email_to": args.email_to or None}
        if args.language:
            payload["language"] = args.language
        show(agent.generate(DocumentRequest(**payload)))
    except NotFound as exc:
        sys.exit(f"Not found: {exc}")
    except ValueError as exc:
        sys.exit(f"Error: {exc}")


if __name__ == "__main__":
    main()
