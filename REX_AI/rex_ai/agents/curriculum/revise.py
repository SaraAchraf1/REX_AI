"""Apply a reviewer's modification request to a curriculum (brief, Agent 3 FR 25) - no LLM needed.

One instruction per line (or separated by ';'). Understood, in English and Arabic:
    add module <title>                      add a module (hours taken proportionally from the others)
    remove module <n | title>               remove a module (its hours go to the others)
    rename module <n | title> to <title>
    add topic <topic> [to module <n | title>]
    remove topic <topic>
    add outcome <text>          remove outcome <text | n>
    add exercise <text> [to module <n | title>]
    capstone: <text>                        replace the capstone project
    set hours to <n>                        rescale every module to the new total
Anything else is kept as a reviewer comment and reported as "not applied" (an LLM applies free text when
configured, see agent.revise).
"""
from __future__ import annotations

import copy
import re
from typing import Optional

from rex_ai.agents.curriculum.generator import _clean_title, _overlap, _same, rescale_hours, split_hours
from rex_ai.agents.rex.text_utils import normalize

ADD = r"(?:add|include|أضف|اضف|إضافة|اضافة)"
REMOVE = r"(?:remove|delete|drop|احذف|حذف|أزل|ازل)"
MODULE = r"(?:module|unit|وحدة|الوحدة)"
TOPIC = r"(?:topics?|موضوع|الموضوع)"
OUTCOME = r"(?:(?:learning\s+)?outcome|نتيجة|ناتج)"
EXERCISE = r"(?:exercise|lab|activity|تمرين|نشاط)"
SEP = r"\s*(?:[:\-–]|(?:on|about|for|عن|حول)\b)?\s*"
TARGET = r"(?:\s+(?:to|in|into|إلى|الى|في)\s+(?:the\s+)?(?:" + MODULE + r"\s+)?['\"]?(?P<target>.+?)['\"]?(?:\s+module)?)?$"


def _find_module(modules: list[dict], ref: Optional[str]) -> Optional[int]:
    """Module by number ('2'), exact or closest title; None if nothing resembles it."""
    if not ref:
        return None
    ref = ref.strip().strip("'\"")
    if re.fullmatch(r"\d{1,2}", ref):
        i = int(ref) - 1
        return i if 0 <= i < len(modules) else None
    exact = next((i for i, m in enumerate(modules) if normalize(m["title"]) == normalize(ref)), None)
    if exact is not None:
        return exact
    best = max(range(len(modules)), key=lambda i: _overlap(ref, modules[i]["title"]), default=None)
    return best if best is not None and _overlap(ref, modules[best]["title"]) >= 0.5 else None


def _teaching(modules: list[dict], capstone_title: Optional[str]) -> list[int]:
    return [i for i, m in enumerate(modules) if m["title"] != capstone_title and not m.get("_capstone")]


def apply_instructions(cur: dict, instructions: str, capstone_title: Optional[str] = None,
                       hours_per_module: float = 6) -> tuple[dict, list[str], list[str]]:
    """Returns (new curriculum, applied, not_applied). The input is not modified."""
    cur = copy.deepcopy(cur)
    modules: list[dict] = cur.setdefault("modules", [])
    outcomes: list[str] = cur.setdefault("learning_outcomes", [])
    total = float(cur.get("training_hours") or sum(m.get("hours", 0) for m in modules))
    applied, not_applied = [], []
    lines = [ln.strip(" -•\t") for ln in re.split(r"[\n;؛]+", instructions or "") if ln.strip(" -•\t")]

    def best_module(text: str) -> int:
        teach = _teaching(modules, capstone_title) or list(range(len(modules)))
        return max(teach, key=lambda i: _overlap(text, f"{modules[i]['title']} {' '.join(modules[i]['topics'])}"))

    def target_module(m: re.Match) -> tuple[Optional[int], str]:
        """'add topic X to module 2' -> (1, X). If the 'to/in ...' part names no module ('data cleaning in Excel'),
        it was part of the text: use the whole phrase and the closest module."""
        what = m.group("what")
        if m.group("target"):
            i = _find_module(modules, m.group("target"))
            explicit = m.group("target").strip().isdigit() or re.search(
                MODULE, line[m.end("what"):m.start("target")], re.I)
            if i is not None or explicit:
                return i, what
            what = line[m.start("what"):].strip("'\" ")
        return (best_module(what) if modules else None), what

    for line in lines:
        low = line.lower()
        m = re.match(rf"^(?:set|change)?\s*(?:the\s+)?(?:total\s+)?(?:training\s+)?hours\s*(?:to|=|:)?\s*(\d+(?:\.\d+)?)$",
                     low) or re.match(r"^(\d+(?:\.\d+)?)\s*(?:training\s+)?hours(?:\s+total)?$", low)
        if m:
            total = float(m.group(1))
            cur["training_hours"] = total
            rescale_hours(modules, total)
            applied.append(f"training hours set to {total:g}")
            continue
        m = re.match(rf"^{ADD}\s+(?:a\s+|an\s+|new\s+)*{MODULE}{SEP}['\"]?(?P<title>.+?)['\"]?$", line, re.I)
        if m:
            title = _clean_title(m.group("title"))
            pos = len(modules) - 1 if modules and modules[-1]["title"] == capstone_title else len(modules)
            new = {"title": title, "topics": [f"Key concepts of {title[:1].lower() + title[1:]}",
                                              f"Hands-on practice: {title[:1].lower() + title[1:]}"],
                   "hours": hours_per_module, "exercises": [f"Exercise: apply {title[:1].lower() + title[1:]}"]}
            modules.insert(pos, new)
            rescale_hours(modules, total)
            applied.append(f"module added: {title}")
            continue
        m = re.match(rf"^{REMOVE}\s+(?:the\s+)?{MODULE}{SEP}(?P<ref>.+)$", line, re.I)
        if m:
            i = _find_module(modules, m.group("ref"))
            if i is None or len(modules) <= 1:
                not_applied.append(f"{line} (module not found)")
                continue
            removed = modules.pop(i)
            rescale_hours(modules, total)
            applied.append(f"module removed: {removed['title']}")
            continue
        m = re.match(rf"^rename\s+(?:the\s+)?{MODULE}\s+['\"]?(?P<ref>.+?)['\"]?\s+(?:to|as)\s+['\"]?(?P<title>.+?)['\"]?$",
                     line, re.I)
        if m:
            i = _find_module(modules, m.group("ref"))
            if i is None:
                not_applied.append(f"{line} (module not found)")
                continue
            old, modules[i]["title"] = modules[i]["title"], _clean_title(m.group("title"))
            applied.append(f"module renamed: {old} -> {modules[i]['title']}")
            continue
        m = re.match(rf"^{ADD}\s+(?:a\s+|the\s+)?{TOPIC}{SEP}['\"]?(?P<what>.+?)['\"]?{TARGET}", line, re.I)
        if m:
            i, what = target_module(m)
            if i is None:
                not_applied.append(f"{line} (module not found)")
                continue
            topic = _clean_title(what)
            if not any(_same(topic, t) for t in modules[i]["topics"]):
                modules[i]["topics"].append(topic)
            applied.append(f"topic added to '{modules[i]['title']}': {topic}")
            continue
        m = re.match(rf"^{REMOVE}\s+(?:the\s+)?{TOPIC}{SEP}['\"]?(?P<what>.+?)['\"]?$", line, re.I)
        if m:
            what = normalize(m.group("what"))
            hits = 0
            for mod in modules:
                keep = [t for t in mod["topics"] if what not in normalize(t)]
                hits += len(mod["topics"]) - len(keep)
                mod["topics"] = keep
            (applied if hits else not_applied).append(f"topic removed: {m.group('what')}" if hits
                                                      else f"{line} (topic not found)")
            continue
        m = re.match(rf"^{ADD}\s+(?:a\s+|an\s+|the\s+)?{OUTCOME}{SEP}['\"]?(?P<what>.+?)['\"]?$", line, re.I)
        if m:
            outcome = _clean_title(m.group("what"))
            if not any(_same(outcome, o) for o in outcomes):
                outcomes.append(outcome)
            applied.append(f"outcome added: {outcome}")
            continue
        m = re.match(rf"^{REMOVE}\s+(?:the\s+)?{OUTCOME}{SEP}['\"]?(?P<what>.+?)['\"]?$", line, re.I)
        if m:
            what = m.group("what").strip()
            idx = int(what) - 1 if what.isdigit() else next(
                (k for k, o in enumerate(outcomes) if normalize(what) in normalize(o) or _same(what, o)), None)
            if idx is None or not 0 <= idx < len(outcomes):
                not_applied.append(f"{line} (outcome not found)")
                continue
            applied.append(f"outcome removed: {outcomes.pop(idx)}")
            continue
        m = re.match(rf"^{ADD}\s+(?:a\s+|an\s+|the\s+)?{EXERCISE}{SEP}['\"]?(?P<what>.+?)['\"]?{TARGET}", line, re.I)
        if m:
            i, what = target_module(m)
            if i is None:
                not_applied.append(f"{line} (module not found)")
                continue
            text = _clean_title(what)
            modules[i].setdefault("exercises", []).append(text)
            cur.setdefault("practical_exercises", []).append(text)
            applied.append(f"exercise added to '{modules[i]['title']}': {text}")
            continue
        m = re.match(r"^(?:(?:replace|change|set)\s+(?:the\s+)?capstone(?:\s+project)?\s*(?:with|to|:)|capstone(?:\s+project)?\s*:"
                     r"|المشروع الختامي\s*:)\s*(?P<what>.+)$", line, re.I)
        if m:
            cur["capstone_project"] = m.group("what").strip()
            applied.append("capstone project replaced")
            continue
        not_applied.append(line)

    if modules and not any(m.get("hours") for m in modules):
        for mod, h in zip(modules, split_hours(total, len(modules))):
            mod["hours"] = h
    return cur, applied, not_applied
