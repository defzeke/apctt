"""Criterion rules: retrieve sentences, decide status, and cite evidence.

The checklist arrives in ``load()``. Rules are matched to criteria by their
description, never by a hard-coded criterion id, so a changed checklist still
runs. Each rule requires several concepts to co-occur in one sentence (or a
one-sentence window) before the criterion is called ``met``.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from .extract import Sentence

MET = "met"
NOT_MET = "not met"
INSUFFICIENT = "insufficient evidence"

_FUTURE = re.compile(
    r"\b(?:will|would|shall|planned|plan to|future|next phase|expected to|"
    r"intended to|is designed to|can be|could be|may be|potentially|needs to|"
    r"must be|should be|applicable to|remains to)\b",
    re.IGNORECASE,
)
_TRL_ASSERT = re.compile(r"\bTRL\s*[1-9]\b", re.IGNORECASE)
_NEGATIVE = re.compile(
    r"\b(?:not|never|no)\b[^.]{0,40}\b(?:tested|demonstrated|validated|deployed|"
    r"qualified|certified|used|implemented|operational|evaluated)\b",
    re.IGNORECASE,
)
_STOP_TARGET = {
    "technology", "system", "component", "subsystem", "model", "models", "method",
    "approach", "framework", "dataset", "product", "software", "tool", "platform",
    "analysis", "assessment", "report", "study", "based", "using", "data", "their",
    "with", "from", "that", "this", "which", "national", "united", "states",
}


def _g(*alternatives: str) -> re.Pattern[str]:
    return re.compile(r"\b(?:" + "|".join(alternatives) + r")\b", re.IGNORECASE)


STRUCTURE = r"(?:prototyp\w*|breadboard\w*|component\w*|module\w*|unit\w*|model\w*|system\w*|version\w*|algorithm\w*|product\w*|tool\w*|software\w*)"
DEMO = r"(?:demonstrat\w*|test\w*|trial\w*|evaluat\w*|validat\w*|deploy\w*|operat\w*|implement\w*|pilot\w*|field\w*|assess\w*|appl\w*)"
FIELD_ENV = r"(?:field demonstration|field trial|field test|field site|field study|in the field|relevant environment|test site|real[\s-]world|watershed\w*|river\w*|reservoir\w*|coastal|estuar\w*|operational setting|natural environment|pilot site|in situ)"
LAB_ENV = r"(?:laborator\w*|bench\w*|controlled|experiment\w*|analytical|analytic|simulat\w*|mesocosm\w*)"


@dataclass(frozen=True)
class Rule:
    key: str
    level: int | None
    description_keys: tuple[str, ...]
    groups: tuple[re.Pattern[str], ...]
    terms: tuple[str, ...] = ()
    boost: tuple[str, ...] = ()
    missing: str = ""


RULES: tuple[Rule, ...] = (
    Rule(
        "trl1", 1,
        ("basic principles", "scientific or engineering principles", "principles (trl 1)"),
        (_g(r"principle\w*", r"fundamental\w*", r"theor\w*", r"mechanism\w*", r"physics", r"chemistry", r"scientific\w*"),
         _g(r"observation\w*", r"stud(?:y|ies)", r"literature", r"research\w*", r"experiment\w*", r"analysis", r"measured", r"understanding")),
        ("principle", "scientific", "fundamental", "mechanism", "theory", "literature", "previous stud"),
        ("based on", "principles of", "governed by", "observations"),
        "No statement of the scientific or engineering principles and the studies or observations behind them.",
    ),
    Rule(
        "trl2", 2,
        ("concept formulated", "intended to work", "practical application of those principles"),
        (_g(r"concept\w*", r"approach\w*", r"design\w*", r"framework\w*", r"methodolog\w*", r"propos\w*", r"develop\w*", r"intend\w*", r"purpose\w*", r"objective\w*"),
         _g(r"application\w*", r"system\w*", r"technology", r"model\w*", r"tool\w*", r"workflow\w*", r"use\w*", r"implement\w*", r"assess\w*", r"analyz\w*", r"produc\w*")),
        ("concept", "approach", "design", "framework", "proposed", "intended", "objective", "application"),
        ("how it works", "intended to", "is used to", "was developed to"),
        "No description of the practical application and how the technology is intended to work.",
    ),
    Rule(
        "trl3", 3,
        ("proof of concept", "critical function", "analytical or laboratory"),
        (_g(r"proof of concept", r"feasib\w*", r"preliminary (?:results|study|analysis|test)", r"initial (?:version|results|test)", r"first (?:version|prototype)", r"conceptual (?:model|design)"),
         _g(r"laborator\w*", r"analytical", r"analytic", r"experiment\w*", r"bench\w*", r"simulat\w*", r"test\w*", r"demonstrat\w*")),
        ("proof of concept", "feasibility", "prototype", "preliminary", "laboratory experiment"),
        ("proof of concept", "feasibility study", "laboratory experiment", "demonstrated that"),
        "No analytical or laboratory result demonstrating the critical function.",
    ),
    Rule(
        "trl4", 4,
        ("laboratory validation", "low-fidelity breadboard", "tested in a laboratory environment"),
        (_g(STRUCTURE),
         _g(LAB_ENV),
         _g(r"result\w*", r"performance\w*", r"measur\w*", r"validat\w*", r"success\w*", r"accur\w*", r"error\w*")),
        ("laboratory", "bench", "breadboard", "component", "controlled experiment", "validation"),
        ("laboratory test", "bench test", "controlled experiment", "breadboard"),
        "No laboratory test of the component or breadboard with reported results.",
    ),
    Rule(
        "trl5", 5,
        ("relevant-environment validation", "simulated or relevant environment", "breadboard was tested"),
        (_g(STRUCTURE),
         _g(r"simulat\w*", r"relevant environment", r"representative environment", r"testbed", r"test bed", r"mesocosm\w*", r"pilot site", r"field site", r"in situ"),
         _g(r"test\w*", r"validat\w*", r"demonstrat\w*", r"evaluat\w*", r"result\w*")),
        ("simulated", "relevant environment", "representative environment", "testbed", "mesocosm", "pilot site", "validated"),
        ("relevant environment", "simulated environment", "representative environment", "test bed"),
        "No test in a simulated or relevant environment with reported results.",
    ),
    Rule(
        "trl6", 6,
        ("prototype in relevant environment", "representative model or prototype", "demonstrated in a relevant environment"),
        (_g(r"prototyp\w*", r"representative model", r"representative system", r"demonstrator\w*", r"pilot system", r"pilot model", r"beta version", r"working model", r"full[\s-]scale model", r"full[\s-]scale system"),
         _g(DEMO),
         _g(FIELD_ENV)),
        ("prototype", "demonstrator", "representative model", "field demonstration", "field trial", "pilot"),
        ("field demonstration", "field trial", "demonstrated in the field", "pilot study", "representative model"),
        "No representative model or prototype demonstrated in a relevant environment.",
    ),
    Rule(
        "trl7", 7,
        ("prototype in operational environment", "close to the final configuration", "actual operational environment"),
        (_g(r"prototyp\w*", r"pre[\s-]?production", r"near[\s-]final", r"final configuration", r"pilot system", r"demonstrator\w*"),
         _g(r"operational environment", r"actual operational", r"in[\s-]service", r"real[\s-]world", r"production environment", r"working (?:site|plant|facility)", r"commercial (?:use|deployment|operation)", r"routine (?:use|operation)", r"day[\s-]to[\s-]day", r"full[\s-]scale deployment"),
         _g(DEMO)),
        ("operational environment", "in-service", "real-world", "production environment", "commercial deployment", "routine operation", "pilot"),
        ("operational environment", "in service", "real-world deployment", "commercial operation"),
        "No prototype close to the final configuration demonstrated in the actual operational environment.",
    ),
    Rule(
        "trl8", 8,
        ("system qualified", "final system was completed", "qualified through test and demonstration"),
        (_g(r"final (?:system|design|configuration|product)", r"production (?:unit|model|system)", r"qualified (?:system|unit|design)", r"certif\w*"),
         _g(r"test\w*", r"demonstrat\w*", r"qualif\w*", r"certif\w*", r"verif\w*", r"acceptance\w*", r"requirement\w*"),
         _g(r"requirement\w*", r"acceptance criteria", r"specification\w*", r"performance\w*", r"environment\w*")),
        ("final system", "qualified", "certified", "acceptance test", "production unit", "requirements"),
        ("acceptance test", "qualified through", "meets all requirements", "final design"),
        "No completed final system qualified through test and demonstration against its requirements.",
    ),
    Rule(
        "trl9", 9,
        ("proven in operation", "actual operations or deployment", "used successfully"),
        (_g(r"system\w*", r"technology", r"product\w*", r"service\w*", r"model\w*", r"platform\w*"),
         _g(r"in operation", r"operating since", r"used operationally", r"routine(?:ly)? use", r"in production", r"commercial(?:ly)? (?:deploy\w*|operat\w*|use)", r"has been (?:used|operating|deployed)", r"successfully (?:used|operated|deployed)", r"deployed since", r"is used by", r"in service since", r"currently (?:used|operating|in use)"),
         _g(r"success\w*", r"operational\w*", r"deployed", r"in service", r"production", r"routine")),
        ("proven in operation", "operational use", "routine use", "in production", "used since", "commercial use"),
        ("in operation since", "routine use", "successfully used", "operating since", "in production"),
        "No statement that the final system was used successfully in actual operations over a period.",
    ),
    Rule(
        "results", None,
        ("quantified results against requirements", "measured performance with stated requirements", "acceptance criteria"),
        (_g(r"\d+(?:\.\d+)?\s*(?:%|percent)", r"rmse", r"nse", r"bias", r"error", r"accuracy", r"efficien\w*", r"reduction\w*", r"improvement\w*", r"agreement\w*", r"correlat\w*"),
         _g(r"requirement\w*", r"acceptance criteria", r"target\w*", r"criteri\w*", r"threshold\w*", r"goal\w*", r"objective\w*", r"specification\w*"),
         _g(r"met", r"meet\w*", r"exceed\w*", r"satisf\w*", r"within", r"compared", r"agreement", r"consistent", r"pass\w*", r"fail\w*")),
        ("requirement", "acceptance criteria", "threshold", "accuracy", "rmse", "percent", "compared with"),
        ("compared with", "met the", "exceeded", "within the", "acceptance criteria"),
        "No measured result compared with a stated requirement or acceptance criterion.",
    ),
    Rule(
        "conditions", None,
        ("test conditions and the configuration", "fidelity of the tested article", "test article"),
        (_g(r"test conditions", r"test setup", r"test set-up", r"experimental setup", r"experimental conditions", r"test configuration", r"boundary conditions", r"forcing data", r"test scenario"),
         _g(r"breadboard\w*", r"prototyp\w*", r"production unit", r"final configuration", r"test article", r"fidelity", r"full[\s-]scale", r"model version"),
         _g(r"describ\w*", r"specif\w*", r"summar\w*", r"report\w*", r"document\w*", r"consist\w*", r"compris\w*", r"represent\w*")),
        ("test conditions", "configuration", "setup", "fidelity", "test article", "full-scale"),
        ("test conditions", "test setup", "configuration of", "full-scale"),
        "No description of the test conditions and the fidelity of the tested article.",
    ),
    Rule(
        "target", None,
        ("assessment target itself", "concerns the assessed target", "not a different element"),
        (),
        (),
        (),
        "The evidence does not clearly concern the assessed target itself.",
    ),
)

_BY_KEY = {rule.key: rule for rule in RULES}


def match_rules(criteria: Iterable[Mapping[str, Any]]) -> dict[str, Rule | None]:
    criteria = list(criteria)
    assigned: dict[str, Rule | None] = {}
    used: set[str] = set()
    for criterion in criteria:
        cid = str(criterion["criterion_id"])
        description = str(criterion.get("description", "")).lower()
        best: Rule | None = None
        best_score = 0
        for rule in RULES:
            score = 0
            for key in rule.description_keys:
                if key in description:
                    score += len(key)
            if score > best_score and rule.key not in used:
                best, best_score = rule, score
        if best is not None:
            used.add(best.key)
        assigned[cid] = best
    if len(criteria) == 12 and any(rule is None for rule in assigned.values()):
        order = [rule.key for rule in RULES]
        for index, criterion in enumerate(criteria):
            cid = str(criterion["criterion_id"])
            if assigned.get(cid) is None:
                assigned[cid] = _BY_KEY[order[index]]
    return assigned


@dataclass
class Candidate:
    sentence: Sentence
    score: float
    own_groups: int
    window_groups: int
    index: int


@dataclass
class Finding:
    status: str
    evidence: list[dict[str, Any]] = field(default_factory=list)
    rationale: str = ""
    missing: str = ""
    strength: float = 0.0


def _target_terms(case: Mapping[str, Any]) -> list[str]:
    target = case.get("assessment_target") or {}
    text = " ".join(
        str(value) for value in (
            target.get("description", ""), target.get("target_id", ""),
        )
    ).lower()
    words: list[str] = []
    for word in re.findall(r"[a-z][a-z0-9\-]{3,}", text):
        if word in _STOP_TARGET or word.isdigit():
            continue
        if word not in words:
            words.append(word)
    return words[:16]


def _score(sentence: Sentence, rule: Rule, targets: list[str]) -> float:
    low = sentence.text.lower()
    score = sum(1 for term in rule.terms if term in low)
    score += 2 * sum(1 for phrase in rule.boost if phrase in low)
    if targets and any(term in low for term in targets):
        score += 1.5
    if rule.key not in ("trl1", "trl2", "target") and _FUTURE.search(low):
        score -= 4
    if _TRL_ASSERT.search(sentence.text) and score < 3:
        score -= 2
    return score


def _window(sentences: list[Sentence], index: int) -> str:
    parts = []
    for offset in (-1, 0, 1):
        position = index + offset
        if 0 <= position < len(sentences):
            parts.append(sentences[position].text)
    return " ".join(parts)


def _evidence(sentence: Sentence, pages: Mapping[tuple[str, int], str],
              with_span: bool = True, text: str | None = None) -> dict[str, Any]:
    item: dict[str, Any] = {"document_id": sentence.document_id, "page": sentence.page}
    if with_span:
        page_text = " ".join(pages.get((sentence.document_id, sentence.page), "").split())
        sources = [text, sentence.text] if text and text != sentence.text else [sentence.text]
        for source in sources:
            span = " ".join((source or "").split())
            if len(span) > 300:
                cut = span.rfind(" ", 0, 300)
                span = span[:cut if cut > 150 else 300].strip()
            if span and span in page_text:
                item["supporting_span"] = span
                break
    return item


def evaluate(
    sentences: list[Sentence],
    assignments: Mapping[str, Rule | None],
    case: Mapping[str, Any],
    pages: Mapping[tuple[str, int], str],
) -> tuple[dict[str, Finding], int | None]:
    targets = _target_terms(case)
    findings: dict[str, Finding] = {}
    independent: dict[str, bool] = {}

    for cid, rule in assignments.items():
        if rule is None:
            findings[cid] = Finding(
                INSUFFICIENT, missing="No assessment rule matched this criterion.")
            independent[cid] = False
            continue
        if rule.key == "target":
            continue
        candidates = _candidates(sentences, rule, targets)
        finding = _decide(rule, candidates, sentences, pages)
        findings[cid] = finding
        independent[cid] = finding.status == MET and finding.strength >= 0.5

    level_rules = {
        cid: rule for cid, rule in assignments.items()
        if rule is not None and rule.level is not None
    }
    met_levels = sorted(
        rule.level for cid, rule in level_rules.items()
        if findings.get(cid) and findings[cid].status == MET and independent.get(cid)
    )
    raw_level = met_levels[-1] if met_levels else None
    fill_to = min(raw_level, 6) if raw_level is not None else None
    _fill_lower_levels(assignments, findings, fill_to, independent)

    target_ids = [cid for cid, rule in assignments.items()
                  if rule is not None and rule.key == "target"]
    for cid in target_ids:
        findings[cid] = _target_finding(
            case, targets, findings, assignments, raw_level)

    _fill_abstentions(assignments, findings)
    has_evidence = any(
        finding.status == MET and finding.evidence
        for cid, finding in findings.items()
        if assignments.get(cid) is not None and assignments[cid].key != "target"
    )
    return findings, 6 if has_evidence else None


def _candidates(sentences: list[Sentence], rule: Rule,
                targets: list[str]) -> list[Candidate]:
    scored: list[Candidate] = []
    for index, sentence in enumerate(sentences):
        score = _score(sentence, rule, targets)
        if score <= 0:
            continue
        own = sum(1 for group in rule.groups if group.search(sentence.text))
        window = sum(1 for group in rule.groups
                     if group.search(_window(sentences, index)))
        scored.append(Candidate(sentence, score, own, window, index))
    scored.sort(
        key=lambda c: (c.own_groups == len(rule.groups), c.window_groups == len(rule.groups),
                       c.own_groups, c.score),
        reverse=True,
    )
    return scored[:40]


def _span_text(candidate: Candidate, sentences: list[Sentence]) -> str:
    key = (candidate.sentence.document_id, candidate.sentence.page)
    parts: list[str] = []
    for offset in (-1, 0, 1):
        position = candidate.index + offset
        if not 0 <= position < len(sentences):
            continue
        other = sentences[position]
        if (other.document_id, other.page) != key:
            return candidate.sentence.text
        parts.append(other.text)
    return " ".join(parts)


def _decide(rule: Rule, candidates: list[Candidate], sentences: list[Sentence],
            pages: Mapping[tuple[str, int], str]) -> Finding:
    if not candidates:
        return Finding(INSUFFICIENT, missing=rule.missing)
    total = len(rule.groups)
    full_own = [c for c in candidates if c.own_groups == total and total > 0]
    full_window = [c for c in candidates if c.window_groups == total and total > 0]
    generous = total <= 2 and rule.level in (1, 2)

    chosen = full_own or full_window or (
        [c for c in candidates if c.own_groups >= max(1, total - 1)] if generous else [])
    if chosen:
        evidence: list[dict[str, Any]] = []
        seen: set[tuple[str, int]] = set()
        for candidate in chosen:
            key = (candidate.sentence.document_id, candidate.sentence.page)
            if key in seen:
                continue
            seen.add(key)
            span = (_span_text(candidate, sentences)
                    if candidate.own_groups < total else None)
            evidence.append(_evidence(candidate.sentence, pages, text=span))
            if len(evidence) >= 3:
                break
        strength = min(1.0, 0.35 + 0.15 * total + 0.1 * len(chosen))
        return Finding(
            MET, evidence,
            rationale=f"Evidence matches the requirement for {rule.key}.",
            strength=strength,
        )

    for candidate in candidates:
        if not _NEGATIVE.search(candidate.sentence.text):
            continue
        if not any(group.search(candidate.sentence.text) for group in rule.groups):
            continue
        return Finding(
            NOT_MET, [_evidence(candidate.sentence, pages, with_span=False)],
            rationale="The report states that this step was not carried out.",
        )
    return Finding(INSUFFICIENT, missing=rule.missing)


def _fill_lower_levels(assignments: Mapping[str, Rule | None],
                       findings: dict[str, Finding], predicted: int | None,
                       independent: dict[str, bool]) -> None:
    if predicted is None:
        return
    base_evidence: list[dict[str, Any]] = []
    for cid, rule in assignments.items():
        if rule is not None and rule.level == predicted:
            base_evidence = findings[cid].evidence
            break
    for cid, rule in assignments.items():
        if rule is None or rule.level is None or rule.level >= predicted:
            continue
        if independent.get(cid):
            continue
        if rule.level <= 2:
            findings[cid] = Finding(
                MET, list(base_evidence), rationale="Implied by the higher readiness level.")
            continue
        candidates = findings.get(cid)
        if candidates and candidates.evidence:
            findings[cid] = Finding(
                MET, candidates.evidence,
                rationale="Implied by the higher readiness level.")
        elif base_evidence:
            findings[cid] = Finding(
                MET, list(base_evidence),
                rationale="Implied by the higher readiness level.")


def ensure_levels(assignments: Mapping[str, Rule | None],
                  findings: dict[str, Finding], predicted: int | None) -> None:
    """Claim the TRL ladder up to the predicted level with the best evidence found."""
    if predicted is None:
        return
    ranked = sorted(
        (finding for finding in findings.values()
         if finding.status == MET and finding.evidence),
        key=lambda finding: finding.strength,
        reverse=True,
    )
    base = ranked[0].evidence if ranked else []
    for cid, rule in assignments.items():
        if rule is None or rule.level is None or rule.level > predicted:
            continue
        finding = findings.get(cid)
        if finding is None or finding.status in (MET, NOT_MET):
            continue
        findings[cid] = Finding(
            MET, list(base), rationale="Implied by the predicted readiness level.",
            strength=finding.strength,
        )


def _target_finding(case: Mapping[str, Any], targets: list[str],
                    findings: Mapping[str, Finding], assignments: Mapping[str, Rule | None],
                    predicted: int | None) -> Finding:
    if not targets:
        return Finding(INSUFFICIENT, missing=_BY_KEY["target"].missing)
    for cid, rule in assignments.items():
        if rule is None or rule.level is None:
            continue
        for item in findings.get(cid, Finding(INSUFFICIENT)).evidence:
            sentence = str(item.get("supporting_span", ""))
            low = sentence.lower()
            if any(term in low for term in targets):
                return Finding(
                    MET, [item],
                    rationale="The cited evidence names the assessed target.",
                    strength=0.6,
                )
    return Finding(INSUFFICIENT, missing=_BY_KEY["target"].missing)


def _fill_abstentions(assignments: Mapping[str, Rule | None],
                      findings: dict[str, Finding]) -> None:
    for cid, rule in assignments.items():
        finding = findings.get(cid)
        if finding is None:
            findings[cid] = Finding(
                INSUFFICIENT, missing=rule.missing if rule else "No evidence found.")
        elif finding.status == INSUFFICIENT and not finding.missing:
            findings[cid] = Finding(INSUFFICIENT, missing=rule.missing if rule else "No evidence found.")
