from __future__ import annotations

import argparse
import json
import re
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import run_template_rendering_experiments as base


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CASES_PATH = BASE_DIR / "outputs" / "shared" / "run_aci_all_v1_realistic_subsets" / "cases_major_omission.jsonl"
DEFAULT_OUTPUT_DIR = BASE_DIR / "outputs" / "problem_state_tracking_major_omission"
DEFAULT_MODEL = base.DEFAULT_MODEL

TRANSCRIPT_CAP_MAX_ITEMS = 64
SUMMARY_CAP_MAX_ITEMS = 40
EXTRACTION_MAX_TOKENS = 2600
TRANSCRIPT_CHUNK_TURNS = 12
TRANSCRIPT_CHUNK_OVERLAP = 3
TRANSCRIPT_CHUNK_MAX_CHARS = 3500
TRANSCRIPT_SHORT_CASE_MAX_TURNS = 18
TRANSCRIPT_RESCUE_MIN_CAPS = 8
SUMMARY_CHUNK_LINES = 8

CURRENT_CAP_RETRY_SUFFIX = """
[Retry Constraints]
- Return compact valid JSON only.
- Prefer fewer CAPs over malformed JSON.
- If the chunk contains any clinically meaningful assertion, do NOT return an empty `caps` list.
- Do not output entity lists or underspecified tuples.
- Keep one CAP = exactly one clinical assertion.
""".strip()

LEGACY_ATOMIC_CAP_RETRY_SUFFIX = """
[Retry Constraints]
- Return compact flat JSON only.
- Return at most 16 atomic propositions.
- Include only these keys for each item:
  prop_id, category, speaker, predicate, status, temporality, claim_type_tags, proposition_text
- Do not include nested objects.
- `filtered_nonverifiable_units` must be a flat list of strings.
- If clinically central content is present, do NOT return an empty `atomic_propositions` list.
""".strip()


TRANSCRIPT_TO_CAP_PROMPT = """
You are an expert clinical information extraction system.

Your task is to convert a clinician-patient dialogue transcript into Clinical Information Units (CIUs), represented here as atomic Clinical Atomic Propositions (CAPs).

[CIU Definition]
A CIU/CAP is:
- one clinically meaningful assertion,
- self-contained and independently verifiable against the transcript,
- explicit about subject and content source when possible,
- and preserves the minimum clinically necessary attributes:
  1. content
  2. certainty
  3. temporality
  4. modality (how the information was ascertained)

[Core Goal]
- Extract the atomic clinical assertions first.
- Do NOT merge multiple distinct assertions into one CAP.
- State/event abstraction will happen downstream; do not over-compress at this stage.

[Why CIU Instead of Simpler Extraction]
- Do NOT behave like a pure entity extractor.
  X: "chest pain", "heparin", "pancreatic cancer"
  O: "The patient denies chest pain." / "Heparin will be started only if D-dimer is elevated." / "The patient's mother died of pancreatic cancer."
- Do NOT behave like a bare subject-predicate-object tuple extractor that drops modifiers.
  X: (Patient, has, chest pain)
  O: "The patient denies chest pain."
- Do NOT split context so aggressively that condition, source, temporality, or management intent is lost.
  X: "MRI" / "epidural"
  O: "A cervical MRI is planned before epidural placement to localize the compressed nerve root."

[CIU Quality Standard]
- A valid CAP must remain clinically safe when read on its own.
- This means the CAP should preserve the minimum context needed for 1:1 comparison with the source dialogue:
  content + certainty + temporality + modality/source.

[CAP Schema]
Each CAP may contain:
- `cap_id`: unique id such as CAP1
- `cap_type`: one of Demographics, ChiefComplaint, Symptom, Problem, ProblemHistory, Allergy, Diagnosis, Impression, ExamFinding, TestResult, MedicationStatement, MedicationRequest, Order, FollowUp, Counseling
- `content_source`: patient, clinician, test_result, or unknown
- `modality`: interview, exam, test_result, assessment, plan, historical_record, or unknown
- `canonical_concept`: normalized clinical concept string
- `verification_status`: one of confirmed, refuted, unconfirmed
- `clinical_status`: one of active, stable, improving, worsening, resolved, historical, planned
- `temporality`: free-text summary of timing or course
- `attributes`: flat object for clinically important modifiers such as severity, location, laterality, dose, frequency, value, unit, course
- `proposition_text`: one standalone sentence capturing exactly one clinical assertion
- `linked_problem`: optional downstream metadata for later state/event grouping
- `event_cluster_id`: optional downstream metadata; use null for now if unknown
- `evidence`: optional supporting references with `turn_id`; `turn_speaker` and `span_text` are optional

[Atomicity Rules - CRITICAL]
1. One CAP = exactly one clinical assertion.
2. If a span contains multiple distinct assertions, split them into separate CAPs.
3. You may keep multiple modifiers inside one CAP only when they belong to the same subject and same clinical event.
4. Do not merge a symptom, a test result, and a plan into one CAP.
5. The goal is not to maximize compactness. The goal is to preserve clinically safe, verifiable atomic propositions.

[Extraction Rules]
1. Resolve pronouns and deictic phrases where possible.
2. Preserve negation, uncertainty, temporality, laterality, quantitative values, severity, and management intent.
3. Do not output pure questions, commands, or vague unresolved fragments as CAPs.
4. Evidence is preferred but not mandatory for initial extraction; if exact grounding is uncertain, you may leave `evidence` empty and it will be attached later.
5. Medication current state and medication change should be represented separately when both appear.
6. Distinguish current findings from future plans.
7. Include clinically important negatives when they matter for diagnostic reasoning, safety, or plan.
8. Prioritize HPI, abnormal exam findings, key test results, diagnoses/impressions, medication state/change, concrete plans/orders, and follow-up.
9. Lifestyle details should be extracted only if clinically consequential.
10. If any clinically central information is present, return at least 1 CAP.
11. Only return an empty list if the chunk truly contains no clinically meaningful assertion.
12. Use `Symptom` for patient-reported symptoms/ROS (including negated symptoms); do not default these to `Problem`.
13. Use `Problem` or `Diagnosis` only for condition-level concepts framed as clinical problems/impressions.
14. If a plan sentence contains multiple independent actions (e.g., order + referral + medication change + follow-up), split it into multiple CAPs.
15. Do not extract pure social chatter or preference anecdotes unless directly tied to diagnosis, risk, or management.
16. Keep JSON compact: for `evidence`, include only `turn_id` when possible.
17. Do not emit long quoted source spans; avoid `span_text` unless strictly necessary.
18. Keep `attributes` minimal and flat; omit empty or verbose attribute payloads.

[Clinical Signal vs Noise - STRICT]
Clinical signal is information that contributes to at least one of:
- problem identification
- current/past clinical state (symptom, diagnosis, finding, negation, uncertainty, temporality)
- diagnostic reasoning evidence
- management decision (medication change, order, follow-up, counseling)
- clinically relevant risk context that changes interpretation or plan

Clinical noise is information that does not affect clinical interpretation or management:
- social small talk, pleasantries, conversational fillers
- narrative anecdotes without medical implication
- preference details that do not change risk, diagnosis, or plan

Decision test:
- If removing a candidate assertion would NOT change diagnosis, reasoning, or plan, treat it as noise and do not output it as a CAP.
- If a lifestyle/context detail clearly modifies risk stratification or management, keep it.

[Output Format]
Return valid JSON only:
{{
  "caps": [
    {{
      "cap_id": "CAP1",
      "cap_type": "Problem",
      "content_source": "patient",
      "modality": "interview",
      "canonical_concept": "left hand pain",
      "verification_status": "confirmed",
      "clinical_status": "active",
      "temporality": "current",
      "attributes": {{"laterality": "left"}},
      "proposition_text": "The patient reports pain in the left hand.",
      "evidence": [{{"turn_id": "T7"}}]
    }}
  ],
  "filtered_nonverifiable_units": ["Is the pain worse at night?"]
}}

[Transcript Turns]
{turn_block}
""".strip()


TRANSCRIPT_TO_CAP_RESCUE_PROMPT = """
You are an expert clinical information extraction system performing a rescue extraction pass.

Your task is to recover clinically central atomic CIUs/CAPs from a clinician-patient dialogue transcript chunk when an earlier strict pass returned too few results.

[Core Goal]
- Extract the most clinically important visit content even if wording is messy, spread across multiple turns, or evidence spans are not exact.
- Favor recall of clinically central states over over-pruning.
- Return atomic one-assertion CAPs using the same schema as the primary extractor.
- Preserve content, certainty, temporality, and modality/source whenever possible.

[Must-Keep Priorities]
If present in the transcript, prioritize these before anything else:
1. chief complaint or main active problem
2. symptom onset, mechanism, location, severity, course, or associated neurologic findings
3. abnormal exam findings or key negative exam findings
4. key test results
5. medication current state or medication change
6. concrete order, referral, or follow-up plan
7. clinically meaningful negated symptoms that constrain differential or safety

[Rescue Rules]
1. Return 6 to 16 high-value CAPs when possible.
2. Do not return an empty list for a chunk containing a clear complaint, diagnosis, medication change, test result, order, or follow-up.
3. Evidence is optional in this rescue pass. If exact evidence is uncertain, leave `evidence` empty.
4. Do not focus only on follow-up or refills if more central HPI, problem, exam, result, or plan content is present.
5. Prefer broad clinically valid propositions over missing the visit's main problem.
6. One CAP = exactly one clinical assertion.
7. Do not revert to entity lists or underspecified tuples; each CAP should still read like a standalone clinical proposition.
8. Use `Symptom` for patient-reported symptoms/ROS including explicit negations.
9. Split multi-action plan statements into multiple CAPs.
10. Exclude non-clinical chatter unless it directly changes risk or management.
11. Apply the same clinical signal vs noise test as the primary extractor: if removal does not change diagnosis, reasoning, or plan, do not output it.

[Output Format]
Return valid JSON only:
{{
  "caps": [
    {{
      "cap_id": "CAP1",
      "cap_type": "ChiefComplaint",
      "content_source": "patient",
      "modality": "interview",
      "canonical_concept": "left arm pain",
      "verification_status": "confirmed",
      "clinical_status": "active",
      "temporality": "current",
      "proposition_text": "The patient reports left arm pain."
    }},
    {{
      "cap_id": "CAP2",
      "cap_type": "Order",
      "content_source": "clinician",
      "modality": "plan",
      "canonical_concept": "cervical MRI",
      "verification_status": "confirmed",
      "clinical_status": "planned",
      "temporality": "future",
      "proposition_text": "The clinician plans a cervical MRI."
    }}
  ],
  "filtered_nonverifiable_units": []
}}

[Transcript Turns]
{turn_block}
""".strip()


SUMMARY_TO_PROBLEM_CAP_PROMPT = """
You are an expert clinical information extraction system.

Convert the reference clinical note into atomic Clinical Information Units (CIUs), represented here as CAPs.

[Goal]
- Extract self-contained, one-assertion clinical units from the note.
- Preserve content, certainty, temporality, and modality.
- Do not over-compress multiple clinical facts into one CAP.

[Output Format]
Return valid JSON only:
{{
  "caps": [
    {{
      "cap_id": "CAP1",
      "cap_type": "Diagnosis",
      "content_source": "clinician",
      "modality": "assessment",
      "canonical_concept": "anemia",
      "verification_status": "confirmed",
      "clinical_status": "active",
      "temporality": "current",
      "proposition_text": "The clinician assesses new anemia."
    }}
  ],
  "filtered_nonverifiable_units": []
}}

[Reference Note]
{summary_text}
""".strip()


LEGACY_ATOMIC_TRANSCRIPT_PROMPT = """
You are an expert clinical information extraction system.

Your task is to convert a clinician-patient dialogue transcript into dialogue-aware Clinical Atomic Propositions (CAPs).

[Definition of CAP]
A Clinical Atomic Proposition (CAP) is:
- one clinically meaningful assertion,
- standalone and self-contained,
- verifiable against transcript evidence,
- explicit about content source and subject,
- and preserves clinically important modifiers such as negation, temporality, laterality, numeric value, severity, and management intent.

Only include VALID, VERIFIABLE declarative claims in `atomic_propositions`.

[Validity Rules - CRITICAL]
Do NOT include invalid / non-verifiable units as CAPs.
Instead, place them in `filtered_nonverifiable_units`.

Invalidity types:
- imperative: instruction / command / directive without a truth-evaluable factual claim
- interrogative: question
- incomplete: fragment without a complete assertion
- vague: unresolved subject, unresolved referent, or ambiguous clinical object that cannot be resolved from context

[Contextual Enrichment Rules - CRITICAL]
1. Resolve deictic expressions and contextual references when possible.
2. Replace pronouns in `proposition_text` with explicit subjects.
3. Preserve patient-reported symptom persistence, progression, worsening, or change over time when explicitly stated.
4. Preserve explicit treatment plans, referrals, follow-up timing, return precautions, and patient-facing management instructions as high-priority propositions.

[Extraction Rules]
1. One CAP = exactly one clinical assertion.
2. Make content_source explicit.
3. Capture negation and uncertainty explicitly.
4. Preserve clinically important laterality, temporality, severity, and management intent inside `proposition_text`.
5. `proposition_text` must be a standalone English sentence.
6. Do not infer facts not explicitly grounded in the transcript.
7. Distinguish current findings from future plans.
8. Return compact, flat JSON only. Omit null fields entirely. Do not include nested objects.
9. Return at most 16 atomic propositions for this transcript chunk. Prefer the most clinically salient propositions.
10. If clinically central content is present, do NOT return an empty list.

[Output Requirements]
Return valid JSON only.
Return one JSON object with this structure:
{{
  "atomic_propositions": [
    {{
      "prop_id": "P1",
      "category": "Symptom",
      "content_source": "patient",
      "predicate": "reports",
      "status": "affirmed",
      "temporality": "current",
      "claim_type_tags": ["symptom"],
      "proposition_text": "The patient reports pain in the left hand."
    }}
  ],
  "filtered_nonverifiable_units": ["Is the pain worse at night?"]
}}

[Input Transcript]
{turn_block}
[/Input Transcript]
""".strip()


EXTRACTION_GUIDANCE = """
[Extraction Guidance]
- Prefer clinically central CIUs rather than low-value fragments.
- One CAP should capture exactly one clinical assertion.
- Focus on chief complaint, HPI, active problems, diagnosis impressions, exam findings, key test results, medication states/changes, concrete orders, and follow-up.
- On dense clinical chunks, returning 8 to 20 CAPs is acceptable if each CAP is atomic and clinically meaningful.
- Keep `canonical_concept` short and normalized.
- `proposition_text` should be a plain factual sentence, not a copy of the source turn.
- Evidence may list only the supporting turn ids if span extraction is difficult.
- Never invent details that are not explicitly supported by the provided text.
- Do not create CAPs from pure questions, acknowledgements, scheduling chatter, consent language, or generic agreement phrases such as "I agree" or "okay".
- Do not emit low-value CAPs such as "heart exam performed", "patient is present", "provider asked about X", or vague normals without a clinically meaningful finding.
- For HPI, prioritize onset, mechanism, location, severity, course, associated neurologic symptoms, and clinically important negatives.
- Prefer clinically central HPI, exam, result, medication-change, and order/follow-up content over minor preference details.
- Use `Symptom` for symptom/ROS assertions (including negated findings); reserve `Problem` for condition-level abstractions.
- If a single plan sentence contains multiple actions, split them into separate CAPs.
- Avoid capturing conversational narrative details (small talk, social anecdotes) unless clinically consequential.
- Treat a candidate as noise if removing it would not change diagnosis, reasoning, risk assessment, or management plan.
- Keep lifestyle/context details only when they are explicitly linked to risk, differential interpretation, or plan.
- Keep output JSON compact and stable: prefer `evidence` with only `turn_id`.
- Avoid long `span_text` copies in evidence.
- Keep `attributes` short and flat; do not generate verbose attribute objects.
""".strip()


ALLOWED_CAP_TYPES = {
    "Demographics",
    "ChiefComplaint",
    "Symptom",
    "Problem",
    "ProblemHistory",
    "Allergy",
    "Diagnosis",
    "Impression",
    "ExamFinding",
    "TestResult",
    "MedicationStatement",
    "MedicationRequest",
    "Order",
    "FollowUp",
    "Counseling",
}
ALLOWED_VERIFICATION = {"confirmed", "refuted", "unconfirmed"}
ALLOWED_CLINICAL_STATUS = {"active", "stable", "improving", "worsening", "resolved", "historical", "planned"}
LOW_VALUE_CONCEPTS = {
    "patient presence",
    "patient is present",
    "doctor change",
    "optometry",
    "appointment",
    "follow up visit",
    "heart exam",
    "condition normal",
    "normal condition",
    "bread consumption",
    "wheat bread",
    "club soda",
    "water intake",
    "food intolerance",
    "alcohol consumption",
}
LOW_VALUE_TEXT_PATTERNS = (
    "the patient is present",
    "heart exam was performed",
    "the patient's condition is normal",
    "provider asked",
    "doctor asked",
    "patient agreed",
    "the patient agrees",
)

NON_CLINICAL_CHATTER_PATTERNS = (
    "nice to meet you",
    "thank you",
    "my pleasure",
    "how are you",
    "good morning",
    "sounds good",
    "it was nice meeting you",
    "see you again",
    "free food",
    "old guys",
    "boss",
    "physics class",
)

CLINICAL_SIGNAL_TOKENS = (
    "pain",
    "ache",
    "sore",
    "swelling",
    "weakness",
    "numb",
    "tingling",
    "dizzy",
    "fatigue",
    "fever",
    "chills",
    "cough",
    "sob",
    "shortness of breath",
    "orthopnea",
    "lying flat",
    "headache",
    "vision",
    "discharge",
    "bleeding",
    "blood",
    "stool",
    "urine",
    "hematuria",
    "appetite",
    "weight",
    "bp",
    "blood pressure",
    "heart rate",
    "murmur",
    "edema",
    "a1c",
    "hemoglobin",
    "mri",
    "x-ray",
    "ct",
    "ultrasound",
    "echo",
    "emg",
    "ncv",
    "diagnos",
    "impression",
    "assessment",
    "prescrib",
    "medication",
    "dose",
    "mg",
    "order",
    "referral",
    "follow up",
    "follow-up",
    "return",
    "monitor",
    "advise",
    "recommend",
    "counsel",
    "plan",
)

PLAN_ACTION_REGEX = re.compile(
    r"\b(start|continue|stop|increase|decrease|switch|refill|order|refer|schedule|arrange|obtain|get|perform|check|monitor|follow[- ]?up|return|advise|recommend|instruct|counsel|call|hydrate|watch|avoid)\b",
    flags=re.IGNORECASE,
)

NEGATION_PHRASE_REGEX = re.compile(
    r"\b(?:deny|denies|denied)\b\s+([^.;]+)",
    flags=re.IGNORECASE,
)

CAP_TYPE_PRIORITY = {
    "ChiefComplaint": 12,
    "Diagnosis": 11,
    "Impression": 10,
    "Problem": 10,
    "Symptom": 10,
    "ExamFinding": 9,
    "TestResult": 9,
    "MedicationRequest": 9,
    "Order": 9,
    "FollowUp": 8,
    "MedicationStatement": 8,
    "ProblemHistory": 7,
    "Allergy": 7,
    "Demographics": 6,
    "Counseling": 5,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run CAP-based problem list and visit state tracking experiments."
    )
    parser.add_argument("--cases-path", type=Path, default=DEFAULT_CASES_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--case-ids", nargs="*", default=None)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--api-base-url", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--extractor-model", default=None)
    parser.add_argument("--extractor-api-base-url", default=None)
    parser.add_argument("--extractor-api-key", default=None)
    parser.add_argument("--request-timeout", type=float, default=180.0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_extraction_tokens", type=int, default=EXTRACTION_MAX_TOKENS)
    parser.add_argument(
        "--cap-extraction-mode",
        choices=("single_call", "robust"),
        default="single_call",
        help=(
            "CAP extraction mode. "
            "'single_call' = chunking + one LLM call per chunk + minimal normalization (paper-friendly). "
            "'robust' = chunking/retry/rescue/fallback + heuristic post-processing."
        ),
    )
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--deployment-cost-only",
        action="store_true",
        help=(
            "Skip reference-note CAP extraction/evaluation and run transcript-side CAP extraction only. "
            "Use this for deployment-path latency/token accounting (X->C)."
        ),
    )
    return parser.parse_args()


def infer_extractor_api_base_url(explicit_base_url: Optional[str], extractor_model: str, fallback_base_url: str) -> str:
    if explicit_base_url:
        return explicit_base_url.rstrip("/")
    model_name = base.safe_text(extractor_model)
    if model_name.startswith("gpt-") or model_name.startswith("o1") or model_name.startswith("o3") or model_name.startswith("o4"):
        return "https://api.openai.com/v1"
    return fallback_base_url.rstrip("/")


def transcript_cap_schema(max_items: int) -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "caps": {
                "type": "array",
                "maxItems": max_items,
                "items": {
                    "type": "object",
                    "properties": {
                        "cap_id": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                        "cap_type": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                        "content_source": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                        "speaker": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                        "modality": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                        "canonical_concept": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                        "verification_status": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                        "clinical_status": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                        "temporality": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                        "linked_problem": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                        "event_cluster_id": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                        "attributes": {
                            "type": "object",
                            "additionalProperties": {"type": ["string", "number", "boolean", "null"]},
                        },
                        "proposition_text": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                        "evidence": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "turn_id": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                                    "turn_speaker": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                                    "speaker": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                                    "span_text": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                                },
                                "required": ["turn_id"],
                                "additionalProperties": False,
                            },
                        },
                    },
                    "required": ["canonical_concept", "proposition_text"],
                    "additionalProperties": False,
                },
            },
            "filtered_nonverifiable_units": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["caps", "filtered_nonverifiable_units"],
        "additionalProperties": False,
    }


def legacy_atomic_cap_schema(max_items: int = 16) -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "atomic_propositions": {
                "type": "array",
                "maxItems": max_items,
                "items": {
                    "type": "object",
                    "properties": {
                        "prop_id": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                        "category": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                        "content_source": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                        "speaker": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                        "predicate": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                        "status": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                        "temporality": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                        "claim_type_tags": {"type": "array", "items": {"type": "string"}},
                        "proposition_text": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                    },
                    "required": ["proposition_text"],
                    "additionalProperties": False,
                },
            },
            "filtered_nonverifiable_units": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["atomic_propositions", "filtered_nonverifiable_units"],
        "additionalProperties": False,
    }


def parse_transcript_turns(transcript: str) -> List[Dict[str, Any]]:
    turns: List[Dict[str, Any]] = []
    cursor = 0
    for idx, raw_line in enumerate(base.safe_text(transcript).splitlines()):
        line = raw_line.strip()
        if not line:
            cursor += len(raw_line) + 1
            continue
        match = re.match(r"^\[(doctor|patient|clinician)\]\s*(.*)$", line, flags=re.IGNORECASE)
        if match:
            speaker = match.group(1).lower()
            text = match.group(2).strip()
        else:
            speaker = "unknown"
            text = line
        start_char = cursor + raw_line.find(text) if text and text in raw_line else cursor
        end_char = start_char + len(text)
        turns.append(
            {
                "turn_id": f"T{idx}",
                "speaker": speaker,
                "text": text,
                "start_char": start_char,
                "end_char": end_char,
            }
        )
        cursor += len(raw_line) + 1
    return turns


def split_note_sentences(note_text: str) -> List[Dict[str, Any]]:
    cleaned = base.safe_text(note_text)
    lines = [ln.strip() for ln in cleaned.splitlines() if ln.strip()]
    out: List[Dict[str, Any]] = []
    for idx, line in enumerate(lines):
        out.append({"turn_id": f"S{idx}", "speaker": "note", "text": line})
    return out


def format_turn_block(turns: Sequence[Dict[str, Any]]) -> str:
    return "\n".join(f"{turn['turn_id']} [{turn['speaker']}] {turn['text']}" for turn in turns)


def render_turn_line(turn: Dict[str, Any]) -> str:
    return f"{turn['turn_id']} [{turn['speaker']}] {turn['text']}"


def chunk_turns(
    turns: Sequence[Dict[str, Any]],
    *,
    chunk_size: int = TRANSCRIPT_CHUNK_TURNS,
    overlap: int = TRANSCRIPT_CHUNK_OVERLAP,
    max_chars: int = TRANSCRIPT_CHUNK_MAX_CHARS,
    short_case_turn_limit: int = TRANSCRIPT_SHORT_CASE_MAX_TURNS,
) -> List[List[Dict[str, Any]]]:
    if not turns:
        return []
    if len(turns) <= short_case_turn_limit:
        return [list(turns)]
    out: List[List[Dict[str, Any]]] = []
    start = 0
    while start < len(turns):
        current: List[Dict[str, Any]] = []
        current_chars = 0
        end = start
        while end < len(turns):
            turn = turns[end]
            turn_line = render_turn_line(turn)
            projected_chars = current_chars + (1 if current else 0) + len(turn_line)
            if current and (len(current) >= chunk_size or projected_chars > max_chars):
                break
            current.append(turn)
            current_chars = projected_chars
            end += 1
        if not current:
            current.append(dict(turns[start]))
            end = start + 1
        out.append(current)
        if end >= len(turns):
            break
        start = max(end - max(0, overlap), start + 1)
    return out


def chunk_note_units(units: Sequence[Dict[str, Any]], *, chunk_size: int = SUMMARY_CHUNK_LINES) -> List[List[Dict[str, Any]]]:
    if not units:
        return []
    return [list(units[i : i + chunk_size]) for i in range(0, len(units), chunk_size)]


def normalize_problem_cap_obj(obj: Any) -> Dict[str, Any]:
    if not isinstance(obj, dict):
        return {"caps": [], "filtered_nonverifiable_units": []}
    caps: List[Dict[str, Any]] = []
    for idx, item in enumerate(obj.get("caps") or [], start=1):
        if not isinstance(item, dict):
            continue
        proposition_text = base.safe_text(item.get("proposition_text"))
        canonical_concept = base.safe_text(item.get("canonical_concept"))
        if not proposition_text or not canonical_concept:
            continue
        evidence: List[Dict[str, Any]] = []
        for ev in item.get("evidence") or []:
            if not isinstance(ev, dict):
                continue
            turn_id = base.safe_text(ev.get("turn_id"))
            span_text = base.safe_text(ev.get("span_text"))
            if not turn_id:
                continue
            evidence.append(
                {
                    "turn_id": turn_id,
                    "turn_speaker": base.safe_text(ev.get("turn_speaker") or ev.get("speaker")) or None,
                    "span_text": span_text or None,
                }
            )
        cap = {
            "cap_id": base.safe_text(item.get("cap_id") or f"CAP{idx}") or f"CAP{idx}",
            "cap_type": base.safe_text(item.get("cap_type")) or None,
            "content_source": base.safe_text(item.get("content_source") or item.get("speaker")) or None,
            "modality": base.safe_text(item.get("modality")) or None,
            "canonical_concept": canonical_concept,
            "verification_status": base.safe_text(item.get("verification_status")) or base.safe_text(item.get("assertion")) or "confirmed",
            "clinical_status": base.safe_text(item.get("clinical_status")) or base.safe_text(item.get("visit_state")) or "active",
            "temporality": base.safe_text(item.get("temporality")) or None,
            "linked_problem": base.safe_text(item.get("linked_problem")) or None,
            "event_cluster_id": base.safe_text(item.get("event_cluster_id")) or None,
            "attributes": item.get("attributes") if isinstance(item.get("attributes"), dict) else {},
            "proposition_text": proposition_text,
            "evidence": evidence,
        }
        caps.append({k: v for k, v in cap.items() if v not in (None, [], {}, "")})
    filtered_nonverifiable_units = [base.safe_text(x) for x in (obj.get("filtered_nonverifiable_units") or []) if base.safe_text(x)]
    return {"caps": caps, "filtered_nonverifiable_units": filtered_nonverifiable_units}


def legacy_status_to_verification(status: str) -> str:
    folded = re.sub(r"[^a-z]", "", base.safe_text(status).lower())
    if folded in {"affirmed", "present", "positive"}:
        return "confirmed"
    if folded in {"negated", "denied", "negative", "absent"}:
        return "refuted"
    return "unconfirmed"


def infer_cap_type_from_legacy_category(category: str, proposition_text: str, speaker: str) -> str:
    norm = re.sub(r"[^a-z]", "", base.safe_text(category).lower())
    text = base.safe_text(proposition_text).lower()
    if norm in {"demographic", "demographics"}:
        return "Demographics"
    if norm == "chiefcomplaint":
        return "ChiefComplaint"
    if norm == "allergy":
        return "Allergy"
    if norm == "diagnosis":
        return "Diagnosis"
    if norm in {"medicationplan"}:
        return "MedicationRequest"
    if norm in {"testplan"}:
        return "Order"
    if norm in {"followupplan"}:
        return "FollowUp"
    if norm in {"history"}:
        if "allerg" in text:
            return "Allergy"
        return "ProblemHistory"
    if norm in {"finding"}:
        if any(token in text for token in ("mri", "x-ray", "ct ", "echo", "hemoglobin", "a1c", "lab", "emg", "ncv", "ultrasound", "%")):
            return "TestResult"
        if speaker == "patient":
            return "Symptom"
        return "ExamFinding"
    if norm in {"symptom"}:
        return "Symptom"
    if norm in {"uncertainornoise"}:
        return "Problem"
    if any(token in text for token in ("follow up", "follow-up", "return in", "come back")):
        return "FollowUp"
    if any(token in text for token in ("order", "referral", "schedule", "mri", "emg", "ncv", "ultrasound", "epidural")):
        return "Order"
    if any(token in text for token in ("prescribe", "start ", "continue ", "stop ", "refill", "dose", "mg")):
        return "MedicationRequest"
    if any(token in text for token in ("pain", "weakness", "numb", "tingl", "swelling", "dizzy", "fatigue", "denies", "no ")):
        return "Symptom"
    return "Problem"


def infer_modality_from_legacy(category: str, cap_type: str, speaker: str, predicate: str) -> str:
    norm = re.sub(r"[^a-z]", "", base.safe_text(category).lower())
    pred = re.sub(r"[^a-z]", "", base.safe_text(predicate).lower())
    if norm in {"finding"} and speaker == "clinician":
        return "exam"
    if cap_type == "TestResult":
        return "test_result"
    if cap_type in {"Diagnosis", "Impression"} or pred in {"diagnoses", "suspects", "notes"}:
        return "assessment"
    if cap_type in {"MedicationRequest", "Order", "FollowUp", "Counseling"} or pred in {"orders", "plans", "prescribes", "recommends", "instructs"}:
        return "plan"
    if cap_type in {"ProblemHistory", "Demographics", "Allergy"}:
        return "historical_record" if speaker in {"clinician", "test_result"} else "interview"
    return "interview" if speaker == "patient" else "unknown"


def derive_canonical_concept_from_text(proposition_text: str) -> str:
    text = base.safe_text(proposition_text).strip()
    if not text:
        return ""
    lowered = text.lower()
    patterns = (
        r"^(the patient|patient|she|he|they)\s+(reports|denies|has|states|notes|describes)\s+",
        r"^(the clinician|clinician|doctor|provider)\s+(notes|states|assesses|diagnoses|suspects|plans|orders|recommends|prescribes)\s+",
    )
    for pattern in patterns:
        lowered = re.sub(pattern, "", lowered)
    lowered = re.sub(r"\b(will|is|are|was|were|be|being|been)\b", " ", lowered)
    lowered = re.sub(r"[^a-z0-9%/+\- ]", " ", lowered)
    lowered = re.sub(r"\s+", " ", lowered).strip()
    words = lowered.split()
    return " ".join(words[:8]).strip()


def normalize_legacy_atomic_cap_obj(obj: Any) -> Dict[str, Any]:
    if not isinstance(obj, dict):
        return {"caps": [], "filtered_nonverifiable_units": []}
    caps: List[Dict[str, Any]] = []
    for idx, item in enumerate(obj.get("atomic_propositions") or [], start=1):
        if not isinstance(item, dict):
            continue
        proposition_text = base.safe_text(item.get("proposition_text") or item.get("fact_text") or item.get("text"))
        if not proposition_text:
            continue
        speaker = base.safe_text(item.get("content_source") or item.get("speaker")) or "unknown"
        predicate = base.safe_text(item.get("predicate"))
        category = base.safe_text(item.get("category"))
        cap_type = infer_cap_type_from_legacy_category(category, proposition_text, speaker)
        verification_status = legacy_status_to_verification(base.safe_text(item.get("status")))
        temporality = base.safe_text(item.get("temporality")) or None
        clinical_status = normalize_clinical_status("", verification_status, temporality or "")
        caps.append(
            {
                "cap_id": base.safe_text(item.get("prop_id") or f"CAP{idx}") or f"CAP{idx}",
                "cap_type": cap_type,
                "content_source": speaker,
                "modality": infer_modality_from_legacy(category, cap_type, speaker, predicate),
                "canonical_concept": derive_canonical_concept_from_text(proposition_text),
                "verification_status": verification_status,
                "clinical_status": clinical_status,
                "temporality": temporality,
                "proposition_text": proposition_text,
                "attributes": {},
                "evidence": [],
            }
        )
    filtered = [base.safe_text(x) for x in (obj.get("filtered_nonverifiable_units") or []) if base.safe_text(x)]
    return {"caps": caps, "filtered_nonverifiable_units": filtered}


def normalize_problem_cap_type(value: str) -> Optional[str]:
    norm = base.safe_text(value)
    aliases = {
        "demographic": "Demographics",
        "demographics": "Demographics",
        "problem": "Problem",
        "problemsymptom": "Symptom",
        "problemorsymptom": "Symptom",
        "symptom": "Symptom",
        "history": "ProblemHistory",
        "diagnosis": "Diagnosis",
        "impression": "Impression",
        "allergy": "Allergy",
        "finding": "ExamFinding",
        "examfinding": "ExamFinding",
        "test": "TestResult",
        "result": "TestResult",
        "testresult": "TestResult",
        "medication": "MedicationStatement",
        "medicationchange": "MedicationRequest",
        "medicationstate": "MedicationStatement",
        "medicationstatement": "MedicationStatement",
        "medicationrequest": "MedicationRequest",
        "plan": "Order",
        "order": "Order",
        "followup": "FollowUp",
        "chiefcomplaint": "ChiefComplaint",
        "counseling": "Counseling",
    }
    if norm in ALLOWED_CAP_TYPES:
        return norm
    folded = re.sub(r"[^a-z]", "", norm.lower())
    if folded in aliases:
        return aliases[folded]
    return None


def normalize_verification_status(value: str) -> str:
    folded = re.sub(r"[^a-z_]", "", base.safe_text(value).lower())
    aliases = {
        "affirmed": "confirmed",
        "present": "confirmed",
        "positive": "confirmed",
        "active": "confirmed",
        "historical": "confirmed",
        "denied": "refuted",
        "absent": "refuted",
        "negative": "refuted",
        "negated": "refuted",
        "ruledout": "refuted",
        "none": "refuted",
        "unknown": "unconfirmed",
        "uncertain": "unconfirmed",
        "possible": "unconfirmed",
        "suspected": "unconfirmed",
        "suspect": "unconfirmed",
        "hypothetical": "unconfirmed",
        "planned": "unconfirmed",
        "plan": "unconfirmed",
    }
    normalized = aliases.get(folded, folded or "confirmed")
    return normalized if normalized in ALLOWED_VERIFICATION else "confirmed"


def normalize_clinical_status(value: str, verification_status: str, temporality: str = "") -> str:
    folded = re.sub(r"[^a-z_]", "", base.safe_text(value).lower())
    aliases = {
        "current": "active",
        "present": "active",
        "ongoing": "active",
        "improved": "improving",
        "better": "improving",
        "worse": "worsening",
        "worsened": "worsening",
        "past": "historical",
        "previous": "historical",
        "ruledout": "resolved",
        "negative": "resolved",
        "negated": "resolved",
    }
    normalized = aliases.get(folded, folded or "")
    if normalized in ALLOWED_CLINICAL_STATUS:
        return normalized
    temp = base.safe_text(temporality).lower()
    if "follow" in temp or "next" in temp or "will" in temp:
        return "planned"
    if verification_status == "refuted":
        return "resolved"
    if "history" in temp or "previous" in temp or "ago" in temp:
        return "historical"
    if verification_status == "unconfirmed":
        return "active"
    return "active"


def normalize_modality(value: str, cap_type: str, speaker: str) -> str:
    folded = re.sub(r"[^a-z_]", "", base.safe_text(value).lower())
    aliases = {
        "interview": "interview",
        "history": "interview",
        "patientreport": "interview",
        "exam": "exam",
        "physicalexam": "exam",
        "testresult": "test_result",
        "lab": "test_result",
        "imaging": "test_result",
        "assessment": "assessment",
        "plan": "plan",
        "historicalrecord": "historical_record",
        "record": "historical_record",
    }
    normalized = aliases.get(folded, "")
    if normalized:
        return normalized
    if cap_type == "ExamFinding":
        return "exam"
    if cap_type == "TestResult":
        return "test_result"
    if cap_type in {"MedicationRequest", "Order", "FollowUp", "Counseling"}:
        return "plan"
    if cap_type in {"Diagnosis", "Impression"}:
        return "assessment"
    if cap_type == "Symptom":
        return "interview" if speaker in {"patient", "unknown"} else "exam"
    if cap_type in {"ProblemHistory", "Demographics", "Allergy"}:
        return "historical_record" if speaker in {"clinician", "test_result"} else "interview"
    return "interview" if speaker == "patient" else "unknown"


def verification_status_to_assertion(verification_status: str, clinical_status: str) -> str:
    if clinical_status == "historical":
        return "historical"
    if clinical_status == "planned":
        return "hypothetical"
    if verification_status == "confirmed":
        return "present"
    if verification_status == "refuted":
        return "absent"
    return "uncertain"


def looks_like_question(text: str) -> bool:
    lowered = base.safe_text(text).strip().lower()
    if not lowered:
        return False
    if "?" in lowered:
        return True
    return any(
        lowered.startswith(prefix)
        for prefix in (
            "what ",
            "when ",
            "where ",
            "how ",
            "why ",
            "do ",
            "did ",
            "does ",
            "have ",
            "has ",
            "is ",
            "are ",
            "any ",
            "can ",
            "could ",
            "would ",
        )
    )


def has_clinical_signal(text: str) -> bool:
    lowered = base.safe_text(text).lower()
    if not lowered:
        return False
    return any(token in lowered for token in CLINICAL_SIGNAL_TOKENS)


def is_non_clinical_chatter(text: str) -> bool:
    lowered = base.safe_text(text).lower().strip()
    if not lowered:
        return True
    if any(pattern in lowered for pattern in NON_CLINICAL_CHATTER_PATTERNS):
        return not has_clinical_signal(lowered)
    if len(lowered.split()) <= 4 and lowered in {"okay", "sounds good", "thank you", "all right", "i agree"}:
        return True
    if any(token in lowered for token in ("cheeseburger", "french fries", "old guys", "wine")) and not has_clinical_signal(lowered):
        return True
    return False


def is_plan_like_text(text: str) -> bool:
    lowered = base.safe_text(text).lower()
    if not lowered:
        return False
    return bool(PLAN_ACTION_REGEX.search(lowered) or "plan" in lowered)


def is_test_result_like_text(text: str) -> bool:
    lowered = base.safe_text(text).lower()
    return any(
        token in lowered
        for token in (
            "x-ray",
            "mri",
            "ct ",
            "ultrasound",
            "echo",
            "hemoglobin",
            "a1c",
            "biopsy",
            "pathology",
            "lab",
            "test result",
            "revealed",
            "showed",
            "negative for",
            "positive for",
        )
    )


def is_exam_finding_like_text(text: str) -> bool:
    lowered = base.safe_text(text).lower()
    return any(
        token in lowered
        for token in (
            "exam",
            "physical",
            "vital",
            "blood pressure",
            "heart rate",
            "murmur",
            "spurling",
            "tender",
            "edema",
            "strength",
            "sensation",
            "reflex",
        )
    )


def is_diagnosis_like_text(text: str) -> bool:
    lowered = base.safe_text(text).lower()
    return any(
        token in lowered
        for token in (
            "diagnos",
            "impression",
            "assessment",
            "consistent with",
            "suspicious for",
            "likely",
            "probable",
            "history of",
        )
    )


def is_symptom_like_text(text: str) -> bool:
    lowered = base.safe_text(text).lower()
    return any(
        token in lowered
        for token in (
            "pain",
            "weakness",
            "numb",
            "tingl",
            "swelling",
            "dizzy",
            "fatigue",
            "cough",
            "headache",
            "fever",
            "chills",
            "shortness of breath",
            "orthopnea",
            "denies",
            "no ",
            "reports",
        )
    )


def infer_plan_cap_type_from_text(text: str) -> str:
    lowered = base.safe_text(text).lower()
    if any(token in lowered for token in ("follow up", "follow-up", "return in", "return to clinic", "see you in", "rtc")):
        return "FollowUp"
    if any(token in lowered for token in ("advise", "recommend", "instruct", "counsel", "call if", "watch ", "avoid ", "hydrate", "diet", "salt intake", "weight daily")):
        return "Counseling"
    if any(token in lowered for token in ("start ", "continue ", "stop ", "increase ", "decrease ", "switch ", "refill ", "mg", "dose", "tablet", "capsule")):
        return "MedicationRequest"
    return "Order"


def retype_cap_by_text_rules(cap: Dict[str, Any]) -> Dict[str, Any]:
    updated = dict(cap)
    cap_type = base.safe_text(updated.get("cap_type"))
    modality = base.safe_text(updated.get("modality")).lower()
    source = base.safe_text(updated.get("content_source")).lower()
    text = base.safe_text(updated.get("proposition_text"))
    verification = base.safe_text(updated.get("verification_status")).lower()

    if cap_type in {"Problem", "Symptom"} and is_plan_like_text(text):
        updated["cap_type"] = infer_plan_cap_type_from_text(text)
    elif cap_type in {"Problem", "Symptom"} and is_test_result_like_text(text):
        updated["cap_type"] = "TestResult"
    elif cap_type in {"Problem", "Symptom"} and is_exam_finding_like_text(text) and source in {"clinician", "doctor"}:
        updated["cap_type"] = "ExamFinding"
    elif cap_type == "Problem" and (source == "patient" or verification == "refuted" or is_symptom_like_text(text)):
        updated["cap_type"] = "Symptom"
    elif cap_type in {"Problem", "Impression"} and source in {"clinician", "doctor"} and is_diagnosis_like_text(text):
        updated["cap_type"] = "Diagnosis"

    if modality == "plan" and updated.get("cap_type") in {"Problem", "Symptom"}:
        updated["cap_type"] = infer_plan_cap_type_from_text(text)
    return updated


def split_plan_cap_if_needed(cap: Dict[str, Any]) -> List[Dict[str, Any]]:
    cap_type = base.safe_text(cap.get("cap_type"))
    modality = base.safe_text(cap.get("modality")).lower()
    text = base.safe_text(cap.get("proposition_text")).strip()
    if not text:
        return []
    if cap_type not in {"Order", "FollowUp", "Counseling", "MedicationRequest"} and modality != "plan":
        return [cap]

    matches = list(PLAN_ACTION_REGEX.finditer(text))
    if len(matches) <= 1:
        return [cap]

    clauses: List[str] = []
    for idx, match in enumerate(matches):
        start = match.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        chunk = text[start:end].strip(" ,;.")
        chunk = re.sub(r"\b(and|then)\b\s*$", "", chunk, flags=re.IGNORECASE).strip(" ,;.")
        if len(chunk.split()) < 2:
            continue
        if not chunk.endswith("."):
            if not re.match(r"^(The patient|Patient|The clinician|Clinician|Doctor)\b", chunk):
                chunk = f"The clinician will {chunk}"
            chunk = chunk.rstrip(".") + "."
        clauses.append(chunk)

    deduped: List[str] = []
    seen: set[str] = set()
    for clause in clauses:
        norm = base.normalize_text(clause)
        if not norm or norm in seen:
            continue
        seen.add(norm)
        deduped.append(clause)
    if len(deduped) <= 1:
        return [cap]

    split_caps: List[Dict[str, Any]] = []
    for clause in deduped[:4]:
        new_cap = dict(cap)
        new_cap["proposition_text"] = clause
        new_cap["canonical_concept"] = derive_canonical_concept_from_text(clause)
        new_cap["cap_type"] = infer_plan_cap_type_from_text(clause)
        split_caps.append(new_cap)
    return split_caps


def extract_negated_phrases_from_turn_text(text: str) -> List[str]:
    lowered = base.safe_text(text).lower()
    if not lowered:
        return []
    phrases: List[str] = []
    for match in NEGATION_PHRASE_REGEX.finditer(lowered):
        phrases.append(match.group(1))
    if lowered.startswith("no "):
        phrases.append(lowered[3:])

    candidates: List[str] = []
    for phrase in phrases:
        phrase = re.split(r"\bbut\b|\bexcept\b", phrase)[0]
        for piece in re.split(r",|/| and | or ", phrase):
            chunk = piece.strip(" .:;")
            if len(chunk.split()) < 2:
                continue
            if chunk in {"nothing", "nothing like that", "none", "issues", "problem"}:
                continue
            if not has_clinical_signal(chunk):
                continue
            candidates.append(chunk)
    deduped: List[str] = []
    seen: set[str] = set()
    for chunk in candidates:
        norm = base.normalize_text(chunk)
        if not norm or norm in seen:
            continue
        seen.add(norm)
        deduped.append(chunk)
    return deduped[:4]


def synthesize_negation_caps_from_turns(
    turns: Sequence[Dict[str, Any]],
    existing_caps: Sequence[Dict[str, Any]],
    *,
    max_new_caps: int = 8,
) -> List[Dict[str, Any]]:
    if not turns:
        return []
    existing_keys = {
        (
            base.normalize_text(base.safe_text(cap.get("canonical_concept"))),
            base.safe_text(cap.get("verification_status")).lower(),
        )
        for cap in existing_caps
    }
    new_caps: List[Dict[str, Any]] = []
    for turn in turns:
        speaker = base.safe_text(turn.get("speaker")).lower()
        if speaker not in {"patient", "unknown"}:
            continue
        text = base.safe_text(turn.get("text"))
        neg_chunks = extract_negated_phrases_from_turn_text(text)
        for chunk in neg_chunks:
            concept = chunk.strip()
            key = (base.normalize_text(concept), "refuted")
            if key in existing_keys:
                continue
            existing_keys.add(key)
            new_caps.append(
                {
                    "cap_type": "Symptom",
                    "content_source": "patient",
                    "modality": "interview",
                    "canonical_concept": concept,
                    "verification_status": "refuted",
                    "clinical_status": "active",
                    "assertion": "absent",
                    "visit_state": "active",
                    "temporality": "current",
                    "proposition_text": f"The patient denies {concept}.",
                    "evidence": [
                        {
                            "turn_id": base.safe_text(turn.get("turn_id")),
                            "turn_speaker": speaker,
                            "span_text": text,
                            "char_start": turn.get("start_char", 0),
                            "char_end": turn.get("end_char", len(text)),
                        }
                    ],
                }
            )
            if len(new_caps) >= max_new_caps:
                return new_caps
    return new_caps


def is_low_value_problem_cap(cap: Dict[str, Any]) -> bool:
    concept = base.normalize_text(base.safe_text(cap.get("canonical_concept")))
    text = base.normalize_text(base.safe_text(cap.get("proposition_text")))
    cap_type = base.safe_text(cap.get("cap_type"))
    if not concept or not text:
        return True
    if is_non_clinical_chatter(text):
        return True
    if concept in LOW_VALUE_CONCEPTS:
        return True
    if cap_type in {"Problem", "Symptom", "ProblemHistory", "ChiefComplaint"} and any(
        token in concept for token in ("bread", "pasta", "soda", "water", "club soda")
    ):
        return True
    if any(token in text for token in ("free food", "boss", "physics class", "old guys", "wine")) and not has_clinical_signal(text):
        return True
    if any(pattern in text for pattern in LOW_VALUE_TEXT_PATTERNS):
        return True
    if len(text.split()) < 3:
        return True
    evidence = cap.get("evidence") or []
    if evidence:
        spans = [base.safe_text(ev.get("span_text")) for ev in evidence if base.safe_text(ev.get("span_text"))]
        if spans and all(looks_like_question(span) for span in spans):
            if cap_type not in {"Order", "FollowUp", "Counseling", "MedicationRequest"}:
                return True
    return False


def is_protected_high_value_cap(cap: Dict[str, Any]) -> bool:
    cap_type = base.safe_text(cap.get("cap_type"))
    concept = base.normalize_text(base.safe_text(cap.get("canonical_concept")))
    if cap_type not in {"ChiefComplaint", "Symptom", "Problem", "Diagnosis", "Impression", "ExamFinding", "TestResult", "MedicationRequest", "Order"}:
        return False
    if not concept or concept in LOW_VALUE_CONCEPTS:
        return False
    if any(token in concept for token in ("bread", "pasta", "soda", "water", "club soda")):
        return False
    return True


def overlap_score(a: str, b: str) -> float:
    return base.token_f1(a, b)


def score_turn_for_cap(cap: Dict[str, Any], turn: Dict[str, Any]) -> float:
    concept = base.safe_text(cap.get("canonical_concept"))
    proposition = base.safe_text(cap.get("proposition_text"))
    text = base.safe_text(turn.get("text"))
    score = 0.55 * base.token_f1(concept, text) + 0.45 * base.token_f1(proposition, text)
    lowered = text.lower()
    if concept and concept.lower() in lowered:
        score += 0.2
    if looks_like_question(text):
        score -= 0.45
    if any(marker in lowered for marker in ("consents to have this visit recorded", "date of birth")):
        score -= 0.25
    if len(lowered.split()) <= 3 and lowered in {"okay", "all right", "sounds good", "thank you"}:
        score -= 0.35
    speaker = base.safe_text(turn.get("speaker")).lower()
    cap_type = base.safe_text(cap.get("cap_type"))
    verification = base.safe_text(cap.get("verification_status")).lower()
    modality = base.safe_text(cap.get("modality")).lower()
    if speaker == "patient" and cap_type in {"ChiefComplaint", "Symptom", "Problem", "ProblemHistory"}:
        score += 0.14
    if speaker in {"doctor", "clinician"} and cap_type in {"ExamFinding", "TestResult", "MedicationRequest", "Order", "FollowUp"}:
        score += 0.14
    if verification == "refuted" and any(token in lowered for token in ("denies", "deny", "no ", "without", "negative for")):
        score += 0.24
    if modality == "plan" and any(token in lowered for token in ("plan", "order", "refer", "start", "continue", "follow up", "return", "recommend", "advise")):
        score += 0.2
    if is_non_clinical_chatter(lowered):
        score -= 0.2
    return score


def attach_evidence_metadata(cap_obj: Dict[str, Any], turns: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    turn_map = {turn["turn_id"]: turn for turn in turns}
    for cap in cap_obj.get("caps", []):
        fixed_evidence: List[Dict[str, Any]] = []
        for ev in cap.get("evidence", []):
            turn_id = base.safe_text(ev.get("turn_id"))
            span_text = base.safe_text(ev.get("span_text"))
            turn = turn_map.get(turn_id)
            current_score = score_turn_for_cap(cap, turn) if turn else 0.0
            if turn is None or current_score < 0.2:
                best = None
                best_score = 0.0
                for candidate in turns:
                    lexical = overlap_score(span_text, candidate["text"])
                    semantic = score_turn_for_cap(cap, candidate)
                    score = 0.35 * lexical + 0.65 * semantic
                    if score > best_score:
                        best = candidate
                        best_score = score
                if best is not None and best_score > current_score:
                    turn = best
                    turn_id = turn["turn_id"]
            if turn:
                text = turn["text"]
                span_lower = span_text.lower()
                text_lower = text.lower()
                idx = text_lower.find(span_lower) if span_lower else -1
                if idx == -1:
                    idx = 0
                fixed_evidence.append(
                    {
                        "turn_id": turn_id,
                        "turn_speaker": turn["speaker"],
                        "span_text": span_text or text,
                        "char_start": turn["start_char"] + idx if "start_char" in turn else idx,
                        "char_end": turn["start_char"] + idx + len(span_text or text) if "start_char" in turn else idx + len(span_text or text),
                    }
                )
        if not fixed_evidence:
            best = None
            best_score = 0.0
            for candidate in turns:
                score = score_turn_for_cap(cap, candidate)
                if score > best_score:
                    best = candidate
                    best_score = score
            if best:
                fixed_evidence.append(
                    {
                        "turn_id": best["turn_id"],
                        "turn_speaker": best["speaker"],
                        "span_text": best["text"],
                        "char_start": best.get("start_char", 0),
                        "char_end": best.get("end_char", len(best["text"])),
                    }
                )
        cap["evidence"] = fixed_evidence
    return cap_obj


def sanitize_problem_cap_obj(cap_obj: Dict[str, Any], *, turns: Optional[Sequence[Dict[str, Any]]] = None) -> Dict[str, Any]:
    staged_caps: List[Dict[str, Any]] = []
    filtered_units = list(cap_obj.get("filtered_nonverifiable_units", []))
    for cap in cap_obj.get("caps", []):
        if not isinstance(cap, dict):
            continue
        normalized_cap = dict(cap)
        normalized_cap.pop("speaker", None)
        normalized_cap["cap_type"] = normalize_problem_cap_type(base.safe_text(cap.get("cap_type"))) or "Problem"
        verification_status = normalize_verification_status(
            base.safe_text(cap.get("verification_status")) or base.safe_text(cap.get("assertion"))
        )
        clinical_status = normalize_clinical_status(
            base.safe_text(cap.get("clinical_status")) or base.safe_text(cap.get("visit_state")),
            verification_status,
            base.safe_text(cap.get("temporality")),
        )
        normalized_cap["verification_status"] = verification_status
        normalized_cap["clinical_status"] = clinical_status
        normalized_cap["assertion"] = verification_status_to_assertion(verification_status, clinical_status)
        normalized_cap["visit_state"] = clinical_status
        normalized_cap["content_source"] = base.safe_text(cap.get("content_source") or cap.get("speaker")) or (
            base.safe_text((cap.get("evidence") or [{}])[0].get("turn_speaker") or (cap.get("evidence") or [{}])[0].get("speaker")) if cap.get("evidence") else None
        )
        normalized_cap["canonical_concept"] = base.safe_text(cap.get("canonical_concept")).strip(" .,:;")
        normalized_cap["proposition_text"] = base.safe_text(cap.get("proposition_text")).strip()
        normalized_cap["linked_problem"] = base.safe_text(cap.get("linked_problem")) or None
        normalized_cap = retype_cap_by_text_rules(normalized_cap)
        normalized_cap["modality"] = normalize_modality(
            base.safe_text(cap.get("modality")),
            normalized_cap["cap_type"],
            base.safe_text(normalized_cap.get("content_source")),
        )
        if not normalized_cap["proposition_text"] or not normalized_cap["canonical_concept"]:
            continue
        if is_low_value_problem_cap(normalized_cap) and not is_protected_high_value_cap(normalized_cap):
            filtered_units.append(normalized_cap["proposition_text"])
            continue
        for split_cap in split_plan_cap_if_needed(normalized_cap):
            if is_low_value_problem_cap(split_cap) and not is_protected_high_value_cap(split_cap):
                filtered_units.append(base.safe_text(split_cap.get("proposition_text")))
                continue
            staged_caps.append(split_cap)

    if turns:
        staged_caps.extend(synthesize_negation_caps_from_turns(turns, staged_caps))

    deduped_caps: List[Dict[str, Any]] = []
    seen: set[Tuple[str, str, str, str, str]] = set()
    for cap in staged_caps:
        key = (
            base.safe_text(cap.get("cap_type")),
            base.normalize_text(base.safe_text(cap.get("canonical_concept"))),
            base.normalize_text(base.safe_text(cap.get("proposition_text"))),
            base.safe_text(cap.get("verification_status")),
            base.normalize_text(base.safe_text(cap.get("temporality"))),
        )
        if not key[1] or not key[2] or key in seen:
            continue
        seen.add(key)
        deduped_caps.append(cap)

    for idx, cap in enumerate(deduped_caps, start=1):
        cap["cap_id"] = f"CAP{idx}"
    return {
        "caps": deduped_caps,
        "filtered_nonverifiable_units": [x for x in filtered_units if base.safe_text(x)],
    }


def sanitize_problem_cap_obj_minimal(cap_obj: Dict[str, Any]) -> Dict[str, Any]:
    sanitized_caps: List[Dict[str, Any]] = []
    filtered_units = [x for x in cap_obj.get("filtered_nonverifiable_units", []) if base.safe_text(x)]
    for cap in cap_obj.get("caps", []):
        if not isinstance(cap, dict):
            continue
        normalized_cap = dict(cap)
        normalized_cap.pop("speaker", None)
        normalized_cap["cap_type"] = normalize_problem_cap_type(base.safe_text(cap.get("cap_type"))) or "Problem"
        verification_status = normalize_verification_status(
            base.safe_text(cap.get("verification_status")) or base.safe_text(cap.get("assertion"))
        )
        clinical_status = normalize_clinical_status(
            base.safe_text(cap.get("clinical_status") or cap.get("visit_state")),
            verification_status,
            base.safe_text(cap.get("temporality")),
        )
        normalized_cap["verification_status"] = verification_status
        normalized_cap["clinical_status"] = clinical_status
        normalized_cap["assertion"] = verification_status_to_assertion(verification_status, clinical_status)
        normalized_cap["visit_state"] = clinical_status
        normalized_cap["content_source"] = base.safe_text(cap.get("content_source") or cap.get("speaker")) or (
            base.safe_text((cap.get("evidence") or [{}])[0].get("turn_speaker") or (cap.get("evidence") or [{}])[0].get("speaker")) if cap.get("evidence") else None
        )
        normalized_cap["modality"] = normalize_modality(
            base.safe_text(cap.get("modality")),
            normalized_cap["cap_type"],
            base.safe_text(normalized_cap.get("content_source")),
        )
        normalized_cap["canonical_concept"] = base.safe_text(cap.get("canonical_concept")).strip(" .,:;")
        normalized_cap["proposition_text"] = base.safe_text(cap.get("proposition_text")).strip()
        normalized_cap["linked_problem"] = base.safe_text(cap.get("linked_problem")) or None

        if not normalized_cap["canonical_concept"] or not normalized_cap["proposition_text"]:
            continue
        sanitized_caps.append(normalized_cap)

    deduped: List[Dict[str, Any]] = []
    seen: set[Tuple[str, str, str, str, str]] = set()
    for cap in sanitized_caps:
        key = (
            base.safe_text(cap.get("cap_type")),
            base.normalize_text(base.safe_text(cap.get("canonical_concept"))),
            base.normalize_text(base.safe_text(cap.get("proposition_text"))),
            base.safe_text(cap.get("verification_status")),
            base.normalize_text(base.safe_text(cap.get("temporality"))),
        )
        if not key[1] or not key[2] or key in seen:
            continue
        seen.add(key)
        deduped.append(cap)

    for idx, cap in enumerate(deduped, start=1):
        cap["cap_id"] = f"CAP{idx}"
    return {
        "caps": deduped,
        "filtered_nonverifiable_units": [x for x in filtered_units if base.safe_text(x)],
    }


def merge_problem_cap_objects(cap_objs: Sequence[Dict[str, Any]], *, max_caps: int = TRANSCRIPT_CAP_MAX_ITEMS) -> Dict[str, Any]:
    candidate_caps: List[Dict[str, Any]] = []
    merged_filtered: List[str] = []
    for cap_obj in cap_objs:
        for cap in cap_obj.get("caps", []):
            candidate_caps.append(dict(cap))
        for unit in cap_obj.get("filtered_nonverifiable_units", []):
            unit_text = base.safe_text(unit)
            if unit_text and unit_text not in merged_filtered:
                merged_filtered.append(unit_text)
    merged_caps: List[Dict[str, Any]] = []
    seen: set[Tuple[str, str, str, str, str, str, str]] = set()
    def cap_priority(cap: Dict[str, Any]) -> Tuple[int, int, int, int, int]:
        cap_type = base.safe_text(cap.get("cap_type"))
        priority = CAP_TYPE_PRIORITY.get(cap_type, 0)
        evidence_count = len(cap.get("evidence") or [])
        has_link = 1 if base.safe_text(cap.get("linked_problem")) else 0
        has_temp = 1 if base.safe_text(cap.get("temporality")) else 0
        text_len = len(base.safe_text(cap.get("proposition_text")).split())
        return (priority, evidence_count, has_link, has_temp, text_len)
    for cap in sorted(candidate_caps, key=cap_priority, reverse=True):
        key = (
            base.safe_text(cap.get("cap_type")).lower(),
            base.normalize_text(base.safe_text(cap.get("proposition_text")) or base.safe_text(cap.get("canonical_concept"))),
            base.safe_text(cap.get("assertion")).lower(),
            base.safe_text(cap.get("visit_state")).lower(),
            base.normalize_text(base.safe_text(cap.get("temporality"))),
            base.normalize_text(base.safe_text(cap.get("linked_problem"))),
            base.safe_text(cap.get("modality")).lower(),
        )
        if not key[1] or key in seen:
            continue
        seen.add(key)
        merged_caps.append(cap)
        if len(merged_caps) >= max_caps:
            break
    for idx, cap in enumerate(merged_caps, start=1):
        cap["cap_id"] = f"CAP{idx}"
    return {"caps": merged_caps[:max_caps], "filtered_nonverifiable_units": merged_filtered}


def turns_have_clinical_signal(turns: Sequence[Dict[str, Any]]) -> bool:
    text = " ".join(base.safe_text(turn.get("text")) for turn in turns).lower()
    if not text.strip():
        return False
    strong_markers = (
        "pain",
        "numb",
        "ting",
        "weakness",
        "dizzy",
        "fatigue",
        "swelling",
        "fracture",
        "diabetes",
        "a1c",
        "hemoglobin",
        "mri",
        "x-ray",
        "emg",
        "ncv",
        "ultrasound",
        "referral",
        "follow up",
        "follow-up",
        "prescribe",
        "continue",
        "start",
        "stop",
        "increase",
        "decrease",
        "order",
        "plan",
        "diagnosis",
        "impression",
    )
    if any(marker in text for marker in strong_markers):
        return True
    meaningful_turns = [turn for turn in turns if len(base.safe_text(turn.get("text")).split()) >= 5]
    return len(meaningful_turns) >= 3


def is_problem_cap_obj_suspicious(cap_obj: Dict[str, Any], turns: Sequence[Dict[str, Any]]) -> bool:
    caps = cap_obj.get("caps") or []
    if not isinstance(caps, list):
        return True
    if not caps:
        return turns_have_clinical_signal(turns)
    for cap in caps:
        if not isinstance(cap, dict):
            return True
        proposition_text = base.safe_text(cap.get("proposition_text"))
        canonical_concept = base.safe_text(cap.get("canonical_concept"))
        if not proposition_text or not canonical_concept:
            return True
        if len(proposition_text) > 500:
            return True
    return False


def parse_structured_json_with_repair(
    client: base.OpenAICompatClient,
    *,
    model: str,
    raw: str,
    schema_obj: Dict[str, Any],
    max_tokens: int,
) -> Any:
    try:
        return base.safe_json_extract(raw)
    except Exception:
        return base.repair_json_via_llm(
            client,
            model=model,
            raw_text=raw,
            schema_obj=schema_obj,
            max_tokens=max_tokens,
        )


def call_problem_cap_extraction(
    client: base.OpenAICompatClient,
    *,
    model: str,
    prompt: str,
    schema_obj: Dict[str, Any],
    max_tokens: int,
    temperature: float,
) -> Any:
    raw = base.call_llm(
        client,
        model=model,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        force_json=True,
        json_schema_obj=schema_obj,
        prefer_json_object=False,
    )
    return parse_structured_json_with_repair(
        client,
        model=model,
        raw=raw,
        schema_obj=schema_obj,
        max_tokens=max_tokens,
    )


def build_legacy_atomic_prompt(turns: Sequence[Dict[str, Any]]) -> str:
    return LEGACY_ATOMIC_TRANSCRIPT_PROMPT.format(turn_block=format_turn_block(turns))


def extract_caps_via_legacy_atomic_fallback(
    client: base.OpenAICompatClient,
    *,
    model: str,
    turns: Sequence[Dict[str, Any]],
    max_tokens: int,
    temperature: float,
) -> Dict[str, Any]:
    schema_obj = legacy_atomic_cap_schema(16)
    attempts = [
        build_legacy_atomic_prompt(turns),
        build_legacy_atomic_prompt(turns) + "\n\n" + LEGACY_ATOMIC_CAP_RETRY_SUFFIX,
        build_legacy_atomic_prompt(turns) + "\n\n" + LEGACY_ATOMIC_CAP_RETRY_SUFFIX + "\nReturn fewer propositions if needed.",
    ]
    last_obj: Dict[str, Any] = {"caps": [], "filtered_nonverifiable_units": []}
    parse_failures: List[Dict[str, Any]] = []
    for prompt in attempts:
        try:
            parsed = call_problem_cap_extraction(
                client,
                model=model,
                prompt=prompt,
                schema_obj=schema_obj,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        except Exception as exc:
            parse_failures.append({"stage": "legacy_atomic", "error": base.safe_text(exc)})
            continue
        cap_obj = normalize_legacy_atomic_cap_obj(parsed)
        cap_obj = attach_evidence_metadata(cap_obj, turns)
        cap_obj = sanitize_problem_cap_obj(cap_obj, turns=turns)
        last_obj = cap_obj
        if not is_problem_cap_obj_suspicious(cap_obj, turns):
            cap_obj["_legacy_fallback_used"] = True
            if parse_failures:
                cap_obj["parse_failures"] = parse_failures
            return cap_obj
    last_obj["_legacy_fallback_used"] = False
    if parse_failures:
        last_obj["parse_failures"] = parse_failures
    return last_obj


def extract_caps_from_text(
    client: base.OpenAICompatClient,
    *,
    model: str,
    prompt: str,
    max_items: int,
    max_tokens: int,
    temperature: float,
    turns: Sequence[Dict[str, Any]],
    allow_legacy_fallback: bool = False,
) -> Dict[str, Any]:
    schema_obj = transcript_cap_schema(max_items)
    attempts = [
        prompt,
        prompt + "\n\n" + CURRENT_CAP_RETRY_SUFFIX,
        prompt + "\n\n" + CURRENT_CAP_RETRY_SUFFIX + "\nReturn fewer CAPs if needed, but do not return an empty list when clinically central assertions are present.",
    ]
    last_obj: Dict[str, Any] = {"caps": [], "filtered_nonverifiable_units": []}
    parse_failures: List[Dict[str, Any]] = []
    for attempt_prompt in attempts:
        try:
            parsed = call_problem_cap_extraction(
                client,
                model=model,
                prompt=attempt_prompt,
                schema_obj=schema_obj,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        except Exception as exc:
            parse_failures.append({"stage": "problem_cap", "error": base.safe_text(exc)})
            continue
        cap_obj = normalize_problem_cap_obj(parsed)
        cap_obj = attach_evidence_metadata(cap_obj, turns)
        cap_obj = sanitize_problem_cap_obj(cap_obj, turns=turns)
        last_obj = cap_obj
        if not is_problem_cap_obj_suspicious(cap_obj, turns):
            cap_obj["_legacy_fallback_used"] = False
            if parse_failures:
                cap_obj["parse_failures"] = parse_failures
            return cap_obj
    if allow_legacy_fallback:
        try:
            legacy_cap_obj = extract_caps_via_legacy_atomic_fallback(
                client,
                model=model,
                turns=turns,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        except Exception as exc:
            parse_failures.append({"stage": "legacy_fallback", "error": base.safe_text(exc)})
            legacy_cap_obj = {"caps": [], "filtered_nonverifiable_units": []}
        if not is_problem_cap_obj_suspicious(legacy_cap_obj, turns):
            if parse_failures:
                legacy_cap_obj["parse_failures"] = parse_failures
            return legacy_cap_obj
    last_obj["_legacy_fallback_used"] = False
    if parse_failures:
        last_obj["parse_failures"] = parse_failures
    return last_obj


def extract_caps_from_text_single_call(
    client: base.OpenAICompatClient,
    *,
    model: str,
    prompt: str,
    max_items: int,
    max_tokens: int,
    temperature: float,
    turns: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    schema_obj = transcript_cap_schema(max_items)
    parsed = call_problem_cap_extraction(
        client,
        model=model,
        prompt=prompt,
        schema_obj=schema_obj,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    cap_obj = normalize_problem_cap_obj(parsed)
    cap_obj = attach_evidence_metadata(cap_obj, turns)
    cap_obj = sanitize_problem_cap_obj_minimal(cap_obj)
    cap_obj["_legacy_fallback_used"] = False
    return cap_obj


def high_value_cap_count(cap_obj: Dict[str, Any]) -> int:
    return sum(
        1
        for cap in cap_obj.get("caps", [])
        if base.safe_text(cap.get("cap_type")) in {"ChiefComplaint", "Symptom", "Problem", "Diagnosis", "Impression", "ExamFinding", "TestResult", "MedicationRequest", "Order"}
    )


def should_run_transcript_rescue(cap_obj: Dict[str, Any], turns: Sequence[Dict[str, Any]]) -> bool:
    n_caps = len(cap_obj.get("caps", []))
    high_value_count = high_value_cap_count(cap_obj)
    if n_caps == 0:
        return True
    if len(turns) > 20 and n_caps < TRANSCRIPT_RESCUE_MIN_CAPS:
        return True
    if len(turns) > 12 and high_value_count == 0:
        return True
    return False


def extract_transcript_caps_rescue(
    client: base.OpenAICompatClient,
    *,
    model: str,
    turns: Sequence[Dict[str, Any]],
    max_tokens: int,
    temperature: float,
) -> Dict[str, Any]:
    rescue_chunks = chunk_turns(
        turns,
        chunk_size=max(18, TRANSCRIPT_CHUNK_TURNS),
        overlap=max(2, TRANSCRIPT_CHUNK_OVERLAP),
        max_chars=max(5000, TRANSCRIPT_CHUNK_MAX_CHARS),
        short_case_turn_limit=max(40, TRANSCRIPT_SHORT_CASE_MAX_TURNS),
    )
    rescue_cap_objs: List[Dict[str, Any]] = []
    legacy_fallback_count = 0
    per_chunk_items = min(64, TRANSCRIPT_CAP_MAX_ITEMS)
    per_chunk_tokens = max(900, min(max_tokens, 1800))
    for chunk in rescue_chunks:
        prompt = (
            TRANSCRIPT_TO_CAP_RESCUE_PROMPT.format(turn_block=format_turn_block(chunk))
            + "\n\n"
            + EXTRACTION_GUIDANCE
        )
        rescue_cap_objs.append(
            extract_caps_from_text(
                client,
                model=model,
                prompt=prompt,
                max_items=per_chunk_items,
                max_tokens=per_chunk_tokens,
                temperature=temperature,
                turns=chunk,
                allow_legacy_fallback=True,
            )
        )
        if rescue_cap_objs[-1].get("_legacy_fallback_used"):
            legacy_fallback_count += 1
    merged = merge_problem_cap_objects(rescue_cap_objs, max_caps=TRANSCRIPT_CAP_MAX_ITEMS)
    merged["chunk_count"] = len(rescue_chunks)
    merged["rescue_used"] = True
    merged["rescue_attempted"] = True
    merged["legacy_fallback_chunk_count"] = legacy_fallback_count
    return merged


def extract_transcript_caps_single_call(
    client: base.OpenAICompatClient,
    *,
    model: str,
    turns: Sequence[Dict[str, Any]],
    max_tokens: int,
    temperature: float,
) -> Dict[str, Any]:
    chunks = chunk_turns(turns)
    cap_objs: List[Dict[str, Any]] = []
    per_chunk_items = min(16, TRANSCRIPT_CAP_MAX_ITEMS)
    per_chunk_tokens = max(900, min(max_tokens, 2600))
    for chunk in chunks:
        prompt = (
            TRANSCRIPT_TO_CAP_PROMPT.format(
                turn_block=format_turn_block(chunk),
                max_items=per_chunk_items,
            )
            + "\n\n"
            + EXTRACTION_GUIDANCE
        )
        cap_obj = extract_caps_from_text_single_call(
            client,
            model=model,
            prompt=prompt,
            max_items=per_chunk_items,
            max_tokens=per_chunk_tokens,
            temperature=temperature,
            turns=chunk,
        )
        cap_objs.append(cap_obj)

    merged = merge_problem_cap_objects(cap_objs, max_caps=TRANSCRIPT_CAP_MAX_ITEMS)
    cap_obj = sanitize_problem_cap_obj_minimal(merged)
    cap_obj = attach_evidence_metadata(cap_obj, turns)
    cap_obj["chunk_count"] = len(chunks)
    cap_obj["rescue_used"] = False
    cap_obj["rescue_attempted"] = False
    cap_obj["legacy_fallback_chunk_count"] = 0
    cap_obj["extraction_mode"] = "single_call"
    cap_obj.pop("_legacy_fallback_used", None)
    return cap_obj


def extract_transcript_caps(
    client: base.OpenAICompatClient,
    *,
    model: str,
    turns: Sequence[Dict[str, Any]],
    max_tokens: int,
    temperature: float,
) -> Dict[str, Any]:
    chunks = chunk_turns(turns)
    cap_objs: List[Dict[str, Any]] = []
    legacy_fallback_count = 0
    per_chunk_items = min(16, TRANSCRIPT_CAP_MAX_ITEMS)
    per_chunk_tokens = max(900, min(max_tokens, 2600))
    for chunk in chunks:
        prompt = (
            TRANSCRIPT_TO_CAP_PROMPT.format(turn_block=format_turn_block(chunk), max_items=per_chunk_items)
            + "\n\n"
            + EXTRACTION_GUIDANCE
        )
        cap_obj = extract_caps_from_text(
            client,
            model=model,
            prompt=prompt,
            max_items=per_chunk_items,
            max_tokens=per_chunk_tokens,
            temperature=temperature,
            turns=chunk,
            allow_legacy_fallback=True,
        )
        cap_objs.append(cap_obj)
        if cap_obj.get("_legacy_fallback_used"):
            legacy_fallback_count += 1
    merged = merge_problem_cap_objects(cap_objs, max_caps=TRANSCRIPT_CAP_MAX_ITEMS)
    merged["chunk_count"] = len(chunks)
    merged["rescue_used"] = False
    merged["rescue_attempted"] = False
    merged["legacy_fallback_chunk_count"] = legacy_fallback_count
    if should_run_transcript_rescue(merged, turns):
        rescue_obj = extract_transcript_caps_rescue(
            client,
            model=model,
            turns=turns,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        merged["rescue_attempted"] = True
        rescue_is_better = (
            len(rescue_obj.get("caps", [])) > len(merged.get("caps", []))
            or high_value_cap_count(rescue_obj) > high_value_cap_count(merged)
        )
        if rescue_is_better:
            rescue_obj["primary_chunk_count"] = len(chunks)
            rescue_obj["primary_legacy_fallback_chunk_count"] = legacy_fallback_count
            return rescue_obj
    if is_problem_cap_obj_suspicious(merged, turns):
        encounter_fallback_obj = extract_caps_via_legacy_atomic_fallback(
            client,
            model=model,
            turns=turns,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        encounter_fallback_obj["chunk_count"] = 1
        encounter_fallback_obj["encounter_legacy_fallback_used"] = bool(encounter_fallback_obj.get("_legacy_fallback_used"))
        encounter_fallback_obj["primary_chunk_count"] = len(chunks)
        encounter_fallback_obj["legacy_fallback_chunk_count"] = legacy_fallback_count
        encounter_fallback_obj["rescue_attempted"] = merged.get("rescue_attempted", False)
        if not is_problem_cap_obj_suspicious(encounter_fallback_obj, turns):
            encounter_fallback_obj.pop("_legacy_fallback_used", None)
            encounter_fallback_obj["rescue_used"] = False
            return encounter_fallback_obj
    merged.pop("_legacy_fallback_used", None)
    return merged


def extract_reference_caps(
    client: base.OpenAICompatClient,
    *,
    model: str,
    note_units: Sequence[Dict[str, Any]],
    max_tokens: int,
    temperature: float,
) -> Dict[str, Any]:
    chunks = chunk_note_units(note_units)
    cap_objs: List[Dict[str, Any]] = []
    per_chunk_items = min(40, SUMMARY_CAP_MAX_ITEMS)
    per_chunk_tokens = max(800, min(max_tokens, 1800))
    for chunk in chunks:
        prompt = (
            SUMMARY_TO_PROBLEM_CAP_PROMPT.format(summary_text=format_turn_block(chunk))
            + "\n\n"
            + EXTRACTION_GUIDANCE
        )
        cap_obj = extract_caps_from_text(
            client,
            model=model,
            prompt=prompt,
            max_items=per_chunk_items,
            max_tokens=per_chunk_tokens,
            temperature=temperature,
            turns=chunk,
        )
        cap_objs.append(cap_obj)
    merged = merge_problem_cap_objects(cap_objs, max_caps=SUMMARY_CAP_MAX_ITEMS)
    merged["chunk_count"] = len(chunks)
    return merged


def extract_reference_caps_single_call(
    client: base.OpenAICompatClient,
    *,
    model: str,
    note_units: Sequence[Dict[str, Any]],
    max_tokens: int,
    temperature: float,
) -> Dict[str, Any]:
    chunks = chunk_note_units(note_units)
    cap_objs: List[Dict[str, Any]] = []
    per_chunk_items = min(40, SUMMARY_CAP_MAX_ITEMS)
    per_chunk_tokens = max(800, min(max_tokens, 1800))
    for chunk in chunks:
        prompt = (
            SUMMARY_TO_PROBLEM_CAP_PROMPT.format(summary_text=format_turn_block(chunk))
            + "\n\n"
            + EXTRACTION_GUIDANCE
        )
        cap_obj = extract_caps_from_text_single_call(
            client,
            model=model,
            prompt=prompt,
            max_items=per_chunk_items,
            max_tokens=per_chunk_tokens,
            temperature=temperature,
            turns=chunk,
        )
        cap_objs.append(cap_obj)

    merged = merge_problem_cap_objects(cap_objs, max_caps=SUMMARY_CAP_MAX_ITEMS)
    cap_obj = sanitize_problem_cap_obj_minimal(merged)
    cap_obj = attach_evidence_metadata(cap_obj, note_units)
    cap_obj["chunk_count"] = len(chunks)
    cap_obj["extraction_mode"] = "single_call"
    cap_obj.pop("_legacy_fallback_used", None)
    return cap_obj


def canonicalize_medication_event(cap: Dict[str, Any]) -> str:
    clinical_status = base.safe_text(cap.get("clinical_status") or cap.get("visit_state")).lower()
    temporality = base.safe_text(cap.get("temporality")).lower()
    text = base.safe_text(cap.get("proposition_text")).lower()
    attributes = cap.get("attributes") if isinstance(cap.get("attributes"), dict) else {}
    action = base.safe_text(attributes.get("action")).lower()
    if clinical_status == "planned":
        return "MedicationChange"
    if action in {"start", "stop", "hold", "change", "increase", "decrease", "switch", "refill", "continue"}:
        return "MedicationChange"
    if any(k in temporality for k in ["plan", "planned", "next", "today start", "will", "going to"]):
        return "MedicationChange"
    if any(k in text for k in ["increase", "decrease", "start", "stop", "hold", "change", "refill", "new prescription", "switch", "will continue", "continue for now"]):
        return "MedicationChange"
    if any(k in text for k in ["taking", "takes", "is on", "continue"]):
        return "MedicationState"
    if base.safe_text(cap.get("cap_type")).lower() == "medicationstatement":
        return "MedicationState"
    return "MedicationState"


def state_cluster_key(cap: Dict[str, Any]) -> Tuple[str, str]:
    cap_type = base.safe_text(cap.get("cap_type"))
    concept = base.safe_text(cap.get("canonical_concept")).lower()
    if cap_type in {"MedicationStatement", "MedicationRequest"}:
        return canonicalize_medication_event(cap), concept
    if cap_type in {"ChiefComplaint", "Symptom", "Problem", "ProblemHistory", "Diagnosis", "Impression", "Allergy"}:
        return "ProblemState", concept
    if cap_type == "Demographics":
        return "Demographics", concept
    if cap_type == "ExamFinding":
        return "ExamFinding", concept
    if cap_type == "TestResult":
        return "TestResult", concept
    if cap_type in {"Order", "Counseling"}:
        return "Plan", concept
    return cap_type or "Other", concept


def first_turn_index(cap: Dict[str, Any]) -> int:
    indices: List[int] = []
    for ev in cap.get("evidence", []):
        turn_id = base.safe_text(ev.get("turn_id"))
        m = re.fullmatch(r"[TS](\d+)", turn_id)
        if m:
            indices.append(int(m.group(1)))
    return min(indices) if indices else 10**9


def resolve_cluster_state(caps: Sequence[Dict[str, Any]]) -> Tuple[str, str]:
    sorted_caps = sorted(caps, key=first_turn_index)
    final_verification = "confirmed"
    final_clinical_status = "active"
    state_rank = {
        "planned": 7,
        "resolved": 6,
        "worsening": 4,
        "improving": 3,
        "stable": 2,
        "active": 1,
        "historical": 1,
    }
    for cap in sorted_caps:
        verification_status = base.safe_text(cap.get("verification_status")) or final_verification
        clinical_status = base.safe_text(cap.get("clinical_status") or cap.get("visit_state")) or final_clinical_status
        final_verification = verification_status
        if state_rank.get(clinical_status, 0) >= state_rank.get(final_clinical_status, 0):
            final_clinical_status = clinical_status
    return final_verification, final_clinical_status


def summarize_cluster_speakers(caps: Sequence[Dict[str, Any]]) -> List[str]:
    speakers = sorted(
        {
            base.safe_text(cap.get("content_source") or cap.get("speaker"))
            for cap in caps
            if base.safe_text(cap.get("content_source") or cap.get("speaker"))
        }
    )
    return speakers


def summarize_cluster_modalities(caps: Sequence[Dict[str, Any]]) -> List[str]:
    modalities = sorted({base.safe_text(cap.get("modality")) for cap in caps if base.safe_text(cap.get("modality"))})
    return modalities


def summarize_cluster_temporalities(caps: Sequence[Dict[str, Any]]) -> List[str]:
    temporality_values = sorted({base.safe_text(cap.get("temporality")) for cap in caps if base.safe_text(cap.get("temporality"))})
    return temporality_values


GENERIC_PROBLEM_CONCEPTS = {
    "pain",
    "weakness",
    "tingling",
    "numbness",
    "swelling",
    "rash",
    "fatigue",
    "dizziness",
    "headache",
    "nausea",
    "congestion",
}


def summarize_cluster_cap_types(caps: Sequence[Dict[str, Any]]) -> List[str]:
    return sorted({base.safe_text(cap.get("cap_type")) for cap in caps if base.safe_text(cap.get("cap_type"))})


def cluster_member_priority(cap: Dict[str, Any]) -> Tuple[int, int, int, int, int]:
    cap_type = base.safe_text(cap.get("cap_type"))
    type_priority = CAP_TYPE_PRIORITY.get(cap_type, 0)
    modality = base.safe_text(cap.get("modality"))
    modality_priority = {
        "assessment": 5,
        "test_result": 4,
        "plan": 4,
        "exam": 3,
        "interview": 2,
        "historical_record": 1,
    }.get(modality, 0)
    concept = base.safe_text(cap.get("canonical_concept"))
    concept_specificity = len(concept.split())
    text_len = len(base.safe_text(cap.get("proposition_text")).split())
    evidence_count = len(cap.get("evidence") or [])
    penalty = -4 if is_low_value_problem_cap(cap) and not is_protected_high_value_cap(cap) else 0
    return (type_priority + penalty, modality_priority, concept_specificity, evidence_count, text_len)


def choose_cluster_concept(members: Sequence[Dict[str, Any]]) -> str:
    best = max(members, key=cluster_member_priority)
    return base.safe_text(best.get("canonical_concept")).lower()


def choose_cluster_summary(members: Sequence[Dict[str, Any]], cluster_type: str) -> str:
    prioritized = sorted(members, key=cluster_member_priority, reverse=True)
    if cluster_type == "MedicationChange":
        planned = [cap for cap in prioritized if canonicalize_medication_event(cap) == "MedicationChange"]
        if planned:
            return base.safe_text(planned[0].get("proposition_text"))
    return base.safe_text(prioritized[0].get("proposition_text"))


def choose_primary_cap_type(members: Sequence[Dict[str, Any]]) -> str:
    best = max(members, key=cluster_member_priority)
    return base.safe_text(best.get("cap_type")) or "Problem"


def concept_is_generic_problem(concept: str) -> bool:
    return base.normalize_text(concept) in GENERIC_PROBLEM_CONCEPTS


def member_turn_span(members: Sequence[Dict[str, Any]]) -> Tuple[int, int]:
    indices = [first_turn_index(cap) for cap in members if first_turn_index(cap) < 10**9]
    if not indices:
        return (10**9, 10**9)
    return (min(indices), max(indices))


def turn_span_gap(span_a: Tuple[int, int], span_b: Tuple[int, int]) -> int:
    if span_a[0] == 10**9 or span_b[0] == 10**9:
        return 10**9
    if span_a[1] < span_b[0]:
        return span_b[0] - span_a[1]
    if span_b[1] < span_a[0]:
        return span_a[0] - span_b[1]
    return 0


def should_merge_member_groups(group_a: Sequence[Dict[str, Any]], group_b: Sequence[Dict[str, Any]]) -> bool:
    type_a, _ = state_cluster_key(group_a[0])
    type_b, _ = state_cluster_key(group_b[0])
    if type_a != type_b:
        return False
    concept_a = choose_cluster_concept(group_a)
    concept_b = choose_cluster_concept(group_b)
    if not concept_a or not concept_b:
        return False
    if concept_a == concept_b:
        return True
    gap = turn_span_gap(member_turn_span(group_a), member_turn_span(group_b))
    if gap > 2:
        return False
    if concept_a in concept_b or concept_b in concept_a:
        return True
    if type_a == "ProblemState":
        generic_a = concept_is_generic_problem(concept_a)
        generic_b = concept_is_generic_problem(concept_b)
        if generic_a and any(token == concept_a for token in concept_b.split()):
            return True
        if generic_b and any(token == concept_b for token in concept_a.split()):
            return True
    return False


def merge_member_groups(member_groups: Sequence[Sequence[Dict[str, Any]]]) -> List[List[Dict[str, Any]]]:
    merged: List[List[Dict[str, Any]]] = []
    for group in sorted(member_groups, key=lambda g: member_turn_span(g)[0]):
        group_list = list(group)
        merged_into_existing = False
        for existing in merged:
            if should_merge_member_groups(existing, group_list):
                seen_ids = {id(cap) for cap in existing}
                for cap in group_list:
                    if id(cap) not in seen_ids:
                        existing.append(cap)
                existing.sort(key=first_turn_index)
                merged_into_existing = True
                break
        if not merged_into_existing:
            merged.append(sorted(group_list, key=first_turn_index))
    return merged


def is_low_value_cluster(cluster_type: str, concept: str, members: Sequence[Dict[str, Any]]) -> bool:
    norm_concept = base.normalize_text(concept)
    if cluster_type != "ProblemState":
        return False
    if norm_concept in LOW_VALUE_CONCEPTS:
        return True
    if any(token in norm_concept for token in ("sandwich", "pasta", "bread", "club soda", "water intake")):
        return True
    if members and all(is_low_value_problem_cap(cap) and not is_protected_high_value_cap(cap) for cap in members):
        return True
    return False


def cluster_problem_states(cap_obj: Dict[str, Any]) -> Dict[str, Any]:
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for cap in cap_obj.get("caps", []):
        grouped[state_cluster_key(cap)].append(cap)

    clusters: List[Dict[str, Any]] = []
    member_groups = merge_member_groups([sorted(members, key=first_turn_index) for members in grouped.values()])
    for idx, members in enumerate(sorted(member_groups, key=lambda g: member_turn_span(g)[0]), start=1):
        cluster_type, _ = state_cluster_key(members[0])
        concept = choose_cluster_concept(members)
        if is_low_value_cluster(cluster_type, concept, members):
            continue
        final_verification, final_clinical_status = resolve_cluster_state(members)
        evidence_turn_ids = sorted({base.safe_text(ev.get("turn_id")) for cap in members for ev in cap.get("evidence", []) if base.safe_text(ev.get("turn_id"))})
        supporting_cap_ids = [base.safe_text(cap.get("cap_id")) for cap in members if base.safe_text(cap.get("cap_id"))]
        updates = []
        current_regimens: List[Dict[str, Any]] = []
        planned_changes: List[Dict[str, Any]] = []
        source_cap_types = summarize_cluster_cap_types(members)
        primary_cap_type = choose_primary_cap_type(members)
        for cap in members:
            updates.append(
                {
                    "cap_id": cap.get("cap_id"),
                    "content_source": cap.get("content_source") or cap.get("speaker"),
                    "modality": cap.get("modality"),
                    "verification_status": cap.get("verification_status"),
                    "clinical_status": cap.get("clinical_status"),
                    "assertion": cap.get("assertion"),
                    "visit_state": cap.get("visit_state"),
                    "temporality": cap.get("temporality"),
                    "turn_ids": [ev.get("turn_id") for ev in cap.get("evidence", [])],
                    "proposition_text": cap.get("proposition_text"),
                }
            )
            if cluster_type in {"MedicationState", "MedicationChange"}:
                event_kind = canonicalize_medication_event(cap)
                regimen = {
                    "proposition_text": cap.get("proposition_text"),
                    "turn_ids": [ev.get("turn_id") for ev in cap.get("evidence", [])],
                    "attributes": cap.get("attributes", {}),
                }
                if event_kind == "MedicationChange":
                    planned_changes.append(regimen)
                else:
                    current_regimens.append(regimen)

        cluster_summary = choose_cluster_summary(members, cluster_type)
        salience_score = max(0, max(cluster_member_priority(cap)[0] for cap in members))
        cluster = {
            "cluster_id": f"CL{idx}",
            "cluster_type": cluster_type,
            "canonical_concept": concept,
            "primary_cap_type": primary_cap_type,
            "source_cap_types": source_cap_types,
            "final_verification_status": final_verification,
            "clinical_status": final_clinical_status,
            "final_assertion": verification_status_to_assertion(final_verification, final_clinical_status),
            "visit_state": final_clinical_status,
            "source_speakers": summarize_cluster_speakers(members),
            "source_modalities": summarize_cluster_modalities(members),
            "temporal_profile": summarize_cluster_temporalities(members),
            "evidence_turn_ids": evidence_turn_ids,
            "supporting_cap_ids": supporting_cap_ids,
            "cluster_summary": cluster_summary,
            "salience_score": salience_score,
            "is_low_value": False,
            "updates": updates,
        }
        if current_regimens:
            cluster["current_regimens"] = current_regimens
        if planned_changes:
            cluster["planned_changes"] = planned_changes
        clusters.append(cluster)
    return {"clusters": clusters}


def assign_event_cluster_ids(cap_obj: Dict[str, Any], cluster_obj: Dict[str, Any]) -> Dict[str, Any]:
    cap_to_cluster: Dict[str, str] = {}
    for cluster in cluster_obj.get("clusters", []):
        cluster_id = base.safe_text(cluster.get("cluster_id"))
        for cap_id in cluster.get("supporting_cap_ids", []):
            cap_id = base.safe_text(cap_id)
            if cap_id and cluster_id:
                cap_to_cluster[cap_id] = cluster_id
    for cap in cap_obj.get("caps", []):
        cap_id = base.safe_text(cap.get("cap_id"))
        if cap_id in cap_to_cluster:
            cap["event_cluster_id"] = cap_to_cluster[cap_id]
    return cap_obj


def render_problem_state_report(case_id: str, cluster_obj: Dict[str, Any]) -> str:
    lines = [f"Case: {case_id}", "", "Problem List and Visit State Tracking", ""]
    problem_clusters = [c for c in cluster_obj.get("clusters", []) if c.get("cluster_type") == "ProblemState"]
    medication_clusters = [c for c in cluster_obj.get("clusters", []) if c.get("cluster_type") in {"MedicationState", "MedicationChange"}]
    other_clusters = [c for c in cluster_obj.get("clusters", []) if c not in problem_clusters and c not in medication_clusters]

    lines.append("Problem List")
    if not problem_clusters:
        lines.append("- None")
    for cluster in problem_clusters:
        lines.append(
            f"- {cluster['canonical_concept']} | verification_status={cluster.get('final_verification_status')} | clinical_status={cluster.get('clinical_status')} | modalities={','.join(cluster.get('source_modalities', []))} | speakers={','.join(cluster.get('source_speakers', []))} | evidence={','.join(cluster.get('evidence_turn_ids', []))}"
        )
        lines.append(f"  Summary: {cluster['cluster_summary']}")
        lines.append(f"  ClusterId: {cluster['cluster_id']}")
        if cluster.get("temporal_profile"):
            lines.append(f"  TemporalProfile: {', '.join(cluster.get('temporal_profile', []))}")
        lines.append("  Updates:")
        for update in cluster.get("updates", []):
            lines.append(
                f"  - {update.get('cap_id')}: content_source={update.get('content_source')} modality={update.get('modality')} verification_status={update.get('verification_status')} clinical_status={update.get('clinical_status')} temporality={update.get('temporality')} turns={','.join(update.get('turn_ids') or [])} :: {update.get('proposition_text')}"
            )

    lines.append("")
    lines.append("Medication Tracking")
    if not medication_clusters:
        lines.append("- None")
    for cluster in medication_clusters:
        lines.append(
            f"- [{cluster['cluster_type']}] {cluster['canonical_concept']} | verification_status={cluster.get('final_verification_status')} | clinical_status={cluster.get('clinical_status')} | modalities={','.join(cluster.get('source_modalities', []))} | speakers={','.join(cluster.get('source_speakers', []))} | evidence={','.join(cluster.get('evidence_turn_ids', []))}"
        )
        lines.append(f"  Summary: {cluster['cluster_summary']}")
        lines.append(f"  ClusterId: {cluster['cluster_id']}")
        if cluster.get("temporal_profile"):
            lines.append(f"  TemporalProfile: {', '.join(cluster.get('temporal_profile', []))}")
        if cluster.get("current_regimens"):
            lines.append("  CurrentRegimens:")
            for item in cluster["current_regimens"]:
                lines.append(f"  - {item['proposition_text']} ({','.join(item.get('turn_ids', []))})")
        if cluster.get("planned_changes"):
            lines.append("  PlannedChanges:")
            for item in cluster["planned_changes"]:
                lines.append(f"  - {item['proposition_text']} ({','.join(item.get('turn_ids', []))})")
        lines.append("  Updates:")
        for update in cluster.get("updates", []):
            lines.append(
                f"  - {update.get('cap_id')}: content_source={update.get('content_source')} modality={update.get('modality')} verification_status={update.get('verification_status')} clinical_status={update.get('clinical_status')} temporality={update.get('temporality')} turns={','.join(update.get('turn_ids') or [])} :: {update.get('proposition_text')}"
            )

    lines.append("")
    lines.append("Other Structured States")
    if not other_clusters:
        lines.append("- None")
    for cluster in other_clusters:
        lines.append(
            f"- [{cluster['cluster_type']}] {cluster['canonical_concept']} | verification_status={cluster.get('final_verification_status')} | clinical_status={cluster.get('clinical_status')} | modalities={','.join(cluster.get('source_modalities', []))} | speakers={','.join(cluster.get('source_speakers', []))} | evidence={','.join(cluster.get('evidence_turn_ids', []))}"
        )
        lines.append(f"  Summary: {cluster['cluster_summary']}")
        lines.append(f"  ClusterId: {cluster['cluster_id']}")
        if cluster.get("temporal_profile"):
            lines.append(f"  TemporalProfile: {', '.join(cluster.get('temporal_profile', []))}")
    return "\n".join(lines).strip() + "\n"


def cluster_signature_set(cluster_obj: Dict[str, Any], *, include_state: bool) -> set[Tuple[str, ...]]:
    out: set[Tuple[str, ...]] = set()
    for cluster in cluster_obj.get("clusters", []):
        concept = base.safe_text(cluster.get("canonical_concept")).lower()
        ctype = base.safe_text(cluster.get("cluster_type"))
        if not concept:
            continue
        if include_state:
            out.add(
                (
                    ctype,
                    concept,
                    base.safe_text(cluster.get("final_verification_status") or cluster.get("final_assertion")),
                    base.safe_text(cluster.get("clinical_status") or cluster.get("visit_state")),
                )
            )
        else:
            out.add((ctype, concept))
    return out


def set_prf(pred_set: set[Tuple[str, ...]], ref_set: set[Tuple[str, ...]]) -> Tuple[float, float, float]:
    if not pred_set:
        return (0.0, 0.0, 0.0)
    overlap = len(pred_set & ref_set)
    precision = overlap / len(pred_set) if pred_set else 0.0
    recall = overlap / len(ref_set) if ref_set else 0.0
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def summarize_rows(rows: Sequence[Dict[str, Any]], metric_names: Sequence[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"n_cases": len(rows)}
    for metric in metric_names:
        vals = [float(row[metric]) for row in rows if row.get(metric) is not None]
        if not vals:
            continue
        out[metric] = round(statistics.mean(vals), 4)
    return out


CASE_FIELDNAMES = [
    "case_id",
    "transcript_cap_count",
    "reference_cap_count",
    "cluster_count",
    "reference_cluster_count",
    "concept_precision",
    "concept_recall",
    "concept_f1",
    "state_precision",
    "state_recall",
    "state_f1",
    "transcript_cap_runtime_seconds",
    "reference_cap_runtime_seconds",
    "transcript_cap_prompt_tokens",
    "transcript_cap_completion_tokens",
    "transcript_cap_total_tokens",
]

AGG_METRICS = [
    "concept_precision",
    "concept_recall",
    "concept_f1",
    "state_precision",
    "state_recall",
    "state_f1",
    "cluster_count",
    "reference_cluster_count",
]

AGG_FIELDNAMES = [
    "n_cases",
    "concept_precision",
    "concept_recall",
    "concept_f1",
    "state_precision",
    "state_recall",
    "state_f1",
    "cluster_count",
    "reference_cluster_count",
]


def flush_progress(output_dir: Path, case_rows: Sequence[Dict[str, Any]]) -> None:
    base.write_jsonl(output_dir / "case_metrics.jsonl", case_rows)
    base.write_csv(output_dir / "case_metrics.csv", case_rows, fieldnames=CASE_FIELDNAMES)
    aggregate = summarize_rows(case_rows, AGG_METRICS)
    base.write_json(output_dir / "aggregate_metrics.json", aggregate)
    base.write_csv(output_dir / "aggregate_metrics.csv", [aggregate], fieldnames=AGG_FIELDNAMES)


def main() -> None:
    base.load_env_file(BASE_DIR / ".env")
    args = parse_args()
    api_base_url = base.infer_api_base_url(args.api_base_url)
    api_key = args.api_key or base.os.getenv("RUNPOD_API_KEY") or base.os.getenv("OPENAI_API_KEY", "")
    client = base.OpenAICompatClient(base_url=api_base_url, api_key=api_key, timeout=args.request_timeout)
    extractor_model = args.extractor_model or args.model
    extractor_api_base_url = infer_extractor_api_base_url(args.extractor_api_base_url, extractor_model, api_base_url)
    extractor_api_key = args.extractor_api_key or (
        base.os.getenv("OPENAI_API_KEY", "") if extractor_api_base_url == "https://api.openai.com/v1" else api_key
    )
    extractor_client = base.OpenAICompatClient(
        base_url=extractor_api_base_url,
        api_key=extractor_api_key,
        timeout=args.request_timeout,
    )

    cases = base.load_cases(args.cases_path, args.limit, args.case_ids)
    if not cases:
        raise SystemExit("No cases found.")

    output_dir = base.ensure_dir(args.output_dir)
    transcript_cap_dir = base.ensure_dir(output_dir / "transcript_caps")
    reference_cap_dir = base.ensure_dir(output_dir / "reference_caps")
    cluster_dir = base.ensure_dir(output_dir / "problem_clusters")
    report_dir = base.ensure_dir(output_dir / "reports")

    case_rows: List[Dict[str, Any]] = []
    print(
        f"[INFO] Using endpoint={api_base_url} model={args.model} "
        f"extractor_endpoint={extractor_api_base_url} extractor_model={extractor_model} "
        f"cases={len(cases)} cap_extraction_mode={args.cap_extraction_mode}",
        flush=True,
    )

    for idx, (case_id, case) in enumerate(cases.items(), start=1):
        transcript = base.safe_text(case.get("transcript"))
        reference_summary = base.safe_text(case.get("summary_gt_note"))
        if not transcript:
            continue
        if not args.deployment_cost_only and not reference_summary:
            continue
        print(f"[INFO] ({idx}/{len(cases)}) Processing {case_id}", flush=True)

        transcript_caps_path = transcript_cap_dir / f"{case_id}.json"
        reference_caps_path = reference_cap_dir / f"{case_id}.json"
        cluster_path = cluster_dir / f"{case_id}.json"
        report_path = report_dir / f"{case_id}.txt"

        turns = parse_transcript_turns(transcript)
        note_units = split_note_sentences(reference_summary) if reference_summary else []

        if args.deployment_cost_only:
            reuse_existing = args.skip_existing and transcript_caps_path.exists() and cluster_path.exists()
        else:
            reuse_existing = args.skip_existing and transcript_caps_path.exists() and reference_caps_path.exists() and cluster_path.exists()
        if reuse_existing:
            transcript_cap_obj = base.read_json(transcript_caps_path)
            reference_cap_obj = base.read_json(reference_caps_path) if (not args.deployment_cost_only and reference_caps_path.exists()) else None
            cluster_obj = base.read_json(cluster_path)
            if transcript_cap_obj.get("caps"):
                transcript_cap_obj = assign_event_cluster_ids(transcript_cap_obj, cluster_obj)
                base.write_json(transcript_caps_path, transcript_cap_obj)
                if not report_path.exists():
                    report_path.write_text(render_problem_state_report(case_id, cluster_obj), encoding="utf-8")
                if args.deployment_cost_only:
                    print(f"[INFO] {case_id}: reuse existing transcript CAPs (deployment-cost-only)", flush=True)
                else:
                    print(f"[INFO] {case_id}: reuse existing transcript/reference CAPs", flush=True)
            else:
                print(f"[INFO] {case_id}: existing transcript CAPs are empty; re-running extraction", flush=True)
                reuse_existing = False
        if not reuse_existing:
            try:
                t0 = time.perf_counter()
                usage_before = extractor_client.get_usage_totals()
                print(f"[INFO] {case_id}: extracting transcript CAPs", flush=True)
                if args.cap_extraction_mode == "single_call":
                    transcript_cap_obj = extract_transcript_caps_single_call(
                        extractor_client,
                        model=extractor_model,
                        max_tokens=args.max_extraction_tokens,
                        temperature=args.temperature,
                        turns=turns,
                    )
                else:
                    transcript_cap_obj = extract_transcript_caps(
                        extractor_client,
                        model=extractor_model,
                        max_tokens=args.max_extraction_tokens,
                        temperature=args.temperature,
                        turns=turns,
                    )
                transcript_cap_obj["runtime_seconds"] = round(time.perf_counter() - t0, 3)
                usage_after = extractor_client.get_usage_totals()
                transcript_usage = base.usage_delta(usage_after, usage_before)
                transcript_cap_obj["prompt_tokens"] = int(transcript_usage.get("prompt_tokens", 0))
                transcript_cap_obj["completion_tokens"] = int(transcript_usage.get("completion_tokens", 0))
                transcript_cap_obj["total_tokens"] = int(transcript_usage.get("total_tokens", 0))
                print(
                    f"[INFO] {case_id}: transcript CAPs={len(transcript_cap_obj.get('caps', []))} "
                    f"chunks={transcript_cap_obj.get('chunk_count', 1)} "
                    f"mode={transcript_cap_obj.get('extraction_mode', args.cap_extraction_mode)} "
                    f"rescue_used={transcript_cap_obj.get('rescue_used', False)} "
                    f"runtime={transcript_cap_obj['runtime_seconds']}s",
                    flush=True,
                )

                reference_cap_obj = None
                if not args.deployment_cost_only:
                    t0 = time.perf_counter()
                    print(f"[INFO] {case_id}: extracting reference CAPs", flush=True)
                    if args.cap_extraction_mode == "single_call":
                        reference_cap_obj = extract_reference_caps_single_call(
                            extractor_client,
                            model=extractor_model,
                            max_tokens=args.max_extraction_tokens,
                            temperature=args.temperature,
                            note_units=note_units,
                        )
                    else:
                        reference_cap_obj = extract_reference_caps(
                            extractor_client,
                            model=extractor_model,
                            max_tokens=args.max_extraction_tokens,
                            temperature=args.temperature,
                            note_units=note_units,
                        )
                    reference_cap_obj["runtime_seconds"] = round(time.perf_counter() - t0, 3)
                    print(
                        f"[INFO] {case_id}: reference CAPs={len(reference_cap_obj.get('caps', []))} "
                        f"chunks={reference_cap_obj.get('chunk_count', 1)} "
                        f"mode={reference_cap_obj.get('extraction_mode', args.cap_extraction_mode)} "
                        f"runtime={reference_cap_obj['runtime_seconds']}s",
                        flush=True,
                    )
                    base.write_json(reference_caps_path, reference_cap_obj)

                cluster_obj = cluster_problem_states(transcript_cap_obj)
                transcript_cap_obj = assign_event_cluster_ids(transcript_cap_obj, cluster_obj)
                base.write_json(transcript_caps_path, transcript_cap_obj)
                base.write_json(cluster_path, cluster_obj)
                report_path.write_text(render_problem_state_report(case_id, cluster_obj), encoding="utf-8")
            except Exception as exc:
                print(f"[WARN] {case_id}: extraction failed; skipping case ({base.safe_text(exc)[:300]})", flush=True)
                error_payload = {
                    "case_id": case_id,
                    "error": base.safe_text(exc),
                }
                base.write_json(report_dir / f"{case_id}_error.json", error_payload)
                continue

        if args.deployment_cost_only:
            concept_p = concept_r = concept_f1 = None
            state_p = state_r = state_f1 = None
            reference_cap_count = None
            reference_cap_runtime = None
            reference_cluster_count = None
        else:
            ref_cluster_obj = cluster_problem_states(reference_cap_obj)
            concept_p, concept_r, concept_f1 = set_prf(
                cluster_signature_set(cluster_obj, include_state=False),
                cluster_signature_set(ref_cluster_obj, include_state=False),
            )
            state_p, state_r, state_f1 = set_prf(
                cluster_signature_set(cluster_obj, include_state=True),
                cluster_signature_set(ref_cluster_obj, include_state=True),
            )
            reference_cap_count = len(reference_cap_obj.get("caps", []))
            reference_cap_runtime = reference_cap_obj.get("runtime_seconds")
            reference_cluster_count = len(ref_cluster_obj.get("clusters", []))

        row = {
            "case_id": case_id,
            "transcript_cap_count": len(transcript_cap_obj.get("caps", [])),
            "reference_cap_count": reference_cap_count,
            "cluster_count": len(cluster_obj.get("clusters", [])),
            "reference_cluster_count": reference_cluster_count,
            "concept_precision": round(concept_p, 4) if concept_p is not None else None,
            "concept_recall": round(concept_r, 4) if concept_r is not None else None,
            "concept_f1": round(concept_f1, 4) if concept_f1 is not None else None,
            "state_precision": round(state_p, 4) if state_p is not None else None,
            "state_recall": round(state_r, 4) if state_r is not None else None,
            "state_f1": round(state_f1, 4) if state_f1 is not None else None,
            "transcript_cap_runtime_seconds": transcript_cap_obj.get("runtime_seconds"),
            "reference_cap_runtime_seconds": reference_cap_runtime,
            "transcript_cap_prompt_tokens": int(transcript_cap_obj.get("prompt_tokens", 0) or 0),
            "transcript_cap_completion_tokens": int(transcript_cap_obj.get("completion_tokens", 0) or 0),
            "transcript_cap_total_tokens": int(transcript_cap_obj.get("total_tokens", 0) or 0),
        }
        case_rows.append(row)
        flush_progress(output_dir, case_rows)

    flush_progress(output_dir, case_rows)
    print(f"[INFO] Wrote results to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
