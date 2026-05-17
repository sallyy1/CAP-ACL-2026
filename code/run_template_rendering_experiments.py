from __future__ import annotations

import argparse
import csv
import json
import os
import re
import statistics
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib import error as urllib_error
from urllib import request as urllib_request

try:
    from openai import OpenAI
except ModuleNotFoundError:  # pragma: no cover
    OpenAI = None


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = BASE_DIR / "outputs" / "shared" / "run_aci_all_v1_realistic_subsets"
_DEFAULT_CASES_PRIMARY = BASE_DIR / "outputs" / "shared" / "run_aci_all_v1_realistic" / "cases.jsonl"
_DEFAULT_CASES_FALLBACK = BASE_DIR / "outputs" / "shared" / "run_aci_all_v1_realistic" / "cases.jsonl"
DEFAULT_CASES_PATH = _DEFAULT_CASES_PRIMARY if _DEFAULT_CASES_PRIMARY.exists() else _DEFAULT_CASES_FALLBACK
DEFAULT_METHOD_A_CSV = BASE_DIR / "outputs" / "shared" / "run_aci_all_v1_realistic_subsets" / "STRESS_method_A_major_omission.csv"
DEFAULT_METHOD_B_CSV = BASE_DIR / "outputs" / "shared" / "run_aci_all_v1_realistic_subsets" / "STRESS_method_B_major_omission.csv"
DEFAULT_METHOD_C_CSV = BASE_DIR / "outputs" / "shared" / "run_aci_all_v1_realistic_subsets" / "STRESS_method_C_major_omission.csv"
DEFAULT_OUTPUT_DIR = BASE_DIR / "outputs" / "template_rendering_runs"
DEFAULT_LEGACY_EXTRACTION_DIR = BASE_DIR / "outputs" / "output_aci_total_no-es_207"

METHOD_ALIASES = {
    "A": "medsum_ent",
    "B": "cluster2sent",
    "C": "cap",
    "C_event": "cap_event",
    "C_only": "cap_only",
    "C_event_only": "cap_event_only",
}

SUPPORTED_METHODS = (
    "direct",
    "medsum_ent",
    "cluster2sent",
    "cap",
    "cap_event",
    "cap_only",
    "cap_event_only",
    "A",
    "B",
    "C",
    "C_event",
    "C_only",
    "C_event_only",
)
SUPPORTED_TEMPLATES = ("soap", "sectioned", "brief")
EXPERIMENT_METHOD_SETS = {
    "all": ["direct", "medsum_ent", "cluster2sent", "cap", "cap_event", "cap_only", "cap_event_only"],
    "main": ["direct", "medsum_ent", "cluster2sent", "cap", "cap_event"],
    "ablation": ["direct", "cap", "cap_only", "cap_event", "cap_event_only"],
}
DEFAULT_MODEL = os.getenv("RUNPOD_MODEL", "ISTA-DASLab/gemma-3-4b-it-GPTQ-4b-128g")

CAP_MAX_PROPS = 36
EVENT_MAX_ITEMS = 14
SUMMARY_MAX_TOKENS = 1600
EXTRACTION_MAX_TOKENS = 2200
EVENT_MAX_TOKENS = 2200
CONSOLIDATION_STRATEGY = "single_stage"
ENABLE_ORPHAN_BACKFILL = False
ENABLE_SURFACE_SYMPTOM_BUNDLE_MERGE = True
GPT_F1_MAX_ITEMS_PER_CATEGORY = 12
NAIR_MAX_CONCEPTS_PER_SECTION = 20
NAIR_MAX_VERIFICATION_ITEMS = 24
JUDGE_MAX_TOKENS = 1800
SEMANTIC_PARTIAL_CREDIT = 0.5

PDSQI_CORE_KEYS = (
    "accurate",
    "thorough",
    "useful",
    "organized",
    "comprehensible",
    "succinct",
)

GPT_F1_CATEGORIES = (
    "active_problems_symptoms",
    "negated_findings",
    "uncertain_findings",
    "medication_treatment",
    "plan_followup",
)

DIRECT_PROMPT_BASE = """
You are a clinical documentation assistant.

Your task is to generate a clinically faithful visit note directly from the transcript.

[Rules]
- Use only information explicitly supported by the transcript.
- Preserve negation, temporality, uncertainty, laterality, numeric values, and management intent.
- Prefer conservative phrasing when the transcript is incomplete.
- Merge redundant utterances into natural clinical prose.
- Do not invent diagnoses, medications, test results, or plans.
- Output only the final note.
- Do not add any preamble, explanation, apology, or meta-commentary.
- Do not mention the transcript, CAPs, evidence units, or event plan.
- Do not use markdown bold, code fences, or source citations.

[Rendering Goal]
{template_instruction}

[Transcript]
{transcript}
""".strip()

MEDSUM_ENT_PROMPT_BASE_v0 = """
You are a clinical documentation assistant using a MEDSUM-ENT-inspired scaffold.

You are given the full transcript plus extracted clinically relevant concepts and status cues from the same
patient-provider dialogue. Use the transcript as the source of truth, and use the extracted scaffold to improve
content planning and section organization.

[How To Use The Inputs]
- Use the transcript as the final grounding source.
- Use the evidence list as a planning scaffold, not as an independent source.
- Organize concepts into the requested note template.
- Keep affirmed, negated, uncertain, current, past, and planned content distinct.
- Merge redundant concepts into concise clinical sentences.
- If an extracted concept seems underspecified, phrase it conservatively instead of speculating.
- If the scaffold conflicts with the transcript, follow the transcript.
- Do not add content not recoverable from the transcript.
- Output only the final note.
- Do not add any preamble, explanation, apology, or meta-commentary.
- Do not mention the transcript, evidence scaffold, or extraction process.
- Do not use markdown bold, code fences, or source citations.

[Rendering Goal]
{template_instruction}

[Transcript]
{transcript}

[Evidence Scaffold]
{source_block}
""".strip()

CLUSTER2SENT_PROMPT_BASE = """
You are a clinical documentation assistant using a Cluster2Sent-inspired modular strategy.

You are given section-targeted evidence units extracted from a patient-provider dialogue. Do not assume access to
the full transcript. Treat the evidence units as the only available source material for note generation.

Follow a modular Extract -> Cluster -> Generate mindset:
- Extract: treat each evidence unit as a localized clinical support utterance or fact.
- Cluster: first mentally group related units into coherent local clinical events.
- Generate: write one concise sentence or bullet per event, then assemble them into the requested note template.

[How To Use The Inputs]
- Use only the evidence units below as your source material.
- Group related evidence units before writing.
- Prefer one coherent sentence or bullet per local clinical event rather than copying raw units.
- Keep content in the correct section and avoid section leakage.
- Preserve negation, temporality, and plan intent.
- If an evidence unit is noisy or underspecified, phrase conservatively rather than speculating.
- Do not add content not recoverable from the evidence units.
- Do not mention information that is absent from the evidence units just because it is common in visit notes.
- Keep the note compact and evidence-linked.
- Output only the final note.
- Do not add any preamble, explanation, apology, or meta-commentary.
- Do not mention the evidence units, clustering process, or extraction process.
- Do not use markdown bold, code fences, or source citations.

[Rendering Goal]
{template_instruction}

[Section-Targeted Evidence Units]
{source_block}
""".strip()

MEDSUM_ENT_PROMPT_BASE = """
You are a clinical documentation assistant using a MEDSUM-ENT-inspired scaffold.

You are given the full transcript plus extracted clinically relevant concepts and status cues from the same 
patient-provider dialogue. Use the transcript as the source of truth, and use the extracted scaffold to improve 
content planning and section organization.

[Scaffold Categories]
The extracted scaffold is organized into these 6 planning categories:
1. Demographics and Social Determinants of Health: Patient's age, sex, occupation, and living situation.
2. Patient Intent: The primary reason the patient is seeking care.
3. Pertinent Positives: Symptoms or conditions confirmed as 'present'.
4. Pertinent Negatives: Symptoms or conditions confirmed as 'absent'.
5. Pertinent Unknowns: Symptoms or conditions marked as 'unknown' or unclear.
6. Medical History: Past medical conditions, surgeries, and family history.

[How To Use The Inputs]
- Use the transcript as the final grounding source.
- Use the evidence list as a planning scaffold to ensure all extracted entities are addressed.
- Use the 6 scaffold categories for content planning, but render the final note in the requested target template rather than copying the scaffold headings verbatim unless the template calls for them.
- Distinguish clearly between entities marked as 'present', 'absent', and 'unknown'.
- Place 'present' entities in Pertinent Positives, 'absent' in Pertinent Negatives, and 'unknown' in Pertinent Unknowns.
- Use Demographics and Social Determinants plus Patient Intent as supporting context when they are available.
- Merge redundant concepts into concise clinical sentences.
- If multiple scaffold entries refer to the same clinical concept, treat them as one resolved concept and avoid repetition.
- If the scaffold conflicts with the transcript, follow the transcript.
- Do not add content not recoverable from the transcript.
- Output only the final note without preamble or meta-commentary.

[Rendering Goal]
{template_instruction}

[Transcript]
{transcript}

[Evidence Scaffold]
{source_block}
""".strip()

CAP_PROMPT_BASE = """
You are a clinical documentation assistant using Clinical Atomic Propositions (CAPs) as a canonical content layer.

You are given the full transcript together with Clinical Atomic Propositions (CAPs) extracted from the same
encounter. Use the transcript as the source of truth and the CAPs as a canonical content plan.

[How To Use The Inputs]
- Use the transcript as the final grounding source.
- Use the CAPs as the main scaffold for content selection and organization.
- Interpret CAP types using FHIR-inspired semantics: Problem and ProblemHistory are condition-level state, ExamFinding and TestResult are observation-level evidence, MedicationStatement is current medication state, MedicationRequest is medication change intent, and Order/FollowUp/Counseling are management actions.
- Merge overlapping CAPs into coherent clinical statements.
- Preserve speaker, temporality, negation, uncertainty, laterality, numeric values, and plan intent.
- Prioritize clinically salient current problems, findings, and management decisions.
- Do not repeat every CAP mechanically; synthesize them into a note.
- If a CAP appears underspecified or slightly noisy, resolve conservatively using the transcript.
- Do not add content not recoverable from the transcript.
- Favor problem-oriented summarization over proposition-by-proposition rendering.
- Suppress low-value conversational details unless they materially affect assessment or plan.
- Do not repeat the same medication or plan detail across multiple sections.
- Output only the final note.
- Do not add any preamble, explanation, apology, or meta-commentary.
- Do not mention the transcript, CAPs, or proposition ids.
- Do not use markdown bold, code fences, or source citations.

[Primary Problem Anchoring - CRITICAL]
Before writing, identify one primary problem for this visit:
- either the chief complaint, or
- the main condition actively evaluated or managed in this encounter.

All included content must be anchored to that primary problem.
Include secondary details only when they directly affect:
- diagnosis of the primary problem,
- risk assessment, or
- management planning.

Strict exclusion rule:
- If removing a detail would not change diagnosis, clinical reasoning, or plan for the primary problem, exclude it.

[CAP Relevance Filter - CRITICAL]
Only include CAP-derived content that is directly related to:
- the presenting problem, or
- clinically relevant comorbidities that change diagnosis, risk, or management in this encounter.

Exclude:
- unrelated symptoms,
- incidental findings without impact on current care,
- weakly grounded or ambiguous signals.

[Section Integrity - HARD CONSTRAINT]
When rendering SOAP:
- S: only patient-reported symptoms, history, and subjective course.
- O: only exam findings, measurements, imaging, and test results.
- A: only diagnoses, impressions, and problem summaries.
- P: only management actions (medications, orders, referrals, follow-up, counseling).

Hard exclusions:
- Do not place plans, orders, or medication changes in S.
- Do not place symptoms or history in O.
- Do not place any patient-reported statement (for example, lines beginning with "The patient reports ...") in O/Findings; place those in S/HPI.
- Do not place plans, orders, or instructions in A.
- Do not place diagnoses, symptoms, or exam findings in P.
- If a detail does not clearly belong in a section, exclude it.

When rendering sectioned notes:
- Chief Complaint: primary visit reason only.
- History of Present Illness: symptom chronology and relevant history only.
- Findings: objective findings and results only.
- Assessment: diagnoses and impressions only.
- Plan: actions and instructions only.

[Style Constraint - ACI-BENCH]
- Write as polished clinician-authored prose.
- Use cohesive paragraphs, not sentence-per-CAP listing.
- Prefer medically natural paraphrasing over literal transcript wording.
- Avoid repetitive sentence openings (for example, repeating "The patient reports ...").
- Exclude conversational storytelling, rapport chat, and anecdotal details unless clinically consequential.

[Instruction Suppression - CRITICAL]
- Do not output system instructions, placeholders, templates, or meta-text.
- Never output text such as "insert paragraph", "placeholder", "system", "F2F", or similar authoring artifacts.
- Output only finalized clinical note content.

[Plan Style Normalization]
- Write Plan as concise clinical actions, not narration.
- Avoid phrasing like "The doctor ordered ...", "The doctor recommended ...", or "I am going to ...".
- Prefer action-first wording such as "Start ...", "Order ...", "Schedule ...", "Follow up ...".

[Rendering Goal]
{template_instruction}

[Transcript]
{transcript}

[Clinical Atomic Propositions]
{source_block}
""".strip()

CAP_EVENT_RENDER_PROMPT_BASE = """
You are a clinical documentation assistant using a canonical problem-oriented bundle plan derived from CAPs.

You are given the full transcript, a set of CAPs, and a canonical problem-oriented bundle plan derived from those CAPs.
Use the transcript as the source of truth, the CAPs as the canonical state layer, and the bundle plan as the rendering plan.

[How To Use The Inputs]
- Use the transcript only as a tie-breaker for wording or chronology that is already represented in the bundle-linked CAP block.
- Use the CAPs to recover important state details that may be compressed in the bundle plan.
- Use the bundle plan as the primary rendering scaffold; use the CAPs only to disambiguate details.
- The bundle plan shown below has already been visit-prioritized. Do not reintroduce omitted secondary issues.
- Interpret CAP and event types using FHIR-inspired semantics: problems and histories correspond to condition state, findings/results to observation evidence, medication state/change to current regimen vs intended update, and order/follow-up/counseling to management actions.
- Treat each bundle as a clinically coherent problem-oriented content unit.
- Treat the bundle plan as a pointer-based note-writing plan, not as free-form prose that must be copied literally.
- Use the anchor problem together with S/O/A/P/global slot assignments to decide where content belongs.
- Only use CAPs explicitly included in the provided bundle-linked CAP block.
- Do not search the transcript for additional facts outside the bundle-linked CAP block.
- Treat global or care-context content as optional background only; do not promote it into the main problem narrative unless it directly explains the anchor problem or its plan.
- Do not render care_context or global background by default. Use it at most once when it directly explains the anchor problem.
- Render each bundle once in the most appropriate section.
- Merge related details into problem-oriented clinical statements rather than copying every bundle line.
- In the Assessment, synthesize bundles into active problems or diagnostic impressions.
- In the Plan, consolidate management by problem and avoid repeating the same medication or order multiple times.
- If the same medication appears as both a current state and a planned change, express it as current regimen plus intended update, not as duplicate standalone lines.
- When event details are partially conflicting or overly granular, prefer a conservative higher-level state description over enumerating every variant.
- Preserve negation, temporality, uncertainty, laterality, numeric values, and medication changes.
- If the scaffold and transcript differ, only follow the transcript when the same fact is already recoverable from the bundle-linked CAP block.
- Do not add content not recoverable from the transcript.
- Do not introduce unrelated symptoms, review-of-systems items, lifestyle details, or background context that are not explicitly part of the bundle-linked CAP block.
- For each bundle, keep the note problem-oriented:
  S = symptoms/history directly relevant to the anchor problem
  O = objective findings/tests directly relevant to the anchor problem
  A = the underlying condition, diagnosis, or diagnostic concern
  P = only plans linked to that anchor problem
- If a bundle has no explicit A-cap, use the provided assessment fallback conservatively rather than inventing a new diagnosis.
- Keep the note clinically useful, concise, and problem-oriented.
- Do not render a CAP separately if its content is already covered by a bundle.
- Output only the final note.
- Do not add any preamble, explanation, apology, or meta-commentary.
- Do not mention the transcript, CAPs, or event plan.
- Do not use markdown bold, code fences, or source citations.

[Primary Problem Anchoring - CRITICAL]
Before writing, identify one primary problem for this visit:
- either the chief complaint, or
- the main condition actively evaluated or managed in this encounter.

All included content must be anchored to that primary problem.
Include secondary details only when they directly affect:
- diagnosis of the primary problem,
- risk assessment, or
- management planning.

Strict exclusion rule:
- If removing a detail would not change diagnosis, clinical reasoning, or plan for the primary problem, exclude it.

[CAP Relevance Filter - CRITICAL]
Only include CAP-derived content that is directly related to:
- the presenting problem, or
- clinically relevant comorbidities that change diagnosis, risk, or management in this encounter.

Exclude:
- unrelated symptoms,
- incidental findings without impact on current care,
- weakly grounded or ambiguous signals.

[Strict Source Constraint - CRITICAL]
You may only use information explicitly present in:
- the provided CAP block, and
- the provided bundle-linked CAP block / bundle plan.

You must not:
- introduce symptoms, conditions, history, or plans not explicitly represented in those inputs,
- recover additional facts from transcript-only context outside bundle-linked CAP coverage.

If uncertain whether a detail is supported, exclude it.
Violation of this rule is a critical error.

[Section Integrity - HARD CONSTRAINT]
When rendering SOAP:
- S: only patient-reported symptoms, history, and subjective course.
- O: only exam findings, measurements, imaging, and test results.
- A: only diagnoses, impressions, and problem summaries.
- P: only management actions (medications, orders, referrals, follow-up, counseling).

Hard exclusions:
- Do not place plans, orders, or medication changes in S.
- Do not place symptoms or history in O.
- Do not place any patient-reported statement (for example, lines beginning with "The patient reports ...") in O/Findings; place those in S/HPI.
- Do not place plans, orders, or instructions in A.
- Do not place diagnoses, symptoms, or exam findings in P.
- If a detail does not clearly belong in a section, exclude it.

When rendering sectioned notes:
- Chief Complaint: primary visit reason only.
- History of Present Illness: symptom chronology and relevant history only.
- Findings: objective findings and results only.
- Assessment: diagnoses and impressions only.
- Plan: actions and instructions only.

[Style Constraint - ACI-BENCH]
- Write as polished clinician-authored prose.
- Use cohesive paragraphs, not sentence-per-CAP listing.
- Prefer medically natural paraphrasing over literal transcript wording.
- Avoid repetitive sentence openings (for example, repeating "The patient reports ...").
- Exclude conversational storytelling, rapport chat, and anecdotal details unless clinically consequential.

[Instruction Suppression - CRITICAL]
- Do not output system instructions, placeholders, templates, or meta-text.
- Never output text such as "insert paragraph", "placeholder", "system", "F2F", or similar authoring artifacts.
- Output only finalized clinical note content.

[Plan Style Normalization]
- Write Plan as concise clinical actions, not narration.
- Avoid phrasing like "The doctor ordered ...", "The doctor recommended ...", or "I am going to ...".
- Prefer action-first wording such as "Start ...", "Order ...", "Schedule ...", "Follow up ...".

[Rendering Goal]
{template_instruction}

[Transcript]
{transcript}

[Clinical Atomic Propositions]
{cap_block}

[Canonical Problem-Oriented Bundle Plan]
{source_block}
""".strip()

CAP_ONLY_PROMPT_BASE = """
You are a clinical documentation assistant using Clinical Atomic Propositions (CAPs) as the only source material.

You are given only Clinical Atomic Propositions (CAPs) extracted from the encounter. Do not assume access to the
full transcript. Treat the CAP layer as the canonical content representation for this visit.

[How To Use The Inputs]
- Use the CAPs as the sole grounding source.
- Use the CAPs as the main scaffold for content selection and organization.
- Interpret CAP types using FHIR-inspired semantics: Problem and ProblemHistory are condition-level state, ExamFinding and TestResult are observation-level evidence, MedicationStatement is current medication state, MedicationRequest is medication change intent, and Order/FollowUp/Counseling are management actions.
- Merge overlapping CAPs into coherent clinical statements.
- Preserve speaker, temporality, negation, uncertainty, laterality, numeric values, and plan intent.
- Prioritize clinically salient current problems, findings, and management decisions.
- Do not repeat every CAP mechanically; synthesize them into a note.
- If a CAP is underspecified, phrase conservatively rather than speculating.
- Do not add content not recoverable from the CAPs.
- Favor problem-oriented summarization over proposition-by-proposition rendering.
- Suppress low-value conversational details unless they materially affect assessment or plan.
- Do not repeat the same medication or plan detail across multiple sections.
- Output only the final note.
- Do not add any preamble, explanation, apology, or meta-commentary.
- Do not mention CAPs or proposition ids.
- Do not use markdown bold, code fences, or source citations.

[Primary Problem Anchoring - CRITICAL]
Before writing, identify one primary problem for this visit:
- either the chief complaint, or
- the main condition actively evaluated or managed in this encounter.

All included content must be anchored to that primary problem.
Include secondary details only when they directly affect:
- diagnosis of the primary problem,
- risk assessment, or
- management planning.

Strict exclusion rule:
- If removing a detail would not change diagnosis, clinical reasoning, or plan for the primary problem, exclude it.

[CAP Relevance Filter - CRITICAL]
Only include CAP-derived content that is directly related to:
- the presenting problem, or
- clinically relevant comorbidities that change diagnosis, risk, or management in this encounter.

Exclude:
- unrelated symptoms,
- incidental findings without impact on current care,
- weakly grounded or ambiguous signals.

[Section Integrity - HARD CONSTRAINT]
When rendering SOAP:
- S: only patient-reported symptoms, history, and subjective course.
- O: only exam findings, measurements, imaging, and test results.
- A: only diagnoses, impressions, and problem summaries.
- P: only management actions (medications, orders, referrals, follow-up, counseling).

Hard exclusions:
- Do not place plans, orders, or medication changes in S.
- Do not place symptoms or history in O.
- Do not place any patient-reported statement (for example, lines beginning with "The patient reports ...") in O/Findings; place those in S/HPI.
- Do not place plans, orders, or instructions in A.
- Do not place diagnoses, symptoms, or exam findings in P.
- If a detail does not clearly belong in a section, exclude it.

When rendering sectioned notes:
- Chief Complaint: primary visit reason only.
- History of Present Illness: symptom chronology and relevant history only.
- Findings: objective findings and results only.
- Assessment: diagnoses and impressions only.
- Plan: actions and instructions only.

[Style Constraint - ACI-BENCH]
- Write as polished clinician-authored prose.
- Use cohesive paragraphs, not sentence-per-CAP listing.
- Prefer medically natural paraphrasing over literal CAP wording.
- Avoid repetitive sentence openings (for example, repeating "The patient reports ...").
- Exclude conversational storytelling, rapport chat, and anecdotal details unless clinically consequential.

[Instruction Suppression - CRITICAL]
- Do not output system instructions, placeholders, templates, or meta-text.
- Never output text such as "insert paragraph", "placeholder", "system", "F2F", or similar authoring artifacts.
- Output only finalized clinical note content.

[Plan Style Normalization]
- Write Plan as concise clinical actions, not narration.
- Avoid phrasing like "The doctor ordered ...", "The doctor recommended ...", or "I am going to ...".
- Prefer action-first wording such as "Start ...", "Order ...", "Schedule ...", "Follow up ...".

[Rendering Goal]
{template_instruction}

[Clinical Atomic Propositions]
{source_block}
""".strip()

CAP_EVENT_ONLY_RENDER_PROMPT_BASE = """
You are a clinical documentation assistant using a canonical problem-oriented bundle plan derived from CAPs as the only source material.

You are given a set of CAPs and a canonical problem-oriented bundle plan derived from those CAPs. Do not assume access to the full
transcript. Treat the bundle plan as the primary rendering scaffold and the CAPs as the supporting canonical state layer.

[How To Use The Inputs]
- Use the bundle plan as the primary rendering scaffold.
- Use the CAPs only to disambiguate details that are compressed in the bundle plan.
- The bundle plan shown below has already been visit-prioritized. Do not reintroduce omitted secondary issues.
- Interpret CAP and event types using FHIR-inspired semantics: problems and histories correspond to condition state, findings/results to observation evidence, medication state/change to current regimen vs intended update, and order/follow-up/counseling to management actions.
- Treat each bundle as a clinically coherent problem-oriented content unit.
- Treat the bundle plan as a pointer-based note-writing plan, not as free-form prose that must be copied literally.
- Use the anchor problem together with S/O/A/P/global slot assignments to decide where content belongs.
- Only use CAPs explicitly included in the provided bundle-linked CAP block.
- Do not introduce new facts from outside the bundle-linked CAP block.
- Treat global or care-context content as optional background only; do not promote it into the main problem narrative unless it directly explains the anchor problem or its plan.
- Do not render care_context or global background by default. Use it at most once when it directly explains the anchor problem.
- Render each bundle once in the most appropriate section.
- Merge related details into problem-oriented clinical statements rather than copying every bundle line.
- In the Assessment, synthesize bundles into active problems or diagnostic impressions.
- In the Plan, consolidate management by problem and avoid repeating the same medication or order multiple times.
- If the same medication appears as both a current state and a planned change, express it as current regimen plus intended update, not as duplicate standalone lines.
- When event details are partially conflicting or overly granular, prefer a conservative higher-level state description over enumerating every variant.
- Preserve negation, temporality, uncertainty, laterality, numeric values, and medication changes.
- Do not add content not recoverable from the CAPs or event plan.
- Do not include unrelated symptoms, generic lifestyle details, or background context that are not explicitly part of the bundle-linked CAP block.
- For each bundle, keep the note problem-oriented:
  S = symptoms/history directly relevant to the anchor problem
  O = objective findings/tests directly relevant to the anchor problem
  A = the underlying condition, diagnosis, or diagnostic concern
  P = only plans linked to that anchor problem
- If a bundle has no explicit A-cap, use the provided assessment fallback conservatively rather than inventing a new diagnosis.
- Keep the note clinically useful, concise, and problem-oriented.
- Do not render a CAP separately if its content is already covered by a bundle.
- Output only the final note.
- Do not add any preamble, explanation, apology, or meta-commentary.
- Do not mention CAPs or the bundle plan.
- Do not use markdown bold, code fences, or source citations.

[Primary Problem Anchoring - CRITICAL]
Before writing, identify one primary problem for this visit:
- either the chief complaint, or
- the main condition actively evaluated or managed in this encounter.

All included content must be anchored to that primary problem.
Include secondary details only when they directly affect:
- diagnosis of the primary problem,
- risk assessment, or
- management planning.

Strict exclusion rule:
- If removing a detail would not change diagnosis, clinical reasoning, or plan for the primary problem, exclude it.

[CAP Relevance Filter - CRITICAL]
Only include CAP-derived content that is directly related to:
- the presenting problem, or
- clinically relevant comorbidities that change diagnosis, risk, or management in this encounter.

Exclude:
- unrelated symptoms,
- incidental findings without impact on current care,
- weakly grounded or ambiguous signals.

[Strict Source Constraint - CRITICAL]
You may only use information explicitly present in:
- the provided CAP block, and
- the provided bundle-linked CAP block / bundle plan.

You must not:
- introduce symptoms, conditions, history, or plans not explicitly represented in those inputs.

If uncertain whether a detail is supported, exclude it.
Violation of this rule is a critical error.

[Section Integrity - HARD CONSTRAINT]
When rendering SOAP:
- S: only patient-reported symptoms, history, and subjective course.
- O: only exam findings, measurements, imaging, and test results.
- A: only diagnoses, impressions, and problem summaries.
- P: only management actions (medications, orders, referrals, follow-up, counseling).

Hard exclusions:
- Do not place plans, orders, or medication changes in S.
- Do not place symptoms or history in O.
- Do not place any patient-reported statement (for example, lines beginning with "The patient reports ...") in O/Findings; place those in S/HPI.
- Do not place plans, orders, or instructions in A.
- Do not place diagnoses, symptoms, or exam findings in P.
- If a detail does not clearly belong in a section, exclude it.

When rendering sectioned notes:
- Chief Complaint: primary visit reason only.
- History of Present Illness: symptom chronology and relevant history only.
- Findings: objective findings and results only.
- Assessment: diagnoses and impressions only.
- Plan: actions and instructions only.

[Style Constraint - ACI-BENCH]
- Write as polished clinician-authored prose.
- Use cohesive paragraphs, not sentence-per-CAP listing.
- Prefer medically natural paraphrasing over literal CAP wording.
- Avoid repetitive sentence openings (for example, repeating "The patient reports ...").
- Exclude conversational storytelling, rapport chat, and anecdotal details unless clinically consequential.

[Instruction Suppression - CRITICAL]
- Do not output system instructions, placeholders, templates, or meta-text.
- Never output text such as "insert paragraph", "placeholder", "system", "F2F", or similar authoring artifacts.
- Output only finalized clinical note content.

[Plan Style Normalization]
- Write Plan as concise clinical actions, not narration.
- Avoid phrasing like "The doctor ordered ...", "The doctor recommended ...", or "I am going to ...".
- Prefer action-first wording such as "Start ...", "Order ...", "Schedule ...", "Follow up ...".

[Rendering Goal]
{template_instruction}

[Clinical Atomic Propositions]
{cap_block}

[Canonical Problem-Oriented Bundle Plan]
{source_block}
""".strip()

FACT_SOURCE_BLOCK = """
{fact_lines}
""".strip()

CAP_SOURCE_BLOCK = """
{cap_lines}
""".strip()

EVENT_SOURCE_BLOCK = """
{event_lines}
""".strip()

TEMPLATE_INSTRUCTIONS = {
    "soap": """
Generate content for a SOAP note (S/O/A/P) in one pass.
The exact output JSON schema is provided later; follow that schema exactly.

Requirements:
- First, identify the single primary problem for this visit and anchor the note to it.
- Include secondary details only if they directly change diagnosis, risk assessment, or management for the primary problem.
- Exclude unrelated symptoms, incidental low-impact details, and weakly grounded ambiguous signals.
- Put patient-reported history, symptoms, and subjective interval changes under S.
- Put patient-reported current medications, adherence, home monitoring logs, and symptom course under S unless they are explicit new orders or medication changes.
- Put exam findings, observed behaviors, measurements, and test results under O.
- Put clinician impressions, differential, or active problems under A.
- Put medications, orders, follow-up, counseling, and instructions under P.
- Objective (O) must contain only physical exam findings, imaging results, lab results, and measured values.
- Do not place current medication state, patient-reported symptoms, or ROS content under P.
- Do not place referrals, medication changes, orders, return precautions, or follow-up scheduling under A.
- Do not place diagnoses or active problems under P unless they are explicitly embedded inside a plan sentence.
- Do not place exam findings, vital signs, imaging, lab results, or measured observations under S.
- If a statement is based on patient report, it must be placed in S, not O.
- Do not force content into a section if unsupported.
- If a detail does not clearly belong in one section, exclude it rather than forcing placement.
- Exclude low-impact background details that do not change diagnosis, reasoning, or plan.
- Do not output instructions, placeholders, or meta-text (for example, "insert paragraph", "placeholder", "system", "F2F").
- Use concise, complete clinical prose.
- Do not write transcript-style sentence fragments.
- Prefer consistent third-person clinical style.
- When the patient is the subject, prefer wording that begins with "The patient ..." rather than "you," "we," or alternating pronouns.
- Avoid repeating "The patient ..." at the beginning of every line; vary openings naturally.
- Avoid explicit narrator phrases like "The doctor ..."; prefer action-oriented plan wording (for example, "Order ...", "Recommend ...", "Arrange follow-up ...").
- In Plan, use concise action-first phrasing (for example, "Start ...", "Order ...", "Schedule ...", "Follow up ...").
- Do not narrate Plan as transcript-style dialogue.
- Do not begin any line with a comma, conjunction fragment, or dangling continuation.
- Do not use markdown bullets, numbered lists, or bold formatting.
- Do not repeat the same fact in multiple sections.
- Keep Assessment problem-oriented and keep Plan action-oriented.
""".strip(),
    "sectioned": """
Generate content for a sectioned clinical note in one pass.
The exact output JSON schema is provided later; follow that schema exactly.

Requirements:
- Keep the note concise but clinically complete, in polished ACI-BENCH-like style.
- First, identify the single primary problem for this visit and anchor the note to it.
- Include secondary details only if they directly change diagnosis, risk assessment, or management for the primary problem.
- Exclude unrelated symptoms, incidental low-impact details, and weakly grounded ambiguous signals.
- Put the main visit reason under Chief Complaint.
- Put patient-reported symptoms, chronology, relevant history, and current medication state under History of Present Illness.
- Put exam findings, vitals, imaging, lab results, and observed findings under Findings.
- Findings must contain only physical exam findings, imaging results, lab results, and measured values.
- Put diagnoses, impressions, and active problems under Assessment.
- Put medication changes, orders, referrals, follow-up, counseling, and instructions under Plan.
- Do not place referrals or treatment recommendations under Assessment.
- Do not place diagnoses or active problems under Plan unless they are part of a management sentence.
- Do not place any patient-reported statement (for example, lines beginning with "The patient reports ...") in Findings; place those in History of Present Illness.
- Use concise complete clinical prose; do not emit sentence fragments.
- Prefer consistent third-person clinical style.
- When the patient is the subject, prefer wording that begins with "The patient ..." rather than "you," "we," or alternating pronouns.
- Avoid repeating "The patient ..." at the beginning of every line; vary openings naturally.
- Avoid explicit narrator phrases like "The doctor ..."; prefer action-oriented plan wording (for example, "Order ...", "Recommend ...", "Arrange follow-up ...").
- Do not begin any line with a comma, conjunction fragment, or dangling continuation.
- Preserve medical meaning without adding unsupported detail.
- If a detail does not clearly belong in one section, exclude it rather than forcing placement.
- Exclude low-impact background details that do not change diagnosis, reasoning, or plan.
- Exclude social chatter or anecdotal conversation unless clinically consequential.
- Do not output instructions, placeholders, or meta-text (for example, "insert paragraph", "placeholder", "system", "F2F").
- In Plan, use concise action-first phrasing (for example, "Start ...", "Order ...", "Schedule ...", "Follow up ...").
- Do not narrate Plan as transcript-style dialogue.
- Do not add markdown bold or a prefatory sentence.
""".strip(),
    "brief": """
Write the output as a concise clinician-facing brief note.

Format:
- Use short bullet points.
- Standard medical abbreviations are allowed when common and unambiguous.
- Organize into these labels if supported: CC, HPI, Findings, Assessment, Plan.
- Keep it terse without dropping clinically important facts.
- Do not add a prefatory sentence or explanation.
""".strip(),
}

SUMMARY_TO_CAP_PROMPT = """
You are an expert clinical information extraction system.

Convert the clinical summary note into Clinical Atomic Propositions (CAPs).

[Definition of CAP]
A CAP is:
- one clinically meaningful assertion,
- standalone and self-contained,
- verifiable from the summary text,
- explicit about speaker or subject when possible,
- faithful to negation, temporality, laterality, numeric values, severity, and plan intent.

[Validity Rules - CRITICAL]
Do NOT include invalid or non-verifiable units as CAPs.
Instead, place them in `filtered_nonverifiable_units`.

Invalidity types:
- imperative without a factual claim
- interrogative
- incomplete fragment
- vague statement with unresolved subject or clinical referent

[Contextual Enrichment Rules - CRITICAL]
To keep propositions reusable for downstream event clustering:
1. Resolve pronouns and deictic expressions whenever the summary makes the referent clear.
2. Make the speaker or clinical subject explicit in `proposition_text`.
3. Preserve persistence, progression, improvement, worsening, and stability cues when explicitly stated.
4. Preserve explicit treatment plans, referrals, follow-up timing, counseling, return precautions, and patient instructions.
5. Do not convert speculation or possibility into affirmed diagnosis or plan.

[Schema]
Use CAP type semantics consistent with the FHIR-inspired source schema when recoverable:
- ChiefComplaint
- Problem
- ProblemHistory
- ExamFinding
- TestResult
- MedicationStatement
- MedicationRequest
- Order
- FollowUp
- Counseling

Valid category values:
- Demographic
- ChiefComplaint
- History
- Allergy
- Symptom
- Finding
- Diagnosis
- MedicationPlan
- TestPlan
- FollowUpPlan
- UncertainOrNoise

Valid speaker values:
- patient
- clinician
- test_result
- unknown

Valid status values:
- affirmed
- negated
- uncertain

Valid temporality values:
- past
- current
- future
- relative

Valid predicate values:
- reports
- denies
- has
- shows
- diagnoses
- suspects
- prescribes
- orders
- recommends
- instructs
- plans
- notes

Valid claim_type_tags values:
- temporal
- quantitative
- causal
- medication
- plan
- diagnostic
- symptom
- exam
- safety_relevant

[Extraction Rules]
1. One CAP = exactly one clinical assertion.
2. Preserve the summary meaning; do not repair unsupported content.
3. Make speaker explicit when recoverable.
4. Preserve negation and uncertainty explicitly.
5. Do not output section headers as CAPs.
6. Do not add facts missing from the summary.
7. Return at most {max_props} atomic propositions.
8. If a sentence contains multiple independent claims, split them.
9. Do NOT aggressively split routine coordinated checklist items when they function as one single summary event.
10. Return compact flat JSON only. Omit null fields entirely.

[Output Format]
Return valid JSON only:
{{
  "atomic_propositions": [
    {{
      "prop_id": "P1",
      "category": "Symptom",
      "speaker": "patient",
      "predicate": "reports",
      "status": "affirmed",
      "temporality": "current",
      "claim_type_tags": ["symptom"],
      "proposition_text": "The patient reports right knee pain."
    }}
  ],
  "filtered_nonverifiable_units": []
}}

<Clinical Summary>
{summary_text}
</Clinical Summary>
""".strip()

CAP_TO_EVENT_PROMPT = """
You are an expert clinical content planner.

Convert the Clinical Atomic Propositions (CAPs) into a compact dialogue-aware problem-oriented care bundle plan.

[Goal]
- Group related CAPs into clinically coherent problem-oriented care bundles.
- Separate content planning from note rendering.
- Preserve negation, temporality, uncertainty, medication changes, and follow-up intent.
- Make the resulting structure useful for note rendering, problem-linked action grouping, and auditability.

[Care Bundle Framing]
- Each event should behave like a problem-oriented care bundle rather than a raw dialogue cluster.
- The bundle should identify an anchor problem and organize linked evidence, current state, planned actions, and care context.
- This abstraction is inspired by FHIR workflow semantics such as Condition, Observation, MedicationStatement, MedicationRequest, ServiceRequest, and CarePlan.
- Do not attempt strict FHIR resource extraction. Produce a dialogue-grounded planning abstraction.

[Event Types]
- ChiefComplaint
- SymptomCourse
- PMH
- MedicationState
- MedicationChange
- ExamFinding
- TestResult
- DiagnosisImpression
- PlanOrder
- FollowUp
- PatientInstruction

[Status Values]
- affirmed
- negated
- uncertain

[Temporality Values]
- past
- current
- future
- relative

[Speaker Source Values]
- patient
- clinician
- test_result
- mixed

[Clinical Priority Values]
- high
- medium
- low

[Rules]
1. Construct one-stage problem-oriented note bundles rather than free-form event summaries.
2. Each bundle must be centered on a single active problem or diagnostic concern.
3. Anchor eligibility is narrow:
   - allowed: ChiefComplaint, Problem
   - optionally allowed: Diagnosis, Impression
   - disallowed: ProblemHistory, MedicationStatement, ExamFinding, TestResult, Order, FollowUp, Counseling, Allergy
4. Use pointer-style assignments whenever possible:
   - anchor_problem_cap_id
   - S_cap_ids
   - O_cap_ids
   - A_cap_ids
   - P_cap_ids
   - global_cap_ids
5. Slot rules:
   - S_caps: patient-reported symptoms, ROS, symptom course, patient-reported medication use
   - O_caps: exam findings, test results, quantified vitals, clinician-observed abnormalities
   - A_caps: diagnostic impressions, problem summaries, clinically interpreted state
   - P_caps: medication changes, orders, referrals, follow-up, counseling
   - global_caps: PMH, allergies, family history, social history, stable background medications, screening history
6. Each CAP may attach to at most one anchor bundle.
7. If no anchor fit is clear, send the CAP to global_cap_ids rather than inventing a new anchor.
8. Do not create standalone anchors for labs, orders, referrals, medication changes, allergies, or background history.
9. Do not over-segment sibling CAPs that describe the same problem.
10. Return at most {max_events} bundles.

[Output Format]
Return valid JSON only:
{{
  "events": [
    {{
      "event_id": "E1",
      "event_type": "ProblemOrientedNoteBundle",
      "status": "affirmed",
      "temporality": "current",
      "speaker_source": "clinician",
      "clinical_priority": "high",
      "title": "Type 2 diabetes",
      "event_summary": "Problem-oriented bundle for type 2 diabetes.",
      "supporting_cap_ids": ["P2", "P3", "P4"],
      "anchor_problem_cap_id": "P2",
      "anchor_problem": "Type 2 diabetes",
      "S_cap_ids": ["P3"],
      "O_cap_ids": [],
      "A_cap_ids": ["P2"],
      "P_cap_ids": ["P4"],
      "global_cap_ids": ["P8"],
      "supporting_evidence": ["Home glucose readings remain elevated."],
      "current_state": ["The patient is taking glimepiride 2 mg twice daily."],
      "planned_actions": ["Increase glimepiride to 4 mg twice daily."],
      "care_context": ["Continue diet and exercise counseling."],
      "key_slots": {{
        "bundle_design": "problem_oriented_note_bundle"
      }},
      "render_sections": ["S", "O", "A", "P", "History of Present Illness", "Findings", "Assessment", "Plan"]
    }}
  ]
}}

[Clinical Atomic Propositions]
{cap_lines}
""".strip()

CAP_EXTRACTION_GUIDANCE = """
[CAP Design Principles]
- CAPs must be standalone, dialogue-grounded, and clinically reusable.
- A good CAP keeps speaker, assertion status, temporality, and clinically meaningful modifiers intact.
- CAPs should prefer central clinical meaning over conversational filler.
- Low-value scaffolding, questions, and incomplete references should be filtered rather than forced into CAPs.
""".strip()

EVENT_CLUSTERING_GUIDANCE = """
[Problem-Oriented Bundle Principles]
- The bundle is the unit of recoverability, omission control, faithful rendering, and problem-linked action grouping.
- Treat the anchor as a clinical decision problem that would reasonably appear on a problem-oriented assessment or problem list.
- Do not write free-form event summaries first and then attach evidence later.
- Prefer pointer-only note-writing bundles with one anchor problem and SOAP-aligned slot assignments.
- Do not over-count multiple CAPs from the same local symptom or plan bundle.
- Merge parent/child support details into one problem-oriented bundle when they represent the same underlying clinical state.
- Prefer one bundle per active problem or diagnostic concern.
- Think in terms of note-writing bundles: anchor problem, S, O, A, P, and global context.
- When in doubt, attach findings, tests, and plans to the nearest clinically relevant problem bundle instead of emitting them as standalone bundles.
- Prefer the underlying clinical condition over individual symptom mentions when multiple related symptoms, findings, or plans describe the same evolving issue.
- Prefer organizing problems at the level of an underlying condition rather than individual symptom mentions.
- Synthetic examples:
  - multiple symptoms sharing a common anatomical region and temporal pattern -> group into one problem
  - symptoms distributed across turns but describing the same evolving issue -> consolidate into one problem
  - causally linked findings (symptom -> exam -> plan) -> group under one problem-oriented bundle
  - risk factors or historical context -> attach as related factors, not as primary problems
- Avoid:
  - splitting closely related symptoms into multiple problems
  - creating anchors from minor or clinically peripheral mentions
  - using general context or lifestyle factors as primary problems
  - mixing historical and current states in the same primary bundle
""".strip()

GPT_F1_EXTRACTION_PROMPT = """
You are an expert clinical content annotator.

Extract clinically salient checklist items from the note for GPT-F1 style evaluation.

[Categories]
- `active_problems_symptoms`: current active diseases, symptoms, abnormal findings, or clinically active complaints
- `negated_findings`: explicitly denied symptoms or ruled-out findings
- `uncertain_findings`: suspected, possible, unclear, or uncertain diagnoses/findings
- `medication_treatment`: medication states, medication changes, ongoing treatments, or treatment recommendations
- `plan_followup`: orders, labs, referrals, follow-up timing, monitoring, counseling, patient instructions, and return precautions

[Rules]
- Use only information explicitly present in the note.
- Extract short standalone items, not full paragraphs.
- Deduplicate near-duplicates.
- Keep each item in the single best category.
- If a category is not present, return an empty list for it.
- Do not invent items.

[Output Format]
Return valid JSON only:
{{
  "categories": {{
    "active_problems_symptoms": [],
    "negated_findings": [],
    "uncertain_findings": [],
    "medication_treatment": [],
    "plan_followup": []
  }}
}}

[Clinical Note]
{note_text}
""".strip()

NAIR_CONCEPT_EXTRACTION_PROMPT = """
Given the following snippet of a medical dialogue summary, extract the medical
concepts (symptoms, diseases, conditions, allergies, lab tests, etc.) present.

The heading of the section from which the summary was extracted will also be
provided.

--- Example 1 ---
Pertinent Negatives: Patient reports no fever, no chest pain, shortness of breath,
and cough. Patient also reports having no trouble with swallowing.

Medical Concepts: ["fever", "chest pain", "shortness of breath", "cough", "swallowing difficulty"]
--- Example 1 ---

--- Example 2 ---
Pertinent Positives: Patient ongoing abdominal pain for the past 5 days, nausea,
and some diarrhea. Patient had colonoscopy done in May 2021.

Medical Concepts: ["abdominal pain", "nausea", "diarrhea", "colonoscopy"]
--- Example 2 ---

--- Example 3 ---
Pertinent Unknowns: Patient is unsure about penicillin allergy and family history of diabetes.

Medical Concepts: ["penicillin allergy", "family history of diabetes"]
--- Example 3 ---

--- Example 4 ---
Medical History: Patient reports some seizure disorder in the past, and had last MRI on DATE_1.

Medical Concepts: ["seizure disorder", "MRI"]
--- Example 4 ---

Here is the example to extract medical concepts from:

{section_heading}: {section_value}

Return valid JSON only:
{{
  "medical_concepts": []
}}
""".strip()

NAIR_CONCEPT_VERIFICATION_PROMPT = """
Given a snippet (snippet) from a medical dialogue summary and a corresponding list
(list_a) of medical concepts extracted from that snippet, evaluate what medical
concepts from a separate list (list_b) can be found in either list_a or snippet.

Note that on some occasions a medical concept from list_b may not be found in list_a,
but can be appropriate to be present given the snippet. This could include
rephrasings of medical concepts that are clinically equivalent (for example:
COVID and COVID-19).

--- Example ---
snippet: Patient reports cough, fever, and no chest pain. COVID-19 test ordered.
list_a: ["cough", "fever", "chest pain", "COVID-19 test"]
list_b: ["cough", "fever", "COVID", "wheezing"]

found_b: ["cough", "fever", "COVID"]
not_found_b: ["wheezing"]
--- Example ---

Here is the snippet and list_a. Evaluate the medical concepts in list_b as above.

snippet: {snippet}
list_a: {list_a}
list_b: {list_b}

Return valid JSON only:
{{
  "found_b": [],
  "not_found_b": []
}}
""".strip()

SEMANTIC_CAP_AUDIT_PROMPT = """
You are an expert clinical faithfulness auditor.

Evaluate whether a generated clinical note preserves the meaning of the source Clinical Atomic Propositions (CAPs).

[Goal]
- Judge semantic recall of source CAPs in the generated note.
- Judge semantic precision of generated summary CAPs relative to the source CAP set.
- Use semantic support, not exact lexical overlap.

[Judgment Labels For Source CAP Recall]
- supported: clearly preserved in the note
- partial: partly preserved but clinically incomplete or less specific
- missing: clinically meaningful CAP not recoverable from the note
- contradicted: note expresses the opposite meaning
- low_value: trivial or low-value CAP that should not count strongly against recall

[Judgment Labels For Summary CAP Precision]
- supported: grounded by the source CAP set
- partial: mostly grounded but overspecified or slightly generalized
- unsupported: not grounded by the source CAP set
- contradicted: conflicts with the source CAP set
- low_value: stylistic or low-value mismatch that should not count as a major hallucination

[Rules]
- Be conservative and semantic.
- Do not penalize paraphrases.
- Preserve negation, uncertainty, temporality, laterality, numeric values, and action intent.
- Use low_value only for clearly trivial or non-central mismatches.
- Focus on clinically meaningful content.

[Output Format]
Return valid JSON only:
{{
  "source_cap_recall": [
    {{
      "cap_id": "P1",
      "judgment": "supported",
      "reason": "..."
    }}
  ],
  "summary_cap_precision": [
    {{
      "cap_id": "P1",
      "judgment": "supported",
      "reason": "..."
    }}
  ]
}}

[Source CAPs]
{source_caps}

[Generated Summary CAPs]
{summary_caps}

[Generated Note]
{summary_text}
""".strip()

SEMANTIC_CAP_RECALL_CHUNK_PROMPT = """
You are an expert clinical faithfulness auditor.

Evaluate whether each source Clinical Atomic Proposition (CAP) is semantically preserved in the generated clinical note.

[Judgment Labels]
- supported: clearly preserved in the note
- partial: partly preserved but clinically incomplete or less specific
- missing: clinically meaningful CAP not recoverable from the note
- contradicted: note expresses the opposite meaning
- low_value: trivial CAP that should not count strongly against recall

[Rules]
- Be conservative and semantic.
- Do not penalize paraphrases.
- Preserve negation, uncertainty, temporality, laterality, numeric values, and action intent.
- Use low_value only for clearly trivial or non-central CAPs.
- Judge only the listed source CAPs.
- Return only the requested JSON fields.
- Do not include explanations outside JSON.

[Output Format]
Return valid JSON only:
{{
  "source_cap_recall": [
    {{
      "cap_id": "P1",
      "judgment": "supported"
    }}
  ]
}}

[Source CAPs To Judge]
{source_caps}

[Generated Note]
{summary_text}
""".strip()

SEMANTIC_CAP_PRECISION_CHUNK_PROMPT = """
You are an expert clinical faithfulness auditor.

Evaluate whether each summary Clinical Atomic Proposition (CAP) is semantically grounded by the source CAP set.

[Judgment Labels]
- supported: grounded by the source CAP set
- partial: mostly grounded but overspecified or slightly generalized
- unsupported: not grounded by the source CAP set
- contradicted: conflicts with the source CAP set
- low_value: trivial or stylistic mismatch that should not count as a major hallucination

[Rules]
- Be conservative and semantic.
- Do not penalize paraphrases.
- Preserve negation, uncertainty, temporality, laterality, numeric values, and action intent.
- Judge only the listed summary CAPs.
- Return only the requested JSON fields.
- Do not include explanations outside JSON.

[Output Format]
Return valid JSON only:
{{
  "summary_cap_precision": [
    {{
      "cap_id": "P1",
      "judgment": "supported"
    }}
  ]
}}

[Source CAPs]
{source_caps}

[Summary CAPs To Judge]
{summary_caps}
""".strip()

ASGARI_SAFETY_PROMPT = """
You are an expert clinical safety auditor.

Your task is to evaluate the clinical faithfulness of a generated medical summary given the original doctor-patient conversation transcript.

[Asgari-Inspired Error Taxonomy]
A hallucination is any clinically meaningful statement in the summary that is unsupported, contradicted, or dangerously distorted relative to the transcript.
Hallucination types:
- fabrication: entirely unsupported new symptom, diagnosis, medication, result, or plan
- negation: polarity or assertion reversal such as omitted denial or invented denial
- contextual: topics or details are mixed into the wrong context, such as wrong source, wrong timeline, wrong topic linkage, or a context-conflated summary sentence
- causality: unsupported causal, explanatory, or associative relationship between facts that were mentioned independently

An omission is clinically important information present in the transcript but missing from the summary.
Omission types:
- current_issues: clinically relevant details of the current presentation, including symptoms, symptom course, associated positives, and important negatives
- pmfs_issues: past medical history, medications including allergies, family history, and social history including smoking, drinking, and drug use
- information_plan: explanations, communication, counseling, management plan, safety-netting, follow-up, referrals, investigations, and return instructions

[Supplementary Guidance]
- A fabricated urgent referral that never appeared in the transcript is `fabrication`.
- Turning a present symptom into a denial, or erasing an important reported symptom with "no other significant symptoms", is `negation`.
- Mixing workplace exposure with home exposure or mixing past history with the current problem is `contextual`.
- Adding "stress-related" or another explanatory cause that was never explicitly stated is `causality`.
- Missing palpitations or blood in sputum from the current presentation is `current_issues`.
- Missing "no allergies" or an important smoking history is `pmfs_issues`.
- Missing advice such as "if the pregnancy test is positive, call back immediately" is `information_plan`.

[Severity]
- major: if left uncorrected, could affect diagnosis, treatment, management, or clinical decision-making
- minor: unlikely to affect clinical decision-making

[Instructions]
- Compare the transcript and summary carefully.
- Identify hallucinations and clinically important omissions.
- Do not penalize paraphrases or harmless wording changes.
- Focus on chief complaint, current symptom course, important negatives, PMFS details, medications and changes, allergies, orders, follow-up, communication, and safety-netting.
- Use the smallest clinically meaningful error unit.
- For omissions, always assign the single best omission type.
- For hallucinations, always assign the single best hallucination type.
- Do not create extra omissions for stylistic compression alone; only clinically relevant omissions should be listed.
- Assess errors relative to what matters clinically, not just lexical mismatch.

[Output Format]
Return valid JSON only:
{{
  "hallucinations": [
    {{
      "text": "...",
      "type": "fabrication",
      "severity": "major",
      "reason": "...",
      "evidence": "relevant transcript span or not found"
    }}
  ],
  "omissions": [
    {{
      "text": "...",
      "type": "information_plan",
      "severity": "major",
      "reason": "...",
      "evidence": "relevant transcript span"
    }}
  ]
}}

[Transcript]
{transcript}

[Generated Summary]
{summary}
""".strip()

ASGARI_HALLUCINATION_PROMPT = """
You are an expert clinical safety auditor.

Evaluate hallucinations only in the generated medical summary given the original doctor-patient conversation transcript.

[Asgari-Inspired Hallucination Taxonomy]
A hallucination is any clinically meaningful statement in the summary that is unsupported, contradicted, or dangerously distorted relative to the transcript.
Types:
- fabrication: entirely unsupported new symptom, diagnosis, medication, result, or plan
- negation: polarity or assertion reversal such as omitted denial or invented denial
- contextual: wrong source, wrong timeline, wrong topic linkage, or a context-conflated statement
- causality: unsupported causal, explanatory, or associative relationship

[Severity]
- major: if left uncorrected, could affect diagnosis, treatment, management, or clinical decision-making
- minor: unlikely to affect clinical decision-making

[Rules]
- Compare the transcript and summary carefully.
- Identify only clinically meaningful hallucinations.
- Do not penalize paraphrases or harmless wording changes.
- Use the smallest clinically meaningful error unit.
- Always assign the single best hallucination type.

[Output Format]
Return valid JSON only:
{{
  "hallucinations": [
    {{
      "text": "...",
      "type": "fabrication",
      "severity": "major",
      "reason": "...",
      "evidence": "relevant transcript span or not found"
    }}
  ]
}}

[Transcript]
{transcript}

[Generated Summary]
{summary}
""".strip()

ASGARI_OMISSION_PROMPT = """
You are an expert clinical safety auditor.

Evaluate omissions only in the generated medical summary given the original doctor-patient conversation transcript.

[Asgari-Inspired Omission Taxonomy]
An omission is clinically important information present in the transcript but missing from the summary.
Types:
- current_issues: clinically relevant details of the current presentation, including symptoms, symptom course, associated positives, and important negatives
- pmfs_issues: past medical history, medications including allergies, family history, and social history including smoking, drinking, and drug use
- information_plan: explanations, communication, counseling, management plan, safety-netting, follow-up, referrals, investigations, and return instructions

[Severity]
- major: if left uncorrected, could affect diagnosis, treatment, management, or clinical decision-making
- minor: unlikely to affect clinical decision-making

[Rules]
- Compare the transcript and summary carefully.
- Identify only clinically important omissions.
- Do not create extra omissions for stylistic compression alone.
- Use the smallest clinically meaningful omitted unit.
- Always assign the single best omission type.

[Output Format]
Return valid JSON only:
{{
  "omissions": [
    {{
      "text": "...",
      "type": "information_plan",
      "severity": "major",
      "reason": "...",
      "evidence": "relevant transcript span"
    }}
  ]
}}

[Transcript]
{transcript}

[Generated Summary]
{summary}
""".strip()

PDSQI_TRANSCRIPT_PROMPT = """
You are a clinical summarization quality expert specializing in evaluating generated visit notes from doctor-patient conversations.

Read the following visit transcript as the source of truth.

<VISIT_TRANSCRIPT>
{transcript}
</VISIT_TRANSCRIPT>

Read the following generated visit note.

<GENERATED_VISIT_NOTE>
{generated_note}
</GENERATED_VISIT_NOTE>

Grade the generated note for a clinician with specialty {target_specialty}.

[Rubric]
- accurate: factual accuracy relative to the transcript
- thorough: inclusion of clinically important content
- useful: relevance to downstream clinician use
- organized: logical structure and grouping
- comprehensible: clinical clarity
- succinct: concise without harmful redundancy
- synthesized: integrates the visit rather than copying dialogue fragments
- voice_summ: stigmatizing language in the note
- voice_note: stigmatizing language already present in the transcript

[Scoring]
- citation_applicable: 0 or 1
- citation: 0 to 5
- accurate: 1 to 5
- thorough: 1 to 5
- useful: 1 to 5
- organized: 1 to 5
- comprehensible: 1 to 5
- succinct: 1 to 5
- abstraction: 0 or 1
- synthesized: 0 to 5
- voice_summ: 0 or 1
- voice_note: 0 or 1

[Rules]
- Do not reward unsupported abstraction.
- Do not penalize paraphrasing if clinically faithful.
- In transcript-to-note evaluation, inline citations are generally not required.
- Return JSON only.

[Output Format]
{{
  "citation_applicable": 0,
  "citation": 0,
  "accurate": 1,
  "thorough": 1,
  "useful": 1,
  "organized": 1,
  "comprehensible": 1,
  "succinct": 1,
  "abstraction": 1,
  "synthesized": 1,
  "voice_summ": 0,
  "voice_note": 0
}}
""".strip()

SECTION_APPROPRIATENESS_PROMPT = """
You are a clinical documentation quality evaluator.

Review the generated note and determine whether statements are placed in the correct section for the requested template.

[Template]
{template_name}

[Section Semantics]
{section_semantics}

[Generated Note]
{generated_note}

[Instructions]
- Identify only clear section miscategorization errors.
- A statement is a major miscategorization when it is clearly in the wrong clinical section in a way that could mislead a reader.
- Minor miscategorization is a softer organizational issue that does not substantially distort meaning.
- Ignore harmless phrasing differences when the section placement is still acceptable.
- If the note omits a section entirely, do not count that as a miscategorization here.
- Return JSON only.

[Output Format]
{{
  "section_appropriateness_score": 1.0,
  "section_miscategorizations": [
    {{
      "text": "The patient is taking baby aspirin every day and insulin.",
      "current_section": "P",
      "expected_section": "S",
      "severity": "major",
      "reason": "This is current medication state reported by the patient rather than a plan."
    }}
  ]
}}
""".strip()

LLM_CHECKLIST_PROMPT = """
You are an expert clinical note quality evaluator.

Your task is to evaluate a generated clinical note against the source transcript using a structured checklist focused on failure modes of clinical dialogue summarization.

[Goal]
- Judge whether the note preserves clinically salient information from the transcript.
- Judge whether the note preserves state, temporality, and problem-linked organization.
- Judge whether the note is safe and usable as clinical documentation.
- Return JSON only.

[Scoring Rubric]
- Use integers 1 to 5 for scored items.
- 1 = very poor
- 2 = poor
- 3 = acceptable / mixed
- 4 = good
- 5 = excellent
- Use "Yes" or "No" for the safety flags.

[Checklist Items]
1. content_preservation_core
Does the note preserve the clinically salient core information from the dialogue without omitting central visit content?

2. noise_suppression
Does the note avoid including small talk, repetitive questioning, or peripheral conversational details that do not belong in the clinical note?

3. lay_expression_preservation
Does the note preserve clinically meaningful patient-reported nuance without being too raw or too aggressively abstracted? Consider severity, uncertainty, and symptom nuance.

4. state_update_fidelity
Does the note correctly reflect the final resolved clinical state when the dialogue contains corrections, clarifications, unknown-to-known resolution, or current medication versus intended change?

5. temporality_preservation
Does the note keep past history, current findings/status, and future plan clearly separated without temporal confusion?

6. problem_evidence_linkage
Are symptoms, exam findings, tests, and relevant history linked to the correct clinical problem or concern?

7. state_plan_separation
Does the note clearly separate current state/current medications from new orders, medication changes, referrals, and follow-up plans?

8. section_organization_appropriateness
Are facts placed in clinically appropriate sections for the requested note structure?

9. clinically_meaningful_omission
Is there a clinically meaningful omission?
Return "Yes" if a meaningful omission exists, otherwise "No".

10. clinically_concerning_hallucination
Is there a clinically concerning hallucination or unsupported statement?
Return "Yes" if such a hallucination exists, otherwise "No".

11. overall_clinical_usability
Would this note be usable as a clinical documentation draft or review note?

[Rules]
- Evaluate clinical content, not writing style preference.
- Do not reward unsupported abstraction.
- Do not penalize reasonable paraphrasing if the clinical meaning is preserved.
- Score based on the transcript as the source of truth.
- Be strict about state, temporality, and problem-plan linkage.

[Transcript]
{transcript}

[Generated Note]
{generated_note}
""".strip()

JSON_REPAIR_PROMPT = """
You are repairing malformed JSON produced by a clinical NLP system.

[Goal]
- Return valid JSON only.
- Preserve the original content as much as possible.
- Do not add new clinical information.
- Only fix formatting, quoting, escaping, commas, brackets, and schema conformance issues.

[Expected Schema]
{schema_hint}

[Malformed JSON]
{raw_text}
""".strip()


def load_env_file(env_path: Path) -> None:
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if " #" in value:
            value = value.split(" #", 1)[0].rstrip()
        elif "\t#" in value:
            value = value.split("\t#", 1)[0].rstrip()
        if value.startswith(("'", '"')) and value.endswith(("'", '"')) and len(value) >= 2:
            value = value[1:-1]
        else:
            value = value.strip("'").strip('"').strip()
        os.environ.setdefault(key, value)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Any]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
    return rows


def safe_text(value: Any) -> str:
    return str(value or "").strip()


def normalize_json_like_text(text: str) -> str:
    s = safe_text(text)
    if not s:
        return s
    replacements = {
        "\u201c": '"',
        "\u201d": '"',
        "\u2018": "'",
        "\u2019": "'",
        "\u2013": "-",
        "\u2014": "-",
        "\u00a0": " ",
    }
    for src, dst in replacements.items():
        s = s.replace(src, dst)
    return s


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run reframed note-generation experiments with direct, MEDSUM-ENT-inspired, Cluster2Sent-lite, CAP, and CAP+Event methods."
    )
    parser.add_argument("--cases-path", type=Path, default=DEFAULT_CASES_PATH)
    parser.add_argument("--method-a-csv", type=Path, default=DEFAULT_METHOD_A_CSV)
    parser.add_argument("--method-b-csv", type=Path, default=DEFAULT_METHOD_B_CSV)
    parser.add_argument("--method-c-csv", type=Path, default=DEFAULT_METHOD_C_CSV)
    parser.add_argument("--legacy-extraction-dir", type=Path, default=DEFAULT_LEGACY_EXTRACTION_DIR)
    parser.add_argument("--problem-state-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--experiment-set", choices=tuple(EXPERIMENT_METHOD_SETS), default="all")
    parser.add_argument("--methods", nargs="+", default=["direct", "medsum_ent", "cluster2sent", "cap", "cap_event"])
    parser.add_argument("--templates", nargs="+", default=["soap", "sectioned", "brief"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--case-ids", nargs="*", default=None)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--api-base-url", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--request-timeout", type=float, default=180.0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-summary-tokens", type=int, default=SUMMARY_MAX_TOKENS)
    parser.add_argument("--max-extraction-tokens", type=int, default=EXTRACTION_MAX_TOKENS)
    parser.add_argument("--max-event-tokens", type=int, default=EVENT_MAX_TOKENS)
    parser.add_argument("--judge-model", default=None)
    parser.add_argument("--judge-api-base-url", default=None)
    parser.add_argument("--judge-api-key", default=None)
    parser.add_argument("--judge-max-tokens", type=int, default=JUDGE_MAX_TOKENS)
    parser.add_argument("--eval-workers", type=int, default=4)
    parser.add_argument("--chunk-workers", type=int, default=4)
    parser.add_argument("--task-workers", type=int, default=1)
    parser.add_argument("--case-workers", type=int, default=1)
    parser.add_argument("--target-specialty", default="Primary Care")
    parser.add_argument("--disable-semantic-cap-judge", action="store_true")
    parser.add_argument("--disable-safety-judge", action="store_true")
    parser.add_argument("--disable-pdsqi-judge", action="store_true")
    parser.add_argument("--disable-section-judge", action="store_true")
    parser.add_argument("--disable-checklist-judge", action="store_true")
    parser.add_argument("--generation-only", action="store_true")
    parser.add_argument("--evaluation-only", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--write-case-files", action="store_true")
    return parser.parse_args()


def normalize_choices(values: Sequence[str], supported: Sequence[str], label: str) -> List[str]:
    out: List[str] = []
    supported_set = set(supported)
    for value in values:
        key = str(value).strip()
        if key not in supported_set:
            raise ValueError(f"Unsupported {label}: {value}")
        canonical = METHOD_ALIASES.get(key, key) if label == "method" else key
        if canonical not in out:
            out.append(canonical)
    if not out:
        raise ValueError(f"No {label} selected.")
    return out


def infer_api_base_url(explicit_base_url: Optional[str]) -> str:
    if explicit_base_url:
        return explicit_base_url.rstrip("/")
    if os.getenv("OPENAI_BASE_URL"):
        return os.getenv("OPENAI_BASE_URL", "").rstrip("/")

    provider = os.getenv("LLM_PROVIDER", os.getenv("GLOBAL_PROVIDER", "runpod")).strip().lower()
    if "#" in provider:
        provider = provider.split("#", 1)[0].strip()
    provider = provider.strip("'").strip('"').strip()
    if provider == "runpod":
        pod_id = os.getenv("RUNPOD_POD_ID", "").strip()
        port = os.getenv("RUNPOD_PORT", "40080").strip()
        if not pod_id:
            raise RuntimeError("RUNPOD_POD_ID is not set. Pass --api-base-url or set RUNPOD_POD_ID.")
        return f"https://{pod_id}-{port}.proxy.runpod.net/v1"
    if provider == "openai":
        return "https://api.openai.com/v1"
    if provider == "local":
        return os.getenv("LOCAL_LLM_BASE_URL", "http://localhost:8000/v1").rstrip("/")
    raise RuntimeError(f"Unsupported provider: {provider}")


def infer_judge_api_base_url(explicit_base_url: Optional[str], judge_model: str) -> str:
    if explicit_base_url:
        return explicit_base_url.rstrip("/")
    model_name = safe_text(judge_model)
    if model_name.startswith("gpt-") or model_name.startswith("o1") or model_name.startswith("o3") or model_name.startswith("o4"):
        return "https://api.openai.com/v1"
    return infer_api_base_url(None)


class OpenAICompatClient:
    def __init__(self, *, base_url: str, api_key: str, timeout: float = 180.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.sdk_client = None
        if OpenAI is not None:
            self.sdk_client = OpenAI(
                api_key=api_key or "dummy",
                base_url=self.base_url,
                timeout=timeout,
            )
        self.usage_totals = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

    def _extract_usage(self, response_obj: Any) -> Dict[str, int]:
        usage = getattr(response_obj, "usage", None)
        if usage is not None:
            p = int(getattr(usage, "prompt_tokens", 0) or 0)
            c = int(getattr(usage, "completion_tokens", 0) or 0)
            t = int(getattr(usage, "total_tokens", 0) or (p + c))
            return {"prompt_tokens": p, "completion_tokens": c, "total_tokens": t}
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    def _accumulate_usage(self, usage: Dict[str, int]) -> None:
        self.usage_totals["prompt_tokens"] += int(usage.get("prompt_tokens", 0) or 0)
        self.usage_totals["completion_tokens"] += int(usage.get("completion_tokens", 0) or 0)
        self.usage_totals["total_tokens"] += int(usage.get("total_tokens", 0) or 0)

    def get_usage_totals(self) -> Dict[str, int]:
        return dict(self.usage_totals)

    def chat_completion(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if self.sdk_client is not None:
            response = self.sdk_client.chat.completions.create(**payload)
            content = response.choices[0].message.content if response.choices else ""
            usage = self._extract_usage(response)
            self._accumulate_usage(usage)
            return {"choices": [{"message": {"content": content}}], "usage": usage}

        req = urllib_request.Request(
            url=f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "openai-python/run_template_rendering_experiments",
                **({"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}),
            },
        )
        try:
            with urllib_request.urlopen(req, timeout=self.timeout) as resp:
                raw = json.loads(resp.read().decode("utf-8"))
                usage_obj = raw.get("usage") or {}
                usage = {
                    "prompt_tokens": int(usage_obj.get("prompt_tokens", 0) or 0),
                    "completion_tokens": int(usage_obj.get("completion_tokens", 0) or 0),
                    "total_tokens": int(usage_obj.get("total_tokens", 0) or 0),
                }
                self._accumulate_usage(usage)
                raw["usage"] = usage
                return raw
        except urllib_error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {exc.code} from {req.full_url}: {detail}") from exc


def normalize_message_content(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("text"):
                parts.append(str(item["text"]))
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts).strip()
    return str(content or "").strip()


def extract_completion_text(response: Dict[str, Any]) -> str:
    choices = response.get("choices") or []
    if not choices:
        raise RuntimeError(f"No choices returned: {response}")
    message = choices[0].get("message") or {}
    return normalize_message_content(message.get("content"))


def safe_json_extract(text: str) -> Dict[str, Any]:
    s = normalize_json_like_text(text)
    if not s:
        raise ValueError("Empty response")
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
        s = re.sub(r"\s*```$", "", s)
    candidates: List[str] = [s]
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidates.append(s[start : end + 1])
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except Exception:
            repaired = re.sub(r",(\s*[}\]])", r"\1", candidate)
            repaired = repaired.replace("\x00", "")
            try:
                return json.loads(repaired)
            except Exception:
                continue
    raise ValueError(f"Could not parse JSON from response head:\n{s[:1000]}")


def structured_note_section_keys(template_key: str) -> List[str]:
    if template_key == "soap":
        return ["S", "O", "A", "P"]
    if template_key == "sectioned":
        return ["Chief Complaint", "History of Present Illness", "Findings", "Assessment", "Plan"]
    return []


def salvage_structured_note_json(raw_text: str, template_key: str) -> Dict[str, Any]:
    keys = structured_note_section_keys(template_key)
    if not keys:
        raise ValueError("No structured section keys available for salvage.")
    s = normalize_json_like_text(raw_text)
    out: Dict[str, Any] = {key: "" for key in keys}
    if not s:
        return out

    matches: List[Tuple[str, int]] = []
    for key in keys:
        m = re.search(rf'"{re.escape(key)}"\s*:', s, flags=re.IGNORECASE)
        if m:
            matches.append((key, m.start()))
    if not matches:
        # Fallback for non-JSON text that still contains visible section headings
        # (e.g., ">- Chief Complaint: ... - History of Present Illness: ...").
        heading_matches: List[Tuple[str, int, int]] = []
        for key in keys:
            for m in re.finditer(
                rf'(?im)(?:^|[\n\r])[\s"\'`>\-*]*{re.escape(key)}\s*:?',
                s,
            ):
                heading_matches.append((key, m.start(), m.end()))
                break
        # Secondary fallback: headings may appear inline without line breaks.
        if not heading_matches:
            lowered = s.lower()
            for key in keys:
                idx = lowered.find(key.lower())
                if idx != -1:
                    heading_matches.append((key, idx, idx + len(key)))
        if not heading_matches:
            raise ValueError("Could not find any structured section keys in malformed JSON.")
        heading_matches.sort(key=lambda x: x[1])
        for idx, (key, _start_pos, end_pos) in enumerate(heading_matches):
            next_pos = heading_matches[idx + 1][1] if idx + 1 < len(heading_matches) else len(s)
            value = safe_text(s[end_pos:next_pos])
            value = re.sub(r'^[\s"\'`>\-,:;]+', "", value)
            value = re.sub(r'[\s"\'`{}\-,:;]+$', "", value)
            value = re.sub(r"\s+", " ", value).strip()
            out[key] = value
        return out
    matches.sort(key=lambda x: x[1])

    for idx, (key, start_pos) in enumerate(matches):
        next_pos = matches[idx + 1][1] if idx + 1 < len(matches) else len(s)
        segment = s[start_pos:next_pos]
        if ":" not in segment:
            continue
        value_part = segment.split(":", 1)[1]
        string_literals = re.findall(r'"(?:\\.|[^"\\])*"', value_part)
        items: List[str] = []
        for lit in string_literals:
            try:
                value = json.loads(lit)
            except Exception:
                value = lit.strip('"')
            value = safe_text(value)
            if value:
                items.append(value)
        out[key] = " ".join(items[:8]).strip()
    return out


def repair_json_via_llm(
    client: OpenAICompatClient,
    *,
    model: str,
    raw_text: str,
    schema_obj: Dict[str, Any],
    max_tokens: int,
) -> Dict[str, Any]:
    schema_hint = json.dumps(schema_obj, ensure_ascii=False)
    repair_prompt = JSON_REPAIR_PROMPT.format(
        schema_hint=schema_hint,
        raw_text=raw_text[:24000],
    )
    repaired = call_llm(
        client,
        model=model,
        prompt=repair_prompt,
        max_tokens=max_tokens,
        temperature=0.0,
        force_json=True,
        json_schema_obj=schema_obj,
        prefer_json_object=True,
    )
    return safe_json_extract(repaired)


def cap_schema(max_props: int) -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "atomic_propositions": {
                "type": "array",
                "maxItems": max_props,
                "items": {
                    "type": "object",
                    "properties": {
                        "prop_id": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                        "cap_type": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                        "canonical_concept": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                        "category": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                        "speaker": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                        "predicate": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                        "status": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                        "visit_state": {"anyOf": [{"type": "string"}, {"type": "null"}]},
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


def event_plan_schema(max_events: int) -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "events": {
                "type": "array",
                "maxItems": max_events,
                "items": {
                    "type": "object",
                    "properties": {
                        "event_id": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                        "event_type": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                        "canonical_concept": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                        "visit_state": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                        "status": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                        "temporality": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                        "speaker_source": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                        "clinical_priority": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                        "title": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                        "event_summary": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                        "supporting_cap_ids": {"type": "array", "items": {"type": "string"}},
                        "anchor_problem_cap_id": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                        "anchor_problem": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                        "S_cap_ids": {"type": "array", "items": {"type": "string"}},
                        "O_cap_ids": {"type": "array", "items": {"type": "string"}},
                        "A_cap_ids": {"type": "array", "items": {"type": "string"}},
                        "P_cap_ids": {"type": "array", "items": {"type": "string"}},
                        "global_cap_ids": {"type": "array", "items": {"type": "string"}},
                        "supporting_evidence": {"type": "array", "items": {"type": "string"}},
                        "current_state": {"type": "array", "items": {"type": "string"}},
                        "planned_actions": {"type": "array", "items": {"type": "string"}},
                        "care_context": {"type": "array", "items": {"type": "string"}},
                        "key_slots": {"type": "object", "additionalProperties": {"type": ["string", "number", "boolean", "null"]}},
                        "render_sections": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["event_summary"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["events"],
        "additionalProperties": False,
    }


def gpt_f1_schema(max_items_per_category: int = GPT_F1_MAX_ITEMS_PER_CATEGORY) -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "categories": {
                "type": "object",
                "properties": {
                    key: {
                        "type": "array",
                        "maxItems": max_items_per_category,
                        "items": {"type": "string"},
                    }
                    for key in GPT_F1_CATEGORIES
                },
                "required": list(GPT_F1_CATEGORIES),
                "additionalProperties": False,
            }
        },
        "required": ["categories"],
        "additionalProperties": False,
    }


def nair_concept_extraction_schema(max_items: int = NAIR_MAX_CONCEPTS_PER_SECTION) -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "medical_concepts": {
                "type": "array",
                "maxItems": max_items,
                "items": {"type": "string"},
            }
        },
        "required": ["medical_concepts"],
        "additionalProperties": False,
    }


def nair_concept_verification_schema(max_items: int = NAIR_MAX_VERIFICATION_ITEMS) -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "found_b": {
                "type": "array",
                "maxItems": max_items,
                "items": {"type": "string"},
            },
            "not_found_b": {
                "type": "array",
                "maxItems": max_items,
                "items": {"type": "string"},
            },
        },
        "required": ["found_b", "not_found_b"],
        "additionalProperties": False,
    }


def semantic_cap_audit_schema(max_items: int = CAP_MAX_PROPS) -> Dict[str, Any]:
    item_schema = {
        "type": "object",
        "properties": {
            "cap_id": {"anyOf": [{"type": "string"}, {"type": "null"}]},
            "judgment": {"anyOf": [{"type": "string"}, {"type": "null"}]},
            "reason": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        },
        "required": ["cap_id", "judgment"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "source_cap_recall": {"type": "array", "maxItems": max_items, "items": item_schema},
            "summary_cap_precision": {"type": "array", "maxItems": max_items, "items": item_schema},
        },
        "required": ["source_cap_recall", "summary_cap_precision"],
        "additionalProperties": False,
    }


def semantic_cap_recall_schema(max_items: int = CAP_MAX_PROPS) -> Dict[str, Any]:
    item_schema = {
        "type": "object",
        "properties": {
            "cap_id": {"anyOf": [{"type": "string"}, {"type": "null"}]},
            "judgment": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        },
        "required": ["cap_id", "judgment"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "source_cap_recall": {"type": "array", "maxItems": max_items, "items": item_schema},
        },
        "required": ["source_cap_recall"],
        "additionalProperties": False,
    }


def semantic_cap_precision_schema(max_items: int = CAP_MAX_PROPS) -> Dict[str, Any]:
    item_schema = {
        "type": "object",
        "properties": {
            "cap_id": {"anyOf": [{"type": "string"}, {"type": "null"}]},
            "judgment": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        },
        "required": ["cap_id", "judgment"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "summary_cap_precision": {"type": "array", "maxItems": max_items, "items": item_schema},
        },
        "required": ["summary_cap_precision"],
        "additionalProperties": False,
    }


def safety_judge_schema(max_items: int = CAP_MAX_PROPS) -> Dict[str, Any]:
    hallucination_schema = {
        "type": "object",
        "properties": {
            "text": {"anyOf": [{"type": "string"}, {"type": "null"}]},
            "type": {"anyOf": [{"type": "string"}, {"type": "null"}]},
            "severity": {"anyOf": [{"type": "string"}, {"type": "null"}]},
            "reason": {"anyOf": [{"type": "string"}, {"type": "null"}]},
            "evidence": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        },
        "required": ["text"],
        "additionalProperties": False,
    }
    omission_schema = {
        "type": "object",
        "properties": {
            "text": {"anyOf": [{"type": "string"}, {"type": "null"}]},
            "type": {"anyOf": [{"type": "string"}, {"type": "null"}]},
            "severity": {"anyOf": [{"type": "string"}, {"type": "null"}]},
            "reason": {"anyOf": [{"type": "string"}, {"type": "null"}]},
            "evidence": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        },
        "required": ["text"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "hallucinations": {"type": "array", "maxItems": max_items, "items": hallucination_schema},
            "omissions": {"type": "array", "maxItems": max_items, "items": omission_schema},
        },
        "required": ["hallucinations", "omissions"],
        "additionalProperties": False,
    }


def safety_hallucination_schema(max_items: int = CAP_MAX_PROPS) -> Dict[str, Any]:
    hallucination_schema = {
        "type": "object",
        "properties": {
            "text": {"anyOf": [{"type": "string"}, {"type": "null"}]},
            "type": {"anyOf": [{"type": "string"}, {"type": "null"}]},
            "severity": {"anyOf": [{"type": "string"}, {"type": "null"}]},
            "reason": {"anyOf": [{"type": "string"}, {"type": "null"}]},
            "evidence": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        },
        "required": ["text"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "hallucinations": {"type": "array", "maxItems": max_items, "items": hallucination_schema},
        },
        "required": ["hallucinations"],
        "additionalProperties": False,
    }


def safety_omission_schema(max_items: int = CAP_MAX_PROPS) -> Dict[str, Any]:
    omission_schema = {
        "type": "object",
        "properties": {
            "text": {"anyOf": [{"type": "string"}, {"type": "null"}]},
            "type": {"anyOf": [{"type": "string"}, {"type": "null"}]},
            "severity": {"anyOf": [{"type": "string"}, {"type": "null"}]},
            "reason": {"anyOf": [{"type": "string"}, {"type": "null"}]},
            "evidence": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        },
        "required": ["text"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "omissions": {"type": "array", "maxItems": max_items, "items": omission_schema},
        },
        "required": ["omissions"],
        "additionalProperties": False,
    }


def pdsqi_schema() -> Dict[str, Any]:
    int_or_null = {"anyOf": [{"type": "integer"}, {"type": "null"}]}
    return {
        "type": "object",
        "properties": {
            "citation_applicable": int_or_null,
            "citation": int_or_null,
            "accurate": int_or_null,
            "thorough": int_or_null,
            "useful": int_or_null,
            "organized": int_or_null,
            "comprehensible": int_or_null,
            "succinct": int_or_null,
            "abstraction": int_or_null,
            "synthesized": int_or_null,
            "voice_summ": int_or_null,
            "voice_note": int_or_null,
        },
        "required": [
            "citation_applicable",
            "citation",
            "accurate",
            "thorough",
            "useful",
            "organized",
            "comprehensible",
            "succinct",
            "abstraction",
            "synthesized",
            "voice_summ",
            "voice_note",
        ],
        "additionalProperties": False,
    }


def llm_checklist_schema() -> Dict[str, Any]:
    score_enum = {"type": "integer", "enum": [1, 2, 3, 4, 5]}
    yes_no_enum = {"type": "string", "enum": ["Yes", "No"]}
    return {
        "type": "object",
        "properties": {
            "content_preservation_core": score_enum,
            "noise_suppression": score_enum,
            "lay_expression_preservation": score_enum,
            "state_update_fidelity": score_enum,
            "temporality_preservation": score_enum,
            "problem_evidence_linkage": score_enum,
            "state_plan_separation": score_enum,
            "section_organization_appropriateness": score_enum,
            "clinically_meaningful_omission": yes_no_enum,
            "clinically_concerning_hallucination": yes_no_enum,
            "overall_clinical_usability": score_enum,
        },
        "required": [
            "content_preservation_core",
            "noise_suppression",
            "lay_expression_preservation",
            "state_update_fidelity",
            "temporality_preservation",
            "problem_evidence_linkage",
            "state_plan_separation",
            "section_organization_appropriateness",
            "clinically_meaningful_omission",
            "clinically_concerning_hallucination",
            "overall_clinical_usability",
        ],
        "additionalProperties": False,
    }


def structured_note_schema(template_key: str) -> Optional[Dict[str, Any]]:
    section_text_schema = {
        "type": "string",
        "maxLength": 6000,
    }
    if template_key == "soap":
        return {
            "type": "object",
            "properties": {
                "S": section_text_schema,
                "O": section_text_schema,
                "A": section_text_schema,
                "P": section_text_schema,
            },
            "required": ["S", "O", "A", "P"],
            "additionalProperties": False,
        }
    if template_key == "sectioned":
        return {
            "type": "object",
            "properties": {
                "Chief Complaint": section_text_schema,
                "History of Present Illness": section_text_schema,
                "Findings": section_text_schema,
                "Assessment": section_text_schema,
                "Plan": section_text_schema,
            },
            "required": [
                "Chief Complaint",
                "History of Present Illness",
                "Findings",
                "Assessment",
                "Plan",
            ],
            "additionalProperties": False,
        }
    return None


def structured_note_instruction(template_key: str) -> str:
    if template_key == "soap":
        return """
[Output Format]
Return a JSON object with exactly these keys:
- S
- O
- A
- P

Each value must be a single prose string for that section (paragraph style; no markdown bullets).

[Clinical Style]
- Write in polished clinician-authored style similar to ACI-BENCH reference notes.
- Use concise clinical prose, not transcript-like line-by-line copying.
- Use third-person clinical voice.
- Avoid explicit narrator phrasing like "The doctor ..."; use action-oriented plan language (for example: "Order ...", "Recommend ...", "Arrange follow-up ...").

[Signal vs Noise]
- Keep only information that changes diagnosis, risk stratification, or management.
- Exclude social chatter, rapport dialogue, and anecdotal storytelling unless clinically consequential.
- Do not repeat the same clinical fact across multiple sections.

[Section Semantics]
- S: symptoms/history chronology, pertinent negatives, adherence context.
- O: objective exam/vitals/tests/labs/imaging findings.
- A: clinician assessment, active problems, diagnostic impressions.
- P: medications, orders, referrals, counseling, follow-up, return precautions.

[Completeness Rule]
- Always emit all 4 keys.
- If a section is truly unsupported, return an empty string "".

Example:
{
  "S": "The patient reports intermittent dizziness for one week, worse when standing. He denies chest pain, dyspnea, or syncope.",
  "O": "Blood pressure is mildly low. Heart rate is within normal limits. Neurologic examination is nonfocal.",
  "A": "Symptoms are most consistent with orthostatic dizziness.",
  "P": "Recommend increased oral hydration and slow positional changes. Follow up if symptoms worsen or do not improve."
}
Do not include markdown, code fences, or prefatory text.
""".strip()
    if template_key == "sectioned":
        return """
[Output Format]
Return a JSON object with exactly these keys:
- Chief Complaint
- History of Present Illness
- Findings
- Assessment
- Plan

Each value must be a single prose string for that section (paragraph style; no markdown bullets).

[Clinical Style]
- Write in polished clinician-authored style similar to ACI-BENCH reference notes.
- Use concise clinical prose with natural paraphrasing.
- Do not copy CAP/transcript phrases verbatim when awkward.
- Use third-person clinical voice.
- Avoid explicit narrator phrasing like "The doctor ..."; for management, prefer action-oriented language.

[Signal vs Noise]
- Keep only clinically decision-relevant information.
- Exclude social chatter, restaurant/food anecdotes, and conversational storytelling unless clinically consequential.
- Avoid repeated or redundant statements across sections.

[Section Semantics]
- Chief Complaint: primary reason for visit.
- History of Present Illness: symptom chronology, relevant history, pertinent negatives, management-relevant context.
- Findings: objective exam/vitals/tests/imaging/labs.
- Assessment: diagnosis/impression/problem interpretation.
- Plan: management actions, medication changes, orders, referrals, follow-up, instructions.

[Completeness Rule]
- Always emit all 5 keys.
- If a section is truly unsupported, return an empty string "".

Example:
{
  "Chief Complaint": "Left shoulder pain.",
  "History of Present Illness": "The patient presents with progressive left shoulder pain over six weeks, worse with overhead activity and when lying on the affected side. He denies recent trauma. Acetaminophen provides partial relief.",
  "Findings": "Examination shows reduced active abduction of the left shoulder and pain with impingement testing.",
  "Assessment": "Clinical presentation is most consistent with rotator cuff tendinopathy.",
  "Plan": "Order shoulder radiographs, initiate physical therapy, and continue acetaminophen as needed. Arrange follow-up in four weeks if symptoms persist."
}
Do not include markdown, code fences, or prefatory text.
""".strip()
    return ""


def call_llm(
    client: OpenAICompatClient,
    *,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    force_json: bool = False,
    json_schema_obj: Optional[Dict[str, Any]] = None,
    prefer_json_object: bool = False,
) -> str:
    token_key = "max_tokens"
    model_name = safe_text(model)
    if model_name.startswith("gpt-5") or model_name.startswith("o1") or model_name.startswith("o3") or model_name.startswith("o4"):
        token_key = "max_completion_tokens"
    base_payload = {
        "model": model,
        "temperature": temperature,
        token_key: max_tokens,
        "messages": [
            {
                "role": "system",
                "content": "You are a precise clinical NLP system. Follow the instructions exactly.",
            },
            {"role": "user", "content": prompt},
        ],
    }
    attempts: List[Dict[str, Any]] = []
    base_url_lc = safe_text(getattr(client, "base_url", "")).lower()
    supports_json_schema = "api.openai.com" in base_url_lc
    if force_json and json_schema_obj:
        if not supports_json_schema:
            attempts.append({**base_payload, "response_format": {"type": "json_object"}})
        else:
            if prefer_json_object:
                attempts.append({**base_payload, "response_format": {"type": "json_object"}})
                attempts.append(
                    {
                        **base_payload,
                        "response_format": {
                            "type": "json_schema",
                            "json_schema": {
                                "name": "structured_result",
                                "strict": True,
                                "schema": json_schema_obj,
                            },
                        },
                    }
                )
            else:
                attempts.append(
                    {
                        **base_payload,
                        "response_format": {
                            "type": "json_schema",
                            "json_schema": {
                                "name": "structured_result",
                                "strict": True,
                                "schema": json_schema_obj,
                            },
                        },
                    }
                )
                attempts.append({**base_payload, "response_format": {"type": "json_object"}})
    attempts.append(base_payload)

    last_exc: Optional[Exception] = None
    response_format_unsupported = False
    for payload in attempts:
        if response_format_unsupported and "response_format" in payload:
            continue
        try:
            response = client.chat_completion(payload)
            return extract_completion_text(response)
        except Exception as exc:
            last_exc = exc
            exc_text = safe_text(exc).lower()
            if "404" in exc_text or "response_format" in exc_text or "json_schema" in exc_text:
                response_format_unsupported = True
    raise RuntimeError(f"LLM call failed after retries: {last_exc}")


def normalize_cap_obj(obj: Any, *, max_props: int = CAP_MAX_PROPS) -> Dict[str, Any]:
    if not isinstance(obj, dict):
        return {"atomic_propositions": [], "filtered_nonverifiable_units": []}
    props: List[Dict[str, Any]] = []
    raw_props = obj.get("atomic_propositions")
    if isinstance(raw_props, list):
        for idx, item in enumerate(raw_props, start=1):
            if not isinstance(item, dict):
                continue
            proposition_text = safe_text(item.get("proposition_text") or item.get("fact_text") or item.get("text"))
            if not proposition_text:
                continue
            claim_type_tags = item.get("claim_type_tags") if isinstance(item.get("claim_type_tags"), list) else []
            normalized = {
                "prop_id": safe_text(item.get("prop_id") or f"P{idx}") or f"P{idx}",
                "cap_type": safe_text(item.get("cap_type")) or None,
                "canonical_concept": safe_text(item.get("canonical_concept")) or None,
                "category": safe_text(item.get("category")) or None,
                "speaker": safe_text(item.get("content_source") or item.get("speaker")) or None,
                "predicate": safe_text(item.get("predicate")) or None,
                "status": safe_text(item.get("status")) or None,
                "visit_state": safe_text(item.get("visit_state")) or None,
                "temporality": safe_text(item.get("temporality")) or None,
                "claim_type_tags": [safe_text(x) for x in claim_type_tags if safe_text(x)],
                "proposition_text": proposition_text,
            }
            props.append({k: v for k, v in normalized.items() if v not in (None, [], "")})
            if len(props) >= max_props:
                break
    filtered_units: List[str] = []
    raw_filtered = obj.get("filtered_nonverifiable_units")
    if isinstance(raw_filtered, list):
        for item in raw_filtered:
            text = safe_text(item.get("unit_text") if isinstance(item, dict) else item)
            if text and text not in filtered_units:
                filtered_units.append(text)
    out = {"atomic_propositions": props, "filtered_nonverifiable_units": filtered_units}
    runtime_seconds = obj.get("runtime_seconds")
    if isinstance(runtime_seconds, (int, float)):
        out["runtime_seconds"] = runtime_seconds
    return out


EVENT_STOPWORDS = {
    "the",
    "patient",
    "clinician",
    "doctor",
    "reports",
    "report",
    "has",
    "have",
    "had",
    "notes",
    "note",
    "states",
    "state",
    "that",
    "with",
    "from",
    "into",
    "this",
    "there",
    "their",
    "them",
    "will",
    "would",
    "should",
    "about",
    "very",
    "mildly",
    "slightly",
    "currently",
}


def infer_cap_speaker(text: str) -> Optional[str]:
    lowered = text.lower()
    if lowered.startswith("the patient") or lowered.startswith("patient ") or " the patient " in f" {lowered} ":
        return "patient"
    if lowered.startswith("the clinician") or lowered.startswith("the doctor") or " the doctor " in f" {lowered} ":
        return "clinician"
    if "lab" in lowered or "hemoglobin" in lowered or "a1c" in lowered or "creatinine" in lowered:
        return "test_result"
    return None


def infer_cap_predicate(text: str) -> Optional[str]:
    lowered = text.lower()
    for needle, predicate in (
        ("denies", "denies"),
        ("reports", "reports"),
        ("notes", "notes"),
        ("shows", "shows"),
        ("diagnoses", "diagnoses"),
        ("suspects", "suspects"),
        ("prescribes", "prescribes"),
        ("orders", "orders"),
        ("recommends", "recommends"),
        ("instructs", "instructs"),
        ("plans", "plans"),
    ):
        if needle in lowered:
            return predicate
    if any(k in lowered for k in ["start ", "continue ", "increase ", "decrease ", "stop ", "take "]):
        return "plans" if any(k in lowered for k in ["start ", "increase ", "decrease ", "stop "]) else "has"
    return None


def infer_cap_status(text: str) -> Optional[str]:
    lowered = text.lower()
    if any(k in lowered for k in ["denies", "no ", "not ", "without "]):
        return "negated"
    if any(k in lowered for k in ["possible", "possibly", "suspects", "may ", "might ", "unclear", "likely"]):
        return "uncertain"
    return "affirmed"


def infer_cap_temporality(text: str) -> Optional[str]:
    lowered = text.lower()
    if any(k in lowered for k in ["will ", "plan", "planned", "follow up", "follow-up", "return", "recheck", "start ", "increase ", "decrease "]):
        return "future"
    if any(k in lowered for k in ["previously", "ago", "last time", "last week", "past ", "history of", "had "]):
        return "past"
    if any(k in lowered for k in ["today", "currently", "now", "this morning", "stable", "still "]):
        return "current"
    return None


def infer_cap_category(text: str, predicate: Optional[str], speaker: Optional[str]) -> Optional[str]:
    lowered = text.lower()
    if any(k in lowered for k in ["allergy", "allergic"]):
        return "Allergy"
    if any(k in lowered for k in ["chief complaint", "here for", "follow-up for"]):
        return "ChiefComplaint"
    if predicate in {"prescribes", "plans", "orders", "recommends", "instructs"} and any(
        k in lowered for k in ["mg", "tablet", "capsule", "medication", "metformin", "glimepiride", "atorvastatin", "amlodipine", "lisinopril", "metoprolol"]
    ):
        return "MedicationPlan"
    if any(k in lowered for k in ["taking ", "takes ", "on metoprolol", "on lisinopril", "on amlodipine", "on atorvastatin"]):
        return "MedicationPlan"
    if predicate in {"orders", "plans"} and any(
        k in lowered for k in ["test", "blood work", "lab", "a1c", "kidney function", "cholesterol", "mri", "x-ray", "scan"]
    ):
        return "TestPlan"
    if any(k in lowered for k in ["follow up", "follow-up", "return", "call the office", "return precaution"]):
        return "FollowUpPlan"
    if predicate in {"suspects", "diagnoses"} or any(k in lowered for k in ["diagnosis", "neuropathy", "hypertension", "hyperlipidemia", "diabetes"]):
        return "Diagnosis"
    if speaker == "test_result" or any(k in lowered for k in ["hemoglobin", "a1c", "creatinine", "bp ", "blood pressure", "result", "lab"]):
        return "Finding"
    if any(k in lowered for k in ["pain", "cough", "dizziness", "numb", "tingling", "swelling", "fatigue", "nocturia"]):
        return "Symptom"
    if any(k in lowered for k in ["exam", "notes", "thickened nails", "dry skin", "sensation", "pulses", "edema"]):
        return "Finding"
    if any(k in lowered for k in ["history", "past medical", "smoking", "social history", "family history"]):
        return "History"
    return "History"


def infer_claim_type_tags(text: str, category: Optional[str], predicate: Optional[str]) -> List[str]:
    lowered = text.lower()
    tags: List[str] = []
    if any(ch.isdigit() for ch in lowered):
        tags.append("quantitative")
    if any(k in lowered for k in ["past", "current", "future", "stable", "improved", "worse", "follow up", "follow-up", "this morning"]):
        tags.append("temporal")
    if any(k in lowered for k in ["because", "due to", "secondary to", "related to"]):
        tags.append("causal")
    if category == "MedicationPlan" or any(k in lowered for k in ["mg", "tablet", "dose", "medication", "glimepiride", "metformin"]):
        tags.append("medication")
    if category in {"TestPlan", "FollowUpPlan"} or predicate in {"orders", "recommends", "instructs", "plans"}:
        tags.append("plan")
    if category == "Diagnosis":
        tags.append("diagnostic")
    if category == "Symptom":
        tags.append("symptom")
    if category == "Finding":
        tags.append("exam")
    if any(k in lowered for k in ["hypoglycemia", "return precaution", "call the office", "seek care", "severe"]):
        tags.append("safety_relevant")
    return sorted(dict.fromkeys(tags))


def infer_visit_state(text: str, status: Optional[str], temporality: Optional[str]) -> str:
    lowered = text.lower()
    status_norm = safe_text(status).lower()
    temporality_norm = safe_text(temporality).lower()
    if status_norm == "negated":
        return "ruled_out"
    if status_norm == "uncertain":
        return "uncertain"
    if any(k in lowered for k in ["resolved", "went away", "improved", "got better"]):
        return "improving" if "improved" in lowered or "got better" in lowered else "resolved"
    if any(k in lowered for k in ["worse", "worsening", "progressive", "more severe"]):
        return "worsening"
    if any(k in lowered for k in ["stable", "unchanged", "same"]):
        return "stable"
    if temporality_norm == "past":
        return "historical"
    return "active"


def infer_canonical_concept(text: str, category: Optional[str]) -> str:
    lowered = text.lower()
    for med in ["glimepiride", "metformin", "metoprolol", "lisinopril", "hydrochlorothiazide", "amlodipine", "atorvastatin"]:
        if med in lowered:
            return med
    for diagnosis in ["peripheral neuropathy", "diabetes", "hypertension", "hyperlipidemia", "cataract"]:
        if diagnosis in lowered:
            return diagnosis
    if "blood pressure" in lowered:
        return "blood pressure"
    if "kidney function" in lowered:
        return "kidney function test"
    if "cholesterol" in lowered:
        return "cholesterol test"
    if "blood work" in lowered:
        return "blood work"
    if "pain" in lowered:
        body_sites = [term for term in ["foot", "feet", "toe", "toes", "hip", "leg", "hand", "back", "knee"] if term in lowered]
        if body_sites:
            return f"pain:{body_sites[0]}"
        return "pain"
    if "cough" in lowered:
        return "cough"
    if "sensation" in lowered or "numb" in lowered or "tingling" in lowered:
        return "sensory change"
    tokens = event_concept_tokens(text)[:4]
    if tokens:
        return " ".join(tokens)
    return safe_text(category) or "clinical_state"


def enrich_cap_obj(cap_obj: Dict[str, Any]) -> Dict[str, Any]:
    enriched: List[Dict[str, Any]] = []
    for idx, item in enumerate(cap_obj.get("atomic_propositions", []), start=1):
        text = safe_text(item.get("proposition_text"))
        if not text:
            continue
        speaker = safe_text(item.get("content_source") or item.get("speaker")) or infer_cap_speaker(text)
        predicate = safe_text(item.get("predicate")) or infer_cap_predicate(text)
        status = safe_text(item.get("status")) or infer_cap_status(text)
        temporality = safe_text(item.get("temporality")) or infer_cap_temporality(text)
        category = safe_text(item.get("category")) or infer_cap_category(text, predicate, speaker)
        claim_type_tags = item.get("claim_type_tags") if isinstance(item.get("claim_type_tags"), list) else []
        if not claim_type_tags:
            claim_type_tags = infer_claim_type_tags(text, category, predicate)
        visit_state = safe_text(item.get("visit_state")) or infer_visit_state(text, status, temporality)
        canonical_concept = safe_text(item.get("canonical_concept")) or infer_canonical_concept(text, category)
        normalized = {
            "prop_id": safe_text(item.get("prop_id") or f"P{idx}") or f"P{idx}",
            "cap_type": category or None,
            "canonical_concept": canonical_concept or None,
            "category": category or None,
            "speaker": speaker or None,
            "predicate": predicate or None,
            "status": status or None,
            "visit_state": visit_state or None,
            "temporality": temporality or None,
            "claim_type_tags": [safe_text(x) for x in claim_type_tags if safe_text(x)],
            "proposition_text": text,
        }
        enriched.append({k: v for k, v in normalized.items() if v not in (None, [], "")})
    out = {
        "atomic_propositions": enriched,
        "filtered_nonverifiable_units": list(cap_obj.get("filtered_nonverifiable_units", [])),
    }
    if isinstance(cap_obj.get("runtime_seconds"), (int, float)):
        out["runtime_seconds"] = cap_obj["runtime_seconds"]
    return out


def event_type_from_cap(rec: Dict[str, Any]) -> str:
    category = safe_text(rec.get("category")).lower()
    predicate = safe_text(rec.get("predicate")).lower()
    text = safe_text(rec.get("proposition_text")).lower()
    if category == "chiefcomplaint":
        return "ChiefComplaint"
    if category in {"symptom"}:
        return "SymptomCourse"
    if category in {"history", "demographic", "allergy"}:
        return "PMH"
    if category == "diagnosis":
        return "DiagnosisImpression"
    if category == "finding":
        if any(k in text for k in ["a1c", "hemoglobin", "creatinine", "lab", "blood pressure", "result"]):
            return "TestResult"
        return "ExamFinding"
    if category == "testplan":
        return "PlanOrder"
    if category == "followupplan":
        if any(k in text for k in ["call", "warning", "return precaution", "seek care"]):
            return "PatientInstruction"
        return "FollowUp"
    if category == "medicationplan":
        if any(k in text for k in ["taking ", "takes ", "continue ", "continues ", "is on ", "on metoprolol", "on lisinopril", "on amlodipine", "on atorvastatin"]):
            return "MedicationState"
        if any(k in text for k in ["increase", "decrease", "start", "stop", "hold", "change", "new prescription", "refill"]):
            return "MedicationChange"
        if predicate in {"instructs", "recommends"}:
            return "PatientInstruction"
        if predicate in {"prescribes", "plans", "orders"}:
            return "MedicationChange"
        return "MedicationState"
    if any(k in text for k in ["follow up", "follow-up", "return in", "come back"]):
        return "FollowUp"
    if any(k in text for k in ["call the office", "watch for", "monitor", "counsel", "avoid", "cut down"]):
        return "PatientInstruction"
    return "PMH"


def event_priority_from_type(event_type: str) -> str:
    if event_type in {"MedicationChange", "PlanOrder", "FollowUp", "PatientInstruction", "DiagnosisImpression", "TestResult"}:
        return "high"
    if event_type in {"ExamFinding", "SymptomCourse", "MedicationState"}:
        return "medium"
    return "low"


def event_render_sections(event_type: str) -> List[str]:
    mapping = {
        "ChiefComplaint": ["S", "Chief Complaint"],
        "SymptomCourse": ["S", "History of Present Illness"],
        "PMH": ["S", "History of Present Illness"],
        "MedicationState": ["S", "Plan"],
        "MedicationChange": ["P", "Plan"],
        "ExamFinding": ["O", "Findings"],
        "TestResult": ["O", "Findings"],
        "DiagnosisImpression": ["A", "Assessment"],
        "PlanOrder": ["P", "Plan"],
        "FollowUp": ["P", "Plan"],
        "PatientInstruction": ["P", "Plan"],
    }
    return mapping.get(event_type, ["S", "Plan"])


def event_concept_tokens(text: str) -> List[str]:
    lowered = text.lower()
    lowered = re.sub(r"\b\d+(?:\.\d+)?\b", " ", lowered)
    lowered = re.sub(r"[^a-z0-9%/+\-]+", " ", lowered)
    tokens: List[str] = []
    for tok in lowered.split():
        if len(tok) <= 2 or tok in EVENT_STOPWORDS:
            continue
        tokens.append(tok)
    return tokens


def extract_key_slots_from_cap(rec: Dict[str, Any]) -> Dict[str, Any]:
    text = safe_text(rec.get("proposition_text"))
    lowered = text.lower()
    slots: Dict[str, Any] = {}

    for med in ["glimepiride", "metformin", "metoprolol", "lisinopril", "hydrochlorothiazide", "amlodipine", "atorvastatin"]:
        if med in lowered:
            slots["medication"] = med
            break

    laterality = None
    if "bilateral" in lowered:
        laterality = "bilateral"
    elif "left" in lowered:
        laterality = "left"
    elif "right" in lowered:
        laterality = "right"
    if laterality:
        slots["laterality"] = laterality

    values = re.findall(r"\b\d+(?:\.\d+)?\s*(?:mg|mcg|g|cm|mm|%|hours?|days?|weeks?|months?)\b", lowered)
    if values:
        slots["values"] = values[:3]
        med_values = [v for v in values if any(unit in v for unit in ["mg", "mcg", "g"])]
        if med_values:
            slots["dose"] = med_values[0]

    frequency = None
    for pattern in [
        "once a day",
        "twice a day",
        "three times a day",
        "daily",
        "bid",
        "tid",
        "qhs",
        "every six hours",
        "every 6 hours",
        "as needed",
    ]:
        if pattern in lowered:
            frequency = pattern
            break
    if frequency:
        slots["frequency"] = frequency

    body_sites = []
    for term in ["foot", "feet", "toe", "toes", "hip", "leg", "hand", "eye", "skin", "nails", "blood pressure"]:
        if term in lowered:
            body_sites.append(term)
    if body_sites:
        slots["body_sites"] = sorted(dict.fromkeys(body_sites))

    if any(k in lowered for k in ["increase", "decrease", "start", "stop", "continue", "refill"]):
        actions = [k for k in ["increase", "decrease", "start", "stop", "continue", "refill"] if k in lowered]
        slots["actions"] = actions
        if actions:
            slots["action"] = actions[0]
    elif any(k in lowered for k in ["taking ", "takes ", "on "]):
        slots["action"] = "continue"
    elif any(k in lowered for k in ["prescribes", "prescribe", "new prescription"]):
        slots["action"] = "prescribe"

    if safe_text(rec.get("temporality")):
        slots["temporality"] = safe_text(rec.get("temporality"))
    if safe_text(rec.get("status")):
        slots["assertion"] = safe_text(rec.get("status"))
    return slots


def event_signature_from_cap(rec: Dict[str, Any], event_type: str) -> Tuple[str, str]:
    slots = extract_key_slots_from_cap(rec)
    concept = safe_text(rec.get("canonical_concept"))
    visit_state = safe_text(rec.get("visit_state"))
    speaker = safe_text(rec.get("speaker"))
    predicate = safe_text(rec.get("predicate"))
    primary: List[str] = []
    if concept:
        primary.append(concept)
    for key in ["medication", "laterality"]:
        if key in slots:
            primary.append(str(slots[key]))
    for key in ["body_sites", "actions", "values"]:
        values = slots.get(key)
        if isinstance(values, list):
            primary.extend([str(v) for v in values[:2]])
    if event_type in {"MedicationState", "MedicationChange"}:
        primary.extend([speaker or "unknown", predicate or "unknown", visit_state or "unknown"])
    elif event_type in {"PlanOrder", "FollowUp", "PatientInstruction"}:
        primary.extend([predicate or "unknown"])
    elif event_type == "SymptomCourse" and visit_state:
        primary.append(visit_state)
    if not primary:
        primary = event_concept_tokens(safe_text(rec.get("proposition_text")))[:4]
    signature = " ".join(primary).strip() or safe_text(rec.get("proposition_text")).lower()[:60]
    return event_type, signature


def build_event_title(event_type: str, slots: Dict[str, Any], anchor: str) -> str:
    if slots.get("medication") and event_type in {"MedicationState", "MedicationChange"}:
        action = ""
        if isinstance(slots.get("actions"), list) and slots["actions"]:
            action = f" {'/'.join(slots['actions'][:2])}"
        return f"{slots['medication']}{action}".strip()
    if isinstance(slots.get("body_sites"), list) and slots["body_sites"]:
        return f"{event_type}: {'/'.join(slots['body_sites'][:2])}"
    anchor_tokens = event_concept_tokens(anchor)[:4]
    if anchor_tokens:
        return " ".join(anchor_tokens)
    return event_type


def classify_bundle_statement(text: str) -> str:
    lowered = normalize_text(text)
    if not lowered:
        return "supporting_evidence"
    if any(
        token in lowered
        for token in (
            "return precaution",
            "return if",
            "go to er",
            "go to the er",
            "seek care",
            "call back",
            "monitor",
            "keep a log",
            "bp log",
            "blood pressure log",
            "hydration",
            "push fluids",
            "diet",
            "low salt",
            "counsel",
            "advised",
            "encouraged",
        )
    ):
        return "care_context"
    if any(
        token in lowered
        for token in (
            "prescribe",
            "start ",
            "stop ",
            "continue ",
            "increase",
            "decrease",
            "adjust",
            "order",
            "refer",
            "referral",
            "follow up",
            "repeat",
            "schedule",
            "mri",
            "ct",
            "x ray",
            "ultrasound",
            "lab",
            "labs",
        )
    ):
        return "planned_actions"
    if any(
        token in lowered
        for token in (
            "taking ",
            "currently taking",
            "is on ",
            "history of",
            "past medical history",
            "allergy",
            "allergic",
        )
    ):
        return "current_state"
    return "supporting_evidence"


def build_care_bundle_fields(
    event_type: str,
    *,
    canonical_concept: str,
    title: str,
    summary: str,
    statements: Sequence[str],
) -> Dict[str, Any]:
    anchor_problem = canonical_concept or title or None
    supporting_evidence: List[str] = []
    current_state: List[str] = []
    planned_actions: List[str] = []
    care_context: List[str] = []

    for text in statements:
        if not text:
            continue
        bucket = classify_bundle_statement(text)
        if event_type in {"MedicationChange", "PlanOrder", "FollowUp"}:
            bucket = "planned_actions"
        elif event_type in {"MedicationState", "PMH"}:
            bucket = "current_state"
        elif event_type == "PatientInstruction":
            bucket = "care_context"
        elif event_type in {"ExamFinding", "TestResult", "SymptomCourse", "ChiefComplaint", "DiagnosisImpression"}:
            bucket = "supporting_evidence"
        if bucket == "current_state" and text not in current_state:
            current_state.append(text)
        elif bucket == "planned_actions" and text not in planned_actions:
            planned_actions.append(text)
        elif bucket == "care_context" and text not in care_context:
            care_context.append(text)
        elif text not in supporting_evidence:
            supporting_evidence.append(text)

    if not supporting_evidence and summary:
        if event_type in {"MedicationState", "PMH"}:
            current_state = current_state or [summary]
        elif event_type in {"MedicationChange", "PlanOrder", "FollowUp"}:
            planned_actions = planned_actions or [summary]
        elif event_type == "PatientInstruction":
            care_context = care_context or [summary]
        else:
            supporting_evidence = [summary]

    return {
        "anchor_problem": anchor_problem,
        "supporting_evidence": supporting_evidence[:4],
        "current_state": current_state[:4],
        "planned_actions": planned_actions[:4],
        "care_context": care_context[:4],
    }


def summarize_event_group(event_type: str, members: List[Dict[str, Any]]) -> Tuple[str, Dict[str, Any], List[str]]:
    anchor = safe_text(members[0].get("proposition_text"))
    merged_slots: Dict[str, Any] = {}
    statements: List[str] = []
    for rec in members:
        text = safe_text(rec.get("proposition_text"))
        if text and text not in statements:
            statements.append(text)
        slots = extract_key_slots_from_cap(rec)
        for key, value in slots.items():
            if key not in merged_slots:
                merged_slots[key] = value
            elif isinstance(value, list) and isinstance(merged_slots[key], list):
                merged_slots[key] = sorted(dict.fromkeys(merged_slots[key] + value))
    if event_type == "MedicationChange":
        medication = safe_text(merged_slots.get("medication"))
        dose_options = []
        freq_options = []
        for rec in members:
            slots = extract_key_slots_from_cap(rec)
            if safe_text(slots.get("dose")):
                dose_options.append(safe_text(slots.get("dose")))
            if safe_text(slots.get("frequency")):
                freq_options.append(safe_text(slots.get("frequency")))
        dose_options = sorted(dict.fromkeys(dose_options))
        freq_options = sorted(dict.fromkeys(freq_options))
        if medication and len(dose_options) <= 1:
            summary = f"The clinician plans a medication change for {medication}."
            if dose_options:
                summary = f"The clinician plans to adjust {medication} to {dose_options[0]}."
            if freq_options:
                summary = summary.rstrip(".") + f" ({freq_options[0]})."
        elif medication:
            merged_slots["dose_options"] = dose_options
            merged_slots["frequency_options"] = freq_options
            summary = f"The clinician plans a dosing update for {medication}."
        else:
            summary = "; ".join(statements[:2])
    elif event_type == "MedicationState":
        medication = safe_text(merged_slots.get("medication"))
        dose = safe_text(merged_slots.get("dose"))
        frequency = safe_text(merged_slots.get("frequency"))
        if medication:
            summary = f"The patient is currently taking {medication}"
            if dose:
                summary += f" {dose}"
            if frequency:
                summary += f" {frequency}"
            summary += "."
        else:
            summary = "; ".join(statements[:2])
    elif event_type in {"PlanOrder", "FollowUp", "PatientInstruction"}:
        summary = "; ".join(statements[:3])
    elif event_type in {"ExamFinding", "TestResult"}:
        summary = "; ".join(statements[:3])
    else:
        summary = "; ".join(statements[:2])
    return summary or anchor, merged_slots, statements


def build_deterministic_event_plan(cap_obj: Dict[str, Any]) -> Dict[str, Any]:
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for rec in cap_obj.get("atomic_propositions", []):
        event_type = event_type_from_cap(rec)
        signature = event_signature_from_cap(rec, event_type)
        grouped.setdefault(signature, []).append(rec)

    events: List[Dict[str, Any]] = []
    for idx, ((event_type, _signature), members) in enumerate(grouped.items(), start=1):
        summary, merged_slots, statements = summarize_event_group(event_type, members)
        representative = members[0]
        if event_type == "MedicationState":
            dosage_values = []
            for rec in members:
                dosage_values.extend(re.findall(r"\b\d+(?:\.\d+)?\s*(?:mg|mcg|g)\b", safe_text(rec.get("proposition_text")).lower()))
            if len(sorted(dict.fromkeys(dosage_values))) >= 2:
                event_type = "MedicationChange"
        title = build_event_title(event_type, merged_slots, summary)
        care_bundle = build_care_bundle_fields(
            event_type,
            canonical_concept=safe_text(representative.get("canonical_concept")) or title,
            title=title,
            summary=summary,
            statements=statements,
        )
        event = {
            "event_id": f"E{idx}",
            "event_type": event_type,
            "canonical_concept": safe_text(representative.get("canonical_concept")) or title,
            "visit_state": safe_text(representative.get("visit_state")) or infer_visit_state(summary, representative.get("status"), representative.get("temporality")),
            "status": safe_text(representative.get("status")) or "affirmed",
            "temporality": safe_text(representative.get("temporality")) or "current",
            "speaker_source": safe_text(representative.get("speaker")) or infer_cap_speaker(summary) or "mixed",
            "clinical_priority": event_priority_from_type(event_type),
            "title": title,
            "event_summary": summary,
            "supporting_cap_ids": [safe_text(x.get("prop_id")) for x in members if safe_text(x.get("prop_id"))],
            "anchor_problem": care_bundle.get("anchor_problem"),
            "supporting_evidence": care_bundle.get("supporting_evidence"),
            "current_state": care_bundle.get("current_state"),
            "planned_actions": care_bundle.get("planned_actions"),
            "care_context": care_bundle.get("care_context"),
            "key_slots": merged_slots,
            "render_sections": event_render_sections(event_type),
        }
        events.append({k: v for k, v in event.items() if v not in (None, [], {}, "")})
        if len(events) >= EVENT_MAX_ITEMS:
            break
    return {"events": events}


def normalize_event_plan(obj: Any, *, max_events: int = EVENT_MAX_ITEMS) -> Dict[str, Any]:
    if not isinstance(obj, dict):
        return {"events": []}
    events: List[Dict[str, Any]] = []
    raw_events = obj.get("events")
    if isinstance(raw_events, list):
        for idx, item in enumerate(raw_events, start=1):
            if not isinstance(item, dict):
                continue
            event_summary = safe_text(item.get("event_summary"))
            if not event_summary:
                continue
            normalized = {
                "event_id": safe_text(item.get("event_id") or f"E{idx}") or f"E{idx}",
                "event_type": safe_text(item.get("event_type")) or None,
                "canonical_concept": safe_text(item.get("canonical_concept")) or None,
                "visit_state": safe_text(item.get("visit_state")) or None,
                "status": safe_text(item.get("status")) or None,
                "temporality": safe_text(item.get("temporality")) or None,
                "speaker_source": safe_text(item.get("speaker_source")) or None,
                "clinical_priority": safe_text(item.get("clinical_priority")) or None,
                "title": safe_text(item.get("title")) or None,
                "event_summary": event_summary,
                "supporting_cap_ids": [
                    safe_text(x) for x in (item.get("supporting_cap_ids") or []) if safe_text(x)
                ],
                "anchor_problem_cap_id": safe_text(item.get("anchor_problem_cap_id")) or None,
                "anchor_problem": safe_text(item.get("anchor_problem")) or safe_text(item.get("canonical_concept")) or None,
                "S_cap_ids": [safe_text(x) for x in (item.get("S_cap_ids") or []) if safe_text(x)],
                "O_cap_ids": [safe_text(x) for x in (item.get("O_cap_ids") or []) if safe_text(x)],
                "A_cap_ids": [safe_text(x) for x in (item.get("A_cap_ids") or []) if safe_text(x)],
                "P_cap_ids": [safe_text(x) for x in (item.get("P_cap_ids") or []) if safe_text(x)],
                "global_cap_ids": [safe_text(x) for x in (item.get("global_cap_ids") or []) if safe_text(x)],
                "supporting_evidence": [safe_text(x) for x in (item.get("supporting_evidence") or []) if safe_text(x)],
                "current_state": [safe_text(x) for x in (item.get("current_state") or []) if safe_text(x)],
                "planned_actions": [safe_text(x) for x in (item.get("planned_actions") or []) if safe_text(x)],
                "care_context": [safe_text(x) for x in (item.get("care_context") or []) if safe_text(x)],
                "key_slots": item.get("key_slots") if isinstance(item.get("key_slots"), dict) else {},
                "render_sections": [safe_text(x) for x in (item.get("render_sections") or []) if safe_text(x)],
            }
            events.append({k: v for k, v in normalized.items() if v not in (None, [], {}, "")})
            if len(events) >= max_events:
                break
    out = {"events": events}
    runtime_seconds = obj.get("runtime_seconds")
    if isinstance(runtime_seconds, (int, float)):
        out["runtime_seconds"] = runtime_seconds
    return out


def event_plan_uses_problem_oriented_bundles(event_plan: Dict[str, Any]) -> bool:
    events = event_plan.get("events")
    if not isinstance(events, list) or not events:
        return False
    for item in events:
        if not isinstance(item, dict):
            continue
        key_slots = item.get("key_slots") if isinstance(item.get("key_slots"), dict) else {}
        if safe_text(key_slots.get("bundle_design")) == "problem_oriented_note_bundle":
            return True
    return False


def split_bullet_facts(text: str) -> List[str]:
    lines: List[str] = []
    for raw in safe_text(text).splitlines():
        line = re.sub(r"^\s*[*\-•]+\s*", "", raw).strip()
        if line:
            lines.append(line)
    return lines


def format_fact_lines(text: str) -> str:
    facts = split_bullet_facts(text)
    if not facts:
        return "(no facts available)"
    return "\n".join(f"- {fact}" for fact in facts)


def format_caps_for_prompt(cap_obj: Dict[str, Any]) -> str:
    lines: List[str] = []
    for idx, item in enumerate(cap_obj.get("atomic_propositions", []), start=1):
        cap_type = safe_text(item.get("cap_type") or item.get("category")) or "NA"
        concept = safe_text(item.get("canonical_concept")) or "NA"
        category = safe_text(item.get("category")) or "NA"
        speaker = safe_text(item.get("content_source") or item.get("speaker")) or "NA"
        modality = safe_text(item.get("modality")) or "NA"
        predicate = safe_text(item.get("predicate")) or "NA"
        status = safe_text(item.get("status")) or "NA"
        visit_state = safe_text(item.get("visit_state")) or "NA"
        temporality = safe_text(item.get("temporality")) or "NA"
        text = safe_text(item.get("proposition_text"))
        lines.append(
            f"P{idx} | cap_type={cap_type} | concept={concept} | category={category} | content_source={speaker} | modality={modality} | "
            f"predicate={predicate} | verification_status={status} | clinical_status={visit_state} | temporality={temporality} | text={text}"
        )
    return "\n".join(lines) if lines else "(no CAPs available)"


def collect_event_plan_cap_ids(event_plan: Optional[Dict[str, Any]], *, include_global: bool = True) -> List[str]:
    if not isinstance(event_plan, dict):
        return []
    ordered_ids: List[str] = []
    seen: set[str] = set()
    for item in event_plan.get("events", []):
        if not isinstance(item, dict):
            continue
        keys = [
            "anchor_problem_cap_id",
            "supporting_cap_ids",
            "S_cap_ids",
            "O_cap_ids",
            "A_cap_ids",
            "P_cap_ids",
        ]
        if include_global:
            keys.append("global_cap_ids")
        for key in keys:
            values = item.get(key)
            if isinstance(values, list):
                for cap_id in values:
                    cap_id = safe_text(cap_id)
                    if cap_id and cap_id not in seen:
                        seen.add(cap_id)
                        ordered_ids.append(cap_id)
            else:
                cap_id = safe_text(values)
                if cap_id and cap_id not in seen:
                    seen.add(cap_id)
                    ordered_ids.append(cap_id)
    return ordered_ids


def filter_event_plan_for_render(event_plan: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(event_plan, dict):
        return {"events": []}
    events: List[Dict[str, Any]] = []
    for item in event_plan.get("events", []):
        if not isinstance(item, dict):
            continue
        key_slots = item.get("key_slots") if isinstance(item.get("key_slots"), dict) else {}
        if safe_text(key_slots.get("render_in_note")).lower() == "false":
            continue
        events.append(item)
    out = {**event_plan, "events": events}
    return out


def filter_cap_obj_to_ids(cap_obj: Dict[str, Any], cap_ids: Sequence[str]) -> Dict[str, Any]:
    wanted = {safe_text(cap_id) for cap_id in cap_ids if safe_text(cap_id)}
    if not wanted:
        return {"atomic_propositions": []}
    return {
        **cap_obj,
        "atomic_propositions": [
            item
            for item in cap_obj.get("atomic_propositions", [])
            if isinstance(item, dict) and safe_text(item.get("prop_id")) in wanted
        ],
    }


def format_event_plan_for_prompt(event_plan: Dict[str, Any]) -> str:
    lines: List[str] = []
    for idx, item in enumerate(event_plan.get("events", []), start=1):
        event_id = safe_text(item.get("event_id") or f"E{idx}")
        event_type = safe_text(item.get("event_type")) or "NA"
        canonical_concept = safe_text(item.get("canonical_concept")) or "NA"
        visit_state = safe_text(item.get("visit_state")) or "NA"
        status = safe_text(item.get("status")) or "NA"
        temporality = safe_text(item.get("temporality")) or "NA"
        priority = safe_text(item.get("clinical_priority")) or "NA"
        title = safe_text(item.get("title")) or "NA"
        summary = safe_text(item.get("event_summary"))
        support = ", ".join(item.get("supporting_cap_ids") or []) or "NA"
        anchor_problem_cap_id = safe_text(item.get("anchor_problem_cap_id")) or "NA"
        s_cap_ids = ", ".join(item.get("S_cap_ids") or []) or "NA"
        o_cap_ids = ", ".join(item.get("O_cap_ids") or []) or "NA"
        a_cap_ids = ", ".join(item.get("A_cap_ids") or []) or "NA"
        p_cap_ids = ", ".join(item.get("P_cap_ids") or []) or "NA"
        global_cap_ids = ", ".join(item.get("global_cap_ids") or []) or "NA"
        render_sections = ", ".join(item.get("render_sections") or []) or "NA"
        anchor_problem = safe_text(item.get("anchor_problem")) or canonical_concept
        supporting_evidence = "; ".join(item.get("supporting_evidence") or []) or "NA"
        current_state = "; ".join(item.get("current_state") or []) or "NA"
        planned_actions = "; ".join(item.get("planned_actions") or []) or "NA"
        care_context = "; ".join(item.get("care_context") or []) or "NA"
        key_slots = item.get("key_slots") or {}
        slots_text = ", ".join(f"{safe_text(k)}={safe_text(v)}" for k, v in key_slots.items()) or "NA"
        lines.append(
            f"{event_id} | type={event_type} | concept={canonical_concept} | clinical_status={visit_state} | verification_status={status} | temporality={temporality} | "
            f"priority={priority} | title={title} | sections={render_sections} | support={support} | "
            f"anchor_problem={anchor_problem} | anchor_problem_cap_id={anchor_problem_cap_id} | "
            f"S_cap_ids={s_cap_ids} | O_cap_ids={o_cap_ids} | A_cap_ids={a_cap_ids} | P_cap_ids={p_cap_ids} | global_cap_ids={global_cap_ids} | "
            f"supporting_evidence={supporting_evidence} | current_state={current_state} | "
            f"planned_actions={planned_actions} | care_context={care_context} | slots={slots_text} | summary={summary}"
        )
    return "\n".join(lines) if lines else "(no events available)"


def build_prompt_for_method(
    method_key: str,
    template_key: str,
    *,
    transcript: str,
    medsum_fact_text: str,
    cluster_fact_text: str,
    cap_obj: Dict[str, Any],
    event_plan: Optional[Dict[str, Any]],
) -> str:
    template_instruction = TEMPLATE_INSTRUCTIONS[template_key]
    render_event_plan = filter_event_plan_for_render(event_plan) if event_plan else event_plan
    event_linked_cap_obj = (
        filter_cap_obj_to_ids(cap_obj, collect_event_plan_cap_ids(render_event_plan, include_global=False))
        if render_event_plan
        else cap_obj
    )
    if method_key == "direct":
        return DIRECT_PROMPT_BASE.format(template_instruction=template_instruction, transcript=transcript)
    if method_key == "medsum_ent":
        return MEDSUM_ENT_PROMPT_BASE.format(
            template_instruction=template_instruction,
            transcript=transcript,
            source_block=FACT_SOURCE_BLOCK.format(fact_lines=format_fact_lines(medsum_fact_text)),
        )
    if method_key == "cluster2sent":
        return CLUSTER2SENT_PROMPT_BASE.format(
            template_instruction=template_instruction,
            source_block=FACT_SOURCE_BLOCK.format(fact_lines=format_fact_lines(cluster_fact_text)),
        )
    if method_key == "cap":
        return CAP_PROMPT_BASE.format(
            template_instruction=template_instruction,
            transcript=transcript,
            source_block=CAP_SOURCE_BLOCK.format(cap_lines=format_caps_for_prompt(cap_obj)),
        )
    if method_key == "cap_only":
        return CAP_ONLY_PROMPT_BASE.format(
            template_instruction=template_instruction,
            source_block=CAP_SOURCE_BLOCK.format(cap_lines=format_caps_for_prompt(cap_obj)),
        )
    if method_key == "cap_event":
        return CAP_EVENT_RENDER_PROMPT_BASE.format(
            template_instruction=template_instruction,
            transcript=transcript,
            cap_block=CAP_SOURCE_BLOCK.format(cap_lines=format_caps_for_prompt(event_linked_cap_obj)),
            source_block=EVENT_SOURCE_BLOCK.format(event_lines=format_event_plan_for_prompt(render_event_plan or {})),
        )
    if method_key == "cap_event_only":
        return CAP_EVENT_ONLY_RENDER_PROMPT_BASE.format(
            template_instruction=template_instruction,
            cap_block=CAP_SOURCE_BLOCK.format(cap_lines=format_caps_for_prompt(event_linked_cap_obj)),
            source_block=EVENT_SOURCE_BLOCK.format(event_lines=format_event_plan_for_prompt(render_event_plan or {})),
        )
    raise ValueError(f"Unsupported method: {method_key}")


def normalize_generated_summary(text: str) -> str:
    cleaned = safe_text(text)
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:text|markdown)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    lines = [ln.rstrip() for ln in cleaned.splitlines()]
    preamble_patterns = (
        r"^here(?:'s|’s| is) (?:a |the )?(?:concise )?(?:clinical note|soap note|medical note|summary)(?: summarizing the (?:patient )?encounter)?(?:, based on the provided information)?[:,]?\s*$",
        r"^here(?:'s|’s| is) the clinical note(?:, generated from the provided transcript and evidence units)?(?:, based on the provided (?:transcript|information|transcript and clinical atomic propositions))?[:,]?\s*$",
        r"^below is (?:a |the )?(?:clinical note|soap note|summary)[:,]?\s*$",
    )
    while lines and any(re.match(pattern, lines[0].strip(), flags=re.IGNORECASE) for pattern in preamble_patterns):
        lines.pop(0)
        while lines and not lines[0].strip():
            lines.pop(0)
    cleaned = "\n".join(lines).strip()
    return cleaned


def cleanup_note_line_style(line: str) -> str:
    cleaned = safe_text(line).strip()
    cleaned = re.sub(r"^\s*[,;:]+\s*", "", cleaned)
    cleaned = re.sub(r"^\s*(?:and|but|also)\s+", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned:
        return ""
    if cleaned[0].islower():
        cleaned = cleaned[0].upper() + cleaned[1:]
    return cleaned


def reduce_repetitive_subject_openers(lines: List[str]) -> List[str]:
    """Reduce repetitive 'The patient ...' / 'The doctor ...' openings for readability."""
    if not lines:
        return lines

    rewrite_rules = [
        (r"^\s*The patient reports\s+", "Reports "),
        (r"^\s*The patient states\s+", "States "),
        (r"^\s*The patient denies\s+", "Denies "),
        (r"^\s*The patient has\s+", "Has "),
        (r"^\s*The patient is\s+", "Is "),
        (r"^\s*The patient underwent\s+", "Underwent "),
        (r"^\s*The patient should\s+", "Should "),
        (r"^\s*The patient will\s+", "Will "),
        (r"^\s*The patient can\s+", "Can "),
        (r"^\s*The patient's\s+", "Patient's "),
        (r"^\s*The doctor recommends\s+", "Recommend "),
        (r"^\s*The doctor orders\s+", "Order "),
        (r"^\s*The doctor discussed\s+", "Discussed "),
    ]

    out: List[str] = []
    for idx, line in enumerate(lines):
        rewritten = line
        if idx > 0:
            for pattern, replacement in rewrite_rules:
                new_line = re.sub(pattern, replacement, rewritten, flags=re.IGNORECASE)
                if new_line != rewritten:
                    rewritten = new_line
                    break
        rewritten = cleanup_note_line_style(rewritten)
        out.append(rewritten)
    return out


def normalize_note_section_text(text: Any, *, template_key: str, section_label: str) -> str:
    raw_items: List[str]
    if isinstance(text, list):
        raw_items = [safe_text(item) for item in text]
    else:
        raw_items = [safe_text(text)]

    deduped_items: List[str] = []
    seen: set[str] = set()
    for item in raw_items:
        cleaned = normalize_generated_summary(item)
        cleaned = re.sub(r"^\s*\*\*(.+?)\*\*\s*$", r"\1", cleaned, flags=re.MULTILINE)
        cleaned = re.sub(rf"^\s*{re.escape(section_label)}\s*:?\s*", "", cleaned, flags=re.IGNORECASE)
        lines = [re.sub(r"^\s*[*\-•]+\s*", "", ln).strip() for ln in cleaned.splitlines()]
        for line in lines:
            line = cleanup_note_line_style(line)
            norm = normalize_text(line)
            if not line or not norm or norm in seen:
                continue
            seen.add(norm)
            deduped_items.append(line)

    deduped_items = reduce_repetitive_subject_openers(deduped_items)

    if template_key == "soap":
        return "\n".join(deduped_items).strip()
    return "\n".join(deduped_items).strip()


def render_structured_note(template_key: str, obj: Dict[str, Any]) -> str:
    if template_key == "soap":
        sections = [
            (
                "S:",
                normalize_note_section_text(
                    obj.get("S", obj.get("subjective")),
                    template_key=template_key,
                    section_label="S",
                ),
            ),
            (
                "O:",
                normalize_note_section_text(
                    obj.get("O", obj.get("objective")),
                    template_key=template_key,
                    section_label="O",
                ),
            ),
            (
                "A:",
                normalize_note_section_text(
                    obj.get("A", obj.get("assessment")),
                    template_key=template_key,
                    section_label="A",
                ),
            ),
            (
                "P:",
                normalize_note_section_text(
                    obj.get("P", obj.get("plan")),
                    template_key=template_key,
                    section_label="P",
                ),
            ),
        ]
    elif template_key == "sectioned":
        sections = [
            (
                "Chief Complaint",
                normalize_note_section_text(
                    obj.get("Chief Complaint", obj.get("chief_complaint")),
                    template_key=template_key,
                    section_label="Chief Complaint",
                ),
            ),
            (
                "History of Present Illness",
                normalize_note_section_text(
                    obj.get("History of Present Illness", obj.get("history_of_present_illness")),
                    template_key=template_key,
                    section_label="History of Present Illness",
                ),
            ),
            (
                "Findings",
                normalize_note_section_text(
                    obj.get("Findings", obj.get("findings")),
                    template_key=template_key,
                    section_label="Findings",
                ),
            ),
            (
                "Assessment",
                normalize_note_section_text(
                    obj.get("Assessment", obj.get("assessment")),
                    template_key=template_key,
                    section_label="Assessment",
                ),
            ),
            (
                "Plan",
                normalize_note_section_text(
                    obj.get("Plan", obj.get("plan")),
                    template_key=template_key,
                    section_label="Plan",
                ),
            ),
        ]
    else:
        return normalize_generated_summary(json.dumps(obj, ensure_ascii=False))

    parts: List[str] = []
    for header, body in sections:
        if template_key == "soap":
            parts.append(f"{header}\n{body}".rstrip())
        else:
            parts.append(f"{header}\n{body}".rstrip())
    return "\n\n".join(parts).strip()


def normalize_template_summary(summary_text: str, template_key: str) -> str:
    cleaned = normalize_generated_summary(summary_text)
    if template_key == "soap":
        replacements = {
            "**S**": "S:",
            "**O**": "O:",
            "**A**": "A:",
            "**P**": "P:",
        }
        for src, dst in replacements.items():
            cleaned = cleaned.replace(src, dst)
        cleaned = re.sub(r"^\s*\*\*(S|O|A|P)\*\*\s*$", r"\1:", cleaned, flags=re.MULTILINE)
    elif template_key == "sectioned":
        section_headers = [
            "Chief Complaint",
            "History of Present Illness",
            "Findings",
            "Assessment",
            "Plan",
        ]
        for header in section_headers:
            cleaned = re.sub(
                rf"^\s*\*\*{re.escape(header)}\*\*\s*$",
                header,
                cleaned,
                flags=re.MULTILINE,
            )
    return cleaned.strip()


def summary_matches_template(summary_text: str, template_key: str) -> bool:
    text = normalize_template_summary(summary_text, template_key)
    if template_key == "soap":
        return bool(re.search(r"(?m)^S:\s*$", text)) and bool(re.search(r"(?m)^O:\s*$", text)) and bool(re.search(r"(?m)^A:\s*$", text)) and bool(re.search(r"(?m)^P:\s*$", text))
    if template_key == "sectioned":
        headers = [
            "Chief Complaint",
            "History of Present Illness",
            "Findings",
            "Assessment",
            "Plan",
        ]
        return all(re.search(rf"(?m)^{re.escape(header)}\s*$", text) for header in headers)
    return True


def summary_has_pathological_repetition(summary_text: str) -> bool:
    text = normalize_generated_summary(summary_text)
    sentences = [
        normalize_text(part)
        for part in re.split(r"(?<=[.!?])\s+", text)
        if normalize_text(part)
    ]
    counts: Dict[str, int] = {}
    for sent in sentences:
        counts[sent] = counts.get(sent, 0) + 1
        if counts[sent] >= 3:
            return True
    return False


def split_summary_for_cap_extraction(summary_text: str, *, max_chars: int = 1200, max_lines: int = 8) -> List[str]:
    lines = [ln.strip() for ln in safe_text(summary_text).splitlines() if ln.strip()]
    if not lines:
        return [safe_text(summary_text)]
    chunks: List[str] = []
    current: List[str] = []
    current_chars = 0
    for line in lines:
        projected = current_chars + len(line) + (1 if current else 0)
        if current and (len(current) >= max_lines or projected > max_chars):
            chunks.append("\n".join(current))
            current = [line]
            current_chars = len(line)
        else:
            current.append(line)
            current_chars = projected
    if current:
        chunks.append("\n".join(current))
    return chunks or [safe_text(summary_text)]


def merge_cap_objects(cap_objs: Sequence[Dict[str, Any]], *, max_props: int = CAP_MAX_PROPS) -> Dict[str, Any]:
    merged_props: List[Dict[str, Any]] = []
    merged_filtered: List[str] = []
    seen_keys: set[Tuple[str, str, str, str]] = set()
    for cap_obj in cap_objs:
        for item in cap_obj.get("atomic_propositions", []):
            text = safe_text(item.get("proposition_text"))
            if not text:
                continue
            key = (
                normalize_text(text),
                safe_text(item.get("status")).lower(),
                safe_text(item.get("temporality")).lower(),
                safe_text(item.get("speaker")).lower(),
            )
            if key in seen_keys:
                continue
            seen_keys.add(key)
            merged_props.append(item)
            if len(merged_props) >= max_props:
                break
        for unit in cap_obj.get("filtered_nonverifiable_units", []):
            unit_text = safe_text(unit)
            if unit_text and unit_text not in merged_filtered:
                merged_filtered.append(unit_text)
        if len(merged_props) >= max_props:
            break
    for idx, item in enumerate(merged_props, start=1):
        item["prop_id"] = f"P{idx}"
    return {
        "atomic_propositions": merged_props[:max_props],
        "filtered_nonverifiable_units": merged_filtered,
    }


def split_note_for_gpt_f1(note_text: str, *, max_chars: int = 1600, max_lines: int = 10) -> List[str]:
    lines = [ln.strip() for ln in safe_text(note_text).splitlines() if ln.strip()]
    if not lines:
        return [safe_text(note_text)]
    chunks: List[str] = []
    current: List[str] = []
    current_chars = 0
    for line in lines:
        projected = current_chars + len(line) + (1 if current else 0)
        if current and (len(current) >= max_lines or projected > max_chars):
            chunks.append("\n".join(current))
            current = [line]
            current_chars = len(line)
        else:
            current.append(line)
            current_chars = projected
    if current:
        chunks.append("\n".join(current))
    return chunks or [safe_text(note_text)]


def split_transcript_for_safety(transcript_text: str, *, max_chars: int = 2600, max_lines: int = 24) -> List[str]:
    lines = [ln.strip() for ln in safe_text(transcript_text).splitlines() if ln.strip()]
    if not lines:
        return [safe_text(transcript_text)]
    chunks: List[str] = []
    current: List[str] = []
    current_chars = 0
    for line in lines:
        projected = current_chars + len(line) + (1 if current else 0)
        if current and (len(current) >= max_lines or projected > max_chars):
            chunks.append("\n".join(current))
            current = [line]
            current_chars = len(line)
        else:
            current.append(line)
            current_chars = projected
    if current:
        chunks.append("\n".join(current))
    return chunks or [safe_text(transcript_text)]


def split_note_into_heading_sections(
    note_text: str,
    *,
    max_chars: int = 1400,
    max_lines: int = 10,
) -> List[Tuple[str, str]]:
    lines = [ln.rstrip() for ln in safe_text(note_text).splitlines() if ln.strip()]
    if not lines:
        return [("Clinical Note", safe_text(note_text))]

    sections: List[Tuple[str, List[str]]] = []
    current_heading = "Clinical Note"
    current_lines: List[str] = []

    def flush_section() -> None:
        if current_lines:
            sections.append((current_heading, list(current_lines)))

    for line in lines:
        stripped = line.strip()
        is_heading = (
            stripped.endswith(":")
            and len(stripped) <= 80
            and not stripped.startswith("-")
            and not stripped.startswith("*")
            and stripped.count(":") == 1
        )
        if is_heading:
            flush_section()
            current_heading = stripped[:-1].strip() or "Clinical Note"
            current_lines = []
            continue
        current_lines.append(stripped)
    flush_section()

    chunked_sections: List[Tuple[str, str]] = []
    for heading, raw_lines in sections or [("Clinical Note", lines)]:
        current_chunk: List[str] = []
        current_chars = 0
        for line in raw_lines:
            projected = current_chars + len(line) + (1 if current_chunk else 0)
            if current_chunk and (len(current_chunk) >= max_lines or projected > max_chars):
                chunked_sections.append((heading, "\n".join(current_chunk)))
                current_chunk = [line]
                current_chars = len(line)
            else:
                current_chunk.append(line)
                current_chars = projected
        if current_chunk:
            chunked_sections.append((heading, "\n".join(current_chunk)))
    return chunked_sections or [("Clinical Note", safe_text(note_text))]


def normalize_string_list(items: Any, *, limit: Optional[int] = None) -> List[str]:
    if not isinstance(items, list):
        return []
    deduped: List[str] = []
    seen: set[str] = set()
    for item in items:
        text = safe_text(item)
        norm = normalize_text(text)
        if not text or not norm or norm in seen:
            continue
        seen.add(norm)
        deduped.append(text)
        if limit is not None and len(deduped) >= limit:
            break
    return deduped


def normalize_gpt_f1_items(obj: Any) -> Dict[str, List[str]]:
    categories = {key: [] for key in GPT_F1_CATEGORIES}
    if not isinstance(obj, dict):
        return categories
    raw_categories = obj.get("categories")
    if not isinstance(raw_categories, dict):
        return categories
    for key in GPT_F1_CATEGORIES:
        raw_items = raw_categories.get(key)
        if not isinstance(raw_items, list):
            continue
        deduped: List[str] = []
        seen: set[str] = set()
        for item in raw_items:
            text = safe_text(item)
            norm = normalize_text(text)
            if not text or not norm or norm in seen:
                continue
            seen.add(norm)
            deduped.append(text)
        categories[key] = deduped
    return categories


def merge_gpt_f1_category_objs(objs: Sequence[Dict[str, List[str]]]) -> Dict[str, List[str]]:
    merged = {key: [] for key in GPT_F1_CATEGORIES}
    seen = {key: set() for key in GPT_F1_CATEGORIES}
    for obj in objs:
        for key in GPT_F1_CATEGORIES:
            for item in obj.get(key, []):
                norm = normalize_text(item)
                if not norm or norm in seen[key]:
                    continue
                seen[key].add(norm)
                merged[key].append(item)
    return merged


def normalize_nair_extraction(obj: Any) -> Dict[str, List[str]]:
    return {"medical_concepts": normalize_string_list((obj or {}).get("medical_concepts"), limit=NAIR_MAX_CONCEPTS_PER_SECTION)}


def merge_nair_extractions(
    sections: Sequence[Tuple[str, str]],
    objs: Sequence[Dict[str, List[str]]],
    *,
    max_total_items: int = 96,
) -> Dict[str, Any]:
    merged_concepts: List[str] = []
    seen: set[str] = set()
    rendered_sections: List[Dict[str, Any]] = []
    for (heading, section_text), obj in zip(sections, objs):
        section_concepts = []
        for item in obj.get("medical_concepts", []):
            norm = normalize_text(item)
            if not norm:
                continue
            section_concepts.append(item)
            if norm in seen or len(merged_concepts) >= max_total_items:
                continue
            seen.add(norm)
            merged_concepts.append(item)
        rendered_sections.append(
            {
                "heading": heading,
                "section_value": section_text,
                "medical_concepts": section_concepts,
            }
        )
    return {
        "sections": rendered_sections,
        "medical_concepts": merged_concepts,
    }


def slice_cap_obj(cap_obj: Dict[str, Any], start: int, end: int) -> Dict[str, Any]:
    props = list(cap_obj.get("atomic_propositions", []))[start:end]
    return {
        "atomic_propositions": props,
        "filtered_nonverifiable_units": [],
    }


def chunk_cap_obj(cap_obj: Dict[str, Any], chunk_size: int) -> List[Dict[str, Any]]:
    props = list(cap_obj.get("atomic_propositions", []))
    if not props:
        return [{"atomic_propositions": [], "filtered_nonverifiable_units": []}]
    chunks: List[Dict[str, Any]] = []
    for idx in range(0, len(props), max(1, chunk_size)):
        chunks.append(slice_cap_obj(cap_obj, idx, idx + max(1, chunk_size)))
    return chunks


def bounded_workers(requested: int, item_count: int) -> int:
    return max(1, min(int(requested or 1), max(1, item_count)))


def extract_summary_caps(
    client: OpenAICompatClient,
    *,
    model: str,
    summary_text: str,
    max_tokens: int,
    temperature: float,
    chunk_workers: int = 1,
) -> Dict[str, Any]:
    schema_obj = cap_schema(CAP_MAX_PROPS)
    chunks = split_summary_for_cap_extraction(summary_text)
    cap_objs: List[Dict[str, Any]] = []
    parse_failures: List[Dict[str, Any]] = []
    per_chunk_max_tokens = max(700, min(max_tokens, 1200))
    per_chunk_max_props = max(8, min(CAP_MAX_PROPS, 14))
    def process_chunk(chunk_idx: int, chunk: str) -> Tuple[int, Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        prompt = (
            SUMMARY_TO_CAP_PROMPT.format(summary_text=chunk, max_props=per_chunk_max_props)
            + "\n\n"
            + CAP_EXTRACTION_GUIDANCE
        )
        try:
            raw = call_llm(
                client,
                model=model,
                prompt=prompt,
                max_tokens=per_chunk_max_tokens,
                temperature=temperature,
                force_json=True,
                json_schema_obj=schema_obj,
                prefer_json_object=True,
            )
        except Exception as exc:
            return chunk_idx, None, {"chunk_index": chunk_idx, "stage": "call_llm", "error": safe_text(exc)}
        try:
            parsed = safe_json_extract(raw)
        except Exception:
            try:
                parsed = repair_json_via_llm(
                    client,
                    model=model,
                    raw_text=raw,
                    schema_obj=schema_obj,
                    max_tokens=per_chunk_max_tokens,
                )
            except Exception as exc:
                return chunk_idx, None, {
                    "chunk_index": chunk_idx,
                    "stage": "repair_json",
                    "error": safe_text(exc),
                    "raw_excerpt": safe_text(raw)[:400],
                }
        return chunk_idx, normalize_cap_obj(parsed, max_props=per_chunk_max_props), None

    results: List[Tuple[int, Optional[Dict[str, Any]], Optional[Dict[str, Any]]]] = []
    with ThreadPoolExecutor(max_workers=bounded_workers(chunk_workers, len(chunks))) as executor:
        futures = [executor.submit(process_chunk, chunk_idx, chunk) for chunk_idx, chunk in enumerate(chunks, start=1)]
        for future in futures:
            results.append(future.result())
    for chunk_idx, cap_obj_chunk, failure in sorted(results, key=lambda x: x[0]):
        if failure:
            parse_failures.append(failure)
            continue
        if cap_obj_chunk:
            cap_objs.append(cap_obj_chunk)
    merged = merge_cap_objects(cap_objs, max_props=CAP_MAX_PROPS) if cap_objs else {
        "atomic_propositions": [],
        "filtered_nonverifiable_units": [],
    }
    merged["chunk_count"] = len(chunks)
    if parse_failures:
        merged["parse_failures"] = parse_failures
    return enrich_cap_obj(merged)


def build_event_plan(
    client: OpenAICompatClient,
    *,
    model: str,
    cap_obj: Dict[str, Any],
    max_tokens: int,
    temperature: float,
) -> Dict[str, Any]:
    enriched_cap = enrich_cap_obj(cap_obj)
    deterministic_plan = build_deterministic_event_plan(enriched_cap)
    if deterministic_plan.get("events"):
        return deterministic_plan

    schema_obj = event_plan_schema(EVENT_MAX_ITEMS)

    prompt = (
        CAP_TO_EVENT_PROMPT.format(
            max_events=EVENT_MAX_ITEMS,
            cap_lines=format_caps_for_prompt(enriched_cap),
        )
        + "\n\n"
        + EVENT_CLUSTERING_GUIDANCE
    )
    raw = call_llm(
        client,
        model=model,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        force_json=True,
        json_schema_obj=schema_obj,
    )
    try:
        parsed = safe_json_extract(raw)
    except Exception:
        parsed = repair_json_via_llm(
            client,
            model=model,
            raw_text=raw,
            schema_obj=schema_obj,
            max_tokens=max_tokens,
        )
    return normalize_event_plan(parsed, max_events=EVENT_MAX_ITEMS)


def extract_gpt_f1_items(
    client: OpenAICompatClient,
    *,
    model: str,
    note_text: str,
    max_tokens: int,
    temperature: float,
    chunk_workers: int = 1,
) -> Dict[str, List[str]]:
    schema_obj = gpt_f1_schema()
    chunks = split_note_for_gpt_f1(note_text)
    outputs: List[Dict[str, List[str]]] = []
    parse_failures: List[Dict[str, Any]] = []
    per_chunk_tokens = max(500, min(max_tokens, 900))
    def process_chunk(chunk_idx: int, chunk: str) -> Tuple[int, Optional[Dict[str, List[str]]], Optional[Dict[str, Any]]]:
        prompt = GPT_F1_EXTRACTION_PROMPT.format(note_text=chunk)
        try:
            raw = call_llm(
                client,
                model=model,
                prompt=prompt,
                max_tokens=per_chunk_tokens,
                temperature=temperature,
                force_json=True,
                json_schema_obj=schema_obj,
                prefer_json_object=True,
            )
        except Exception as exc:
            return chunk_idx, None, {"chunk_index": chunk_idx, "stage": "call_llm", "error": safe_text(exc)}
        try:
            parsed = safe_json_extract(raw)
        except Exception:
            try:
                parsed = repair_json_via_llm(
                    client,
                    model=model,
                    raw_text=raw,
                    schema_obj=schema_obj,
                    max_tokens=per_chunk_tokens,
                )
            except Exception as exc:
                return chunk_idx, None, {
                    "chunk_index": chunk_idx,
                    "stage": "repair_json",
                    "error": safe_text(exc),
                    "raw_excerpt": safe_text(raw)[:400],
                }
        return chunk_idx, normalize_gpt_f1_items(parsed), None

    results: List[Tuple[int, Optional[Dict[str, List[str]]], Optional[Dict[str, Any]]]] = []
    with ThreadPoolExecutor(max_workers=bounded_workers(chunk_workers, len(chunks))) as executor:
        futures = [executor.submit(process_chunk, chunk_idx, chunk) for chunk_idx, chunk in enumerate(chunks, start=1)]
        for future in futures:
            results.append(future.result())
    for chunk_idx, chunk_output, failure in sorted(results, key=lambda x: x[0]):
        if failure:
            parse_failures.append(failure)
            continue
        if chunk_output:
            outputs.append(chunk_output)
    merged = merge_gpt_f1_category_objs(outputs)
    merged["_chunk_count"] = [str(len(chunks))]
    if parse_failures:
        merged["_parse_failures"] = [json.dumps(x, ensure_ascii=False) for x in parse_failures]
    return merged


def extract_nair_medical_concepts(
    client: OpenAICompatClient,
    *,
    model: str,
    note_text: str,
    max_tokens: int,
    temperature: float,
    chunk_workers: int = 1,
) -> Dict[str, Any]:
    schema_obj = nair_concept_extraction_schema()
    sections = split_note_into_heading_sections(note_text)
    successful_sections: List[Tuple[str, str]] = []
    outputs: List[Dict[str, List[str]]] = []
    parse_failures: List[Dict[str, Any]] = []
    per_section_tokens = max(400, min(max_tokens, 800))
    def process_section(section_idx: int, heading: str, section_value: str) -> Tuple[int, Optional[Tuple[str, str]], Optional[Dict[str, List[str]]], Optional[Dict[str, Any]]]:
        prompt = NAIR_CONCEPT_EXTRACTION_PROMPT.format(
            section_heading=heading or "Clinical Note",
            section_value=section_value,
        )
        try:
            raw = call_llm(
                client,
                model=model,
                prompt=prompt,
                max_tokens=per_section_tokens,
                temperature=temperature,
                force_json=True,
                json_schema_obj=schema_obj,
                prefer_json_object=True,
            )
        except Exception as exc:
            return section_idx, None, None, {"section_index": section_idx, "stage": "call_llm", "error": safe_text(exc)}
        try:
            parsed = safe_json_extract(raw)
        except Exception:
            try:
                parsed = repair_json_via_llm(
                    client,
                    model=model,
                    raw_text=raw,
                    schema_obj=schema_obj,
                    max_tokens=per_section_tokens,
                )
            except Exception as exc:
                return section_idx, None, None, {
                    "section_index": section_idx,
                    "stage": "repair_json",
                    "error": safe_text(exc),
                    "raw_excerpt": safe_text(raw)[:400],
                }
        return section_idx, (heading, section_value), normalize_nair_extraction(parsed), None

    results: List[Tuple[int, Optional[Tuple[str, str]], Optional[Dict[str, List[str]]], Optional[Dict[str, Any]]]] = []
    with ThreadPoolExecutor(max_workers=bounded_workers(chunk_workers, len(sections))) as executor:
        futures = [
            executor.submit(process_section, section_idx, heading, section_value)
            for section_idx, (heading, section_value) in enumerate(sections, start=1)
        ]
        for future in futures:
            results.append(future.result())
    for section_idx, successful_section, output_obj, failure in sorted(results, key=lambda x: x[0]):
        if failure:
            parse_failures.append(failure)
            continue
        if successful_section and output_obj:
            successful_sections.append(successful_section)
            outputs.append(output_obj)
    merged = merge_nair_extractions(successful_sections, outputs)
    merged["section_count"] = len(sections)
    if parse_failures:
        merged["parse_failures"] = parse_failures
    return merged


def normalize_text(text: Any) -> str:
    text = safe_text(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> List[str]:
    return [tok for tok in normalize_text(text).split() if tok]


def token_f1(a: str, b: str) -> float:
    toks_a = tokenize(a)
    toks_b = tokenize(b)
    if not toks_a or not toks_b:
        return 0.0
    overlap = sum((Counter(toks_a) & Counter(toks_b)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(toks_a)
    recall = overlap / len(toks_b)
    return 2 * precision * recall / (precision + recall)


ALIGNMENT_STOPWORDS = {
    "the",
    "a",
    "an",
    "patient",
    "doctor",
    "clinician",
    "reports",
    "report",
    "reported",
    "states",
    "state",
    "notes",
    "noted",
    "has",
    "have",
    "had",
    "with",
    "for",
    "from",
    "that",
    "this",
    "there",
    "their",
    "them",
    "into",
    "current",
}


def normalize_alignment_text(text: str) -> str:
    norm = normalize_text(text)
    tokens: List[str] = []
    for tok in norm.split():
        if tok in ALIGNMENT_STOPWORDS:
            continue
        if tok.endswith("ies") and len(tok) > 4:
            tok = tok[:-3] + "y"
        elif tok.endswith("s") and len(tok) > 4 and not tok.endswith("ss"):
            tok = tok[:-1]
        tokens.append(tok)
    return " ".join(tokens)


def relaxed_token_f1(a: str, b: str) -> float:
    return token_f1(normalize_alignment_text(a), normalize_alignment_text(b))


def proposition_similarity(a: Dict[str, Any], b: Dict[str, Any]) -> float:
    text_a = safe_text(a.get("proposition_text"))
    text_b = safe_text(b.get("proposition_text"))
    concept_a = safe_text(a.get("canonical_concept")) or text_a
    concept_b = safe_text(b.get("canonical_concept")) or text_b
    score = 0.45 * token_f1(text_a, text_b)
    score += 0.35 * relaxed_token_f1(concept_a, concept_b)
    if concept_a and concept_b:
        norm_a = normalize_alignment_text(concept_a)
        norm_b = normalize_alignment_text(concept_b)
        if norm_a and norm_b and (norm_a == norm_b or norm_a in norm_b or norm_b in norm_a):
            score += 0.14
    for key, bonus in (
        ("category", 0.06),
        ("status", 0.05),
        ("visit_state", 0.05),
        ("temporality", 0.03),
        ("speaker", 0.03),
        ("predicate", 0.02),
    ):
        aval = safe_text(a.get(key)).lower()
        bval = safe_text(b.get(key)).lower()
        if aval and bval and aval == bval:
            score += bonus
    return min(score, 1.0)


def greedy_cap_alignment(
    source_cap: Dict[str, Any],
    summary_cap: Dict[str, Any],
    *,
    threshold: float = 0.58,
    strong_text_threshold: float = 0.72,
) -> Dict[str, Any]:
    source_props = source_cap.get("atomic_propositions", [])
    summary_props = summary_cap.get("atomic_propositions", [])
    candidate_pairs: List[Tuple[float, int, int]] = []
    for s_idx, s_prop in enumerate(summary_props):
        for t_idx, t_prop in enumerate(source_props):
            text_score = token_f1(
                safe_text(s_prop.get("proposition_text")),
                safe_text(t_prop.get("proposition_text")),
            )
            relaxed_score = relaxed_token_f1(
                safe_text(s_prop.get("canonical_concept")) or safe_text(s_prop.get("proposition_text")),
                safe_text(t_prop.get("canonical_concept")) or safe_text(t_prop.get("proposition_text")),
            )
            full_score = proposition_similarity(s_prop, t_prop)
            if full_score >= threshold or text_score >= strong_text_threshold or relaxed_score >= strong_text_threshold:
                candidate_pairs.append((max(full_score, text_score, relaxed_score), s_idx, t_idx))

    candidate_pairs.sort(reverse=True)
    used_summary = set()
    used_source = set()
    matches: List[Dict[str, Any]] = []
    for score, s_idx, t_idx in candidate_pairs:
        if s_idx in used_summary or t_idx in used_source:
            continue
        used_summary.add(s_idx)
        used_source.add(t_idx)
        matches.append(
            {
                "score": round(score, 4),
                "summary_prop_id": summary_props[s_idx].get("prop_id") or f"P{s_idx+1}",
                "source_prop_id": source_props[t_idx].get("prop_id") or f"P{t_idx+1}",
                "summary_text": safe_text(summary_props[s_idx].get("proposition_text")),
                "source_text": safe_text(source_props[t_idx].get("proposition_text")),
            }
        )

    precision = len(matches) / len(summary_props) if summary_props else 0.0
    recall = len(matches) / len(source_props) if source_props else 0.0
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {
        "cap_precision": precision,
        "cap_recall": recall,
        "cap_f1": f1,
        "matched_count": len(matches),
        "summary_prop_count": len(summary_props),
        "source_prop_count": len(source_props),
        "matched_pairs": matches,
        "unsupported_summary_props": [summary_props[i] for i in range(len(summary_props)) if i not in used_summary],
        "missing_source_props": [source_props[i] for i in range(len(source_props)) if i not in used_source],
    }


def greedy_text_alignment(
    pred_items: Sequence[str],
    ref_items: Sequence[str],
    *,
    threshold: float = 0.58,
) -> Dict[str, Any]:
    candidate_pairs: List[Tuple[float, int, int]] = []
    for p_idx, pred in enumerate(pred_items):
        for r_idx, ref in enumerate(ref_items):
            score = max(token_f1(pred, ref), relaxed_token_f1(pred, ref))
            if score >= threshold:
                candidate_pairs.append((score, p_idx, r_idx))
    candidate_pairs.sort(reverse=True)
    used_pred = set()
    used_ref = set()
    matches: List[Dict[str, Any]] = []
    for score, p_idx, r_idx in candidate_pairs:
        if p_idx in used_pred or r_idx in used_ref:
            continue
        used_pred.add(p_idx)
        used_ref.add(r_idx)
        matches.append(
            {
                "score": round(score, 4),
                "pred_text": pred_items[p_idx],
                "ref_text": ref_items[r_idx],
            }
        )
    precision = len(matches) / len(pred_items) if pred_items else 0.0
    recall = len(matches) / len(ref_items) if ref_items else 0.0
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "matched_count": len(matches),
        "pred_count": len(pred_items),
        "ref_count": len(ref_items),
        "matches": matches,
        "unsupported_pred_items": [pred_items[i] for i in range(len(pred_items)) if i not in used_pred],
        "missing_ref_items": [ref_items[i] for i in range(len(ref_items)) if i not in used_ref],
    }


def compute_gpt_f1_metrics(
    pred_categories: Dict[str, List[str]],
    ref_categories: Dict[str, List[str]],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    metrics: Dict[str, Any] = {}
    alignment: Dict[str, Any] = {"by_category": {}}
    macro_vals: List[float] = []
    for key in GPT_F1_CATEGORIES:
        result = greedy_text_alignment(pred_categories.get(key, []), ref_categories.get(key, []))
        alignment["by_category"][key] = result
        metrics[f"gptf1_{key}_precision"] = round(result["precision"], 4)
        metrics[f"gptf1_{key}_recall"] = round(result["recall"], 4)
        metrics[f"gptf1_{key}_f1"] = round(result["f1"], 4)
        metrics[f"gptf1_{key}_pred_count"] = result["pred_count"]
        metrics[f"gptf1_{key}_ref_count"] = result["ref_count"]
        if result["ref_count"] > 0 or result["pred_count"] > 0:
            macro_vals.append(result["f1"])
    metrics["gptf1_macro_f1"] = round(sum(macro_vals) / len(macro_vals), 4) if macro_vals else 0.0
    return metrics, alignment


def normalize_nair_verification_result(obj: Any) -> Dict[str, List[str]]:
    if not isinstance(obj, dict):
        return {"found_b": [], "not_found_b": []}
    return {
        "found_b": normalize_string_list(obj.get("found_b"), limit=NAIR_MAX_VERIFICATION_ITEMS),
        "not_found_b": normalize_string_list(obj.get("not_found_b"), limit=NAIR_MAX_VERIFICATION_ITEMS),
    }


def chunk_list(items: Sequence[str], size: int) -> List[List[str]]:
    if size <= 0:
        return [list(items)]
    return [list(items[idx : idx + size]) for idx in range(0, len(items), size)]


def verify_nair_concepts(
    client: OpenAICompatClient,
    *,
    model: str,
    snippet: str,
    list_a: Sequence[str],
    list_b: Sequence[str],
    max_tokens: int,
) -> Dict[str, Any]:
    if not list_b:
        return {"found_b": [], "not_found_b": [], "chunk_count": 0}
    schema_obj = nair_concept_verification_schema()
    found: List[str] = []
    not_found: List[str] = []
    found_seen: set[str] = set()
    not_found_seen: set[str] = set()
    chunk_results: List[Dict[str, Any]] = []
    for chunk in chunk_list(list_b, NAIR_MAX_VERIFICATION_ITEMS):
        prompt = NAIR_CONCEPT_VERIFICATION_PROMPT.format(
            snippet=snippet,
            list_a=json.dumps(list(list_a), ensure_ascii=False),
            list_b=json.dumps(chunk, ensure_ascii=False),
        )
        parsed = call_structured_judge(
            client,
            model=model,
            prompt=prompt,
            schema_obj=schema_obj,
            max_tokens=max(400, min(max_tokens, 900)),
        )
        normalized = normalize_nair_verification_result(parsed)
        chunk_results.append(normalized)
        found_norms = {normalize_text(x) for x in normalized["found_b"]}
        for item in chunk:
            norm = normalize_text(item)
            if not norm:
                continue
            if norm in found_norms:
                if norm not in found_seen:
                    found_seen.add(norm)
                    found.append(item)
            elif norm not in not_found_seen:
                not_found_seen.add(norm)
                not_found.append(item)
    return {
        "found_b": found,
        "not_found_b": not_found,
        "chunk_count": len(chunk_results),
        "chunk_results": chunk_results,
    }


def compute_nair_gpt_metrics(
    client: OpenAICompatClient,
    *,
    model: str,
    pred_note: str,
    ref_note: str,
    pred_concepts_obj: Dict[str, Any],
    ref_concepts_obj: Dict[str, Any],
    max_tokens: int,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    pred_concepts = normalize_string_list(pred_concepts_obj.get("medical_concepts"), limit=256)
    ref_concepts = normalize_string_list(ref_concepts_obj.get("medical_concepts"), limit=256)

    recall_alignment = verify_nair_concepts(
        client,
        model=model,
        snippet=pred_note,
        list_a=pred_concepts,
        list_b=ref_concepts,
        max_tokens=max_tokens,
    )
    precision_alignment = verify_nair_concepts(
        client,
        model=model,
        snippet=ref_note,
        list_a=ref_concepts,
        list_b=pred_concepts,
        max_tokens=max_tokens,
    )

    precision = len(precision_alignment["found_b"]) / len(pred_concepts) if pred_concepts else 0.0
    recall = len(recall_alignment["found_b"]) / len(ref_concepts) if ref_concepts else 0.0
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)

    metrics = {
        "nair_gpt_precision": round(precision, 4),
        "nair_gpt_recall": round(recall, 4),
        "nair_gpt_f1": round(f1, 4),
        "nair_pred_concept_count": len(pred_concepts),
        "nair_ref_concept_count": len(ref_concepts),
        "nair_precision_match_count": len(precision_alignment["found_b"]),
        "nair_recall_match_count": len(recall_alignment["found_b"]),
    }
    alignment = {
        "pred_concepts": pred_concepts,
        "ref_concepts": ref_concepts,
        "precision_alignment": precision_alignment,
        "recall_alignment": recall_alignment,
    }
    return metrics, alignment


def call_structured_judge(
    client: OpenAICompatClient,
    *,
    model: str,
    prompt: str,
    schema_obj: Dict[str, Any],
    max_tokens: int,
) -> Dict[str, Any]:
    raw = call_llm(
        client,
        model=model,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=0.0,
        force_json=True,
        json_schema_obj=schema_obj,
        prefer_json_object=True,
    )
    try:
        return safe_json_extract(raw)
    except Exception:
        return repair_json_via_llm(
            client,
            model=model,
            raw_text=raw,
            schema_obj=schema_obj,
            max_tokens=max_tokens,
        )


def section_appropriateness_schema(max_items: int = 40) -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "section_appropriateness_score": {"anyOf": [{"type": "number"}, {"type": "null"}]},
            "section_miscategorizations": {
                "type": "array",
                "maxItems": max_items,
                "items": {
                    "type": "object",
                    "properties": {
                        "text": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                        "current_section": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                        "expected_section": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                        "severity": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                        "reason": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                    },
                    "required": ["text"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["section_miscategorizations"],
        "additionalProperties": False,
    }


def section_semantics_for_template(template_key: str) -> Optional[str]:
    if template_key == "soap":
        return """
- S: patient-reported symptoms, history, subjective changes, ROS, home measurements, and current medication state reported by the patient.
- O: observed findings, physical exam, clinician-measured vitals, imaging, lab results, and procedures already performed.
- A: clinician assessment, diagnostic impression, differential, or active problem synthesis.
- P: medication changes, new prescriptions, orders, referrals, follow-up, counseling, return precautions, and explicit management actions.
- Current medications the patient is already taking belong in S unless framed as a treatment change or order.
- Referrals, medication changes, and follow-up plans belong in P, not A.
""".strip()
    if template_key == "sectioned":
        return """
- Chief Complaint: primary reason for visit.
- History of Present Illness: patient-reported symptoms, chronology, relevant history, and current medication state.
- Findings: exam findings, observed behaviors, vitals, imaging, and lab results.
- Assessment: diagnostic impression, clinician synthesis, and active problems.
- Plan: medication changes, orders, referrals, follow-up, counseling, and instructions.
""".strip()
    if template_key == "brief":
        return """
- CC/HPI lines should contain the visit reason and patient-reported symptom history.
- Findings lines should contain exam or test evidence.
- Assessment lines should contain diagnoses or impressions.
- Plan lines should contain treatment changes, orders, referrals, counseling, and follow-up.
""".strip()
    return None


def normalize_section_judgment(obj: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    judgment = obj or {}
    issues: List[Dict[str, Any]] = []
    for raw in judgment.get("section_miscategorizations") or []:
        if not isinstance(raw, dict):
            continue
        severity = safe_text(raw.get("severity")).lower()
        if severity not in {"major", "minor"}:
            severity = "minor"
        issues.append(
            {
                "text": safe_text(raw.get("text")),
                "current_section": safe_text(raw.get("current_section")) or None,
                "expected_section": safe_text(raw.get("expected_section")) or None,
                "severity": severity,
                "reason": safe_text(raw.get("reason")) or None,
            }
        )
    score = judgment.get("section_appropriateness_score")
    try:
        score = float(score) if score is not None else None
    except Exception:
        score = None
    if score is not None:
        score = max(0.0, min(1.0, score))
    summary_statistics = {
        "section_appropriateness_score": score,
        "section_miscategorization_count": len(issues),
        "major_section_miscategorization_count": sum(1 for issue in issues if issue["severity"] == "major"),
    }
    return {
        "section_miscategorizations": issues,
        "summary_statistics": summary_statistics,
    }


def evaluate_section_appropriateness_judge(
    client: OpenAICompatClient,
    *,
    model: str,
    template_key: str,
    summary_text: str,
    max_tokens: int,
) -> Dict[str, Any]:
    section_semantics = section_semantics_for_template(template_key)
    if not section_semantics:
        return {"section_miscategorizations": [], "summary_statistics": {}}
    prompt = SECTION_APPROPRIATENESS_PROMPT.format(
        template_name=template_key,
        section_semantics=section_semantics,
        generated_note=summary_text,
    )
    parsed = call_structured_judge(
        client,
        model=model,
        prompt=prompt,
        schema_obj=section_appropriateness_schema(),
        max_tokens=max_tokens,
    )
    return normalize_section_judgment(parsed)


def normalize_semantic_judgment(label: str, allowed: Sequence[str], default: str) -> str:
    value = re.sub(r"[^a-z_]", "", safe_text(label).lower())
    aliases = {
        "supported": "supported",
        "support": "supported",
        "partial": "partial",
        "partiallysupported": "partial",
        "partiallysupported": "partial",
        "missing": "missing",
        "unsupported": "unsupported",
        "contradicted": "contradicted",
        "contradiction": "contradicted",
        "lowvalue": "low_value",
        "lowvaluemismatch": "low_value",
        "trivial": "low_value",
    }
    normalized = aliases.get(value, value or default)
    return normalized if normalized in set(allowed) else default


def normalize_semantic_cap_audit(obj: Any) -> Dict[str, Any]:
    out = {"source_cap_recall": [], "summary_cap_precision": []}
    if not isinstance(obj, dict):
        return out
    recall_allowed = ("supported", "partial", "missing", "contradicted", "low_value")
    precision_allowed = ("supported", "partial", "unsupported", "contradicted", "low_value")
    for key, allowed, default in (
        ("source_cap_recall", recall_allowed, "missing"),
        ("summary_cap_precision", precision_allowed, "unsupported"),
    ):
        rows: List[Dict[str, Any]] = []
        for item in obj.get(key) or []:
            if not isinstance(item, dict):
                continue
            cap_id = safe_text(item.get("cap_id"))
            if not cap_id:
                continue
            rows.append(
                {
                    "cap_id": cap_id,
                    "judgment": normalize_semantic_judgment(item.get("judgment"), allowed, default),
                    "reason": safe_text(item.get("reason")) or None,
                }
            )
        out[key] = rows
    return out


def semantic_cap_metrics(audit: Dict[str, Any]) -> Dict[str, Any]:
    recall_rows = audit.get("source_cap_recall") or []
    precision_rows = audit.get("summary_cap_precision") or []

    def weighted_score(rows: Sequence[Dict[str, Any]], supported: str, partial: str, excluded: str) -> Tuple[float, int, int, int]:
        filtered = [row for row in rows if row.get("judgment") != excluded]
        supported_count = sum(1 for row in filtered if row.get("judgment") == supported)
        partial_count = sum(1 for row in filtered if row.get("judgment") == partial)
        denom = len(filtered)
        score = 0.0 if denom == 0 else (supported_count + SEMANTIC_PARTIAL_CREDIT * partial_count) / denom
        return score, supported_count, partial_count, denom

    recall, recall_supported, recall_partial, recall_denom = weighted_score(recall_rows, "supported", "partial", "low_value")
    precision, precision_supported, precision_partial, precision_denom = weighted_score(precision_rows, "supported", "partial", "low_value")
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)

    return {
        "semantic_cap_precision": round(precision, 4),
        "semantic_cap_recall": round(recall, 4),
        "semantic_cap_f1": round(f1, 4),
        "semantic_cap_precision_supported_count": precision_supported,
        "semantic_cap_precision_partial_count": precision_partial,
        "semantic_cap_precision_denominator": precision_denom,
        "semantic_cap_recall_supported_count": recall_supported,
        "semantic_cap_recall_partial_count": recall_partial,
        "semantic_cap_recall_denominator": recall_denom,
        "semantic_cap_low_value_source_count": sum(1 for row in recall_rows if row.get("judgment") == "low_value"),
        "semantic_cap_low_value_summary_count": sum(1 for row in precision_rows if row.get("judgment") == "low_value"),
        "semantic_cap_missing_count": sum(1 for row in recall_rows if row.get("judgment") == "missing"),
        "semantic_cap_unsupported_count": sum(1 for row in precision_rows if row.get("judgment") == "unsupported"),
        "semantic_cap_contradicted_source_count": sum(1 for row in recall_rows if row.get("judgment") == "contradicted"),
        "semantic_cap_contradicted_summary_count": sum(1 for row in precision_rows if row.get("judgment") == "contradicted"),
    }


def evaluate_semantic_cap_audit(
    client: OpenAICompatClient,
    *,
    model: str,
    source_cap: Dict[str, Any],
    summary_cap: Dict[str, Any],
    summary_text: str,
    max_tokens: int,
    chunk_workers: int = 1,
) -> Dict[str, Any]:
    source_chunks = chunk_cap_obj(source_cap, 16)
    summary_chunks = chunk_cap_obj(summary_cap, 16)

    def semantic_chunk_judge(
        *,
        prompt: str,
        schema_obj: Dict[str, Any],
        fallback_key: str,
    ) -> Dict[str, Any]:
        try:
            return call_structured_judge(
                client,
                model=model,
                prompt=prompt,
                schema_obj=schema_obj,
                max_tokens=max(500, min(max_tokens, 900)),
            )
        except Exception:
            smaller_prompt = prompt + "\n\n[Retry]\n- Return shorter JSON.\n- Omit explanations.\n- Output only cap_id and judgment.\n"
            try:
                return call_structured_judge(
                    client,
                    model=model,
                    prompt=smaller_prompt,
                    schema_obj=schema_obj,
                    max_tokens=max(300, min(max_tokens, 600)),
                )
            except Exception:
                return {fallback_key: []}

    def process_recall_chunk(chunk: Dict[str, Any]) -> List[Dict[str, Any]]:
        prompt = SEMANTIC_CAP_RECALL_CHUNK_PROMPT.format(
            source_caps=format_caps_for_prompt(chunk),
            summary_text=summary_text,
        )
        parsed = semantic_chunk_judge(
            prompt=prompt,
            schema_obj=semantic_cap_recall_schema(max_items=max(1, len(chunk.get("atomic_propositions", [])))),
            fallback_key="source_cap_recall",
        )
        normalized_chunk = normalize_semantic_cap_audit({"source_cap_recall": parsed.get("source_cap_recall", [])})
        return normalized_chunk.get("source_cap_recall", [])

    recall_rows: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=bounded_workers(chunk_workers, len(source_chunks))) as executor:
        for rows in executor.map(process_recall_chunk, source_chunks):
            recall_rows.extend(rows)

    precision_rows: List[Dict[str, Any]] = []
    source_caps_text = format_caps_for_prompt(source_cap)
    def process_precision_chunk(chunk: Dict[str, Any]) -> List[Dict[str, Any]]:
        prompt = SEMANTIC_CAP_PRECISION_CHUNK_PROMPT.format(
            source_caps=source_caps_text,
            summary_caps=format_caps_for_prompt(chunk),
        )
        parsed = semantic_chunk_judge(
            prompt=prompt,
            schema_obj=semantic_cap_precision_schema(max_items=max(1, len(chunk.get("atomic_propositions", [])))),
            fallback_key="summary_cap_precision",
        )
        normalized_chunk = normalize_semantic_cap_audit({"summary_cap_precision": parsed.get("summary_cap_precision", [])})
        return normalized_chunk.get("summary_cap_precision", [])

    with ThreadPoolExecutor(max_workers=bounded_workers(chunk_workers, len(summary_chunks))) as executor:
        for rows in executor.map(process_precision_chunk, summary_chunks):
            precision_rows.extend(rows)

    normalized = normalize_semantic_cap_audit(
        {
            "source_cap_recall": recall_rows,
            "summary_cap_precision": precision_rows,
        }
    )
    normalized["metrics"] = semantic_cap_metrics(normalized)
    return normalized


def normalize_safety_audit(obj: Any) -> Dict[str, Any]:
    def coerce_item(item: Any) -> Optional[Dict[str, Any]]:
        if isinstance(item, dict):
            return item
        if isinstance(item, list):
            if all(isinstance(x, list) and len(x) == 2 for x in item):
                return {safe_text(k): v for k, v in item if safe_text(k)}
            if len(item) % 2 == 0 and all(isinstance(item[i], str) for i in range(0, len(item), 2)):
                return {safe_text(item[i]): item[i + 1] for i in range(0, len(item), 2) if safe_text(item[i])}
        return None

    out = {"hallucinations": [], "omissions": []}
    if not isinstance(obj, dict):
        return out
    for item in obj.get("hallucinations") or []:
        item = coerce_item(item)
        if not isinstance(item, dict):
            continue
        text = safe_text(item.get("text"))
        if not text:
            continue
        err_type = normalize_semantic_judgment(item.get("type"), ("fabrication", "negation", "contextual", "causality"), "fabrication")
        severity = normalize_semantic_judgment(item.get("severity"), ("major", "minor"), "minor")
        out["hallucinations"].append(
            {
                "text": text,
                "type": err_type,
                "severity": severity,
                "reason": safe_text(item.get("reason")) or None,
                "evidence": safe_text(item.get("evidence")) or None,
            }
        )
    for item in obj.get("omissions") or []:
        item = coerce_item(item)
        if not isinstance(item, dict):
            continue
        text = safe_text(item.get("text"))
        if not text:
            continue
        raw_type = safe_text(item.get("type")).lower()
        omission_aliases = {
            "symptom_history": "current_issues",
            "symptomhistory": "current_issues",
            "diagnosis_assessment": "current_issues",
            "diagnosisassessment": "current_issues",
            "exam_result": "current_issues",
            "examresult": "current_issues",
            "medication_treatment": "pmfs_issues",
            "medicationtreatment": "pmfs_issues",
            "plan_followup": "information_plan",
            "planfollowup": "information_plan",
            "safety_risk": "information_plan",
            "safetyrisk": "information_plan",
            "currentissue": "current_issues",
            "currentissues": "current_issues",
            "pmfsissue": "pmfs_issues",
            "pmfsissues": "pmfs_issues",
            "informationandplan": "information_plan",
            "planinformation": "information_plan",
        }
        normalized_raw_type = normalize_text(raw_type).replace(" ", "")
        omission_type = omission_aliases.get(normalized_raw_type, raw_type)
        omission_type = normalize_semantic_judgment(
            omission_type,
            ("current_issues", "pmfs_issues", "information_plan"),
            "current_issues",
        )
        severity = normalize_semantic_judgment(item.get("severity"), ("major", "minor"), "minor")
        out["omissions"].append(
            {
                "text": text,
                "type": omission_type,
                "severity": severity,
                "reason": safe_text(item.get("reason")) or None,
                "evidence": safe_text(item.get("evidence")) or None,
            }
        )
    out["summary_statistics"] = {
        "hallucination_count": len(out["hallucinations"]),
        "major_hallucination_count": sum(1 for item in out["hallucinations"] if item.get("severity") == "major"),
        "omission_count": len(out["omissions"]),
        "major_omission_count": sum(1 for item in out["omissions"] if item.get("severity") == "major"),
        "fabrication_count": sum(1 for item in out["hallucinations"] if item.get("type") == "fabrication"),
        "negation_error_count": sum(1 for item in out["hallucinations"] if item.get("type") == "negation"),
        "contextual_error_count": sum(1 for item in out["hallucinations"] if item.get("type") == "contextual"),
        "causality_error_count": sum(1 for item in out["hallucinations"] if item.get("type") == "causality"),
        "current_issues_omission_count": sum(1 for item in out["omissions"] if item.get("type") == "current_issues"),
        "pmfs_issues_omission_count": sum(1 for item in out["omissions"] if item.get("type") == "pmfs_issues"),
        "information_plan_omission_count": sum(1 for item in out["omissions"] if item.get("type") == "information_plan"),
    }
    return out


def evaluate_safety_judge(
    client: OpenAICompatClient,
    *,
    model: str,
    transcript: str,
    summary_text: str,
    max_tokens: int,
    chunk_workers: int = 1,
) -> Dict[str, Any]:
    transcript_chunks = split_transcript_for_safety(transcript)
    hallucinations: List[Dict[str, Any]] = []
    omissions: List[Dict[str, Any]] = []
    seen_h: set[Tuple[str, str, str]] = set()
    seen_o: set[Tuple[str, str, str]] = set()

    def process_transcript_chunk(chunk: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        hallucination_prompt = ASGARI_HALLUCINATION_PROMPT.format(transcript=chunk, summary=summary_text)
        try:
            hallucination_parsed = call_structured_judge(
                client,
                model=model,
                prompt=hallucination_prompt,
                schema_obj=safety_hallucination_schema(),
                max_tokens=max(700, min(max_tokens, 1400)),
            )
        except Exception:
            hallucination_parsed = {"hallucinations": []}
        normalized_h = normalize_safety_audit({"hallucinations": hallucination_parsed.get("hallucinations", [])})
        for item in normalized_h.get("hallucinations", []):
            key = (normalize_text(item.get("text")), safe_text(item.get("type")).lower(), safe_text(item.get("severity")).lower())
            if key in seen_h:
                continue
            seen_h.add(key)
            hallucinations.append(item)

        omission_prompt = ASGARI_OMISSION_PROMPT.format(transcript=chunk, summary=summary_text)
        try:
            omission_parsed = call_structured_judge(
                client,
                model=model,
                prompt=omission_prompt,
                schema_obj=safety_omission_schema(),
                max_tokens=max(700, min(max_tokens, 1400)),
            )
        except Exception:
            omission_parsed = {"omissions": []}
        normalized_o = normalize_safety_audit({"omissions": omission_parsed.get("omissions", [])})
        return normalized_h.get("hallucinations", []), normalized_o.get("omissions", [])

    with ThreadPoolExecutor(max_workers=bounded_workers(chunk_workers, len(transcript_chunks))) as executor:
        for chunk_h, chunk_o in executor.map(process_transcript_chunk, transcript_chunks):
            for item in chunk_h:
                key = (normalize_text(item.get("text")), safe_text(item.get("type")).lower(), safe_text(item.get("severity")).lower())
                if key in seen_h:
                    continue
                seen_h.add(key)
                hallucinations.append(item)
            for item in chunk_o:
                key = (normalize_text(item.get("text")), safe_text(item.get("type")).lower(), safe_text(item.get("severity")).lower())
                if key in seen_o:
                    continue
                seen_o.add(key)
                omissions.append(item)

    return normalize_safety_audit(
        {
            "hallucinations": hallucinations,
            "omissions": omissions,
        }
    )


def normalize_pdsqi_result(obj: Any) -> Dict[str, int]:
    keys = (
        "citation_applicable",
        "citation",
        "accurate",
        "thorough",
        "useful",
        "organized",
        "comprehensible",
        "succinct",
        "abstraction",
        "synthesized",
        "voice_summ",
        "voice_note",
    )
    out: Dict[str, int] = {}
    if not isinstance(obj, dict):
        return {key: 0 for key in keys}
    for key in keys:
        value = obj.get(key)
        out[key] = int(value) if isinstance(value, int) else 0
    return out


LLM_CHECKLIST_SCORE_KEYS = [
    "content_preservation_core",
    "noise_suppression",
    "lay_expression_preservation",
    "state_update_fidelity",
    "temporality_preservation",
    "problem_evidence_linkage",
    "state_plan_separation",
    "section_organization_appropriateness",
    "overall_clinical_usability",
]

LLM_CHECKLIST_A_KEYS = [
    "content_preservation_core",
    "noise_suppression",
    "lay_expression_preservation",
]
LLM_CHECKLIST_B_KEYS = [
    "state_update_fidelity",
    "temporality_preservation",
]
LLM_CHECKLIST_C_KEYS = [
    "problem_evidence_linkage",
    "state_plan_separation",
    "section_organization_appropriateness",
]


def normalize_llm_checklist_result(obj: Any) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if not isinstance(obj, dict):
        obj = {}
    for key in LLM_CHECKLIST_SCORE_KEYS:
        value = obj.get(key)
        out[key] = int(value) if isinstance(value, int) and value in {1, 2, 3, 4, 5} else 1
    for key in ("clinically_meaningful_omission", "clinically_concerning_hallucination"):
        value = safe_text(obj.get(key)).strip().lower()
        out[key] = "Yes" if value == "yes" else "No"
    values = [float(out[key]) for key in LLM_CHECKLIST_SCORE_KEYS]
    abc_values = [float(out[key]) for key in (LLM_CHECKLIST_A_KEYS + LLM_CHECKLIST_B_KEYS + LLM_CHECKLIST_C_KEYS)]
    out["checklist_a_mean"] = round(sum(float(out[key]) for key in LLM_CHECKLIST_A_KEYS) / len(LLM_CHECKLIST_A_KEYS), 4)
    out["checklist_b_mean"] = round(sum(float(out[key]) for key in LLM_CHECKLIST_B_KEYS) / len(LLM_CHECKLIST_B_KEYS), 4)
    out["checklist_c_mean"] = round(sum(float(out[key]) for key in LLM_CHECKLIST_C_KEYS) / len(LLM_CHECKLIST_C_KEYS), 4)
    out["llm_checklist_abc_mean"] = round(sum(abc_values) / len(abc_values), 4) if abc_values else 0.0
    out["llm_checklist_mean_no_usability"] = round(
        sum(float(out[key]) for key in LLM_CHECKLIST_SCORE_KEYS if key != "overall_clinical_usability")
        / max(1, len([key for key in LLM_CHECKLIST_SCORE_KEYS if key != "overall_clinical_usability"])),
        4,
    )
    out["llm_checklist_total_score"] = int(sum(int(out[key]) for key in LLM_CHECKLIST_SCORE_KEYS))
    out["llm_checklist_mean"] = round(sum(values) / len(values), 4) if values else 0.0
    out["llm_checklist_meaningful_omission_yes"] = 1 if out["clinically_meaningful_omission"] == "Yes" else 0
    out["llm_checklist_concerning_hallucination_yes"] = 1 if out["clinically_concerning_hallucination"] == "Yes" else 0
    return out


def compute_pdsqi_core_mean(result: Dict[str, int]) -> float:
    values = [float(result.get(key, 0)) for key in PDSQI_CORE_KEYS]
    if result.get("abstraction"):
        values.append(float(result.get("synthesized", 0)))
    return round(sum(values) / len(values), 4) if values else 0.0


def evaluate_pdsqi_judge(
    client: OpenAICompatClient,
    *,
    model: str,
    transcript: str,
    summary_text: str,
    target_specialty: str,
    max_tokens: int,
) -> Dict[str, Any]:
    prompt = PDSQI_TRANSCRIPT_PROMPT.format(
        transcript=transcript,
        generated_note=summary_text,
        target_specialty=target_specialty,
    )
    parsed = call_structured_judge(
        client,
        model=model,
        prompt=prompt,
        schema_obj=pdsqi_schema(),
        max_tokens=max_tokens,
    )
    normalized = normalize_pdsqi_result(parsed)
    normalized["pdsqi_core_mean"] = compute_pdsqi_core_mean(normalized)
    return normalized


def evaluate_llm_checklist_judge(
    client: OpenAICompatClient,
    *,
    model: str,
    transcript: str,
    summary_text: str,
    max_tokens: int,
) -> Dict[str, Any]:
    prompt = LLM_CHECKLIST_PROMPT.format(
        transcript=transcript,
        generated_note=summary_text,
    )
    parsed = call_structured_judge(
        client,
        model=model,
        prompt=prompt,
        schema_obj=llm_checklist_schema(),
        max_tokens=max_tokens,
    )
    return normalize_llm_checklist_result(parsed)


def lcs_length(a_tokens: List[str], b_tokens: List[str]) -> int:
    if not a_tokens or not b_tokens:
        return 0
    prev = [0] * (len(b_tokens) + 1)
    for a_tok in a_tokens:
        curr = [0]
        for j, b_tok in enumerate(b_tokens, start=1):
            if a_tok == b_tok:
                curr.append(prev[j - 1] + 1)
            else:
                curr.append(max(prev[j], curr[j - 1]))
        prev = curr
    return prev[-1]


def rouge_l_f1(prediction: str, reference: str) -> float:
    pred_tokens = tokenize(prediction)
    ref_tokens = tokenize(reference)
    if not pred_tokens or not ref_tokens:
        return 0.0
    lcs = lcs_length(pred_tokens, ref_tokens)
    if lcs == 0:
        return 0.0
    precision = lcs / len(pred_tokens)
    recall = lcs / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def load_cases(path: Path, limit: Optional[int], case_ids: Optional[Sequence[str]]) -> Dict[str, Dict[str, Any]]:
    wanted = {str(x) for x in (case_ids or [])}
    out: Dict[str, Dict[str, Any]] = {}
    for row in read_jsonl(path):
        case_id = safe_text(row.get("file") or row.get("case_id") or row.get("id"))
        if not case_id:
            continue
        if wanted and case_id not in wanted:
            continue
        out[case_id] = row
        if limit is not None and len(out) >= limit:
            break
    return out


def load_method_csv(path: Path) -> Dict[str, Dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return {safe_text(row.get("file")): row for row in rows if safe_text(row.get("file"))}


def load_legacy_extraction_result(legacy_dir: Optional[Path], case_id: str) -> Optional[Dict[str, Any]]:
    if legacy_dir is None:
        return None
    path = legacy_dir / f"{case_id}_result.json"
    if not path.exists():
        return None
    try:
        obj = read_json(path)
    except Exception:
        return None
    if not isinstance(obj, dict) or not obj:
        return None
    inner = next(iter(obj.values()))
    return inner if isinstance(inner, dict) else None


def clean_utterance_sentence(text: str) -> str:
    s = safe_text(text)
    if not s:
        return s
    s = re.sub(r"^\[[^\]]+\]\s*", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def is_generic_medsum_prompt_entity(entity_text: str, sentence: str) -> bool:
    ent = normalize_text(entity_text)
    sent = normalize_text(sentence)
    generic_entities = {
        "any medicine",
        "medicine",
        "medicines",
        "medications",
        "any medical problems",
        "medical problems",
        "surgeries",
        "surgery",
        "your pain",
        "pain",
        "other injuries",
        "procedures",
        "your exam",
        "exam",
    }
    if ent in generic_entities and ("?" in sentence or sent.startswith("have you") or sent.startswith("do you") or sent.startswith("and what")):
        return True
    return False


def normalize_medsum_concept(entity_text: str, entity_type: str, sentence: str) -> str:
    text = safe_text(entity_text).lower().strip()
    text = re.sub(r"\b(this|that|these|those|a|an|the|my|your|his|her|their|some|any)\b", " ", text)
    text = re.sub(r"\b(out|before|ever|had)\b", " ", text)
    text = re.sub(r"[^a-z0-9\s/-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if text.endswith(" stones"):
        text = text[:-1]
    if text.endswith(" injuries"):
        text = text[:-3] + "y"
    if text == "appendix":
        text = "appendectomy"
    if "appendix out" in text:
        text = "appendectomy"
    if text in {"right index finger pain", "index finger pain", "finger pain"}:
        return "finger pain"
    if not text:
        inferred = infer_canonical_concept(entity_text, entity_type.title() if entity_type else None)
        return safe_text(inferred) or safe_text(entity_text)
    return text


def infer_medsum_planning_category(entity_type: str, sentence: str) -> str:
    et = normalize_text(entity_type)
    sent = normalize_text(sentence)
    if any(k in sent for k in ["year old", "years old", "works as", "job", "insurance", "lives with", "smokes", "drink", "alcohol", "marijuana"]):
        return "Demographics and Social Determinants of Health"
    if et in {"problem", "bodyloc"} and any(k in sent for k in ["chief complaint", "here because", "came in", "coming in", "reason for visit"]):
        return "Patient Intent"
    return ""


def infer_medsum_bucket(entity_type: str, sentence: str, relation_types: Sequence[str]) -> str:
    et = normalize_text(entity_type)
    sent = normalize_text(sentence)
    rels = {normalize_text(x) for x in relation_types}
    if any(k in sent for k in ["unknown", "unsure", "not sure", "don't know", "doesn't know", "unclear"]):
        return "Pertinent Unknowns"
    if "negation" in rels or sent.startswith("no ") or " denies " in f" {sent} " or " no " in f" {sent} ":
        return "Pertinent Negatives"
    if "uncertain" in sent or "possible" in sent:
        return "Pertinent Unknowns"
    if any(k in sent for k in ["history of", "had ", "when i was", "previous", "family history", "appendix out", "appendectomy"]):
        return "Medical History"
    if et in {"problem", "drug", "treatment", "test", "procedure", "bodyloc", "labvalue"}:
        return "Pertinent Positives"
    return "Medical History"


def infer_cluster_section(entity_type: str, sentence: str, relation_types: Sequence[str]) -> str:
    et = normalize_text(entity_type)
    sent = normalize_text(sentence)
    rels = {normalize_text(x) for x in relation_types}
    if et in {"treatment", "drug"} and any(k in sent for k in ["prescribe", "start", "continue", "increase", "decrease", "follow up", "return", "refer"]):
        return "Plan"
    if et in {"test", "labvalue"} or any(k in sent for k in ["x-ray", "mri", "ct", "normal", "lab", "result", "show"]):
        return "Findings"
    if any(k in sent for k in ["plan", "follow up", "return", "refer", "order", "prescribe", "counsel"]):
        return "Plan"
    if any(k in sent for k in ["impression", "diagnosis", "suspect", "assessment", "sprain", "kidney stone", "hypertension", "diabetes"]):
        return "Assessment"
    if "negation" in rels:
        return "History of Present Illness"
    return "History of Present Illness"


def build_scaffolds_from_legacy_result(legacy_result: Dict[str, Any]) -> Tuple[str, str]:
    entities = legacy_result.get("entities") if isinstance(legacy_result.get("entities"), list) else []
    relations = legacy_result.get("relations") if isinstance(legacy_result.get("relations"), list) else []
    relation_types_by_line: Dict[int, List[str]] = {}
    for rel in relations:
        if not isinstance(rel, dict):
            continue
        line = rel.get("line")
        rel_type = None
        relation_obj = rel.get("relation")
        if isinstance(relation_obj, dict):
            rel_type = relation_obj.get("type")
        elif isinstance(relation_obj, str):
            rel_type = relation_obj
        if isinstance(line, int) and safe_text(rel_type):
            relation_types_by_line.setdefault(line, []).append(safe_text(rel_type))

    medsum_groups: Dict[str, List[str]] = {
        "Demographics and Social Determinants of Health": [],
        "Patient Intent": [],
        "Pertinent Positives": [],
        "Pertinent Negatives": [],
        "Pertinent Unknowns": [],
        "Medical History": [],
    }
    medsum_resolved: Dict[Tuple[str, str], Dict[str, Any]] = {}
    cluster_items: List[Dict[str, Any]] = []
    seen_sentence_keys: set[str] = set()
    first_patient_intent: Optional[str] = None

    for ent in entities:
        if not isinstance(ent, dict):
            continue
        raw_sentence = clean_utterance_sentence(ent.get("sentence"))
        entity_text = clean_utterance_sentence(ent.get("entity"))
        if not raw_sentence or not entity_text:
            continue
        if is_generic_medsum_prompt_entity(entity_text, raw_sentence):
            continue
        line = ent.get("line") if isinstance(ent.get("line"), int) else -1
        relation_types = relation_types_by_line.get(line, [])
        medsum_bucket = infer_medsum_bucket(safe_text(ent.get("type")), raw_sentence, relation_types)
        canonical = normalize_medsum_concept(entity_text, safe_text(ent.get("type")), raw_sentence)
        planning_category = infer_medsum_planning_category(safe_text(ent.get("type")), raw_sentence)
        if not first_patient_intent and medsum_bucket == "Pertinent Positives" and normalize_text(safe_text(ent.get("type"))) == "problem":
            first_patient_intent = canonical
        resolved_key = (medsum_bucket, canonical)
        resolved_entry = medsum_resolved.setdefault(
            resolved_key,
            {
                "bucket": medsum_bucket,
                "canonical": canonical,
                "evidence": [],
                "line_min": line if line >= 0 else 10**9,
            },
        )
        if raw_sentence not in resolved_entry["evidence"]:
            resolved_entry["evidence"].append(raw_sentence)
        if line >= 0:
            resolved_entry["line_min"] = min(resolved_entry["line_min"], line)
        if planning_category:
            planning_line = f"- {canonical} | evidence: {raw_sentence}"
            if planning_line not in medsum_groups[planning_category]:
                medsum_groups[planning_category].append(planning_line)

        sentence_key = f"{line}:{raw_sentence}"
        if sentence_key in seen_sentence_keys:
            continue
        seen_sentence_keys.add(sentence_key)
        cluster_items.append(
            {
                "line": line,
                "section": infer_cluster_section(safe_text(ent.get("type")), raw_sentence, relation_types),
                "sentence": raw_sentence,
            }
        )

    resolved_by_concept: Dict[str, set[str]] = {}
    for (bucket, canonical), entry in medsum_resolved.items():
        resolved_by_concept.setdefault(canonical, set()).add(bucket)

    if first_patient_intent:
        medsum_groups["Patient Intent"].append(f"- {first_patient_intent}")

    for (bucket, canonical), entry in sorted(
        medsum_resolved.items(),
        key=lambda item: (item[1].get("line_min", 10**9), item[0][0], item[0][1]),
    ):
        if bucket == "Pertinent Unknowns" and resolved_by_concept.get(canonical, set()) & {"Pertinent Positives", "Pertinent Negatives"}:
            continue
        evidence = " || ".join(entry["evidence"][:3])
        medsum_groups[bucket].append(f"- {canonical} | evidence: {evidence}")

    medsum_parts: List[str] = []
    for heading in (
        "Demographics and Social Determinants of Health",
        "Patient Intent",
        "Pertinent Positives",
        "Pertinent Negatives",
        "Pertinent Unknowns",
        "Medical History",
    ):
        lines = medsum_groups.get(heading) or []
        if lines:
            medsum_parts.append(f"{heading}:\n" + "\n".join(lines))
    medsum_fact_text = "\n\n".join(medsum_parts).strip()

    cluster_items.sort(key=lambda item: (item["section"], item["line"]))
    clusters: List[str] = []
    current_section = None
    current_lines: List[str] = []
    current_end_line = None
    tau = 2
    for item in cluster_items:
        if (
            current_section is None
            or item["section"] != current_section
            or current_end_line is None
            or (item["line"] - current_end_line) > tau
        ):
            if current_lines:
                clusters.append(f"[{current_section}] " + " ".join(current_lines))
            current_section = item["section"]
            current_lines = [item["sentence"]]
        else:
            current_lines.append(item["sentence"])
        current_end_line = item["line"]
    if current_lines:
        clusters.append(f"[{current_section}] " + " ".join(current_lines))
    cluster_fact_text = "\n".join(f"- {line}" for line in clusters).strip()
    return medsum_fact_text, cluster_fact_text


def assertion_to_status(assertion: str) -> Optional[str]:
    value = safe_text(assertion).lower()
    if value == "present":
        return "affirmed"
    if value == "absent":
        return "negated"
    if value in {"uncertain", "hypothetical"}:
        return "uncertain"
    if value == "historical":
        return "affirmed"
    return None


def verification_to_status(verification_status: str) -> Optional[str]:
    value = safe_text(verification_status).lower()
    if value == "confirmed":
        return "affirmed"
    if value == "refuted":
        return "negated"
    if value == "unconfirmed":
        return "uncertain"
    return None


def convert_problem_state_caps(problem_cap_obj: Dict[str, Any]) -> Dict[str, Any]:
    props: List[Dict[str, Any]] = []
    for idx, item in enumerate(problem_cap_obj.get("caps", []), start=1):
        if not isinstance(item, dict):
            continue
        evidence = item.get("evidence") if isinstance(item.get("evidence"), list) else []
        first_speaker = safe_text(item.get("content_source") or item.get("speaker")) or None
        for ev in evidence:
            if isinstance(ev, dict) and safe_text(ev.get("turn_speaker") or ev.get("speaker")):
                first_speaker = safe_text(ev.get("turn_speaker") or ev.get("speaker"))
                break
        prop = {
            "prop_id": safe_text(item.get("cap_id") or f"P{idx}") or f"P{idx}",
            "cap_type": safe_text(item.get("cap_type")) or None,
            "canonical_concept": safe_text(item.get("canonical_concept")) or None,
            "category": safe_text(item.get("cap_type")) or None,
            "speaker": first_speaker,
            "modality": safe_text(item.get("modality")) or None,
            "status": verification_to_status(item.get("verification_status")) or assertion_to_status(item.get("assertion")),
            "visit_state": safe_text(item.get("clinical_status")) or safe_text(item.get("visit_state")) or None,
            "temporality": safe_text(item.get("temporality")) or None,
            "claim_type_tags": [],
            "proposition_text": safe_text(item.get("proposition_text")),
        }
        if prop["proposition_text"]:
            props.append({k: v for k, v in prop.items() if v not in (None, [], "")})
    return enrich_cap_obj({"atomic_propositions": props, "filtered_nonverifiable_units": []})


def build_cap_lookup(cap_obj: Optional[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    lookup: Dict[str, Dict[str, Any]] = {}
    if not isinstance(cap_obj, dict):
        return lookup
    for item in cap_obj.get("atomic_propositions", []):
        if not isinstance(item, dict):
            continue
        prop_id = safe_text(item.get("prop_id"))
        if prop_id:
            lookup[prop_id] = item
    return lookup


LOW_VALUE_BUNDLE_ANCHOR_CONCEPTS = {
    "allergy",
    "bad combination",
    "blood pressure cuff",
    "bread consumption",
    "cataract surgery",
    "chlamydia",
    "club soda",
    "cold weather",
    "dating her partner for three years",
    "diet",
    "ecog performance status",
    "er positive",
    "er/pr positive",
    "establish care",
    "eye prescription",
    "fluid intake",
    "french fries",
    "gonorrhea",
    "hepatitis c",
    "hiv",
    "iodine",
    "last pap smear",
    "menopausal",
    "new glasses",
    "no evidence of recurrence",
    "not cancerous",
    "pap smear history",
    "pap smear",
    "performance status",
    "pr positive",
    "smoking",
    "smoking cessation",
    "stage zero",
    "syphilis",
    "thinking about it",
    "trichomoniasis",
    "urology",
}

PROBLEM_FAMILY_ALIASES = {
    "high blood pressure": "hypertension",
    "blood pressure": "hypertension",
    "hyperglycemia": "diabetes",
    "a1c": "diabetes",
    "a1 c": "diabetes",
    "diabetes mellitus": "diabetes",
    "high cholesterol": "hyperlipidemia",
    "cholesterol": "hyperlipidemia",
    "kidney stone": "nephrolithiasis",
    "renal stone": "nephrolithiasis",
}

MEDICATION_PROBLEM_HINTS = {
    "amlodipine": {"hypertension"},
    "atorvastatin": {"hyperlipidemia", "vascular disease", "coronary artery disease"},
    "baby aspirin": {"coronary artery disease", "heart disease"},
    "carvedilol": {"hypertension", "heart disease"},
    "glimepiride": {"diabetes", "hyperglycemia"},
    "hctz": {"hypertension"},
    "hydrochlorothiazide": {"hypertension"},
    "insulin": {"diabetes", "hyperglycemia"},
    "lexapro": {"anxiety"},
    "lisinopril": {"hypertension"},
    "lipitor": {"hyperlipidemia", "vascular disease", "coronary artery disease"},
    "metformin": {"diabetes", "hyperglycemia"},
    "metoprolol": {"hypertension", "heart disease"},
    "norvasc": {"hypertension"},
    "oxycodone": {"pain", "nephrolithiasis"},
}

BODY_REGION_GROUPS = {
    "lower_extremity": {"foot", "feet", "leg", "legs", "hip", "hips", "toe", "toes", "lower extremity"},
    "upper_extremity": {"hand", "hands", "wrist", "wrists", "finger", "fingers", "forearm", "forearms", "elbow", "elbows"},
    "urinary": {"flank", "kidney", "urine", "urinary", "hematuria", "cva", "groin"},
    "genitourinary": {"vagina", "vaginal", "cervix", "labia", "pelvic", "discharge"},
    "breast": {"breast", "mammogram", "lumpectomy", "dcis", "nipple", "retroareolar"},
}

SYMPTOM_FAMILY_KEYWORDS = {
    "pain": {"pain", "sore", "throb"},
    "sensory": {"numbness", "tingling", "paresthesia", "sensation", "sensory"},
    "urinary": {"urination", "urinate", "pee", "nocturia", "hematuria", "stream"},
    "screening": {"screen", "screening", "testing", "test"},
}

PROBLEM_DOMAIN_KEYWORDS = {
    "breast_oncology": {"breast", "dcis", "carcinoma", "lumpectomy", "mammogram", "recurrence", "radiation"},
    "diabetes_metabolic": {"diabetes", "hyperglycemia", "a1c", "glimepiride", "metformin", "insulin", "glycemia"},
    "genitourinary": {"vaginal", "pelvic", "cervix", "pap smear", "std", "sexual", "gonorrhea", "chlamydia", "syphilis", "hiv", "hepatitis c"},
    "nephrolithiasis": {"kidney stone", "nephrolithiasis", "hematuria", "flank", "cva", "urology", "urinary"},
    "neuropathy_upper_extremity": {"carpal tunnel", "cubital tunnel", "hand", "wrist", "finger", "forearm", "elbow", "tingling", "numbness", "paresthesia"},
    "eye": {"eye", "vision", "glasses", "optometrist", "cataract", "iodine", "prescription"},
    "vascular_cardiac": {"vascular disease", "coronary", "heart disease", "heart attack", "stent", "atorvastatin", "lipitor", "aspirin", "hypertension", "blood pressure", "metoprolol", "lisinopril", "amlodipine", "carvedilol"},
    "wound_ulcer": {"ulcer", "wound", "podiatry", "granulation", "purulent", "cellulitis", "healing"},
}

GENERIC_SURFACE_ANCHOR_TERMS = {
    "abdominal pain",
    "arm pain",
    "back pain",
    "blood in urine",
    "cough",
    "discharge",
    "discomfort",
    "eye pain",
    "fatigue",
    "flank pain",
    "hand pain",
    "hand weakness",
    "hematuria",
    "leg pain",
    "numbness",
    "pain",
    "pelvic pain",
    "soreness",
    "tingling",
    "vaginal pain",
    "vaginal soreness",
    "weakness",
}

LOW_VALUE_BUNDLE_CONTEXT_PATTERNS = {
    "bad food habits",
    "bread consumption",
    "club soda",
    "drinks in the evening",
    "french fries",
    "new glasses",
    "received an eye prescription",
    "smoking cessation",
    "takes coq10",
    "takes elderberry",
    "takes fish oil",
    "takes vitamin",
    "thinking about it",
    "water intake",
}

ANCHOR_DECISION_WEIGHTS = {
    "preference": 3.0,
    "plan_link": 2.5,
    "non_subjective_support": 1.5,
    "support_score": 1.0,
    "salience": 0.1,
}

SCREENING_OR_TEST_LIKE_TERMS = {
    "ecog",
    "mammogram",
    "pap smear",
    "performance status",
    "screening",
    "test",
    "testing",
    "ultrasound",
}


def canonical_problem_family(text: Any) -> str:
    norm = normalize_alignment_text(text)
    if not norm:
        return ""
    for key, family in PROBLEM_FAMILY_ALIASES.items():
        key_norm = normalize_alignment_text(key)
        if key_norm and (norm == key_norm or key_norm in norm or norm in key_norm):
            return family
    return norm


def concept_matches_any(text: Any, phrases: Sequence[str]) -> bool:
    norm = normalize_text(text)
    return any(normalize_text(phrase) in norm for phrase in phrases if normalize_text(phrase))


def cap_or_cluster_text(*parts: Any) -> str:
    return " ".join(safe_text(part) for part in parts if safe_text(part))


def infer_body_region(text: Any) -> Optional[str]:
    norm = normalize_text(text)
    if not norm:
        return None
    for label, keywords in BODY_REGION_GROUPS.items():
        if any(keyword in norm for keyword in keywords):
            return label
    return None


def infer_symptom_family(text: Any) -> Optional[str]:
    norm = normalize_text(text)
    if not norm:
        return None
    for label, keywords in SYMPTOM_FAMILY_KEYWORDS.items():
        if any(keyword in norm for keyword in keywords):
            return label
    return None


def medication_problem_hints(text: Any) -> set[str]:
    norm = normalize_text(text)
    hints: set[str] = set()
    if not norm:
        return hints
    for med, med_hints in MEDICATION_PROBLEM_HINTS.items():
        med_norm = normalize_text(med)
        if med_norm and med_norm in norm:
            hints.update(med_hints)
    return hints


def problem_domains(text: Any) -> set[str]:
    norm = normalize_text(text)
    if not norm:
        return set()
    domains: set[str] = set()
    for domain, keywords in PROBLEM_DOMAIN_KEYWORDS.items():
        if any(keyword in norm for keyword in keywords):
            domains.add(domain)
    return domains


def cluster_problem_family(cluster: Dict[str, Any]) -> str:
    concept = safe_text(cluster.get("canonical_concept"))
    summary = safe_text(cluster.get("cluster_summary"))
    return canonical_problem_family(concept or summary)


def is_disease_like_problem_text(text: Any) -> bool:
    norm = normalize_text(text)
    return any(
        keyword in norm
        for keyword in [
            "cancer",
            "carcinoma",
            "dcis",
            "diabetes",
            "disease",
            "hyperglycemia",
            "hypertension",
            "infection",
            "kidney stone",
            "neuropathy",
            "syndrome",
            "ulcer",
            "vascular disease",
        ]
    )


def is_chronic_condition_text(text: Any) -> bool:
    norm = normalize_text(text)
    return any(
        keyword in norm
        for keyword in [
            "coronary artery disease",
            "diabetes",
            "heart disease",
            "hyperglycemia",
            "hyperlipidemia",
            "hypertension",
            "neuropathy",
            "peripheral vascular disease",
            "vascular disease",
        ]
    )


def is_screening_or_test_like_text(text: Any) -> bool:
    norm = normalize_text(text)
    if not norm:
        return False
    return any(term in norm for term in SCREENING_OR_TEST_LIKE_TERMS)


def cap_anchor_priority_score(cap_type: str) -> int:
    return {
        "Diagnosis": 5,
        "Impression": 5,
        "Problem": 4,
        "ChiefComplaint": 3,
        "ProblemHistory": 2,
    }.get(cap_type, 0)


def is_surface_symptom_anchor_text(text: Any) -> bool:
    norm = normalize_text(text)
    if not norm or is_disease_like_problem_text(norm):
        return False
    if norm in GENERIC_SURFACE_ANCHOR_TERMS:
        return True
    if infer_symptom_family(norm) and len(norm.split()) <= 4:
        return True
    return False


def is_noise_context_text(text: Any) -> bool:
    norm = normalize_text(text)
    if not norm:
        return False
    if norm in {normalize_text(x) for x in LOW_VALUE_BUNDLE_ANCHOR_CONCEPTS}:
        return True
    return any(pattern in norm for pattern in LOW_VALUE_BUNDLE_CONTEXT_PATTERNS)


def cap_can_name_bundle_problem(prop: Dict[str, Any]) -> bool:
    cap_type = safe_text(prop.get("cap_type") or prop.get("category"))
    concept = safe_text(prop.get("canonical_concept")) or safe_text(prop.get("proposition_text"))
    status = safe_text(prop.get("status"))
    if status == "negated" or not concept:
        return False
    if is_noise_context_text(concept) or is_screening_or_test_like_text(concept):
        return False
    if cap_type in {"Diagnosis", "Impression", "Problem", "ChiefComplaint"}:
        return True
    if cap_type == "ProblemHistory" and is_disease_like_problem_text(concept):
        return True
    return False


def cluster_anchor_preference(cluster: Dict[str, Any]) -> int:
    primary_cap_type = safe_text(cluster.get("primary_cap_type"))
    cluster_type = safe_text(cluster.get("cluster_type"))
    concept_text = cap_or_cluster_text(cluster.get("canonical_concept"), cluster.get("cluster_summary"))
    if primary_cap_type in {"Diagnosis", "Impression"} or cluster_type in {"Diagnosis", "DiagnosisImpression"}:
        return 5
    if is_disease_like_problem_text(concept_text):
        return 4
    if primary_cap_type in {"ChiefComplaint", "Problem"}:
        return 3
    if primary_cap_type == "ProblemHistory":
        return 1
    return 0


def cap_slot_from_prop(prop: Dict[str, Any]) -> str:
    cap_type = safe_text(prop.get("cap_type") or prop.get("category"))
    status = safe_text(prop.get("status"))
    speaker = normalize_text(prop.get("speaker"))
    temporality = normalize_text(prop.get("temporality"))
    visit_state = normalize_text(prop.get("visit_state") or prop.get("clinical_status"))
    text = normalize_text(prop.get("proposition_text"))

    if cap_type in {"Order", "FollowUp", "Counseling", "MedicationRequest"}:
        return "P"
    if cap_type in {"ExamFinding", "TestResult"}:
        return "O"
    if cap_type in {"Diagnosis", "Impression"}:
        return "A"
    if cap_type == "MedicationStatement":
        if visit_state in {"planned", "future"}:
            return "P"
        if any(phrase in text for phrase in ["increase ", "decrease ", "start ", "stop ", "switch ", "change ", "continue ", "refill ", "one pill twice a day", "new prescription"]):
            return "P"
        return "S" if speaker == "patient" else "G"
    if cap_type in {"ProblemHistory", "Allergy", "Demographics"}:
        return "G"
    if cap_type in {"ChiefComplaint", "Problem"}:
        return "S"
    if status == "negated":
        return "S"
    if temporality in {"past", "historical", "history"}:
        return "G"
    return "G"


def cap_is_anchor_eligible(prop: Dict[str, Any]) -> bool:
    cap_type = safe_text(prop.get("cap_type") or prop.get("category"))
    status = safe_text(prop.get("status"))
    temporality = normalize_text(prop.get("temporality"))
    text = normalize_text(prop.get("proposition_text"))
    concept = safe_text(prop.get("canonical_concept")) or safe_text(prop.get("proposition_text"))
    if status == "negated":
        return False
    if cap_type not in {"ChiefComplaint", "Problem", "Diagnosis", "Impression"}:
        return False
    if any(phrase in text for phrase in ["is taking", "takes ", "denies ", "allergy", "family history", "social history", "follow up", "referred", "ordered"]):
        return False
    if canonical_problem_family(prop.get("canonical_concept") or text) in {canonical_problem_family(x) for x in LOW_VALUE_BUNDLE_ANCHOR_CONCEPTS}:
        return False
    if cap_type in {"Diagnosis", "Impression"}:
        return True
    if temporality in {"past", "historical", "history"} and cap_type != "ChiefComplaint":
        return is_disease_like_problem_text(concept)
    return True


def select_anchor_problem_cap_id(cluster: Dict[str, Any], cap_lookup: Dict[str, Dict[str, Any]]) -> Optional[str]:
    best_cap_id: Optional[str] = None
    best_score = -1
    for cap_id in cluster.get("supporting_cap_ids") or []:
        cap_id = safe_text(cap_id)
        prop = cap_lookup.get(cap_id, {})
        if not cap_id or not cap_is_anchor_eligible(prop):
            continue
        cap_type = safe_text(prop.get("cap_type") or prop.get("category"))
        concept_text = safe_text(prop.get("canonical_concept")) or safe_text(prop.get("proposition_text"))
        score = cap_anchor_priority_score(cap_type)
        if is_disease_like_problem_text(concept_text):
            score += 2
        if is_surface_symptom_anchor_text(concept_text):
            score -= 2
        if normalize_text(prop.get("speaker")) == "doctor":
            score += 1
        if score > best_score:
            best_score = score
            best_cap_id = cap_id
    return best_cap_id


def render_sections_from_cluster_type(cluster_type: str) -> List[str]:
    mapping = {
        "ChiefComplaint": ["S", "Chief Complaint", "History of Present Illness"],
        "PMH": ["Past Medical History", "History of Present Illness"],
        "DiagnosisImpression": ["A", "Assessment"],
        "ProblemState": ["S", "A", "History of Present Illness", "Assessment"],
        "MedicationState": ["S", "History of Present Illness"],
        "MedicationChange": ["P", "Plan"],
        "Plan": ["P", "Plan"],
        "FollowUp": ["P", "Plan"],
        "TestOrResult": ["O", "Findings"],
        "TestResult": ["O", "Findings"],
    }
    return mapping.get(cluster_type, ["Assessment"])


def event_type_from_problem_cluster(cluster: Dict[str, Any]) -> str:
    cluster_type = safe_text(cluster.get("cluster_type"))
    primary_cap_type = safe_text(cluster.get("primary_cap_type"))
    if cluster_type in {"MedicationState", "MedicationChange"}:
        return cluster_type
    if cluster_type == "ProblemState":
        if primary_cap_type == "ChiefComplaint":
            return "ChiefComplaint"
        if primary_cap_type == "ProblemHistory":
            return "PMH"
        if primary_cap_type in {"Diagnosis", "Impression", "Allergy"}:
            return "DiagnosisImpression"
        return "SymptomCourse"
    if cluster_type in {"Diagnosis", "DiagnosisImpression"}:
        return "DiagnosisImpression"
    if cluster_type in {"TestOrResult", "TestResult"}:
        return "TestResult"
    if cluster_type == "Plan":
        return "PlanOrder"
    if cluster_type == "FollowUp":
        return "FollowUp"
    return "SymptomCourse"


def clinical_priority_from_problem_cluster(cluster: Dict[str, Any], event_type: str) -> str:
    salience = int(cluster.get("salience_score") or 0)
    primary_cap_type = safe_text(cluster.get("primary_cap_type"))
    if event_type in {"MedicationChange", "PlanOrder", "FollowUp", "DiagnosisImpression", "TestResult", "ChiefComplaint"}:
        return "high"
    if primary_cap_type in {"Order", "MedicationRequest", "Diagnosis", "Impression"}:
        return "high"
    if salience >= 9:
        return "high"
    if salience >= 6:
        return "medium"
    return "low"


def should_emit_problem_cluster_event(cluster: Dict[str, Any]) -> bool:
    if cluster.get("is_low_value"):
        return False
    concept = normalize_text(cluster.get("canonical_concept"))
    if concept in {
        "sandwich",
        "diet",
        "bread consumption",
        "water intake",
        "club soda",
        "french fries",
        "bad food habits",
    }:
        return False
    return True


def render_sections_from_problem_cluster(cluster: Dict[str, Any], event_type: str) -> List[str]:
    if event_type in {"ChiefComplaint", "PMH", "DiagnosisImpression"}:
        return render_sections_from_cluster_type(event_type)
    return render_sections_from_cluster_type(safe_text(cluster.get("cluster_type")))


def cluster_turn_span(cluster: Dict[str, Any]) -> Tuple[int, int]:
    indices: List[int] = []
    for turn_id in cluster.get("evidence_turn_ids") or []:
        m = re.fullmatch(r"[TS](\d+)", safe_text(turn_id))
        if m:
            indices.append(int(m.group(1)))
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


def cluster_assertion_status(cluster: Dict[str, Any]) -> str:
    return (
        verification_to_status(cluster.get("final_verification_status"))
        or assertion_to_status(cluster.get("final_assertion"))
        or "affirmed"
    )


def is_past_only_cluster(cluster: Dict[str, Any]) -> bool:
    values = [normalize_text(x) for x in (cluster.get("temporal_profile") or []) if normalize_text(x)]
    if not values:
        return False
    current_markers = {
        "current",
        "today",
        "this morning",
        "this week",
        "now",
        "weeks",
        "months",
        "days",
        "active",
        "present",
        "recent",
    }
    if any(value in current_markers for value in values):
        return False
    past_markers = {
        "past",
        "previous",
        "historical",
        "history",
        "prior",
        "last december",
        "during accident",
        "2018",
        "2019",
        "2020",
        "2021",
        "age 20",
        "years ago",
        "post-menopausal",
    }
    return all(value in past_markers or re.search(r"\b(19|20)\d{2}\b", value) for value in values)


def is_problem_bundle_anchor(cluster: Dict[str, Any], cap_lookup: Dict[str, Dict[str, Any]]) -> bool:
    concept = normalize_text(cluster.get("canonical_concept"))
    summary = normalize_text(cluster.get("cluster_summary"))
    cluster_type = safe_text(cluster.get("cluster_type"))
    primary_cap_type = safe_text(cluster.get("primary_cap_type"))
    status = cluster_assertion_status(cluster)
    visit_state = normalize_text(cluster.get("clinical_status") or cluster.get("visit_state"))
    speakers = {normalize_text(x) for x in (cluster.get("source_speakers") or []) if normalize_text(x)}
    temporal_values = [normalize_text(x) for x in (cluster.get("temporal_profile") or []) if normalize_text(x)]

    if status == "negated":
        return False
    if cluster.get("is_low_value"):
        return False
    if not select_anchor_problem_cap_id(cluster, cap_lookup):
        return False
    if concept in {normalize_text(x) for x in LOW_VALUE_BUNDLE_ANCHOR_CONCEPTS}:
        return False
    if is_screening_or_test_like_text(concept):
        return False
    if "establish care" in summary or "performance status" in summary:
        return False
    if any(phrase in summary for phrase in ["will order", "ordered", "referred", "prescribed", "consulted", "schedule her for"]):
        return False
    if any(phrase in summary for phrase in ["is taking", "states she takes", "received an eye prescription", "dating her partner", "partner started cheating"]):
        return False
    if cluster_type in {"MedicationState", "MedicationChange", "Plan", "FollowUp", "TestOrResult", "TestResult"}:
        return False
    if primary_cap_type in {"ProblemHistory", "Order", "MedicationRequest", "MedicationStatement", "FollowUp", "Counseling", "ExamFinding", "TestResult", "Allergy"}:
        return False
    if concept_matches_any(summary, ["quit smoking", "still smoking", "partner started cheating", "received an eye prescription", "new glasses", "drinks in the evening"]):
        return False
    if speakers == {"clinician"} and primary_cap_type not in {"Diagnosis", "Impression"}:
        return False
    if temporal_values and all(value == "future" for value in temporal_values) and primary_cap_type not in {"Diagnosis", "Impression"}:
        return False
    return cluster_type in {"ProblemState", "Diagnosis", "DiagnosisImpression"} or primary_cap_type in {
        "ChiefComplaint",
        "Problem",
        "Diagnosis",
        "Impression",
    }


def is_symptom_like_anchor(cluster: Dict[str, Any]) -> bool:
    cluster_type = safe_text(cluster.get("cluster_type"))
    primary_cap_type = safe_text(cluster.get("primary_cap_type"))
    concept_text = cap_or_cluster_text(cluster.get("canonical_concept"), cluster.get("cluster_summary"))
    if is_disease_like_problem_text(concept_text):
        return False
    return cluster_type == "ProblemState" and primary_cap_type in {"ChiefComplaint", "Problem"}


def cluster_temporal_layer(cluster: Dict[str, Any]) -> str:
    temporal_values = [normalize_text(x) for x in (cluster.get("temporal_profile") or []) if normalize_text(x)]
    visit_state = normalize_text(cluster.get("clinical_status") or cluster.get("visit_state"))
    summary = normalize_text(cluster.get("cluster_summary"))
    if visit_state in {"planned", "future"}:
        return "future"
    if temporal_values and all(value == "future" for value in temporal_values):
        return "future"
    if is_past_only_cluster(cluster):
        return "history"
    if any(phrase in summary for phrase in ["history of", "status post", "s/p ", "previously", "years ago", "last mammogram", "prior "]):
        return "history"
    return "current"


def plausible_related_factor(anchor_text: str, candidate_text: str) -> bool:
    anchor_norm = normalize_text(anchor_text)
    cand_norm = normalize_text(candidate_text)
    if not anchor_norm or not cand_norm:
        return False
    if any(term in anchor_norm for term in ["ulcer", "wound"]):
        return any(term in cand_norm for term in ["diabetes", "hyperglycemia", "vascular disease", "blood supply", "smoking", "peripheral vascular disease"])
    if any(term in anchor_norm for term in ["diabetes", "hyperglycemia"]):
        return any(term in cand_norm for term in ["neuropathy", "vascular disease", "hyperlipidemia", "hypertension", "smoking"])
    if any(term in anchor_norm for term in ["vaginal", "pelvic", "std", "sexual"]):
        return any(term in cand_norm for term in ["infidelity", "exposure", "screening", "pap smear", "testing"])
    if any(term in anchor_norm for term in ["breast", "dcis", "carcinoma", "cancer"]):
        return any(term in cand_norm for term in ["lumpectomy", "mammogram", "radiation", "recurrence", "ecog", "lymph node"])
    if is_chronic_condition_text(cand_norm) and any(
        term in anchor_norm for term in ["ulcer", "wound", "kidney stone", "nephrolithiasis", "vaginal", "pelvic"]
    ):
        return True
    return False


def classify_bundle_relation(
    anchor: Dict[str, Any],
    candidate: Dict[str, Any],
    cap_lookup: Optional[Dict[str, Dict[str, Any]]] = None,
) -> str:
    anchor_text = cap_or_cluster_text(anchor.get("canonical_concept"), anchor.get("cluster_summary"))
    cand_text = cap_or_cluster_text(candidate.get("canonical_concept"), candidate.get("cluster_summary"))
    anchor_family = canonical_problem_family(anchor_text)
    cand_family = canonical_problem_family(cand_text)
    if anchor_family and cand_family and anchor_family == cand_family:
        return "direct"
    if relaxed_token_f1(anchor_text, cand_text) >= 0.55:
        return "direct"
    anchor_region = infer_body_region(anchor_text)
    cand_region = infer_body_region(cand_text)
    anchor_symptom = infer_symptom_family(anchor_text)
    cand_symptom = infer_symptom_family(cand_text)
    if anchor_region and cand_region and anchor_region == cand_region:
        if anchor_symptom and cand_symptom and anchor_symptom == cand_symptom:
            return "direct"
        if problem_bundle_slot(candidate) in {"O", "P"}:
            return "direct"
    med_hints = medication_problem_hints(cand_text)
    if med_hints and any(hint in anchor_family for hint in med_hints):
        return "direct"
    if plausible_related_factor(anchor_text, cand_text):
        return "related_factor"
    if (
        is_disease_like_problem_text(anchor_text)
        and is_disease_like_problem_text(cand_text)
        and is_chronic_condition_text(cand_text)
        and anchor_family
        and cand_family
        and anchor_family != cand_family
    ):
        return "related_factor"
    anchor_domains = problem_domains(anchor_text)
    cand_domains = problem_domains(cand_text)
    if anchor_domains and cand_domains and anchor_domains.isdisjoint(cand_domains):
        return "unrelated"
    if anchor_domains and not cand_domains and cluster_anchor_preference(anchor) >= 4 and problem_bundle_slot(candidate) == "G":
        return "unrelated"
    if (
        anchor_domains
        and not cand_domains
        and cluster_anchor_preference(anchor) >= 4
        and safe_text(candidate.get("primary_cap_type")) == "ProblemHistory"
        and not infer_body_region(cand_text)
    ):
        return "unrelated"
    if cluster_anchor_preference(candidate) >= 4 and anchor_family and cand_family and anchor_family != cand_family:
        return "unrelated"
    if problem_bundle_slot(candidate) == "G" and anchor_domains and cand_domains and anchor_domains.isdisjoint(cand_domains):
        return "unrelated"
    return "weak"


def problem_bundle_slot(cluster: Dict[str, Any]) -> str:
    cluster_type = safe_text(cluster.get("cluster_type"))
    primary_cap_type = safe_text(cluster.get("primary_cap_type"))
    status = cluster_assertion_status(cluster)
    speakers = {normalize_text(x) for x in (cluster.get("source_speakers") or []) if normalize_text(x)}
    summary = normalize_text(cluster.get("cluster_summary"))

    if cluster_type in {"Plan", "FollowUp", "MedicationChange"} or primary_cap_type in {"Order", "MedicationRequest", "FollowUp", "Counseling"}:
        return "P"
    if cluster_type in {"TestOrResult", "TestResult"} or primary_cap_type in {"ExamFinding", "TestResult"}:
        return "O"
    if primary_cap_type in {"Diagnosis", "Impression"}:
        return "A"
    if cluster_type == "MedicationState" or primary_cap_type == "MedicationStatement":
        if any(phrase in summary for phrase in ["increase ", "decrease ", "start ", "stop ", "switch ", "continue ", "new prescription", "recommending glimepiride", "one pill twice a day"]):
            return "P"
        return "S"
    if primary_cap_type == "ProblemHistory":
        return "G" if is_past_only_cluster(cluster) else "S"
    if status == "negated":
        return "S"
    if "patient" in speakers or primary_cap_type in {"ChiefComplaint", "Problem"}:
        return "S"
    return "A"


def bundle_anchor_score(anchor: Dict[str, Any], candidate: Dict[str, Any], cap_lookup: Optional[Dict[str, Dict[str, Any]]] = None) -> float:
    anchor_concept = safe_text(anchor.get("canonical_concept"))
    cand_concept = safe_text(candidate.get("canonical_concept"))
    score = 0.0
    anchor_family = cluster_problem_family(anchor)
    cand_family = cluster_problem_family(candidate)
    if anchor_concept and cand_concept:
        overlap = relaxed_token_f1(anchor_concept, cand_concept)
        score += overlap * 2.5
        norm_anchor = normalize_alignment_text(anchor_concept)
        norm_cand = normalize_alignment_text(cand_concept)
        if norm_anchor and norm_cand and (norm_anchor in norm_cand or norm_cand in norm_anchor):
            score += 1.5
        if anchor_family and cand_family and anchor_family == cand_family:
            score += 1.5
    gap = turn_span_gap(cluster_turn_span(anchor), cluster_turn_span(candidate))
    if gap == 0:
        score += 1.5
    elif gap <= 2:
        score += 1.0
    elif gap <= 6:
        score += 0.5
    slot = problem_bundle_slot(candidate)
    if slot in {"O", "P"}:
        score += 0.75
    elif slot == "S":
        score += 0.25
    anchor_text = cap_or_cluster_text(anchor.get("canonical_concept"), anchor.get("cluster_summary"))
    cand_text = cap_or_cluster_text(candidate.get("canonical_concept"), candidate.get("cluster_summary"))
    anchor_region = infer_body_region(anchor_text)
    cand_region = infer_body_region(cand_text)
    anchor_preference = cluster_anchor_preference(anchor)
    if anchor_region and cand_region and anchor_region == cand_region:
        score += 0.75
    anchor_symptom_family = infer_symptom_family(anchor_text)
    cand_symptom_family = infer_symptom_family(cand_text)
    if anchor_symptom_family and cand_symptom_family and anchor_symptom_family == cand_symptom_family:
        score += 0.5
    if anchor_preference >= 5 and slot == "S" and gap <= 4:
        score += 1.0
    med_hints = medication_problem_hints(cand_text)
    if med_hints:
        normalized_anchor_text = canonical_problem_family(anchor_text)
        if any(hint in normalized_anchor_text for hint in med_hints):
            score += 3.0
        elif slot == "P" and anchor_preference <= 3:
            score -= 1.0
    elif slot == "P" and anchor_preference <= 3 and not (anchor_region and cand_region and anchor_region == cand_region):
        score -= 0.75
    if cluster_anchor_preference(anchor) >= 4 and slot in {"O", "P"}:
        score += 0.5
    if problem_bundle_slot(candidate) == "G":
        score -= 0.25
    relation = classify_bundle_relation(anchor, candidate, cap_lookup)
    if relation == "direct":
        score += 0.75
    elif relation == "related_factor":
        score += 0.15
    elif relation == "weak":
        score -= 0.1
    else:
        score -= 4.0
    return score


def should_keep_distinct_anchor(
    existing_anchors: Sequence[Dict[str, Any]],
    candidate: Dict[str, Any],
    cap_lookup: Optional[Dict[str, Dict[str, Any]]] = None,
) -> bool:
    if not existing_anchors:
        return True
    candidate_is_symptom = is_symptom_like_anchor(candidate)
    candidate_is_diagnostic = not candidate_is_symptom
    for anchor in existing_anchors:
        score = bundle_anchor_score(anchor, candidate, cap_lookup)
        same_region = infer_body_region(cap_or_cluster_text(anchor.get("canonical_concept"), anchor.get("cluster_summary"))) and (
            infer_body_region(cap_or_cluster_text(anchor.get("canonical_concept"), anchor.get("cluster_summary")))
            == infer_body_region(cap_or_cluster_text(candidate.get("canonical_concept"), candidate.get("cluster_summary")))
        )
        same_domain = bool(
            problem_domains(cap_or_cluster_text(anchor.get("canonical_concept"), anchor.get("cluster_summary")))
            & problem_domains(cap_or_cluster_text(candidate.get("canonical_concept"), candidate.get("cluster_summary")))
        )
        if candidate_is_symptom and not is_symptom_like_anchor(anchor) and score >= 0.75:
            return False
        if candidate_is_symptom and not is_symptom_like_anchor(anchor) and (same_region or same_domain) and score >= 0.25:
            return False
        if candidate_is_symptom and is_symptom_like_anchor(anchor) and score >= 1.0:
            return False
        if candidate_is_diagnostic and not is_symptom_like_anchor(anchor) and score >= 2.0:
            return False
        if candidate_is_diagnostic and not is_symptom_like_anchor(anchor) and turn_span_gap(cluster_turn_span(anchor), cluster_turn_span(candidate)) <= 2:
            return False
    return True


def bundle_problem_domains(bundle: Dict[str, Any]) -> set[str]:
    text_parts = list(bundle.get("canonical_concepts") or [])
    text_parts.extend(bundle.get("history_caps") or [])
    text_parts.extend(bundle.get("related_factor_caps") or [])
    joined = " ".join(safe_text(part) for part in text_parts if safe_text(part))
    return problem_domains(joined)


def promoted_bundle_problem(bundle: Dict[str, Any], cap_lookup: Dict[str, Dict[str, Any]]) -> Tuple[Optional[str], Optional[str]]:
    candidate_ids: List[str] = []
    seen: set[str] = set()
    for cap_id in (
        (bundle.get("A_cap_ids") or [])
        + (bundle.get("S_cap_ids") or [])
        + (bundle.get("history_cap_ids") or [])
        + (bundle.get("related_factor_cap_ids") or [])
    ):
        cap_id = safe_text(cap_id)
        if cap_id and cap_id not in seen:
            seen.add(cap_id)
            candidate_ids.append(cap_id)

    bundle_domains = bundle_problem_domains(bundle)
    best_cap_id: Optional[str] = safe_text(bundle.get("anchor_problem_cap_id")) or None
    best_concept: Optional[str] = safe_text(bundle.get("anchor_problem")) or None
    best_score = -1.0

    for cap_id in candidate_ids:
        prop = cap_lookup.get(cap_id)
        if not prop:
            continue
        if not cap_can_name_bundle_problem(prop):
            continue
        cap_type = safe_text(prop.get("cap_type") or prop.get("category"))
        concept = safe_text(prop.get("canonical_concept")) or safe_text(prop.get("proposition_text"))
        if not concept or is_noise_context_text(concept) or is_screening_or_test_like_text(concept):
            continue
        score = float(cap_anchor_priority_score(cap_type))
        if is_disease_like_problem_text(concept):
            score += 3.0
        if is_surface_symptom_anchor_text(concept):
            score -= 2.5
        if cap_type == "ProblemHistory":
            score -= 1.0
        if normalize_text(prop.get("speaker")) == "doctor":
            score += 0.5
        candidate_domains = problem_domains(concept)
        if bundle_domains and candidate_domains and not bundle_domains.isdisjoint(candidate_domains):
            score += 1.0
        if cluster_problem_family({"canonical_concept": concept}) == cluster_problem_family({"canonical_concept": bundle.get("anchor_problem")}):
            score += 0.5
        if score > best_score:
            best_score = score
            best_cap_id = cap_id
            best_concept = concept
    return best_cap_id, best_concept


def preferred_bundle_problem(bundle: Dict[str, Any], cap_lookup: Dict[str, Dict[str, Any]]) -> Tuple[Optional[str], Optional[str]]:
    promoted_cap_id, promoted_concept = promoted_bundle_problem(bundle, cap_lookup)
    if promoted_concept:
        return promoted_cap_id, promoted_concept
    return safe_text(bundle.get("anchor_problem_cap_id")) or None, safe_text(bundle.get("anchor_problem")) or None


def should_drop_bundle_cluster(anchor: Dict[str, Any], candidate: Dict[str, Any], cap_lookup: Dict[str, Dict[str, Any]]) -> bool:
    candidate_text = cap_or_cluster_text(candidate.get("canonical_concept"), candidate.get("cluster_summary"))
    if is_noise_context_text(candidate_text):
        return True
    if is_screening_or_test_like_text(candidate_text) and cluster_anchor_preference(candidate) <= 1:
        return True
    relation = classify_bundle_relation(anchor, candidate, cap_lookup)
    if relation == "unrelated":
        return True
    slot = problem_bundle_slot(candidate)
    if relation == "weak" and slot == "G":
        return True
    if relation == "weak" and cluster_anchor_preference(anchor) >= 4 and is_symptom_like_anchor(candidate):
        anchor_text = cap_or_cluster_text(anchor.get("canonical_concept"), anchor.get("cluster_summary"))
        same_region = infer_body_region(anchor_text) and infer_body_region(anchor_text) == infer_body_region(candidate_text)
        same_domain = bool(problem_domains(anchor_text) & problem_domains(candidate_text))
        same_symptom_family = infer_symptom_family(anchor_text) and infer_symptom_family(anchor_text) == infer_symptom_family(candidate_text)
        if not same_region and not same_domain and not same_symptom_family:
            return True
    return False


def anchor_decision_score(
    anchor: Dict[str, Any],
    candidate_clusters: Sequence[Dict[str, Any]],
    cap_lookup: Dict[str, Dict[str, Any]],
) -> float:
    preference, plan_links, support_score, salience = anchor_support_profile(anchor, candidate_clusters, cap_lookup)
    non_subjective_support = anchor_non_subjective_support(anchor, candidate_clusters, cap_lookup)
    concept_text = cap_or_cluster_text(anchor.get("canonical_concept"), anchor.get("cluster_summary"))
    score = 0.0
    score += preference * ANCHOR_DECISION_WEIGHTS["preference"]
    score += min(plan_links, 3) * ANCHOR_DECISION_WEIGHTS["plan_link"]
    score += non_subjective_support * ANCHOR_DECISION_WEIGHTS["non_subjective_support"]
    score += support_score * ANCHOR_DECISION_WEIGHTS["support_score"]
    score += min(salience, 10) * ANCHOR_DECISION_WEIGHTS["salience"]
    if is_disease_like_problem_text(concept_text):
        score += 3.0
    if problem_domains(concept_text):
        score += 0.5
    if is_surface_symptom_anchor_text(concept_text):
        score -= 3.5
    if is_noise_context_text(concept_text) or is_screening_or_test_like_text(concept_text):
        score -= 5.0
    if cluster_temporal_layer(anchor) == "history" and preference < 5:
        score -= 2.0
    return score


def anchor_support_profile(
    anchor: Dict[str, Any],
    candidate_clusters: Sequence[Dict[str, Any]],
    cap_lookup: Dict[str, Dict[str, Any]],
) -> Tuple[int, int, float, int]:
    plan_links = 0
    support_score = 0.0
    for candidate in candidate_clusters:
        if safe_text(candidate.get("cluster_id")) == safe_text(anchor.get("cluster_id")):
            continue
        slot = problem_bundle_slot(candidate)
        score = bundle_anchor_score(anchor, candidate, cap_lookup)
        threshold = 0.4 if slot in {"O", "P"} else 1.0
        if score < threshold:
            continue
        if slot == "P":
            plan_links += 1
            support_score += 2.5
        elif slot == "O":
            support_score += 1.5
        elif slot == "A":
            support_score += 2.0
        elif slot == "S":
            support_score += 0.5
    return (
        cluster_anchor_preference(anchor),
        plan_links,
        support_score,
        int(anchor.get("salience_score") or 0),
    )


def anchor_non_subjective_support(
    anchor: Dict[str, Any],
    candidate_clusters: Sequence[Dict[str, Any]],
    cap_lookup: Dict[str, Dict[str, Any]],
) -> int:
    support = 0
    for candidate in candidate_clusters:
        if safe_text(candidate.get("cluster_id")) == safe_text(anchor.get("cluster_id")):
            continue
        score = bundle_anchor_score(anchor, candidate, cap_lookup)
        slot = problem_bundle_slot(candidate)
        threshold = 0.4 if slot in {"O", "P"} else 1.0
        if score < threshold:
            continue
        if slot in {"O", "A", "P"} or classify_bundle_relation(anchor, candidate, cap_lookup) == "related_factor":
            support += 1
    return support


def init_problem_bundle(anchor: Dict[str, Any], cap_lookup: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    concept = safe_text(anchor.get("canonical_concept"))
    anchor_problem_cap_id = select_anchor_problem_cap_id(anchor, cap_lookup)
    return {
        "anchor_cluster": anchor,
        "anchor_problem_cap_id": anchor_problem_cap_id,
        "anchor_problem": concept,
        "title": concept,
        "event_type": "ProblemOrientedNoteBundle",
        "supporting_cap_ids": [],
        "S_cap_ids": [],
        "O_cap_ids": [],
        "A_cap_ids": [],
        "P_cap_ids": [],
        "global_cap_ids": [],
        "S_caps": [],
        "O_caps": [],
        "A_caps": [],
        "P_caps": [],
        "global_context_caps": [],
        "history_cap_ids": [],
        "history_caps": [],
        "related_factor_cap_ids": [],
        "related_factor_caps": [],
        "temporal_profile": [],
        "source_modalities": [],
        "source_speakers": [],
        "canonical_concepts": [concept] if concept else [],
    }


def append_unique_summary(target: List[str], summary: str) -> None:
    norm = normalize_text(summary)
    if not summary or not norm:
        return
    if any(normalize_text(item) == norm for item in target):
        return
    target.append(summary)


def append_unique_cap_id(target: List[str], cap_id: str) -> None:
    cap_id = safe_text(cap_id)
    if cap_id and cap_id not in target:
        target.append(cap_id)


def add_cluster_to_problem_bundle(bundle: Dict[str, Any], cluster: Dict[str, Any], cap_lookup: Dict[str, Dict[str, Any]]) -> None:
    concept = safe_text(cluster.get("canonical_concept"))
    summary = safe_text(cluster.get("cluster_summary"))
    if not summary:
        return
    if safe_text(cluster.get("cluster_id")) != safe_text(bundle["anchor_cluster"].get("cluster_id")) and should_drop_bundle_cluster(
        bundle["anchor_cluster"], cluster, cap_lookup
    ):
        return
    relation = classify_bundle_relation(bundle["anchor_cluster"], cluster, cap_lookup)
    if relation == "unrelated":
        return
    temporal_layer = cluster_temporal_layer(cluster)
    if concept and concept not in bundle["canonical_concepts"]:
        bundle["canonical_concepts"].append(concept)
    for cap_id in cluster.get("supporting_cap_ids") or []:
        cap_id = safe_text(cap_id)
        if cap_id and cap_id not in bundle["supporting_cap_ids"]:
            bundle["supporting_cap_ids"].append(cap_id)
    for value in [safe_text(x) for x in (cluster.get("temporal_profile") or []) if safe_text(x)]:
        if value not in bundle["temporal_profile"]:
            bundle["temporal_profile"].append(value)
    for value in [safe_text(x) for x in (cluster.get("source_modalities") or []) if safe_text(x)]:
        if value not in bundle["source_modalities"]:
            bundle["source_modalities"].append(value)
    for value in [safe_text(x) for x in (cluster.get("source_speakers") or []) if safe_text(x)]:
        if value not in bundle["source_speakers"]:
            bundle["source_speakers"].append(value)

    slotted_any = False
    for cap_id in cluster.get("supporting_cap_ids") or []:
        cap_id = safe_text(cap_id)
        prop = cap_lookup.get(cap_id)
        if not prop:
            continue
        cap_text = safe_text(prop.get("proposition_text"))
        slot = cap_slot_from_prop(prop)
        if relation == "related_factor" and slot == "P":
            append_unique_cap_id(bundle["P_cap_ids"], cap_id)
            append_unique_summary(bundle["P_caps"], cap_text)
        elif relation == "related_factor":
            append_unique_cap_id(bundle["related_factor_cap_ids"], cap_id)
            append_unique_summary(bundle["related_factor_caps"], cap_text)
        elif relation == "weak" and slot in {"G", "S"} and temporal_layer != "current":
            append_unique_cap_id(bundle["related_factor_cap_ids"], cap_id)
            append_unique_summary(bundle["related_factor_caps"], cap_text)
        elif temporal_layer == "history" and slot != "P":
            append_unique_cap_id(bundle["history_cap_ids"], cap_id)
            append_unique_summary(bundle["history_caps"], cap_text)
        elif slot == "S":
            append_unique_cap_id(bundle["S_cap_ids"], cap_id)
            append_unique_summary(bundle["S_caps"], cap_text)
        elif slot == "O":
            append_unique_cap_id(bundle["O_cap_ids"], cap_id)
            append_unique_summary(bundle["O_caps"], cap_text)
        elif slot == "A":
            append_unique_cap_id(bundle["A_cap_ids"], cap_id)
            append_unique_summary(bundle["A_caps"], cap_text)
        elif slot == "P":
            append_unique_cap_id(bundle["P_cap_ids"], cap_id)
            append_unique_summary(bundle["P_caps"], cap_text)
        else:
            append_unique_cap_id(bundle["global_cap_ids"], cap_id)
            append_unique_summary(bundle["global_context_caps"], cap_text)
        slotted_any = True

    if slotted_any:
        return

    slot = problem_bundle_slot(cluster)
    if relation == "related_factor" and slot == "P":
        append_unique_summary(bundle["P_caps"], summary)
    elif relation == "related_factor":
        append_unique_summary(bundle["related_factor_caps"], summary)
    elif relation == "weak" and slot in {"G", "S"} and temporal_layer != "current":
        append_unique_summary(bundle["related_factor_caps"], summary)
    elif temporal_layer == "history" and slot != "P":
        append_unique_summary(bundle["history_caps"], summary)
    elif slot == "S":
        append_unique_summary(bundle["S_caps"], summary)
    elif slot == "O":
        append_unique_summary(bundle["O_caps"], summary)
    elif slot == "A":
        append_unique_summary(bundle["A_caps"], summary)
    elif slot == "P":
        append_unique_summary(bundle["P_caps"], summary)
    else:
        append_unique_summary(bundle["global_context_caps"], summary)


def merge_problem_bundle_into(target: Dict[str, Any], source: Dict[str, Any]) -> None:
    for key in [
        "supporting_cap_ids",
        "S_cap_ids",
        "O_cap_ids",
        "A_cap_ids",
        "P_cap_ids",
        "global_cap_ids",
        "history_cap_ids",
        "related_factor_cap_ids",
        "canonical_concepts",
        "temporal_profile",
        "source_modalities",
        "source_speakers",
    ]:
        for value in source.get(key) or []:
            if value not in target[key]:
                target[key].append(value)
    for key in [
        "S_caps",
        "O_caps",
        "A_caps",
        "P_caps",
        "global_context_caps",
        "history_caps",
        "related_factor_caps",
    ]:
        for value in source.get(key) or []:
            append_unique_summary(target[key], value)


def should_merge_surface_symptom_bundle(source: Dict[str, Any], target: Dict[str, Any], cap_lookup: Dict[str, Dict[str, Any]]) -> bool:
    source_anchor = source["anchor_cluster"]
    target_anchor = target["anchor_cluster"]
    source_text = safe_text(source.get("anchor_problem")) or cap_or_cluster_text(
        source_anchor.get("canonical_concept"), source_anchor.get("cluster_summary")
    )
    if not is_surface_symptom_anchor_text(source_text):
        return False
    if source.get("O_caps") or source.get("A_caps"):
        return False
    if len(source.get("P_caps") or []) > 1:
        return False
    if cluster_anchor_preference(target_anchor) < 4:
        return False
    relation = classify_bundle_relation(target_anchor, source_anchor, cap_lookup)
    if relation == "unrelated":
        return False
    target_text = safe_text(target.get("anchor_problem")) or cap_or_cluster_text(
        target_anchor.get("canonical_concept"), target_anchor.get("cluster_summary")
    )
    same_region = infer_body_region(target_text) and infer_body_region(target_text) == infer_body_region(source_text)
    same_domain = bool(problem_domains(target_text) & problem_domains(source_text))
    same_symptom_family = infer_symptom_family(target_text) and infer_symptom_family(target_text) == infer_symptom_family(source_text)
    if same_region or same_domain or same_symptom_family:
        return True
    score = bundle_anchor_score(target_anchor, source_anchor, cap_lookup)
    return score >= 0.25


def conservative_assessment_from_bundle(bundle: Dict[str, Any]) -> Optional[str]:
    anchor_text = safe_text(bundle.get("anchor_problem"))
    if not anchor_text:
        return None
    norm = normalize_text(anchor_text)
    if any(term in norm for term in ["dcis", "carcinoma", "cancer", "breast cancer"]):
        return f"The patient is being followed for {anchor_text} without evidence of recurrence."
    if any(term in norm for term in ["kidney stone", "nephrolithiasis"]):
        return f"Findings are consistent with {anchor_text}."
    if any(term in norm for term in ["carpal tunnel", "cubital tunnel", "neuropathy"]):
        return f"Findings are concerning for {anchor_text}."
    if any(term in norm for term in ["std", "infection"]):
        return f"Symptoms are concerning for {anchor_text}."
    if any(term in norm for term in ["ulcer", "wound"]):
        return f"The patient has {anchor_text} requiring ongoing management."
    if is_disease_like_problem_text(anchor_text):
        return f"The patient has {anchor_text}."
    return None


def is_action_like_text(text: Any) -> bool:
    norm = normalize_text(text)
    if not norm:
        return False
    return any(
        phrase in norm
        for phrase in [
            "consult",
            "continue ",
            "follow-up",
            "follow up",
            "increase ",
            "order ",
            "ordered",
            "recommend",
            "referral",
            "refer",
            "schedule",
            "should ",
            "start ",
            "stop ",
            "testing",
            "undergo",
            "will ",
        ]
    )


def bundle_render_profile(bundle: Dict[str, Any]) -> Dict[str, Any]:
    anchor = bundle["anchor_cluster"]
    anchor_text = safe_text(bundle.get("anchor_problem")) or cap_or_cluster_text(
        anchor.get("canonical_concept"), anchor.get("cluster_summary")
    )
    s_count = len(bundle.get("S_cap_ids") or bundle.get("S_caps") or [])
    o_count = len(bundle.get("O_cap_ids") or bundle.get("O_caps") or [])
    a_count = len(bundle.get("A_cap_ids") or bundle.get("A_caps") or [])
    p_count = len(bundle.get("P_cap_ids") or bundle.get("P_caps") or [])
    related_count = len(bundle.get("related_factor_cap_ids") or bundle.get("related_factor_caps") or []) + len(
        bundle.get("history_cap_ids") or bundle.get("history_caps") or []
    )
    related_action_count = sum(1 for item in (bundle.get("related_factor_caps") or []) if is_action_like_text(item))
    support_count = len(bundle.get("supporting_cap_ids") or [])
    primary = False
    if p_count > 0 or a_count > 0:
        primary = True
    elif related_action_count > 0:
        primary = True
    elif cluster_anchor_preference(anchor) >= 4 and (o_count > 0 or related_count > 0):
        primary = True
    elif is_disease_like_problem_text(anchor_text) and support_count >= 2:
        primary = True
    incidental = s_count <= 1 and o_count == 0 and a_count == 0 and p_count == 0 and related_count == 0
    render_in_note = primary or p_count > 0 or a_count > 0 or related_action_count > 0
    if incidental and not (p_count > 0 or a_count > 0):
        render_in_note = False
    return {
        "visit_primary": primary,
        "render_in_note": render_in_note,
        "support_count": support_count,
    }


def finalize_problem_bundle_event(bundle: Dict[str, Any], idx: int, cap_lookup: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    anchor = bundle["anchor_cluster"]
    preferred_cap_id, preferred_concept = preferred_bundle_problem(bundle, cap_lookup)
    if preferred_cap_id:
        bundle["anchor_problem_cap_id"] = preferred_cap_id
    if preferred_concept:
        bundle["anchor_problem"] = preferred_concept
        bundle["title"] = preferred_concept
    visit_state = safe_text(anchor.get("clinical_status") or anchor.get("visit_state")) or "active"
    status = cluster_assertion_status(anchor)
    temporality = safe_text((bundle.get("temporal_profile") or [None])[-1]) or ("current" if visit_state != "planned" else "future")
    title = bundle["title"] or safe_text(anchor.get("cluster_summary")) or f"Bundle {idx}"
    event_summary = f"Problem-oriented bundle for {title}."
    supporting_evidence = (bundle["S_caps"] + bundle["O_caps"] + bundle["A_caps"])[:6]
    current_state = (bundle["A_caps"] + bundle["S_caps"])[:5]
    planned_actions = bundle["P_caps"][:5]
    if not planned_actions:
        planned_actions = [item for item in (bundle["related_factor_caps"] + bundle["global_context_caps"]) if is_action_like_text(item)][:5]
    care_context = [
        item
        for item in (bundle["history_caps"] + bundle["related_factor_caps"] + bundle["global_context_caps"])
        if item not in planned_actions
    ][:6]
    render_profile = bundle_render_profile(bundle)
    assessment_fallback = conservative_assessment_from_bundle(bundle) if not bundle.get("A_cap_ids") else None
    return {
        "event_id": safe_text(anchor.get("cluster_id") or f"E{idx}") or f"E{idx}",
        "event_type": bundle["event_type"],
        "canonical_concept": bundle["anchor_problem"],
        "visit_state": visit_state,
        "status": status,
        "temporality": temporality,
        "speaker_source": ",".join(bundle.get("source_speakers") or []) or "mixed",
        "clinical_priority": clinical_priority_from_problem_cluster(anchor, event_type_from_problem_cluster(anchor)),
        "title": title,
        "event_summary": event_summary,
        "supporting_cap_ids": bundle["supporting_cap_ids"],
        "anchor_problem_cap_id": bundle.get("anchor_problem_cap_id"),
        "anchor_problem": bundle["anchor_problem"],
        "S_cap_ids": bundle["S_cap_ids"],
        "O_cap_ids": bundle["O_cap_ids"],
        "A_cap_ids": bundle["A_cap_ids"],
        "P_cap_ids": bundle["P_cap_ids"],
        "global_cap_ids": bundle["global_cap_ids"],
        "supporting_evidence": supporting_evidence,
        "current_state": current_state,
        "planned_actions": planned_actions,
        "care_context": care_context,
        "key_slots": {
            "source_modalities": ", ".join(bundle.get("source_modalities") or []),
            "temporal_profile": ", ".join(bundle.get("temporal_profile") or []),
            "anchor_cluster_type": safe_text(anchor.get("cluster_type")) or "",
            "anchor_primary_cap_type": safe_text(anchor.get("primary_cap_type")) or "",
            "bundle_concepts": "; ".join(bundle.get("canonical_concepts") or []),
            "history_context": "; ".join(bundle.get("history_caps") or []),
            "related_factors": "; ".join(bundle.get("related_factor_caps") or []),
            "visit_primary": str(render_profile["visit_primary"]).lower(),
            "render_in_note": str(render_profile["render_in_note"]).lower(),
            "assessment_fallback": assessment_fallback or "",
            "bundle_design": "problem_oriented_note_bundle",
        },
        "render_sections": ["S", "O", "A", "P", "History of Present Illness", "Findings", "Assessment", "Plan"],
    }


def convert_problem_clusters_to_event_plan(
    cluster_obj: Dict[str, Any],
    *,
    cap_obj: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    # Single-stage consolidation:
    # 1) select anchor bundles, 2) assign each remaining cluster once to best anchor,
    # 3) optional local symptom-bundle merge, 4) renderability filter.
    cap_lookup = build_cap_lookup(cap_obj)
    candidate_clusters = [
        cluster for cluster in cluster_obj.get("clusters", []) if isinstance(cluster, dict) and should_emit_problem_cluster_event(cluster)
    ]
    sorted_clusters = sorted(
        candidate_clusters,
        key=lambda cluster: (
            {"high": 2, "medium": 1, "low": 0}.get(
                clinical_priority_from_problem_cluster(cluster, event_type_from_problem_cluster(cluster)),
                0,
            ),
            int(cluster.get("salience_score") or 0),
            len(cluster.get("supporting_cap_ids") or []),
        ),
        reverse=True,
    )

    raw_anchor_candidates: List[Dict[str, Any]] = []
    for cluster in sorted_clusters:
        if not is_problem_bundle_anchor(cluster, cap_lookup):
            continue
        raw_anchor_candidates.append(cluster)
    ranked_anchor_candidates = sorted(
        raw_anchor_candidates,
        key=lambda cluster: (
            anchor_decision_score(cluster, sorted_clusters, cap_lookup),
            anchor_support_profile(cluster, sorted_clusters, cap_lookup),
        ),
        reverse=True,
    )

    strong_anchor_present = any(
        anchor_decision_score(cluster, sorted_clusters, cap_lookup) >= 8.0
        or anchor_support_profile(cluster, sorted_clusters, cap_lookup)[0] >= 4
        or anchor_support_profile(cluster, sorted_clusters, cap_lookup)[1] > 0
        for cluster in ranked_anchor_candidates
    )

    anchor_clusters: List[Dict[str, Any]] = []
    for cluster in ranked_anchor_candidates:
        decision_score = anchor_decision_score(cluster, sorted_clusters, cap_lookup)
        preference, plan_links, support_score, _salience = anchor_support_profile(cluster, sorted_clusters, cap_lookup)
        non_subjective_support = anchor_non_subjective_support(cluster, sorted_clusters, cap_lookup)
        if strong_anchor_present and decision_score < 3.0:
            continue
        if strong_anchor_present and is_symptom_like_anchor(cluster) and decision_score < 6.0:
            continue
        if strong_anchor_present and preference <= 3 and plan_links == 0 and support_score < 1.5:
            continue
        if strong_anchor_present and is_symptom_like_anchor(cluster) and plan_links == 0 and non_subjective_support == 0:
            continue
        if is_symptom_like_anchor(cluster):
            concept_text = cap_or_cluster_text(cluster.get("canonical_concept"), cluster.get("cluster_summary"))
            overshadowed = False
            for other in ranked_anchor_candidates:
                if safe_text(other.get("cluster_id")) == safe_text(cluster.get("cluster_id")):
                    continue
                if cluster_anchor_preference(other) < 4:
                    continue
                if classify_bundle_relation(other, cluster, cap_lookup) == "unrelated":
                    continue
                if bundle_anchor_score(other, cluster, cap_lookup) >= 0.25:
                    overshadowed = True
                    break
            if not overshadowed and strong_anchor_present and is_surface_symptom_anchor_text(concept_text) and plan_links == 0:
                continue
            if overshadowed:
                continue
        if should_keep_distinct_anchor(anchor_clusters, cluster, cap_lookup):
            anchor_clusters.append(cluster)
    if not anchor_clusters:
        anchor_clusters = sorted_clusters[:3]

    bundles: List[Dict[str, Any]] = []
    assigned_cluster_ids: set[str] = set()
    orphan_global_clusters: List[Dict[str, Any]] = []
    for anchor in anchor_clusters:
        concept = safe_text(anchor.get("canonical_concept"))
        summary = safe_text(anchor.get("cluster_summary"))
        if not concept or not summary:
            continue
        bundle = init_problem_bundle(anchor, cap_lookup)
        add_cluster_to_problem_bundle(bundle, anchor, cap_lookup)
        assigned_cluster_ids.add(safe_text(anchor.get("cluster_id")))
        bundles.append(bundle)

    for cluster in sorted_clusters:
        cluster_id = safe_text(cluster.get("cluster_id"))
        if cluster_id in assigned_cluster_ids:
            continue
        if bundles and all(should_drop_bundle_cluster(bundle["anchor_cluster"], cluster, cap_lookup) for bundle in bundles):
            orphan_global_clusters.append(cluster)
            continue
        best_bundle = None
        best_score = -1.0
        for bundle in bundles:
            score = bundle_anchor_score(bundle["anchor_cluster"], cluster, cap_lookup)
            if score > best_score:
                best_score = score
                best_bundle = bundle
        if best_bundle is None:
            orphan_global_clusters.append(cluster)
            continue
        if should_drop_bundle_cluster(best_bundle["anchor_cluster"], cluster, cap_lookup):
            orphan_global_clusters.append(cluster)
            continue
        relation = classify_bundle_relation(best_bundle["anchor_cluster"], cluster, cap_lookup)
        threshold = 0.4 if problem_bundle_slot(cluster) in {"O", "P"} else 0.75
        if relation == "related_factor":
            threshold = 0.2
        elif relation == "weak":
            threshold = max(threshold, 1.5 if problem_bundle_slot(cluster) != "G" else 2.5)
        elif relation == "unrelated":
            threshold = 10.0
        if cluster_temporal_layer(cluster) == "history" and relation != "related_factor":
            threshold = max(threshold, 1.75)
        if best_score < threshold:
            orphan_global_clusters.append(cluster)
            continue
        add_cluster_to_problem_bundle(best_bundle, cluster, cap_lookup)
        assigned_cluster_ids.add(cluster_id)

    if ENABLE_ORPHAN_BACKFILL and bundles and orphan_global_clusters:
        for cluster in orphan_global_clusters:
            best_bundle = None
            best_score = -1.0
            for bundle in bundles:
                score = bundle_anchor_score(bundle["anchor_cluster"], cluster, cap_lookup)
                if score > best_score:
                    best_score = score
                    best_bundle = bundle
            target_bundle = best_bundle or bundles[0]
            if should_drop_bundle_cluster(target_bundle["anchor_cluster"], cluster, cap_lookup):
                continue
            relation = classify_bundle_relation(target_bundle["anchor_cluster"], cluster, cap_lookup)
            if relation == "unrelated" or relation == "weak" or best_score < 0.75:
                continue
            for cap_id in cluster.get("supporting_cap_ids") or []:
                cap_id = safe_text(cap_id)
                prop = cap_lookup.get(cap_id)
                if prop:
                    if relation == "related_factor":
                        append_unique_cap_id(target_bundle["related_factor_cap_ids"], cap_id)
                        append_unique_summary(target_bundle["related_factor_caps"], safe_text(prop.get("proposition_text")))
                    else:
                        append_unique_cap_id(target_bundle["global_cap_ids"], cap_id)
                        append_unique_summary(target_bundle["global_context_caps"], safe_text(prop.get("proposition_text")))
            if not cluster.get("supporting_cap_ids"):
                if relation == "related_factor":
                    append_unique_summary(target_bundle["related_factor_caps"], safe_text(cluster.get("cluster_summary")))
                else:
                    append_unique_summary(target_bundle["global_context_caps"], safe_text(cluster.get("cluster_summary")))

    if ENABLE_SURFACE_SYMPTOM_BUNDLE_MERGE:
        merged_bundles: List[Dict[str, Any]] = []
        consumed_bundle_ids: set[str] = set()
        for idx, bundle in enumerate(bundles):
            source_bundle_id = safe_text(bundle["anchor_cluster"].get("cluster_id")) or f"bundle_{idx}"
            if source_bundle_id in consumed_bundle_ids:
                continue
            best_target = None
            best_score = -1.0
            for other_idx, other_bundle in enumerate(bundles):
                target_bundle_id = safe_text(other_bundle["anchor_cluster"].get("cluster_id")) or f"bundle_{other_idx}"
                if target_bundle_id == source_bundle_id or target_bundle_id in consumed_bundle_ids:
                    continue
                if not should_merge_surface_symptom_bundle(bundle, other_bundle, cap_lookup):
                    continue
                score = bundle_anchor_score(other_bundle["anchor_cluster"], bundle["anchor_cluster"], cap_lookup)
                if score > best_score:
                    best_score = score
                    best_target = other_bundle
            if best_target is not None:
                merge_problem_bundle_into(best_target, bundle)
                consumed_bundle_ids.add(source_bundle_id)
                continue
            merged_bundles.append(bundle)
        bundles = merged_bundles

    events: List[Dict[str, Any]] = []
    sorted_bundles = sorted(
        bundles,
        key=lambda bundle: (
            {"high": 2, "medium": 1, "low": 0}.get(
                clinical_priority_from_problem_cluster(bundle["anchor_cluster"], event_type_from_problem_cluster(bundle["anchor_cluster"])),
                0,
            ),
            int(bundle["anchor_cluster"].get("salience_score") or 0),
            len(bundle.get("supporting_cap_ids") or []),
        ),
        reverse=True,
    )
    for idx, bundle in enumerate(sorted_bundles, start=1):
        if not (bundle["S_caps"] or bundle["O_caps"] or bundle["A_caps"] or bundle["P_caps"]):
            continue
        events.append(finalize_problem_bundle_event(bundle, idx, cap_lookup))
        if len(events) >= EVENT_MAX_ITEMS:
            break
    renderable_events = []
    for item in events:
        key_slots = item.get("key_slots") if isinstance(item.get("key_slots"), dict) else {}
        if safe_text(key_slots.get("render_in_note")).lower() == "false":
            continue
        renderable_events.append(item)
    if renderable_events:
        events = renderable_events
    return normalize_event_plan({"events": events}, max_events=EVENT_MAX_ITEMS)


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def summarize_rows(rows: Sequence[Dict[str, Any]], metric_names: Sequence[str]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {"n_cases": len(rows)}
    for metric in metric_names:
        vals = [float(row[metric]) for row in rows if row.get(metric) is not None]
        if not vals:
            continue
        summary[metric] = round(statistics.mean(vals), 4)
        if len(vals) > 1:
            summary[f"{metric}_stdev"] = round(statistics.stdev(vals), 4)
    return summary


def metrics_have_evaluation(metrics: Optional[Dict[str, Any]]) -> bool:
    if not isinstance(metrics, dict):
        return False
    return any(
        metrics.get(key) is not None
        for key in (
            "cap_precision",
            "cap_recall",
            "cap_f1",
            "semantic_cap_precision",
            "semantic_cap_recall",
            "semantic_cap_f1",
            "hallucination_count",
            "pdsqi_core_mean",
            "llm_checklist_mean",
        )
    )


def usage_delta(after: Dict[str, int], before: Dict[str, int]) -> Dict[str, int]:
    return {
        "prompt_tokens": int(after.get("prompt_tokens", 0) or 0) - int(before.get("prompt_tokens", 0) or 0),
        "completion_tokens": int(after.get("completion_tokens", 0) or 0) - int(before.get("completion_tokens", 0) or 0),
        "total_tokens": int(after.get("total_tokens", 0) or 0) - int(before.get("total_tokens", 0) or 0),
    }


CASE_METRIC_FIELDNAMES = [
    "case_id",
    "method",
    "template",
    "source_cap_count",
    "event_plan_count",
    "summary_cap_count",
    "cap_matched_count",
    "cap_precision",
    "cap_recall",
    "cap_f1",
    "unsupported_summary_cap_count",
    "missing_source_cap_count",
    "rouge_l_f1_vs_reference",
    "token_f1_vs_reference",
    "gptf1_macro_f1",
    "nair_gpt_precision",
    "nair_gpt_recall",
    "nair_gpt_f1",
    "nair_pred_concept_count",
    "nair_ref_concept_count",
    "nair_precision_match_count",
    "nair_recall_match_count",
    "gptf1_active_problems_symptoms_precision",
    "gptf1_active_problems_symptoms_recall",
    "gptf1_active_problems_symptoms_f1",
    "gptf1_negated_findings_precision",
    "gptf1_negated_findings_recall",
    "gptf1_negated_findings_f1",
    "gptf1_uncertain_findings_precision",
    "gptf1_uncertain_findings_recall",
    "gptf1_uncertain_findings_f1",
    "gptf1_medication_treatment_precision",
    "gptf1_medication_treatment_recall",
    "gptf1_medication_treatment_f1",
    "gptf1_plan_followup_precision",
    "gptf1_plan_followup_recall",
    "gptf1_plan_followup_f1",
    "semantic_cap_precision",
    "semantic_cap_recall",
    "semantic_cap_f1",
    "semantic_cap_missing_count",
    "semantic_cap_unsupported_count",
    "hallucination_count",
    "major_hallucination_count",
    "omission_count",
    "major_omission_count",
    "section_appropriateness_score",
    "section_miscategorization_count",
    "major_section_miscategorization_count",
    "fabrication_count",
    "negation_error_count",
    "contextual_error_count",
    "causality_error_count",
    "current_issues_omission_count",
    "pmfs_issues_omission_count",
    "information_plan_omission_count",
    "pdsqi_core_mean",
    "pdsqi_accurate",
    "pdsqi_thorough",
    "pdsqi_useful",
    "pdsqi_organized",
    "pdsqi_comprehensible",
    "pdsqi_succinct",
    "pdsqi_synthesized",
    "llm_checklist_total_score",
    "llm_checklist_mean",
    "llm_checklist_mean_no_usability",
    "checklist_a_mean",
    "checklist_b_mean",
    "checklist_c_mean",
    "llm_checklist_abc_mean",
    "content_preservation_core",
    "noise_suppression",
    "lay_expression_preservation",
    "state_update_fidelity",
    "temporality_preservation",
    "problem_evidence_linkage",
    "state_plan_separation",
    "section_organization_appropriateness",
    "overall_clinical_usability",
    "llm_checklist_meaningful_omission_yes",
    "llm_checklist_concerning_hallucination_yes",
    "summary_runtime_seconds",
    "summary_cap_runtime_seconds",
    "event_plan_runtime_seconds",
    "event_plan_prompt_tokens",
    "event_plan_completion_tokens",
    "event_plan_total_tokens",
    "llm_prompt_tokens",
    "llm_completion_tokens",
    "llm_total_tokens",
]

AGGREGATE_METRIC_NAMES = [
    "cap_precision",
    "cap_recall",
    "cap_f1",
    "semantic_cap_precision",
    "semantic_cap_recall",
    "semantic_cap_f1",
    "rouge_l_f1_vs_reference",
    "token_f1_vs_reference",
    "gptf1_macro_f1",
    "nair_gpt_precision",
    "nair_gpt_recall",
    "nair_gpt_f1",
    "gptf1_active_problems_symptoms_f1",
    "gptf1_negated_findings_f1",
    "gptf1_uncertain_findings_f1",
    "gptf1_medication_treatment_f1",
    "gptf1_plan_followup_f1",
    "hallucination_count",
    "major_hallucination_count",
    "omission_count",
    "major_omission_count",
    "section_appropriateness_score",
    "section_miscategorization_count",
    "major_section_miscategorization_count",
    "pdsqi_core_mean",
    "pdsqi_accurate",
    "pdsqi_thorough",
    "llm_checklist_total_score",
    "llm_checklist_mean",
    "llm_checklist_mean_no_usability",
    "checklist_a_mean",
    "checklist_b_mean",
    "checklist_c_mean",
    "llm_checklist_abc_mean",
    "content_preservation_core",
    "noise_suppression",
    "lay_expression_preservation",
    "state_update_fidelity",
    "temporality_preservation",
    "problem_evidence_linkage",
    "state_plan_separation",
    "section_organization_appropriateness",
    "overall_clinical_usability",
    "llm_checklist_meaningful_omission_yes",
    "llm_checklist_concerning_hallucination_yes",
    "event_plan_count",
    "summary_cap_count",
    "unsupported_summary_cap_count",
    "missing_source_cap_count",
    "summary_runtime_seconds",
    "summary_cap_runtime_seconds",
    "event_plan_runtime_seconds",
    "event_plan_prompt_tokens",
    "event_plan_completion_tokens",
    "event_plan_total_tokens",
    "llm_prompt_tokens",
    "llm_completion_tokens",
    "llm_total_tokens",
]

AGGREGATE_FIELDNAMES = [
    "method",
    "template",
    "n_cases",
    "cap_precision",
    "cap_recall",
    "cap_f1",
    "semantic_cap_precision",
    "semantic_cap_recall",
    "semantic_cap_f1",
    "rouge_l_f1_vs_reference",
    "token_f1_vs_reference",
    "gptf1_macro_f1",
    "nair_gpt_precision",
    "nair_gpt_recall",
    "nair_gpt_f1",
    "gptf1_active_problems_symptoms_f1",
    "gptf1_negated_findings_f1",
    "gptf1_uncertain_findings_f1",
    "gptf1_medication_treatment_f1",
    "gptf1_plan_followup_f1",
    "hallucination_count",
    "major_hallucination_count",
    "omission_count",
    "major_omission_count",
    "section_appropriateness_score",
    "section_miscategorization_count",
    "major_section_miscategorization_count",
    "pdsqi_core_mean",
    "pdsqi_accurate",
    "pdsqi_thorough",
    "llm_checklist_total_score",
    "llm_checklist_mean",
    "llm_checklist_mean_no_usability",
    "checklist_a_mean",
    "checklist_b_mean",
    "checklist_c_mean",
    "llm_checklist_abc_mean",
    "content_preservation_core",
    "noise_suppression",
    "lay_expression_preservation",
    "state_update_fidelity",
    "temporality_preservation",
    "problem_evidence_linkage",
    "state_plan_separation",
    "section_organization_appropriateness",
    "overall_clinical_usability",
    "llm_checklist_meaningful_omission_yes",
    "llm_checklist_concerning_hallucination_yes",
    "event_plan_count",
    "summary_cap_count",
    "unsupported_summary_cap_count",
    "missing_source_cap_count",
    "summary_runtime_seconds",
    "summary_cap_runtime_seconds",
    "event_plan_runtime_seconds",
    "event_plan_prompt_tokens",
    "event_plan_completion_tokens",
    "event_plan_total_tokens",
    "llm_prompt_tokens",
    "llm_completion_tokens",
    "llm_total_tokens",
]


def build_aggregate_rows(
    flat_rows: Sequence[Dict[str, Any]],
    *,
    methods: Sequence[str],
    templates: Sequence[str],
) -> List[Dict[str, Any]]:
    aggregate_rows: List[Dict[str, Any]] = []
    for method_key in methods:
        for template_key in templates:
            rows = [row for row in flat_rows if row["method"] == method_key and row["template"] == template_key]
            aggregate_rows.append({"method": method_key, "template": template_key, **summarize_rows(rows, AGGREGATE_METRIC_NAMES)})
    return aggregate_rows


def flush_metric_outputs(
    output_dir: Path,
    *,
    flat_rows: Sequence[Dict[str, Any]],
    methods: Sequence[str],
    templates: Sequence[str],
) -> None:
    write_jsonl(output_dir / "case_metrics.jsonl", flat_rows)
    write_csv(output_dir / "case_metrics.csv", flat_rows, fieldnames=CASE_METRIC_FIELDNAMES)
    aggregate_rows = build_aggregate_rows(flat_rows, methods=methods, templates=templates)
    write_json(output_dir / "aggregate_metrics.json", aggregate_rows)
    write_csv(output_dir / "aggregate_metrics.csv", aggregate_rows, fieldnames=AGGREGATE_FIELDNAMES)


def method_output_stem(case_id: str, method_key: str, template_key: str) -> str:
    return f"{case_id}_{method_key}_{template_key}"


def method_uses_event_plan(method_key: str) -> bool:
    return method_key in {"cap_event", "cap_event_only"}


def process_method_template_task(
    *,
    case_id: str,
    method_key: str,
    template_key: str,
    transcript: str,
    reference_summary: str,
    medsum_fact_text: str,
    cluster_fact_text: str,
    source_cap: Dict[str, Any],
    event_plan_cache: Optional[Dict[str, Any]],
    event_plan_runtime: Optional[float],
    event_plan_usage: Optional[Dict[str, int]],
    ref_gpt_items: Optional[Dict[str, List[str]]],
    ref_nair_concepts: Optional[Dict[str, Any]],
    args: argparse.Namespace,
    client: OpenAICompatClient,
    judge_client: OpenAICompatClient,
    judge_model: str,
    output_dir: Path,
    summary_dir: Path,
    summary_cap_dir: Path,
    gpt_summary_dir: Path,
    nair_summary_dir: Path,
    semantic_cap_dir: Path,
    safety_dir: Path,
    section_dir: Path,
    pdsqi_dir: Path,
    checklist_dir: Path,
    case_dir: Path,
) -> Optional[Dict[str, Any]]:
    generation_usage_before = client.get_usage_totals()
    judge_usage_before = judge_client.get_usage_totals()

    stem = method_output_stem(case_id, method_key, template_key)
    summary_path = summary_dir / f"{stem}.txt"
    summary_cap_path = summary_cap_dir / f"{stem}.json"
    summary_gpt_items_path = gpt_summary_dir / f"{stem}.json"
    summary_nair_concepts_path = nair_summary_dir / f"{stem}.json"
    semantic_cap_path = semantic_cap_dir / f"{stem}.json"
    safety_path = safety_dir / f"{stem}.json"
    section_path = section_dir / f"{stem}.json"
    pdsqi_path = pdsqi_dir / f"{stem}.json"
    checklist_path = checklist_dir / f"{stem}.json"
    case_path = case_dir / f"{stem}.json"

    if args.skip_existing and case_path.exists():
        existing = read_json(case_path)
        existing_metrics = existing.get("metrics", {})
        needs_eval = args.evaluation_only and not metrics_have_evaluation(existing_metrics)
        needs_section_eval = (
            args.evaluation_only
            and not args.disable_section_judge
            and template_key in {"soap", "sectioned", "brief"}
            and existing_metrics.get("section_miscategorization_count") is None
        )
        needs_regen = False
        if args.generation_only and template_key in {"soap", "sectioned"} and summary_path.exists():
            existing_summary = normalize_template_summary(summary_path.read_text(encoding="utf-8"), template_key)
            if not summary_matches_template(existing_summary, template_key) or summary_has_pathological_repetition(existing_summary):
                needs_regen = True
        if needs_eval or needs_section_eval:
            print(
                f"[INFO] {case_id} {method_key}/{template_key}: existing case file is missing required evaluation metrics; re-running evaluation",
                flush=True,
            )
        elif needs_regen:
            print(
                f"[INFO] {case_id} {method_key}/{template_key}: existing case file found, but summary format is inconsistent; re-running generation",
                flush=True,
            )
        else:
            return existing_metrics

    if method_key == "medsum_ent" and not medsum_fact_text:
        print(f"[WARN] Missing MEDSUM-ENT-inspired scaffold for {case_id}; skipping.", flush=True)
        return None
    if method_key == "cluster2sent" and not cluster_fact_text:
        print(f"[WARN] Missing Cluster2Sent-inspired scaffold for {case_id}; skipping.", flush=True)
        return None
    if method_uses_event_plan(method_key) and not event_plan_cache:
        print(f"[WARN] Missing CAP event plan for {case_id}; skipping.", flush=True)
        return None
    if args.evaluation_only and not summary_path.exists():
        print(
            f"[WARN] {case_id} {method_key}/{template_key}: no existing summary found for evaluation-only mode; skipping",
            flush=True,
        )
        return None

    summary_runtime: Optional[float] = None
    should_regenerate = False
    if summary_path.exists():
        raw_summary_text = summary_path.read_text(encoding="utf-8").strip()
        summary_text = normalize_template_summary(raw_summary_text, template_key)
        if summary_text != raw_summary_text.strip():
            summary_path.write_text(summary_text + "\n", encoding="utf-8")
        if not args.evaluation_only and template_key in {"soap", "sectioned"} and (
            not summary_matches_template(summary_text, template_key)
            or summary_has_pathological_repetition(summary_text)
        ):
            print(
                f"[INFO] {case_id} {method_key}/{template_key}: existing summary does not match required template structure or has pathological repetition; regenerating",
                flush=True,
            )
            should_regenerate = True
        else:
            print(f"[INFO] {case_id} {method_key}/{template_key}: reuse existing summary", flush=True)
    if not summary_path.exists() or should_regenerate:
        if args.evaluation_only:
            print(
                f"[WARN] {case_id} {method_key}/{template_key}: evaluation-only mode forbids new generation; skipping",
                flush=True,
            )
            return None
        prompt = build_prompt_for_method(
            method_key,
            template_key,
            transcript=transcript,
            medsum_fact_text=medsum_fact_text,
            cluster_fact_text=cluster_fact_text,
            cap_obj=source_cap,
            event_plan=event_plan_cache,
        )
        schema_obj = structured_note_schema(template_key)
        if schema_obj:
            prompt = f"{prompt}\n\n{structured_note_instruction(template_key)}"
        print(f"[INFO] {case_id} {method_key}/{template_key}: generating summary (prompt_chars={len(prompt)})", flush=True)
        t0 = time.perf_counter()
        raw_summary = call_llm(
            client,
            model=args.model,
            prompt=prompt,
            max_tokens=args.max_summary_tokens,
            temperature=args.temperature,
            force_json=bool(schema_obj),
            json_schema_obj=schema_obj,
            prefer_json_object=True,
        )
        if schema_obj:
            try:
                parsed_summary = safe_json_extract(raw_summary)
            except Exception:
                try:
                    parsed_summary = repair_json_via_llm(
                        client,
                        model=args.model,
                        raw_text=raw_summary,
                        schema_obj=schema_obj,
                        max_tokens=min(args.max_summary_tokens, 1200),
                    )
                except Exception:
                    try:
                        parsed_summary = salvage_structured_note_json(raw_summary, template_key)
                    except Exception as parse_exc:
                        print(
                            f"[WARN] {case_id} {method_key}/{template_key}: structured parse failed after repair/salvage; "
                            f"falling back to normalized raw text ({safe_text(parse_exc)[:160]})",
                            flush=True,
                        )
                        fallback_text = normalize_template_summary(
                            normalize_generated_summary(raw_summary),
                            template_key,
                        )
                        if summary_matches_template(fallback_text, template_key):
                            summary_text = fallback_text
                        else:
                            empty_obj = {key: "" for key in structured_note_section_keys(template_key)}
                            summary_text = render_structured_note(template_key, empty_obj)
                        summary_text = normalize_template_summary(summary_text, template_key)
                        summary_runtime = round(time.perf_counter() - t0, 3)
                        summary_path.write_text(summary_text + "\n", encoding="utf-8")
                        print(f"[INFO] {case_id} {method_key}/{template_key}: summary done in {summary_runtime}s", flush=True)
                        parsed_summary = None
            if parsed_summary is not None:
                summary_text = render_structured_note(template_key, parsed_summary)
        else:
            summary_text = normalize_generated_summary(raw_summary)
        summary_text = normalize_template_summary(summary_text, template_key)
        summary_runtime = round(time.perf_counter() - t0, 3)
        summary_path.write_text(summary_text + "\n", encoding="utf-8")
        print(f"[INFO] {case_id} {method_key}/{template_key}: summary done in {summary_runtime}s", flush=True)

    if args.generation_only:
        generation_usage_after = client.get_usage_totals()
        usage_gen = usage_delta(generation_usage_after, generation_usage_before)
        event_usage = event_plan_usage or {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        if method_key != "cap_event":
            event_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        metrics = {
            "case_id": case_id,
            "method": method_key,
            "template": template_key,
            "source_cap_count": len(source_cap.get("atomic_propositions", [])),
            "event_plan_count": len((event_plan_cache or {}).get("events", [])) if method_uses_event_plan(method_key) else None,
            "summary_cap_count": None,
            "cap_matched_count": None,
            "cap_precision": None,
            "cap_recall": None,
            "cap_f1": None,
            "unsupported_summary_cap_count": None,
            "missing_source_cap_count": None,
            "summary_runtime_seconds": summary_runtime,
            "summary_cap_runtime_seconds": None,
            "event_plan_runtime_seconds": event_plan_runtime if method_uses_event_plan(method_key) else None,
            "event_plan_prompt_tokens": event_usage["prompt_tokens"],
            "event_plan_completion_tokens": event_usage["completion_tokens"],
            "event_plan_total_tokens": event_usage["total_tokens"],
            "llm_prompt_tokens": usage_gen["prompt_tokens"] + event_usage["prompt_tokens"],
            "llm_completion_tokens": usage_gen["completion_tokens"] + event_usage["completion_tokens"],
            "llm_total_tokens": usage_gen["total_tokens"] + event_usage["total_tokens"],
            "rouge_l_f1_vs_reference": round(rouge_l_f1(summary_text, reference_summary), 4) if reference_summary else None,
            "token_f1_vs_reference": round(token_f1(summary_text, reference_summary), 4) if reference_summary else None,
            "gptf1_macro_f1": None,
            "nair_gpt_precision": None,
            "nair_gpt_recall": None,
            "nair_gpt_f1": None,
            "semantic_cap_precision": None,
            "semantic_cap_recall": None,
            "semantic_cap_f1": None,
            "hallucination_count": None,
            "major_hallucination_count": None,
            "omission_count": None,
            "major_omission_count": None,
            "section_appropriateness_score": None,
            "section_miscategorization_count": None,
            "major_section_miscategorization_count": None,
            "pdsqi_core_mean": None,
            "pdsqi_accurate": None,
            "pdsqi_thorough": None,
            "llm_checklist_total_score": None,
            "llm_checklist_mean": None,
            "content_preservation_core": None,
            "noise_suppression": None,
            "lay_expression_preservation": None,
            "state_update_fidelity": None,
            "temporality_preservation": None,
            "problem_evidence_linkage": None,
            "state_plan_separation": None,
            "section_organization_appropriateness": None,
            "overall_clinical_usability": None,
            "llm_checklist_meaningful_omission_yes": None,
            "llm_checklist_concerning_hallucination_yes": None,
        }
        result = {
            "case_id": case_id,
            "method": method_key,
            "template": template_key,
            "summary_text": summary_text,
            "reference_summary": reference_summary,
            "source_cap": source_cap,
            "event_plan": event_plan_cache if method_uses_event_plan(method_key) else None,
            "summary_cap": None,
            "cap_alignment": None,
            "reference_gpt_f1_items": ref_gpt_items,
            "summary_gpt_f1_items": None,
            "gpt_f1_alignment": None,
            "reference_nair_concepts": ref_nair_concepts,
            "summary_nair_concepts": None,
            "nair_alignment": None,
            "semantic_cap_audit": None,
            "safety_judgment": None,
            "section_judgment": None,
            "pdsqi_judgment": None,
            "llm_checklist_judgment": None,
            "metrics": metrics,
        }
        if args.write_case_files:
            write_json(case_path, result)
        return metrics

    if summary_cap_path.exists():
        summary_cap = enrich_cap_obj(normalize_cap_obj(read_json(summary_cap_path), max_props=CAP_MAX_PROPS))
        summary_cap_runtime = summary_cap.get("runtime_seconds")
        print(f"[INFO] {case_id} {method_key}/{template_key}: reuse existing summary CAP", flush=True)
    else:
        chunk_count = len(split_summary_for_cap_extraction(summary_text))
        print(f"[INFO] {case_id} {method_key}/{template_key}: extracting summary CAP", flush=True)
        t0 = time.perf_counter()
        summary_cap = extract_summary_caps(
            client,
            model=args.model,
            summary_text=summary_text,
            max_tokens=args.max_extraction_tokens,
            temperature=args.temperature,
            chunk_workers=args.chunk_workers,
        )
        summary_cap_runtime = round(time.perf_counter() - t0, 3)
        summary_cap["runtime_seconds"] = summary_cap_runtime
        write_json(summary_cap_path, summary_cap)
        print(
            f"[INFO] {case_id} {method_key}/{template_key}: summary CAP done in {summary_cap_runtime}s (chunks={summary_cap.get('chunk_count', chunk_count)})",
            flush=True,
        )
    if summary_cap.get("parse_failures"):
        print(
            f"[WARN] {case_id} {method_key}/{template_key}: summary CAP had {len(summary_cap.get('parse_failures', []))} failed chunk(s); using partial extraction",
            flush=True,
        )

    alignment = greedy_cap_alignment(source_cap, summary_cap)
    print(
        f"[INFO] {case_id} {method_key}/{template_key}: CAP-P={alignment['cap_precision']:.4f} CAP-R={alignment['cap_recall']:.4f}",
        flush=True,
    )
    summary_gpt_items: Optional[Dict[str, List[str]]] = None
    gpt_f1_alignment: Optional[Dict[str, Any]] = None
    gpt_f1_metrics: Dict[str, Any] = {}
    summary_nair_concepts: Optional[Dict[str, Any]] = None
    nair_alignment: Optional[Dict[str, Any]] = None
    nair_metrics: Dict[str, Any] = {}
    semantic_cap_audit: Optional[Dict[str, Any]] = None
    semantic_cap_metrics_dict: Dict[str, Any] = {}
    safety_judgment: Optional[Dict[str, Any]] = None
    safety_metrics: Dict[str, Any] = {}
    section_judgment: Optional[Dict[str, Any]] = None
    section_metrics: Dict[str, Any] = {}
    pdsqi_judgment: Optional[Dict[str, Any]] = None
    pdsqi_metrics: Dict[str, Any] = {}
    checklist_judgment: Optional[Dict[str, Any]] = None
    checklist_metrics: Dict[str, Any] = {}
    eval_futures: Dict[str, Any] = {}
    with ThreadPoolExecutor(max_workers=max(1, args.eval_workers)) as executor:
        if reference_summary:
            if summary_gpt_items_path.exists():
                summary_gpt_items = read_json(summary_gpt_items_path)
            else:
                print(f"[INFO] {case_id} {method_key}/{template_key}: extracting GPT-F1 summary items", flush=True)
                eval_futures["summary_gpt_items"] = executor.submit(
                    extract_gpt_f1_items,
                    client,
                    model=args.model,
                    note_text=summary_text,
                    max_tokens=args.max_extraction_tokens,
                    temperature=args.temperature,
                    chunk_workers=args.chunk_workers,
                )
            if summary_nair_concepts_path.exists():
                summary_nair_concepts = read_json(summary_nair_concepts_path)
            else:
                print(f"[INFO] {case_id} {method_key}/{template_key}: extracting Nair summary concepts", flush=True)
                eval_futures["summary_nair_concepts"] = executor.submit(
                    extract_nair_medical_concepts,
                    judge_client,
                    model=judge_model,
                    note_text=summary_text,
                    max_tokens=args.judge_max_tokens,
                    temperature=0.0,
                    chunk_workers=args.chunk_workers,
                )
        if not args.disable_semantic_cap_judge:
            if semantic_cap_path.exists():
                semantic_cap_audit = read_json(semantic_cap_path)
            else:
                print(f"[INFO] {case_id} {method_key}/{template_key}: running semantic CAP audit", flush=True)
                eval_futures["semantic_cap"] = executor.submit(
                    evaluate_semantic_cap_audit,
                    judge_client,
                    model=judge_model,
                    source_cap=source_cap,
                    summary_cap=summary_cap,
                    summary_text=summary_text,
                    max_tokens=args.judge_max_tokens,
                    chunk_workers=args.chunk_workers,
                )
        if not args.disable_safety_judge:
            if safety_path.exists():
                safety_judgment = read_json(safety_path)
            else:
                print(f"[INFO] {case_id} {method_key}/{template_key}: running safety judge", flush=True)
                eval_futures["safety"] = executor.submit(
                    evaluate_safety_judge,
                    judge_client,
                    model=judge_model,
                    transcript=transcript,
                    summary_text=summary_text,
                    max_tokens=args.judge_max_tokens,
                    chunk_workers=args.chunk_workers,
                )
        if not args.disable_section_judge and template_key in {"soap", "sectioned", "brief"}:
            if section_path.exists():
                section_judgment = read_json(section_path)
            else:
                print(f"[INFO] {case_id} {method_key}/{template_key}: running section appropriateness judge", flush=True)
                eval_futures["section"] = executor.submit(
                    evaluate_section_appropriateness_judge,
                    judge_client,
                    model=judge_model,
                    template_key=template_key,
                    summary_text=summary_text,
                    max_tokens=args.judge_max_tokens,
                )
        if not args.disable_pdsqi_judge:
            if pdsqi_path.exists():
                pdsqi_judgment = read_json(pdsqi_path)
            else:
                print(f"[INFO] {case_id} {method_key}/{template_key}: running PDSQI judge", flush=True)
                eval_futures["pdsqi"] = executor.submit(
                    evaluate_pdsqi_judge,
                    judge_client,
                    model=judge_model,
                    transcript=transcript,
                    summary_text=summary_text,
                    target_specialty=args.target_specialty,
                    max_tokens=args.judge_max_tokens,
                )
        if not args.disable_checklist_judge:
            if checklist_path.exists():
                checklist_judgment = read_json(checklist_path)
            else:
                print(f"[INFO] {case_id} {method_key}/{template_key}: running LLM checklist judge", flush=True)
                eval_futures["checklist"] = executor.submit(
                    evaluate_llm_checklist_judge,
                    judge_client,
                    model=judge_model,
                    transcript=transcript,
                    summary_text=summary_text,
                    max_tokens=args.judge_max_tokens,
                )

    if "summary_gpt_items" in eval_futures:
        try:
            summary_gpt_items = eval_futures["summary_gpt_items"].result()
            write_json(summary_gpt_items_path, summary_gpt_items)
        except Exception as exc:
            print(f"[WARN] {case_id} {method_key}/{template_key}: GPT-F1 extraction failed; continuing without GPT-F1 metrics ({safe_text(exc)[:200]})", flush=True)
            summary_gpt_items = None
    if summary_gpt_items and summary_gpt_items.get("_parse_failures"):
        print(f"[WARN] {case_id} {method_key}/{template_key}: GPT-F1 extraction had {len(summary_gpt_items.get('_parse_failures', []))} failed chunk(s); using partial extraction", flush=True)
    if ref_gpt_items and summary_gpt_items:
        gpt_f1_metrics, gpt_f1_alignment = compute_gpt_f1_metrics(summary_gpt_items, ref_gpt_items)
        print(f"[INFO] {case_id} {method_key}/{template_key}: GPT-F1 macro={gpt_f1_metrics['gptf1_macro_f1']:.4f}", flush=True)

    if "summary_nair_concepts" in eval_futures:
        try:
            summary_nair_concepts = eval_futures["summary_nair_concepts"].result()
            write_json(summary_nair_concepts_path, summary_nair_concepts)
        except Exception as exc:
            print(f"[WARN] {case_id} {method_key}/{template_key}: Nair concept extraction failed; continuing without Nair metrics ({safe_text(exc)[:200]})", flush=True)
            summary_nair_concepts = None
    if summary_nair_concepts and summary_nair_concepts.get("parse_failures"):
        print(f"[WARN] {case_id} {method_key}/{template_key}: Nair extraction had {len(summary_nair_concepts.get('parse_failures', []))} failed section(s); using partial extraction", flush=True)
    if ref_nair_concepts and summary_nair_concepts:
        try:
            nair_metrics, nair_alignment = compute_nair_gpt_metrics(
                judge_client,
                model=judge_model,
                pred_note=summary_text,
                ref_note=reference_summary,
                pred_concepts_obj=summary_nair_concepts,
                ref_concepts_obj=ref_nair_concepts,
                max_tokens=args.judge_max_tokens,
            )
            print(f"[INFO] {case_id} {method_key}/{template_key}: Nair GPT-P={nair_metrics['nair_gpt_precision']:.4f} Nair GPT-R={nair_metrics['nair_gpt_recall']:.4f}", flush=True)
        except Exception as exc:
            print(f"[WARN] {case_id} {method_key}/{template_key}: Nair metric computation failed; continuing without Nair metrics ({safe_text(exc)[:200]})", flush=True)
            nair_metrics = {}
            nair_alignment = None

    if "semantic_cap" in eval_futures:
        try:
            semantic_cap_audit = eval_futures["semantic_cap"].result()
            write_json(semantic_cap_path, semantic_cap_audit)
        except Exception as exc:
            print(f"[WARN] {case_id} {method_key}/{template_key}: semantic CAP audit failed; continuing without semantic CAP metrics ({safe_text(exc)[:200]})", flush=True)
            semantic_cap_audit = None
    if semantic_cap_audit:
        semantic_cap_metrics_dict = dict(semantic_cap_audit.get("metrics") or {})
        if semantic_cap_metrics_dict:
            print(f"[INFO] {case_id} {method_key}/{template_key}: semantic CAP-P={semantic_cap_metrics_dict.get('semantic_cap_precision', 0.0):.4f} semantic CAP-R={semantic_cap_metrics_dict.get('semantic_cap_recall', 0.0):.4f}", flush=True)

    if "safety" in eval_futures:
        try:
            safety_judgment = eval_futures["safety"].result()
            write_json(safety_path, safety_judgment)
        except Exception as exc:
            print(f"[WARN] {case_id} {method_key}/{template_key}: safety judge failed; continuing without safety metrics ({safe_text(exc)[:200]})", flush=True)
            safety_judgment = None
    if safety_judgment:
        safety_metrics = dict(safety_judgment.get("summary_statistics") or {})
        print(f"[INFO] {case_id} {method_key}/{template_key}: hallucinations={safety_metrics.get('hallucination_count', 0)} omissions={safety_metrics.get('omission_count', 0)}", flush=True)

    if "section" in eval_futures:
        try:
            section_judgment = eval_futures["section"].result()
            write_json(section_path, section_judgment)
        except Exception as exc:
            print(f"[WARN] {case_id} {method_key}/{template_key}: section judge failed; continuing without section metrics ({safe_text(exc)[:200]})", flush=True)
            section_judgment = None
    if section_judgment:
        section_metrics = dict(section_judgment.get("summary_statistics") or {})
        print(f"[INFO] {case_id} {method_key}/{template_key}: section-miscat={section_metrics.get('section_miscategorization_count', 0)}", flush=True)

    if "pdsqi" in eval_futures:
        try:
            pdsqi_judgment = eval_futures["pdsqi"].result()
            write_json(pdsqi_path, pdsqi_judgment)
        except Exception as exc:
            print(f"[WARN] {case_id} {method_key}/{template_key}: PDSQI judge failed; continuing without PDSQI metrics ({safe_text(exc)[:200]})", flush=True)
            pdsqi_judgment = None
    if pdsqi_judgment:
        pdsqi_metrics = {
            "pdsqi_core_mean": pdsqi_judgment.get("pdsqi_core_mean"),
            "pdsqi_accurate": pdsqi_judgment.get("accurate"),
            "pdsqi_thorough": pdsqi_judgment.get("thorough"),
            "pdsqi_useful": pdsqi_judgment.get("useful"),
            "pdsqi_organized": pdsqi_judgment.get("organized"),
            "pdsqi_comprehensible": pdsqi_judgment.get("comprehensible"),
            "pdsqi_succinct": pdsqi_judgment.get("succinct"),
            "pdsqi_synthesized": pdsqi_judgment.get("synthesized"),
        }
        print(f"[INFO] {case_id} {method_key}/{template_key}: PDSQI core={pdsqi_metrics.get('pdsqi_core_mean', 0.0):.4f}", flush=True)

    if "checklist" in eval_futures:
        try:
            checklist_judgment = eval_futures["checklist"].result()
            write_json(checklist_path, checklist_judgment)
        except Exception as exc:
            print(f"[WARN] {case_id} {method_key}/{template_key}: checklist judge failed; continuing without checklist metrics ({safe_text(exc)[:200]})", flush=True)
            checklist_judgment = None
    if checklist_judgment:
        checklist_metrics = {
            "llm_checklist_total_score": checklist_judgment.get("llm_checklist_total_score"),
            "llm_checklist_mean": checklist_judgment.get("llm_checklist_mean"),
            "llm_checklist_mean_no_usability": checklist_judgment.get("llm_checklist_mean_no_usability"),
            "checklist_a_mean": checklist_judgment.get("checklist_a_mean"),
            "checklist_b_mean": checklist_judgment.get("checklist_b_mean"),
            "checklist_c_mean": checklist_judgment.get("checklist_c_mean"),
            "llm_checklist_abc_mean": checklist_judgment.get("llm_checklist_abc_mean"),
            "content_preservation_core": checklist_judgment.get("content_preservation_core"),
            "noise_suppression": checklist_judgment.get("noise_suppression"),
            "lay_expression_preservation": checklist_judgment.get("lay_expression_preservation"),
            "state_update_fidelity": checklist_judgment.get("state_update_fidelity"),
            "temporality_preservation": checklist_judgment.get("temporality_preservation"),
            "problem_evidence_linkage": checklist_judgment.get("problem_evidence_linkage"),
            "state_plan_separation": checklist_judgment.get("state_plan_separation"),
            "section_organization_appropriateness": checklist_judgment.get("section_organization_appropriateness"),
            "overall_clinical_usability": checklist_judgment.get("overall_clinical_usability"),
            "llm_checklist_meaningful_omission_yes": checklist_judgment.get("llm_checklist_meaningful_omission_yes"),
            "llm_checklist_concerning_hallucination_yes": checklist_judgment.get("llm_checklist_concerning_hallucination_yes"),
        }
        print(
            f"[INFO] {case_id} {method_key}/{template_key}: checklist total={checklist_metrics.get('llm_checklist_total_score', 0)} "
            f"mean={checklist_metrics.get('llm_checklist_mean', 0.0):.4f} "
            f"omission_yes={checklist_metrics.get('llm_checklist_meaningful_omission_yes', 0)} "
            f"hallucination_yes={checklist_metrics.get('llm_checklist_concerning_hallucination_yes', 0)}",
            flush=True,
        )

    metrics = {
        "case_id": case_id,
        "method": method_key,
        "template": template_key,
        "source_cap_count": len(source_cap.get("atomic_propositions", [])),
        "event_plan_count": len((event_plan_cache or {}).get("events", [])) if method_uses_event_plan(method_key) else None,
        "summary_cap_count": len(summary_cap.get("atomic_propositions", [])),
        "cap_matched_count": alignment["matched_count"],
        "cap_precision": round(alignment["cap_precision"], 4),
        "cap_recall": round(alignment["cap_recall"], 4),
        "cap_f1": round(alignment["cap_f1"], 4),
        "unsupported_summary_cap_count": len(alignment["unsupported_summary_props"]),
        "missing_source_cap_count": len(alignment["missing_source_props"]),
        "summary_runtime_seconds": summary_runtime,
        "summary_cap_runtime_seconds": summary_cap_runtime,
        "event_plan_runtime_seconds": event_plan_runtime if method_uses_event_plan(method_key) else None,
        "rouge_l_f1_vs_reference": round(rouge_l_f1(summary_text, reference_summary), 4) if reference_summary else None,
        "token_f1_vs_reference": round(token_f1(summary_text, reference_summary), 4) if reference_summary else None,
        "gptf1_macro_f1": gpt_f1_metrics.get("gptf1_macro_f1"),
        "nair_gpt_precision": nair_metrics.get("nair_gpt_precision"),
        "nair_gpt_recall": nair_metrics.get("nair_gpt_recall"),
        "nair_gpt_f1": nair_metrics.get("nair_gpt_f1"),
        "semantic_cap_precision": semantic_cap_metrics_dict.get("semantic_cap_precision"),
        "semantic_cap_recall": semantic_cap_metrics_dict.get("semantic_cap_recall"),
        "semantic_cap_f1": semantic_cap_metrics_dict.get("semantic_cap_f1"),
        "hallucination_count": safety_metrics.get("hallucination_count"),
        "major_hallucination_count": safety_metrics.get("major_hallucination_count"),
        "omission_count": safety_metrics.get("omission_count"),
        "major_omission_count": safety_metrics.get("major_omission_count"),
        "section_appropriateness_score": section_metrics.get("section_appropriateness_score"),
        "section_miscategorization_count": section_metrics.get("section_miscategorization_count"),
        "major_section_miscategorization_count": section_metrics.get("major_section_miscategorization_count"),
        "pdsqi_core_mean": pdsqi_metrics.get("pdsqi_core_mean"),
        "pdsqi_accurate": pdsqi_metrics.get("pdsqi_accurate"),
        "pdsqi_thorough": pdsqi_metrics.get("pdsqi_thorough"),
        "llm_checklist_total_score": checklist_metrics.get("llm_checklist_total_score"),
        "llm_checklist_mean": checklist_metrics.get("llm_checklist_mean"),
        "llm_checklist_mean_no_usability": checklist_metrics.get("llm_checklist_mean_no_usability"),
        "checklist_a_mean": checklist_metrics.get("checklist_a_mean"),
        "checklist_b_mean": checklist_metrics.get("checklist_b_mean"),
        "checklist_c_mean": checklist_metrics.get("checklist_c_mean"),
        "llm_checklist_abc_mean": checklist_metrics.get("llm_checklist_abc_mean"),
        "content_preservation_core": checklist_metrics.get("content_preservation_core"),
        "noise_suppression": checklist_metrics.get("noise_suppression"),
        "lay_expression_preservation": checklist_metrics.get("lay_expression_preservation"),
        "state_update_fidelity": checklist_metrics.get("state_update_fidelity"),
        "temporality_preservation": checklist_metrics.get("temporality_preservation"),
        "problem_evidence_linkage": checklist_metrics.get("problem_evidence_linkage"),
        "state_plan_separation": checklist_metrics.get("state_plan_separation"),
        "section_organization_appropriateness": checklist_metrics.get("section_organization_appropriateness"),
        "overall_clinical_usability": checklist_metrics.get("overall_clinical_usability"),
        "llm_checklist_meaningful_omission_yes": checklist_metrics.get("llm_checklist_meaningful_omission_yes"),
        "llm_checklist_concerning_hallucination_yes": checklist_metrics.get("llm_checklist_concerning_hallucination_yes"),
    }
    generation_usage_after = client.get_usage_totals()
    judge_usage_after = judge_client.get_usage_totals()
    usage_gen = usage_delta(generation_usage_after, generation_usage_before)
    usage_judge = usage_delta(judge_usage_after, judge_usage_before)
    event_usage = event_plan_usage or {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    if method_key != "cap_event":
        event_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    metrics["event_plan_prompt_tokens"] = event_usage["prompt_tokens"]
    metrics["event_plan_completion_tokens"] = event_usage["completion_tokens"]
    metrics["event_plan_total_tokens"] = event_usage["total_tokens"]
    metrics["llm_prompt_tokens"] = usage_gen["prompt_tokens"] + usage_judge["prompt_tokens"] + event_usage["prompt_tokens"]
    metrics["llm_completion_tokens"] = usage_gen["completion_tokens"] + usage_judge["completion_tokens"] + event_usage["completion_tokens"]
    metrics["llm_total_tokens"] = usage_gen["total_tokens"] + usage_judge["total_tokens"] + event_usage["total_tokens"]
    metrics.update(gpt_f1_metrics)
    metrics.update(nair_metrics)
    metrics.update({k: v for k, v in semantic_cap_metrics_dict.items() if k not in metrics})
    metrics.update({k: v for k, v in safety_metrics.items() if k not in metrics})
    metrics.update({k: v for k, v in section_metrics.items() if k not in metrics})
    metrics.update({k: v for k, v in pdsqi_metrics.items() if k not in metrics})
    metrics.update({k: v for k, v in checklist_metrics.items() if k not in metrics})

    result = {
        "case_id": case_id,
        "method": method_key,
        "template": template_key,
        "summary_text": summary_text,
        "reference_summary": reference_summary,
        "source_cap": source_cap,
        "event_plan": event_plan_cache if method_uses_event_plan(method_key) else None,
        "summary_cap": summary_cap,
        "cap_alignment": alignment,
        "reference_gpt_f1_items": ref_gpt_items,
        "summary_gpt_f1_items": summary_gpt_items,
        "gpt_f1_alignment": gpt_f1_alignment,
        "reference_nair_concepts": ref_nair_concepts,
        "summary_nair_concepts": summary_nair_concepts,
        "nair_alignment": nair_alignment,
        "semantic_cap_audit": semantic_cap_audit,
        "safety_judgment": safety_judgment,
        "section_judgment": section_judgment,
        "pdsqi_judgment": pdsqi_judgment,
        "llm_checklist_judgment": checklist_judgment,
        "metrics": metrics,
    }
    if args.write_case_files:
        write_json(case_path, result)
    return metrics


def process_case(
    *,
    idx: int,
    total_cases: int,
    case_id: str,
    case: Dict[str, Any],
    methods: Sequence[str],
    templates: Sequence[str],
    args: argparse.Namespace,
    client: OpenAICompatClient,
    judge_client: OpenAICompatClient,
    judge_model: str,
    method_a_rows: Dict[str, Dict[str, str]],
    method_b_rows: Dict[str, Dict[str, str]],
    method_c_rows: Dict[str, Dict[str, str]],
    output_dir: Path,
    summary_dir: Path,
    summary_cap_dir: Path,
    gpt_ref_dir: Path,
    gpt_summary_dir: Path,
    nair_ref_dir: Path,
    nair_summary_dir: Path,
    semantic_cap_dir: Path,
    safety_dir: Path,
    section_dir: Path,
    pdsqi_dir: Path,
    checklist_dir: Path,
    event_plan_dir: Path,
    case_dir: Path,
    medsum_source_block_dir: Path,
    cluster_source_block_dir: Path,
    transcript_cap_input_dir: Optional[Path],
    problem_cluster_input_dir: Optional[Path],
) -> List[Dict[str, Any]]:
    case_rows: List[Dict[str, Any]] = []
    transcript = safe_text(case.get("transcript"))
    reference_summary = safe_text(case.get("summary_gt_note"))
    if not transcript:
        print(f"[WARN] Missing transcript for {case_id}; skipping.", flush=True)
        return case_rows

    print(f"[INFO] ({idx}/{total_cases}) Processing {case_id}", flush=True)

    medsum_fact_text = safe_text(method_a_rows.get(case_id, {}).get("transcript_facts"))
    cluster_fact_text = safe_text(method_b_rows.get(case_id, {}).get("transcript_facts"))
    medsum_source_origin = "method_a_csv" if medsum_fact_text else None
    cluster_source_origin = "method_b_csv" if cluster_fact_text else None
    if not medsum_fact_text or not cluster_fact_text:
        legacy_result = load_legacy_extraction_result(args.legacy_extraction_dir, case_id)
        if legacy_result:
            fallback_medsum, fallback_cluster = build_scaffolds_from_legacy_result(legacy_result)
            if not medsum_fact_text:
                medsum_fact_text = fallback_medsum
                medsum_source_origin = "legacy_fallback"
            if not cluster_fact_text:
                cluster_fact_text = fallback_cluster
                cluster_source_origin = "legacy_fallback"
    if medsum_fact_text:
        medsum_source_block = FACT_SOURCE_BLOCK.format(fact_lines=format_fact_lines(medsum_fact_text))
        write_text(medsum_source_block_dir / f"{case_id}.txt", medsum_source_block)
        write_json(
            medsum_source_block_dir / f"{case_id}.json",
            {
                "case_id": case_id,
                "origin": medsum_source_origin or "unknown",
                "transcript_facts": medsum_fact_text,
                "source_block": medsum_source_block,
            },
        )
    if cluster_fact_text:
        cluster_source_block = FACT_SOURCE_BLOCK.format(fact_lines=format_fact_lines(cluster_fact_text))
        write_text(cluster_source_block_dir / f"{case_id}.txt", cluster_source_block)
        write_json(
            cluster_source_block_dir / f"{case_id}.json",
            {
                "case_id": case_id,
                "origin": cluster_source_origin or "unknown",
                "transcript_facts": cluster_fact_text,
                "source_block": cluster_source_block,
            },
        )
    ref_gpt_items_path = gpt_ref_dir / f"{case_id}.json"
    ref_nair_concepts_path = nair_ref_dir / f"{case_id}.json"
    source_cap: Optional[Dict[str, Any]] = None
    problem_cluster_cache: Optional[Dict[str, Any]] = None
    if transcript_cap_input_dir and problem_cluster_input_dir:
        transcript_cap_path = transcript_cap_input_dir / f"{case_id}.json"
        problem_cluster_path = problem_cluster_input_dir / f"{case_id}.json"
        if transcript_cap_path.exists():
            source_cap = convert_problem_state_caps(read_json(transcript_cap_path))
            if problem_cluster_path.exists():
                problem_cluster_cache = read_json(problem_cluster_path)
                print(f"[INFO] {case_id}: using source CAP from problem-state runner", flush=True)
        else:
            print(f"[WARN] Missing problem-state transcript CAP for {case_id}; falling back to method C source CAP", flush=True)

    if source_cap is None:
        if case_id not in method_c_rows:
            print(f"[WARN] Missing method C source CAP for {case_id}; skipping.", flush=True)
            return case_rows
        source_cap = enrich_cap_obj(
            normalize_cap_obj(
                json.loads(method_c_rows[case_id]["transcript_facts"]),
                max_props=CAP_MAX_PROPS,
            )
        )

    event_plan_cache: Optional[Dict[str, Any]] = None
    event_plan_runtime: Optional[float] = None
    event_plan_usage: Dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    if any(method in methods for method in ("cap_event", "cap_event_only")):
        event_plan_path = event_plan_dir / f"{case_id}_cap_event_plan.json"
        if event_plan_path.exists():
            event_plan_cache = normalize_event_plan(read_json(event_plan_path), max_events=EVENT_MAX_ITEMS)
            event_plan_runtime = event_plan_cache.get("runtime_seconds")
            if event_plan_cache.get("events") and event_plan_uses_problem_oriented_bundles(event_plan_cache):
                print(f"[INFO] {case_id}: reuse existing CAP event plan", flush=True)
            elif event_plan_cache.get("events") and not args.evaluation_only:
                print(f"[INFO] {case_id}: rebuilding cached CAP event plan with problem-oriented bundles", flush=True)
                if problem_cluster_cache and problem_cluster_cache.get("clusters"):
                    event_plan_cache = convert_problem_clusters_to_event_plan(problem_cluster_cache, cap_obj=source_cap)
                    event_plan_runtime = 0.0
                else:
                    usage_before = client.get_usage_totals()
                    t0 = time.perf_counter()
                    event_plan_cache = build_event_plan(
                        client,
                        model=args.model,
                        cap_obj=source_cap,
                        max_tokens=args.max_event_tokens,
                        temperature=args.temperature,
                    )
                    event_plan_runtime = round(time.perf_counter() - t0, 3)
                    usage_after = client.get_usage_totals()
                    event_plan_usage = usage_delta(usage_after, usage_before)
                    event_plan_cache["runtime_seconds"] = event_plan_runtime
                write_json(event_plan_path, event_plan_cache)
            elif not event_plan_cache.get("events") and not args.evaluation_only:
                print(f"[INFO] {case_id}: cached CAP event plan was empty; rebuilding", flush=True)
                usage_before = client.get_usage_totals()
                t0 = time.perf_counter()
                event_plan_cache = build_event_plan(
                    client,
                    model=args.model,
                    cap_obj=source_cap,
                    max_tokens=args.max_event_tokens,
                    temperature=args.temperature,
                )
                event_plan_runtime = round(time.perf_counter() - t0, 3)
                usage_after = client.get_usage_totals()
                event_plan_usage = usage_delta(usage_after, usage_before)
                event_plan_cache["runtime_seconds"] = event_plan_runtime
                write_json(event_plan_path, event_plan_cache)
                print(f"[INFO] {case_id}: CAP event plan done in {event_plan_runtime}s", flush=True)
            else:
                print(f"[WARN] {case_id}: cached CAP event plan is empty in evaluation-only mode", flush=True)
        elif problem_cluster_cache and problem_cluster_cache.get("clusters"):
            event_plan_cache = convert_problem_clusters_to_event_plan(problem_cluster_cache, cap_obj=source_cap)
            event_plan_runtime = 0.0
            write_json(event_plan_path, event_plan_cache)
            print(
                f"[INFO] {case_id}: built CAP event plan from problem-state clusters "
                f"(strategy={CONSOLIDATION_STRATEGY}, orphan_backfill={ENABLE_ORPHAN_BACKFILL}, "
                f"symptom_merge={ENABLE_SURFACE_SYMPTOM_BUNDLE_MERGE})",
                flush=True,
            )
        elif not args.evaluation_only:
            print(f"[INFO] {case_id}: building CAP event plan", flush=True)
            usage_before = client.get_usage_totals()
            t0 = time.perf_counter()
            event_plan_cache = build_event_plan(
                client,
                model=args.model,
                cap_obj=source_cap,
                max_tokens=args.max_event_tokens,
                temperature=args.temperature,
            )
            event_plan_runtime = round(time.perf_counter() - t0, 3)
            usage_after = client.get_usage_totals()
            event_plan_usage = usage_delta(usage_after, usage_before)
            event_plan_cache["runtime_seconds"] = event_plan_runtime
            write_json(event_plan_path, event_plan_cache)
            print(f"[INFO] {case_id}: CAP event plan done in {event_plan_runtime}s", flush=True)
        else:
            print(f"[WARN] {case_id}: cap_event requested but no cached event plan in evaluation-only mode", flush=True)

    ref_gpt_items: Optional[Dict[str, List[str]]] = None
    ref_nair_concepts: Optional[Dict[str, Any]] = None
    if reference_summary:
        if ref_gpt_items_path.exists():
            ref_gpt_items = read_json(ref_gpt_items_path)
        if ref_nair_concepts_path.exists():
            ref_nair_concepts = read_json(ref_nair_concepts_path)
        elif not args.generation_only:
            pass
        if not args.generation_only and (ref_gpt_items is None or ref_nair_concepts is None):
            ref_futures: Dict[str, Any] = {}
            with ThreadPoolExecutor(max_workers=min(max(1, args.eval_workers), 2)) as executor:
                if ref_gpt_items is None:
                    print(f"[INFO] {case_id}: extracting GPT-F1 reference items", flush=True)
                    ref_futures["gpt"] = executor.submit(
                        extract_gpt_f1_items,
                        client,
                        model=args.model,
                        note_text=reference_summary,
                        max_tokens=args.max_extraction_tokens,
                        temperature=args.temperature,
                        chunk_workers=args.chunk_workers,
                    )
                if ref_nair_concepts is None:
                    print(f"[INFO] {case_id}: extracting Nair reference concepts", flush=True)
                    ref_futures["nair"] = executor.submit(
                        extract_nair_medical_concepts,
                        judge_client,
                        model=judge_model,
                        note_text=reference_summary,
                        max_tokens=args.judge_max_tokens,
                        temperature=0.0,
                        chunk_workers=args.chunk_workers,
                    )
            if "gpt" in ref_futures:
                ref_gpt_items = ref_futures["gpt"].result()
                write_json(ref_gpt_items_path, ref_gpt_items)
            if "nair" in ref_futures:
                ref_nair_concepts = ref_futures["nair"].result()
                write_json(ref_nair_concepts_path, ref_nair_concepts)

    tasks = [(method_key, template_key) for method_key in methods for template_key in templates]
    if max(1, args.task_workers) <= 1:
        for method_key, template_key in tasks:
            try:
                metrics = process_method_template_task(
                    case_id=case_id,
                    method_key=method_key,
                    template_key=template_key,
                    transcript=transcript,
                    reference_summary=reference_summary,
                    medsum_fact_text=medsum_fact_text,
                    cluster_fact_text=cluster_fact_text,
                    source_cap=source_cap,
                    event_plan_cache=event_plan_cache,
                    event_plan_runtime=event_plan_runtime,
                    event_plan_usage=event_plan_usage,
                    ref_gpt_items=ref_gpt_items,
                    ref_nair_concepts=ref_nair_concepts,
                    args=args,
                    client=client,
                    judge_client=judge_client,
                    judge_model=judge_model,
                    output_dir=output_dir,
                    summary_dir=summary_dir,
                    summary_cap_dir=summary_cap_dir,
                    gpt_summary_dir=gpt_summary_dir,
                    nair_summary_dir=nair_summary_dir,
                    semantic_cap_dir=semantic_cap_dir,
                    safety_dir=safety_dir,
                    section_dir=section_dir,
                    pdsqi_dir=pdsqi_dir,
                    checklist_dir=checklist_dir,
                    case_dir=case_dir,
                )
            except Exception as exc:
                print(
                    f"[WARN] {case_id} {method_key}/{template_key}: task failed; skipping and continuing "
                    f"({safe_text(exc)[:220]})",
                    flush=True,
                )
                continue
            if metrics:
                case_rows.append(metrics)
    else:
        with ThreadPoolExecutor(max_workers=min(max(1, args.task_workers), len(tasks))) as executor:
            futures = [
                executor.submit(
                    process_method_template_task,
                    case_id=case_id,
                    method_key=method_key,
                    template_key=template_key,
                    transcript=transcript,
                    reference_summary=reference_summary,
                    medsum_fact_text=medsum_fact_text,
                    cluster_fact_text=cluster_fact_text,
                    source_cap=source_cap,
                    event_plan_cache=event_plan_cache,
                    event_plan_runtime=event_plan_runtime,
                    event_plan_usage=event_plan_usage,
                    ref_gpt_items=ref_gpt_items,
                    ref_nair_concepts=ref_nair_concepts,
                    args=args,
                    client=client,
                    judge_client=judge_client,
                    judge_model=judge_model,
                    output_dir=output_dir,
                    summary_dir=summary_dir,
                    summary_cap_dir=summary_cap_dir,
                    gpt_summary_dir=gpt_summary_dir,
                    nair_summary_dir=nair_summary_dir,
                    semantic_cap_dir=semantic_cap_dir,
                    safety_dir=safety_dir,
                    section_dir=section_dir,
                    pdsqi_dir=pdsqi_dir,
                    checklist_dir=checklist_dir,
                    case_dir=case_dir,
                )
                for method_key, template_key in tasks
            ]
            for future in as_completed(futures):
                try:
                    metrics = future.result()
                except Exception as exc:
                    print(
                        f"[WARN] {case_id}: one method/template task failed in parallel mode; skipping "
                        f"({safe_text(exc)[:220]})",
                        flush=True,
                    )
                    continue
                if metrics:
                    case_rows.append(metrics)
    return case_rows


def main() -> None:
    load_env_file(BASE_DIR / ".env")
    args = parse_args()
    if args.generation_only and args.evaluation_only:
        raise SystemExit("Choose only one of --generation-only or --evaluation-only.")

    methods_input = args.methods
    if "--methods" not in sys.argv and args.experiment_set in EXPERIMENT_METHOD_SETS:
        methods_input = EXPERIMENT_METHOD_SETS[args.experiment_set]
    methods = normalize_choices(methods_input, SUPPORTED_METHODS, "method")
    templates = normalize_choices(args.templates, SUPPORTED_TEMPLATES, "template")

    api_base_url = infer_api_base_url(args.api_base_url)
    api_key = args.api_key or os.getenv("RUNPOD_API_KEY") or os.getenv("OPENAI_API_KEY", "")
    client = OpenAICompatClient(base_url=api_base_url, api_key=api_key, timeout=args.request_timeout)
    judge_model = args.judge_model or args.model
    judge_api_base_url = infer_judge_api_base_url(args.judge_api_base_url, judge_model)
    judge_api_key = args.judge_api_key or (os.getenv("OPENAI_API_KEY", "") if judge_api_base_url == "https://api.openai.com/v1" else api_key)
    judge_client = OpenAICompatClient(base_url=judge_api_base_url, api_key=judge_api_key, timeout=args.request_timeout)

    need_method_a = "medsum_ent" in methods
    need_method_b = "cluster2sent" in methods
    need_method_c = (
        any(method in methods for method in ("cap", "cap_only", "cap_event", "cap_event_only"))
        and args.problem_state_dir is None
    )

    method_a_rows: Dict[str, Dict[str, str]] = {}
    method_b_rows: Dict[str, Dict[str, str]] = {}
    method_c_rows: Dict[str, Dict[str, str]] = {}

    if need_method_a:
        if not args.method_a_csv.exists():
            raise SystemExit(
                f"Missing --method-a-csv file required for method 'medsum_ent': {args.method_a_csv}"
            )
        method_a_rows = load_method_csv(args.method_a_csv)
    if need_method_b:
        if not args.method_b_csv.exists():
            raise SystemExit(
                f"Missing --method-b-csv file required for method 'cluster2sent': {args.method_b_csv}"
            )
        method_b_rows = load_method_csv(args.method_b_csv)
    if need_method_c:
        if not args.method_c_csv.exists():
            raise SystemExit(
                "Missing --method-c-csv file required for CAP methods when --problem-state-dir is not provided: "
                f"{args.method_c_csv}"
            )
        method_c_rows = load_method_csv(args.method_c_csv)
    cases = load_cases(args.cases_path, args.limit, args.case_ids)
    transcript_cap_input_dir = args.problem_state_dir / "transcript_caps" if args.problem_state_dir else None
    problem_cluster_input_dir = args.problem_state_dir / "problem_clusters" if args.problem_state_dir else None
    if not cases:
        raise SystemExit("No cases found from the requested JSONL / filters.")

    print(
        f"[INFO] Using endpoint={api_base_url} model={args.model} cases={len(cases)} "
        f"judge_endpoint={judge_api_base_url} judge_model={judge_model} "
        f"methods={methods} templates={templates} timeout={args.request_timeout}s "
        f"generation_only={args.generation_only} evaluation_only={args.evaluation_only} "
        f"case_workers={max(1, args.case_workers)} "
        f"eval_workers={max(1, args.eval_workers)} chunk_workers={max(1, args.chunk_workers)} "
        f"task_workers={max(1, args.task_workers)}",
        flush=True,
    )

    output_dir = ensure_dir(args.output_dir)
    summary_dir = ensure_dir(output_dir / "summaries")
    summary_cap_dir = ensure_dir(output_dir / "summary_caps")
    gpt_ref_dir = ensure_dir(output_dir / "gpt_f1_reference_items")
    gpt_summary_dir = ensure_dir(output_dir / "gpt_f1_summary_items")
    nair_ref_dir = ensure_dir(output_dir / "nair_reference_concepts")
    nair_summary_dir = ensure_dir(output_dir / "nair_summary_concepts")
    semantic_cap_dir = ensure_dir(output_dir / "semantic_cap_audits")
    safety_dir = ensure_dir(output_dir / "safety_judgments")
    section_dir = ensure_dir(output_dir / "section_judgments")
    pdsqi_dir = ensure_dir(output_dir / "pdsqi_judgments")
    checklist_dir = ensure_dir(output_dir / "llm_checklist_judgments")
    event_plan_dir = ensure_dir(output_dir / "event_plans")
    case_dir = ensure_dir(output_dir / "case_results")
    medsum_source_block_dir = ensure_dir(output_dir / "medsum_ent_source_blocks")
    cluster_source_block_dir = ensure_dir(output_dir / "cluster2sent_source_blocks")

    flat_rows: List[Dict[str, Any]] = []
    case_items = list(cases.items())
    case_workers = min(max(1, args.case_workers), len(case_items))

    common_kwargs: Dict[str, Any] = {
        "methods": methods,
        "templates": templates,
        "args": args,
        "client": client,
        "judge_client": judge_client,
        "judge_model": judge_model,
        "method_a_rows": method_a_rows,
        "method_b_rows": method_b_rows,
        "method_c_rows": method_c_rows,
        "output_dir": output_dir,
        "summary_dir": summary_dir,
        "summary_cap_dir": summary_cap_dir,
        "gpt_ref_dir": gpt_ref_dir,
        "gpt_summary_dir": gpt_summary_dir,
        "nair_ref_dir": nair_ref_dir,
        "nair_summary_dir": nair_summary_dir,
        "semantic_cap_dir": semantic_cap_dir,
        "safety_dir": safety_dir,
        "section_dir": section_dir,
        "pdsqi_dir": pdsqi_dir,
        "checklist_dir": checklist_dir,
        "event_plan_dir": event_plan_dir,
        "case_dir": case_dir,
        "medsum_source_block_dir": medsum_source_block_dir,
        "cluster_source_block_dir": cluster_source_block_dir,
        "transcript_cap_input_dir": transcript_cap_input_dir,
        "problem_cluster_input_dir": problem_cluster_input_dir,
    }

    if case_workers <= 1:
        for idx, (case_id, case) in enumerate(case_items, start=1):
            case_rows = process_case(
                idx=idx,
                total_cases=len(case_items),
                case_id=case_id,
                case=case,
                **common_kwargs,
            )
            if case_rows:
                flat_rows.extend(case_rows)
            flush_metric_outputs(output_dir, flat_rows=flat_rows, methods=methods, templates=templates)
    else:
        print(
            f"[INFO] Running case-level parallelism with case_workers={case_workers} "
            f"(task_workers={max(1, args.task_workers)}, eval_workers={max(1, args.eval_workers)})",
            flush=True,
        )
        with ThreadPoolExecutor(max_workers=case_workers) as executor:
            futures = [
                executor.submit(
                    process_case,
                    idx=idx,
                    total_cases=len(case_items),
                    case_id=case_id,
                    case=case,
                    **common_kwargs,
                )
                for idx, (case_id, case) in enumerate(case_items, start=1)
            ]
            for completed, future in enumerate(as_completed(futures), start=1):
                case_rows = future.result()
                if case_rows:
                    flat_rows.extend(case_rows)
                flush_metric_outputs(output_dir, flat_rows=flat_rows, methods=methods, templates=templates)
                print(f"[INFO] Case-level progress: {completed}/{len(case_items)} cases finished", flush=True)
    flush_metric_outputs(output_dir, flat_rows=flat_rows, methods=methods, templates=templates)
    print(f"[INFO] Wrote results to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
