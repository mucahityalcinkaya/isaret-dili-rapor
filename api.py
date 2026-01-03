# synthetic_sft_generator_with_conversation.py
# -*- coding: utf-8 -*-

import os
import re
import json
import time
import random
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any

# =========================
# SPEED / BATCH SETTINGS
# =========================
RANDOM_SEED = 42

NUM_DIALOGUES = 2000
MIN_TURNS = 1
MAX_TURNS = 8

MIN_SIGNS_PER_TURN = 1
MAX_SIGNS_PER_TURN = 4
MAX_ALTS_PER_SIGN = 3

AMBIGUITY_MARGIN = 0.08
TOPIC_SWITCH_PROB = 0.18
EXTRA_TOKEN_PROB = 0.35

AVOID_GLOBAL_DUPLICATES = True
AVOID_LOCAL_DUPLICATES = True

BATCH_SIZE = 300
FLUSH_EVERY_N_TURNS_PRINT = 1

WRITE_DEBUG = True

# =========================
# OPTIONAL API
# =========================
USE_API_TEXT = True
API_KEY_ENV = "api"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"

API_TEMPERATURE = 0.2
API_MAX_TOKENS = 250
API_RETRIES = 3
API_RETRY_SLEEP = 1.2

MAX_WORKERS = 8

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

from concurrent.futures import ThreadPoolExecutor, as_completed


# =========================
# FILES
# =========================
GLOSS_FILE = "gloss.txt"
PRIORS_FILE = "priors.txt"

OUTPUT_JSONL = "synthetic_sft.jsonl"
OUTPUT_DEBUG_JSONL = "synthetic_debug.jsonl"
SEEN_SIGNATURES_FILE = "seen_signatures.txt"


# =========================
# DATA CLASSES
# =========================
@dataclass
class Candidate:
    word: str
    p: float


# =========================
# UTIL
# =========================
def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def safe_float(p: float) -> float:
    if p != p:
        return 0.0
    return clamp(float(p), 0.0, 1.0)


class JsonlBatchWriter:
    def __init__(self, path: str):
        self.path = path
        self.buf: List[str] = []
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    def add(self, obj: dict):
        self.buf.append(json.dumps(obj, ensure_ascii=False))

    def flush(self):
        if not self.buf:
            return
        with open(self.path, "a", encoding="utf-8") as f:
            f.write("\n".join(self.buf) + "\n")
        self.buf.clear()


class LineBatchWriter:
    def __init__(self, path: str):
        self.path = path
        self.buf: List[str] = []
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    def add(self, line: str):
        self.buf.append(line)

    def flush(self):
        if not self.buf:
            return
        with open(self.path, "a", encoding="utf-8") as f:
            f.write("\n".join(self.buf) + "\n")
        self.buf.clear()


# =========================
# LOADERS
# =========================
def load_glosses(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def parse_priors_txt(path: str) -> Tuple[Dict[str, List[str]], Dict[str, str], Dict[str, str]]:
    confusion_map: Dict[str, List[str]] = {}
    second_guess: Dict[str, str] = {}
    third_guess: Dict[str, str] = {}

    current = None
    sec_pat = re.compile(r"^\[(.+)\]$")

    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue

            m = sec_pat.match(line)
            if m:
                sec = m.group(1).strip()
                if sec == "CONFUSION_MAP":
                    current = "confusion"
                elif sec == "HARD_SECOND_GUESS":
                    current = "second"
                elif sec == "HARD_THIRD_GUESS":
                    current = "third"
                else:
                    current = None
                continue

            if current is None or ":" not in line:
                continue

            k, v = line.split(":", 1)
            key = k.strip()
            val = v.strip()

            if current == "confusion":
                if val.startswith("[") and val.endswith("]"):
                    inner = val[1:-1].strip()
                    if not inner:
                        confusion_map[key] = []
                    else:
                        parts = [x.strip() for x in inner.split(",")]
                        parts = [p.strip().strip("'").strip('"') for p in parts]
                        parts = [p for p in parts if p]
                        confusion_map[key] = parts
                else:
                    val = val.strip().strip("'").strip('"')
                    if val:
                        confusion_map[key] = [val]
            elif current == "second":
                second_guess[key] = val.strip().strip("'").strip('"')
            elif current == "third":
                third_guess[key] = val.strip().strip("'").strip('"')

    return confusion_map, second_guess, third_guess


def make_confusion_symmetric(conf_map: Dict[str, List[str]]) -> Dict[str, List[str]]:
    out = {k: list(dict.fromkeys(v)) for k, v in conf_map.items()}
    for a, bs in conf_map.items():
        for b in bs:
            if b == a:
                continue
            if b not in out:
                out[b] = [a]
            else:
                if a not in out[b]:
                    out[b].append(a)
    for k in out:
        out[k] = list(dict.fromkeys(out[k]))
    return out


def load_seen_signatures(path: str) -> set:
    if not os.path.exists(path):
        return set()
    s = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            t = line.strip()
            if t:
                s.add(t)
    return s


# =========================
# LEXICON / STATE
# =========================
def build_lexicon(glosses: List[str]) -> Dict[str, set]:
    gset = set(glosses)

    person = {"i", "you", "my", "myself", "we", "our", "your", "yourself", "their", "me"} & gset
    family = {
        "mother", "mom", "father", "dad", "grandfather", "grandmother", "brother", "sister",
        "son", "daughter", "baby", "husband", "wife", "aunt", "uncle", "cousin", "children"
    } & gset

    time_words = {
        "today", "yesterday", "morning", "afternoon", "evening", "night",
        "weekend", "last week", "last year", "ago", "before", "early", "late", "hour", "minute", "month", "year", "day"
    } & gset

    symptom = {
        "pain", "hurt", "headache", "dizzy", "vomit", "diarrhea", "sick", "ill", "cold", "hot", "thirsty",
        "throat", "eyes", "ear", "tooth", "teeth", "skin", "heart attack", "infection", "pneumonia"
    } & gset

    location = {"head", "back", "stomach", "abdomen", "chest", "shoulder", "brain"} & gset
    request = {"doctor", "nurse", "help", "call", "appointment", "test", "surgery", "hospital", "dentist", "surgeon", "office"} & gset
    admin = {"document", "paper", "form", "insurance", "approve", "require", "pay", "money", "report", "write", "read"} & gset
    comm = {"deaf", "hard of hearing", "hearing aid", "hearing", "blind", "talk", "sign", "answer", "ask", "tell"} & gset
    condition = {"diabetes", "allergy"} & gset

    return {
        "person": person,
        "family": family,
        "time": time_words,
        "symptom": symptom,
        "location": location,
        "request": request,
        "admin": admin,
        "comm": comm,
        "condition": condition,
        "all": gset
    }


def normalize_word(w: str) -> str:
    m = {"hurt": "pain", "ill": "sick", "mom": "mother", "dad": "father", "teeth": "tooth"}
    return m.get(w, w)


def infer_intent(words: List[str], lex: Dict[str, set]) -> str:
    ws = set(words)
    if ws & lex["symptom"]:
        return "symptom_report"
    if ws & lex["request"]:
        return "request_help"
    if ws & lex["admin"]:
        return "admin_issue"
    if ws & lex["comm"]:
        return "communication_status"
    return "general"


def update_state(prior: Dict[str, Any], selected_words: List[str], lex: Dict[str, set]) -> Dict[str, Any]:
    state = json.loads(json.dumps(prior))

    state.setdefault("intent", None)
    state.setdefault("slots", {})
    slots = state["slots"]

    slots.setdefault("person", None)
    slots.setdefault("symptom", None)
    slots.setdefault("location", None)
    slots.setdefault("time", None)
    slots.setdefault("request", None)
    slots.setdefault("conditions", [])
    slots.setdefault("family", [])
    slots.setdefault("admin", [])
    slots.setdefault("communication", [])
    slots.setdefault("other", [])

    norm = [normalize_word(w) for w in selected_words if w]

    new_intent = infer_intent(norm, lex)
    if state["intent"] is None:
        state["intent"] = new_intent
    else:
        if new_intent != "general" and new_intent != state["intent"]:
            state["intent"] = new_intent

    for w in norm:
        if w in lex["person"]:
            slots["person"] = w
        elif w in lex["symptom"]:
            slots["symptom"] = w
        elif w in lex["location"]:
            slots["location"] = w
        elif w in lex["time"]:
            slots["time"] = w
        elif w in lex["request"]:
            slots["request"] = w
        elif w in lex["condition"]:
            if w not in slots["conditions"]:
                slots["conditions"].append(w)
        elif w in lex["family"]:
            if w not in slots["family"]:
                slots["family"].append(w)
        elif w in lex["admin"]:
            if w not in slots["admin"]:
                slots["admin"].append(w)
        elif w in lex["comm"]:
            if w not in slots["communication"]:
                slots["communication"].append(w)
        else:
            slots["other"].append(w)

    slots["conditions"] = slots["conditions"][-5:]
    slots["family"] = slots["family"][-5:]
    slots["admin"] = slots["admin"][-5:]
    slots["communication"] = slots["communication"][-5:]
    slots["other"] = slots["other"][-8:]

    return state


# =========================
# CANDIDATE SIMULATION
# =========================
def sample_prob_desc(n: int, p1_lo: float, p1_hi: float) -> List[float]:
    if n <= 0:
        return []
    p1 = random.uniform(p1_lo, p1_hi)
    ps = [p1]
    for _ in range(1, n):
        ps.append(ps[-1] * random.uniform(0.35, 0.85))
    return [safe_float(p) for p in ps]


def make_candidate_group(
    true_word: str,
    glosses: List[str],
    conf_map: Dict[str, List[str]],
    second_guess: Dict[str, str],
    third_guess: Dict[str, str],
) -> Tuple[List[Candidate], str, bool]:
    true_word = normalize_word(true_word)

    pool: List[str] = [true_word]

    if true_word in second_guess:
        pool.append(normalize_word(second_guess[true_word]))
    if true_word in third_guess:
        pool.append(normalize_word(third_guess[true_word]))

    if true_word in conf_map:
        cs = [normalize_word(x) for x in conf_map[true_word]]
        random.shuffle(cs)
        for x in cs[:2]:
            if x not in pool:
                pool.append(x)

    if len(pool) < 2 and glosses:
        r = normalize_word(random.choice(glosses))
        if r not in pool:
            pool.append(r)

    p_top1_correct = 0.86
    if true_word in conf_map:
        p_top1_correct -= 0.18
    if true_word in second_guess:
        p_top1_correct -= 0.05
    p_top1_correct = clamp(p_top1_correct, 0.45, 0.92)

    top1_is_true = (random.random() < p_top1_correct)

    if top1_is_true:
        ordering = pool[:]
    else:
        wrong_options = []
        if true_word in conf_map:
            wrong_options += [normalize_word(x) for x in conf_map[true_word]]
        if true_word in second_guess:
            wrong_options.append(normalize_word(second_guess[true_word]))
        wrong_options = [w for w in wrong_options if w and w != true_word]
        if not wrong_options:
            wrong_options = [normalize_word(random.choice(glosses))]
        wrong_top1 = random.choice(wrong_options)
        ordering = [wrong_top1] + [w for w in pool if w != wrong_top1]
        if true_word not in ordering:
            ordering.insert(1, true_word)

    ordering = list(dict.fromkeys(ordering))[:MAX_ALTS_PER_SIGN]

    ps = sample_prob_desc(len(ordering), 0.55, 0.95) if top1_is_true else sample_prob_desc(len(ordering), 0.45, 0.78)
    cands = [Candidate(word=w, p=ps[i]) for i, w in enumerate(ordering)]
    return cands, true_word, top1_is_true


def sign_ambiguity(cands: List[Candidate]) -> bool:
    if len(cands) < 2:
        return False
    return (cands[0].p - cands[1].p) < AMBIGUITY_MARGIN


def english_question_for_pair(a: str, b: str) -> str:
    return f"'{a}' or '{b}'?"


def sequence_signature(candidates: List[List[Candidate]]) -> str:
    sign_keys = []
    for sign in candidates:
        ws = sorted([c.word for c in sign])
        sign_keys.append("|".join(ws))
    sign_keys.sort()
    return ";;".join(sign_keys)


# =========================
# GROUND TRUTH BUILDER
# =========================
def pick_from_set(s: set) -> Optional[str]:
    if not s:
        return None
    return random.choice(list(s))


def build_turn_ground_truth(lex: Dict[str, set], prior_state: Dict[str, Any], force_new_topic: bool) -> List[str]:
    words: List[str] = []

    if force_new_topic or (prior_state.get("intent") is None):
        intent = random.choice(["symptom_report", "request_help", "admin_issue", "communication_status"])

        if intent == "symptom_report":
            for w in [pick_from_set(lex["person"]), pick_from_set(lex["symptom"])]:
                if w: words.append(w)
            if random.random() < 0.7:
                w = pick_from_set(lex["location"]);  words.append(w) if w else None
            if random.random() < 0.45:
                w = pick_from_set(lex["time"]);      words.append(w) if w else None
            if random.random() < 0.35:
                w = pick_from_set(lex["request"]);   words.append(w) if w else None

        elif intent == "request_help":
            for w in [pick_from_set(lex["person"]), pick_from_set(lex["request"])]:
                if w: words.append(w)
            if random.random() < 0.4:
                w = pick_from_set(lex["symptom"]); words.append(w) if w else None
            if random.random() < 0.35:
                w = pick_from_set(lex["time"]); words.append(w) if w else None

        elif intent == "admin_issue":
            w = pick_from_set(lex["admin"]);    words.append(w) if w else None
            if random.random() < 0.6:
                w = pick_from_set(lex["request"]); words.append(w) if w else None
            if random.random() < 0.35:
                w = pick_from_set(lex["person"]);  words.append(w) if w else None

        else:
            for w in [pick_from_set(lex["comm"]), pick_from_set(lex["person"])]:
                if w: words.append(w)

        if random.random() < 0.25:
            w = pick_from_set(lex["condition"]); words.append(w) if w else None
        if random.random() < 0.20:
            w = pick_from_set(lex["family"]); words.append(w) if w else None

    else:
        slots = prior_state.get("slots", {})
        missing = []
        if not slots.get("symptom"):  missing.append(("symptom", lex["symptom"]))
        if not slots.get("location"): missing.append(("location", lex["location"]))
        if not slots.get("time"):     missing.append(("time", lex["time"]))
        if not slots.get("request"):  missing.append(("request", lex["request"]))
        if not slots.get("person"):   missing.append(("person", lex["person"]))

        if missing:
            _, s = random.choice(missing)
            w = pick_from_set(s)
            if w: words.append(w)

        if random.random() < EXTRA_TOKEN_PROB:
            extra_pool = []
            if lex["family"]:    extra_pool.append(pick_from_set(lex["family"]))
            if lex["condition"]: extra_pool.append(pick_from_set(lex["condition"]))
            if lex["admin"]:     extra_pool.append(pick_from_set(lex["admin"]))
            extra_pool = [x for x in extra_pool if x]
            if extra_pool:
                words.append(random.choice(extra_pool))

    words = [normalize_word(w) for w in words if w]
    words = list(dict.fromkeys(words))

    if not words:
        w = pick_from_set(lex["all"])
        words = [w] if w else ["unknown"]

    random.shuffle(words)
    L = random.randint(MIN_SIGNS_PER_TURN, MAX_SIGNS_PER_TURN)
    return words[:L]


# =========================
# OPTIONAL API (doctor_text)
# =========================
def get_client():
    if not USE_API_TEXT:
        return None
    if OpenAI is None:
        raise RuntimeError("openai package not found. pip install openai")
    api_key = API_KEY_ENV
    return OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)


def api_generate_doctor_text(client, conversation_history: List[Dict[str, Any]], current_turn_selected: List[str]) -> Dict[str, Any]:
    system = """You are a medical sign language interpreter AI assistant working in a hospital setting.

Your role is to translate sign language recognition outputs into clear, professional English text that doctors and medical staff can understand.

## Core Principles

1. **Accuracy First**: Only use information explicitly present in the conversation history and current turn
2. **No Fabrication**: Never invent symptoms, diagnoses, time frames, or medical details
3. **Literal Translation Priority**: Default to literal word-by-word translation unless context clearly requires integration
4. **Context Awareness**: Consider conversation history but don't force connections
5. **Professional Tone**: Write in clear, medical-appropriate language

## Input Format

You will receive:
- `conversation_history`: Array of previous turns with their selected words and states
- `current_turn_selected`: The words recognized in the current turn (THESE ARE THE PRIMARY FOCUS)

## Output Format

Return ONLY a valid JSON object with exactly these two fields:

{
  "patient_message": "A natural English sentence of what the patient is communicating",
  "structured_summary": "Bullet-pointed structured information for medical records"
}

## CRITICAL TRANSLATION RULES - READ CAREFULLY

### Rule 1: Literal First Approach
When translating current_turn_selected, start with literal translation:
- ["office"] = "Office" or "I need office" (NOT "I need to go to the office")
- ["you"] = "You" or "I mean you" (NOT "I need you to help")
- ["back"] = "My back" or "Back" (NOT "back pain" unless "pain" is also in current turn)
- ["afraid"] = "I am afraid" (NOT "I am afraid for my baby" unless baby is in current turn)

### Rule 2: Context Integration - When and How
ONLY integrate previous context when:
✅ The current word is clearly a continuation (e.g., "pain" after "stomach")
✅ The current word references something previously mentioned explicitly
✅ There's a clear semantic connection

DON'T integrate context when:
❌ Current word can stand alone meaningfully
❌ Current word is unrelated to previous topic
❌ You're adding interpretation not present in the words

### Rule 3: Questions vs Statements
NEVER turn statements into questions unless there's explicit question indicator:
❌ WRONG: ["doctor", "you", "evening"] → "Are you the doctor on duty?"
✅ RIGHT: ["doctor", "you", "evening"] → "I need a doctor this evening" or "Doctor for me this evening"

### Rule 4: Body Parts and Symptoms
- Body part ALONE = location reference: "eyes" → "my eyes", "head" → "my head"
- Body part + symptom word = combined: ["head", "pain"] → "head pain" or "pain in my head"
- Body part in new turn = NEW information, not necessarily related to previous symptoms

### Rule 5: Single Word Turns
Use MINIMAL interpretation for single words:
- "afraid" = "I am afraid" (period. Don't add why)
- "baby" = "My baby" or "Baby" (don't add assumptions)
- "office" = "Office" or context-dependent minimal addition
- "brain" = "My brain" or "Brain issue"

### Rule 6: Family and Pronouns
- "my" + family = clear possessive: ["my", "mother"] → "my mother"
- Family word alone = ["baby"] → "my baby" or "the baby"
- "you" = second person, not third person assumption

### Rule 7: Time and Tense
- Use present tense unless time word indicates past: ["pain"] → "I have pain"
- Time word changes tense: ["sick", "yesterday"] → "I was sick yesterday"
- Don't add time information not present in current or previous turns

## Enhanced Examples with Common Mistakes

### Example 1: Single Word - Office
Input:
{
  "conversation_history": [],
  "current_turn_selected": ["office"]
}

❌ WRONG Output:
{
  "patient_message": "I need to go to the office.",
  "structured_summary": "- Request: office visit"
}

✅ CORRECT Output:
{
  "patient_message": "Office.",
  "structured_summary": "- Location mentioned: office"
}

### Example 2: Single Word with Context - "You"
Input:
{
  "conversation_history": [
    {"turn": 1, "selected": ["office"], "state": {"intent": "request_help", "slots": {"request": "office"}}}
  ],
  "current_turn_selected": ["you"]
}

❌ WRONG Output:
{
  "patient_message": "I need you to help with the office.",
  "structured_summary": "- Request: office assistance"
}

✅ CORRECT Output:
{
  "patient_message": "You for office.",
  "structured_summary": "- Person: you\n- Previous context: office"
}

### Example 3: Body Part Context - "Brain"
Input:
{
  "conversation_history": [
    {"turn": 1, "selected": ["eyes", "doctor"], "state": {"intent": "symptom_report", "slots": {"symptom": "eyes", "request": "doctor"}}}
  ],
  "current_turn_selected": ["brain"]
}

❌ WRONG Output:
{
  "patient_message": "I need a doctor for my eyes and brain.",
  "structured_summary": "- Symptoms: eyes, brain\n- Request: doctor"
}

✅ CORRECT Output:
{
  "patient_message": "Brain. I mentioned eyes before.",
  "structured_summary": "- Current: brain\n- Previous symptoms: eyes\n- Request: doctor"
}

### Example 4: Emotion Word - "Afraid"
Input:
{
  "conversation_history": [
    {"turn": 1, "selected": ["baby", "sick"], "state": {"intent": "symptom_report", "slots": {"symptom": "sick", "family": ["baby"]}}}
  ],
  "current_turn_selected": ["afraid"]
}

❌ WRONG Output:
{
  "patient_message": "I am afraid for my sick baby.",
  "structured_summary": "- Emotion: afraid\n- Concern: baby (sick)"
}

✅ CORRECT Output:
{
  "patient_message": "I am afraid. My baby is sick.",
  "structured_summary": "- Emotion: afraid\n- Family: baby\n- Symptom: sick"
}

### Example 5: Question Assumption Error
Input:
{
  "conversation_history": [],
  "current_turn_selected": ["doctor", "you", "evening"]
}

❌ WRONG Output:
{
  "patient_message": "Are you the doctor on duty this evening?",
  "structured_summary": "- Question about doctor availability\n- Time: evening"
}

✅ CORRECT Output:
{
  "patient_message": "Doctor for me this evening.",
  "structured_summary": "- Request: doctor\n- Time: evening\n- Person: me"
}

### Example 6: Progressive Context Building (GOOD)
Turn 1:
Input:
{
  "conversation_history": [],
  "current_turn_selected": ["i", "pain"]
}

Output:
{
  "patient_message": "I have pain.",
  "structured_summary": "- Patient: I\n- Symptom: pain"
}

Turn 2:
Input:
{
  "conversation_history": [
    {"turn": 1, "selected": ["i", "pain"], "state": {"intent": "symptom_report", "slots": {"person": "i", "symptom": "pain"}}}
  ],
  "current_turn_selected": ["stomach"]
}

Output:
{
  "patient_message": "I have stomach pain.",
  "structured_summary": "- Patient: I\n- Symptom: pain\n- Location: stomach"
}

Turn 3:
Input:
{
  "conversation_history": [
    {"turn": 1, "selected": ["i", "pain"], "state": {"intent": "symptom_report", "slots": {"person": "i", "symptom": "pain"}}},
    {"turn": 2, "selected": ["stomach"], "state": {"intent": "symptom_report", "slots": {"person": "i", "symptom": "pain", "location": "stomach"}}}
  ],
  "current_turn_selected": ["yesterday"]
}

Output:
{
  "patient_message": "I have had stomach pain since yesterday.",
  "structured_summary": "- Patient: I\n- Symptom: stomach pain\n- Time: since yesterday"
}

### Example 7: Separate Topics (DON'T Force Connection)
Input:
{
  "conversation_history": [
    {"turn": 1, "selected": ["headache", "morning"], "state": {"intent": "symptom_report", "slots": {"symptom": "headache", "time": "morning"}}}
  ],
  "current_turn_selected": ["insurance", "form"]
}

❌ WRONG Output:
{
  "patient_message": "I had a headache this morning and need an insurance form.",
  "structured_summary": "- Symptom: headache (morning)\n- Request: insurance form"
}

✅ CORRECT Output:
{
  "patient_message": "I need an insurance form. I had a headache this morning.",
  "structured_summary": "- Current request: insurance form\n- Previous symptom: headache (morning)"
}

### Example 8: Building Related Context (GOOD Integration)
Input:
{
  "conversation_history": [
    {"turn": 1, "selected": ["my", "daughter", "sick"], "state": {"intent": "symptom_report", "slots": {"person": "my", "symptom": "sick", "family": ["daughter"]}}}
  ],
  "current_turn_selected": ["fever", "high"]
}

Output:
{
  "patient_message": "My daughter is sick with a high fever.",
  "structured_summary": "- Patient's daughter\n- Symptoms: sick, high fever"
}

## Word Category Guidelines

**Pronouns**: i, you, my, our, their, me
→ Translate literally, minimal addition

**Body Parts**: head, back, stomach, chest, eyes, ear, throat, brain, shoulder, abdomen
→ Alone = "my [body part]" or just "[body part]"
→ With context = check if related to previous symptom or new issue

**Symptoms**: pain, hurt, sick, dizzy, vomit, headache, cold, infection
→ Present tense unless time word says otherwise

**Time**: today, yesterday, morning, evening, ago, before, day, week, month
→ Affects verb tense appropriately

**Requests**: doctor, nurse, help, hospital, appointment, test, surgery
→ "I need [request]" or "[request] please"

**Family**: mother, father, brother, sister, baby, son, daughter
→ Use possessive "my" if context supports it

**Admin**: insurance, form, document, paper, money, pay
→ Usually "I need" or "help with"

**Emotions**: afraid, worried, sad, happy
→ "I am [emotion]" - don't add elaborate reasoning

## Final Reminders

1. **Primary focus**: Translate current_turn_selected words FIRST
2. **Secondary focus**: Add context ONLY if clearly related
3. **Never assume**: Questions, complex reasons, medical conclusions
4. **Stay literal**: Especially for single words or ambiguous combinations
5. **Separate topics**: Don't force unrelated turns to connect
6. **JSON only**: No explanations, no markdown, pure JSON response
7. **Natural language**: While being literal, still use grammatical English
8. **Medical appropriate**: Professional tone but not overly technical

Your output must be production-ready for real medical documentation systems. Every word you add beyond the literal translation must be justifiable from the input data."""

    user_content = {
        "conversation_history": conversation_history,
        "current_turn_selected": current_turn_selected
    }

    last_err = None
    for _ in range(API_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=DEEPSEEK_MODEL,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": json.dumps(user_content, ensure_ascii=False)}
                ],
                temperature=API_TEMPERATURE,
                max_tokens=API_MAX_TOKENS,
                stream=False
            )
            txt = resp.choices[0].message.content.strip()
            
            # Try to parse JSON
            try:
                result = json.loads(txt)
                return result
            except Exception:
                # Try to extract JSON from markdown code blocks
                m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", txt, flags=re.S)
                if m:
                    return json.loads(m.group(1))
                # Try to find JSON object directly
                m = re.search(r"\{.*\}", txt, flags=re.S)
                if m:
                    return json.loads(m.group(0))
                return {
                    "patient_message": None,
                    "structured_summary": None
                }
        except Exception as e:
            last_err = e
            time.sleep(API_RETRY_SLEEP)

    return {
        "patient_message": None,
        "structured_summary": None,
        "error": str(last_err) if last_err else "unknown"
    }


# =========================
# BUILD ONE TURN RECORD
# =========================
def build_turn_records(
    dialogue_id: int,
    turn_number: int,
    total_turns: int,
    prior_state: Dict[str, Any],
    true_words: List[str],
    glosses: List[str],
    conf_map: Dict[str, List[str]],
    second_guess: Dict[str, str],
    third_guess: Dict[str, str],
    lex: Dict[str, set],
) -> Tuple[dict, dict, str, Dict[str, Any]]:
    sign_groups: List[List[Candidate]] = []
    gold_selected_observed_order: List[str] = []
    top1_true_flags: List[bool] = []

    for w in true_words:
        cands, true_w, top1_is_true = make_candidate_group(
            true_word=w,
            glosses=glosses,
            conf_map=conf_map,
            second_guess=second_guess,
            third_guess=third_guess
        )
        sign_groups.append(cands)
        gold_selected_observed_order.append(true_w)
        top1_true_flags.append(top1_is_true)

    zipped = list(zip(sign_groups, gold_selected_observed_order, top1_true_flags))
    random.shuffle(zipped)
    sign_groups, gold_selected_observed_order, top1_true_flags = map(list, zip(*zipped))

    questions_en: List[str] = []
    need_clar = False
    per_sign_amb = []
    for sg in sign_groups:
        amb = sign_ambiguity(sg)
        per_sign_amb.append(amb)
        if amb:
            need_clar = True
            if len(sg) >= 2:
                questions_en.append(english_question_for_pair(sg[0].word, sg[1].word))

    updated_state = update_state(prior_state, gold_selected_observed_order, lex)

    alternatives = []
    if need_clar:
        alt1 = [sg[0].word for sg in sign_groups]
        alternatives.append({"selected": alt1, "state": update_state(prior_state, alt1, lex)})

        alt2 = []
        for sg, amb in zip(sign_groups, per_sign_amb):
            alt2.append(sg[1].word if amb and len(sg) >= 2 else sg[0].word)
        alternatives.append({"selected": alt2, "state": update_state(prior_state, alt2, lex)})
        alternatives = alternatives[:2]

    avg_top1 = sum([sg[0].p for sg in sign_groups]) / max(1, len(sign_groups))
    conf = safe_float(avg_top1 - (0.12 if need_clar else 0.0))

    input_obj = {
        "prior_state": prior_state,
        "candidates": [
            [{"word": c.word, "p": round(c.p, 4)} for c in sg]
            for sg in sign_groups
        ]
    }
    output_obj = {
        "selected": gold_selected_observed_order,
        "state": updated_state,
        "need_clarification": need_clar,
        "clarification_questions": questions_en,
        "alternatives": alternatives,
        "confidence": round(conf, 4)
    }

    sft_record = {
        "dialogue_id": dialogue_id,
        "turn_number": turn_number,
        "total_turns": total_turns,
        "timestamp": now_iso(),
        "input": input_obj,
        "output": output_obj
    }

    debug_record = {
        "dialogue_id": dialogue_id,
        "turn_number": turn_number,
        "total_turns": total_turns,
        "timestamp": now_iso(),
        "ground_truth_words_pre_shuffle": true_words,
        "ground_truth_selected_observed_order": gold_selected_observed_order,
        "top1_is_true_flags": top1_true_flags,
        "signature": sequence_signature(sign_groups),
        "prior_state": prior_state,
        "updated_state": updated_state,
        "need_clarification": need_clar,
        "clarification_questions": questions_en,
        "candidates": input_obj["candidates"]
    }

    sig = debug_record["signature"]
    return sft_record, debug_record, sig, updated_state


# =========================
# BATCH API ENRICH
# =========================
def enrich_batch_with_api_text(client, batch_sft: List[dict], dialogue_conversations: Dict[int, List[Dict]]) -> None:
    if not USE_API_TEXT or client is None:
        return

    tasks = []
    for idx, rec in enumerate(batch_sft):
        dialogue_id = rec["dialogue_id"]
        turn_number = rec["turn_number"]
        selected_words = rec["output"]["selected"]
        
        # Get conversation history for this dialogue up to current turn
        history = dialogue_conversations.get(dialogue_id, [])
        conv_history = [h for h in history if h["turn"] < turn_number]
        
        tasks.append((idx, conv_history, selected_words))

    results = {}

    def worker(conv_history, selected):
        return api_generate_doctor_text(client, conv_history, selected)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        future_map = {ex.submit(worker, conv, sel): idx for (idx, conv, sel) in tasks}
        for fut in as_completed(future_map):
            idx = future_map[fut]
            try:
                results[idx] = fut.result()
            except Exception as e:
                results[idx] = {"patient_message": None, "structured_summary": None, "error": str(e)}

    for idx, rec in enumerate(batch_sft):
        rec["output"]["doctor_text"] = results.get(idx, None)


# =========================
# MAIN
# =========================
def main():
    random.seed(RANDOM_SEED)

    if not os.path.exists(GLOSS_FILE):
        raise FileNotFoundError(f"{GLOSS_FILE} not found.")
    if not os.path.exists(PRIORS_FILE):
        raise FileNotFoundError(f"{PRIORS_FILE} not found.")

    glosses = load_glosses(GLOSS_FILE)
    conf_map, second_guess, third_guess = parse_priors_txt(PRIORS_FILE)
    conf_map = make_confusion_symmetric(conf_map)
    lex = build_lexicon(glosses)

    seen = load_seen_signatures(SEEN_SIGNATURES_FILE) if AVOID_GLOBAL_DUPLICATES else set()
    seen_writer = LineBatchWriter(SEEN_SIGNATURES_FILE)

    sft_writer = JsonlBatchWriter(OUTPUT_JSONL)
    dbg_writer = JsonlBatchWriter(OUTPUT_DEBUG_JSONL) if WRITE_DEBUG else None

    client = get_client()

    print("Loaded:")
    print(f"- glosses: {len(glosses)}")
    print(f"- confusion_map: {len(conf_map)} (symmetric)")
    print(f"- hard_second: {len(second_guess)}")
    print(f"- hard_third: {len(third_guess)}")
    print(f"- seen signatures: {len(seen)}")
    print(f"- USE_API_TEXT: {USE_API_TEXT} | MAX_WORKERS={MAX_WORKERS} | BATCH_SIZE={BATCH_SIZE}")

    batch_sft: List[dict] = []
    batch_dbg: List[dict] = []
    batch_sigs: List[str] = []
    
    # Track conversation history per dialogue for API calls
    dialogue_conversations: Dict[int, List[Dict]] = {}

    total_written = 0

    for d in range(1, NUM_DIALOGUES + 1):
        turns = random.randint(MIN_TURNS, MAX_TURNS)
        prior_state: Dict[str, Any] = {"intent": None, "slots": {}}
        local_seen = set()
        
        # Initialize conversation history for this dialogue
        dialogue_conversations[d] = []

        if d % 50 == 0:
            print(f"\nDialogue {d}/{NUM_DIALOGUES} ...")

        for t in range(1, turns + 1):
            force_new_topic = (random.random() < TOPIC_SWITCH_PROB) or (prior_state.get("intent") is None)
            true_words = build_turn_ground_truth(lex, prior_state, force_new_topic)

            sft_record, debug_record, sig, updated_state = build_turn_records(
                dialogue_id=d,
                turn_number=t,
                total_turns=turns,
                prior_state=prior_state,
                true_words=true_words,
                glosses=glosses,
                conf_map=conf_map,
                second_guess=second_guess,
                third_guess=third_guess,
                lex=lex
            )

            if AVOID_GLOBAL_DUPLICATES and sig in seen:
                continue
            if AVOID_LOCAL_DUPLICATES and sig in local_seen:
                continue

            batch_sft.append(sft_record)
            if WRITE_DEBUG:
                batch_dbg.append(debug_record)
            batch_sigs.append(sig)
            
            # Add to conversation history for this dialogue
            dialogue_conversations[d].append({
                "turn": t,
                "selected": sft_record["output"]["selected"],
                "state": sft_record["output"]["state"]
            })

            if AVOID_GLOBAL_DUPLICATES:
                seen.add(sig)
                seen_writer.add(sig)
            if AVOID_LOCAL_DUPLICATES:
                local_seen.add(sig)

            prior_state = updated_state

            if (t % FLUSH_EVERY_N_TURNS_PRINT) == 0 and d % 200 == 0:
                top1_words = [sign[0]["word"] for sign in sft_record["input"]["candidates"]]
                gold_words = sft_record["output"]["selected"]
                print(f"  d={d} t={t} top1={' → '.join(top1_words)} | gold={' → '.join(gold_words)}")

            if len(batch_sft) >= BATCH_SIZE:
                enrich_batch_with_api_text(client, batch_sft, dialogue_conversations)

                for rec in batch_sft:
                    sft_writer.add(rec)
                sft_writer.flush()

                if WRITE_DEBUG and dbg_writer:
                    for rec in batch_dbg:
                        dbg_writer.add(rec)
                    dbg_writer.flush()

                seen_writer.flush()

                total_written += len(batch_sft)
                batch_sft.clear()
                batch_dbg.clear()
                batch_sigs.clear()
                
                # Clean up old dialogue histories to save memory
                current_dialogues = {rec["dialogue_id"] for rec in batch_sft}
                for old_d in list(dialogue_conversations.keys()):
                    if old_d not in current_dialogues and old_d < d - 100:
                        del dialogue_conversations[old_d]

    if batch_sft:
        enrich_batch_with_api_text(client, batch_sft, dialogue_conversations)

        for rec in batch_sft:
            sft_writer.add(rec)
        sft_writer.flush()

        if WRITE_DEBUG and dbg_writer:
            for rec in batch_dbg:
                dbg_writer.add(rec)
            dbg_writer.flush()

        seen_writer.flush()

        total_written += len(batch_sft)
        batch_sft.clear()
        batch_dbg.clear()
        batch_sigs.clear()

    print("\nDONE")
    print(f"- wrote: {total_written} examples")
    print(f"- output: {OUTPUT_JSONL}")
    if WRITE_DEBUG:
        print(f"- debug : {OUTPUT_DEBUG_JSONL}")
    print(f"- seen  : {SEEN_SIGNATURES_FILE}")


if __name__ == "__main__":
    main()