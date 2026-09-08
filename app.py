"""
LinguaLoop — Language Practice Coach on LangGraph.

Demonstrates four graded LangGraph topics:
  1. Graph — StateGraph, typed state, nodes, conditional edge, UI visualization
  2. Reducers — add_messages + operator.add (vocab_learned), checkpointer + thread_id
  3. Trimming & filtering — before every model call; UI shows before/after counts
  4. Memory — short-term (MemorySaver / thread) + long-term (InMemoryStore / user profile)

Product idea: a useful practice partner for anyone learning a language —
converse in scenarios, translate, get grammar tips, and quiz words you actually learned.
"""

from __future__ import annotations

import json
import operator
import os
import random
import re
import urllib.error
import urllib.request
from typing import Annotated, Any, Dict, List, Optional, TypedDict

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.store.memory import InMemoryStore
from pydantic import BaseModel

# =====================================================================
# LLM ENGINE (swappable: Gemini / OpenAI / Ollama / Offline Demo)
# =====================================================================


def resolve_llm_model(provider: str, preferred_model: Optional[str]) -> str:
    """Normalize the configured model and default to the local Ollama phi4-mini when needed."""
    provider_name = (provider or "offline").lower()
    model_name = (preferred_model or "").strip()

    if provider_name == "ollama":
        if model_name and model_name.lower() not in {"demo-model", "llama3"}:
            return model_name
        return "phi4-mini:latest"
    if provider_name == "gemini":
        if model_name and model_name.lower() != "demo-model":
            return model_name
        return "gemini-1.5-flash"
    if provider_name == "openai":
        if model_name and model_name.lower() != "demo-model":
            return model_name
        return "gpt-4o-mini"
    return "demo-model"


def detect_initial_provider() -> tuple[str, str, str]:
    if os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
        return (
            "gemini",
            os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or "",
            "gemini-1.5-flash",
        )
    if os.environ.get("OPENAI_API_KEY"):
        return "openai", os.environ.get("OPENAI_API_KEY") or "", "gpt-4o-mini"

    base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    try:
        req = urllib.request.Request(f"{base_url.rstrip('/')}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=3) as response:
            payload = json.loads(response.read().decode("utf-8"))
            models = []
            for item in payload.get("models", []):
                name = item.get("name") or item.get("model") or ""
                if name:
                    models.append(name)
            for name in models:
                if "phi4-mini" in name.lower():
                    return "ollama", "", "phi4-mini:latest"
            if models:
                return "ollama", "", models[0]
    except Exception:
        pass

    return "offline", "", "demo-model"


_init_provider, _init_key, _init_model = detect_initial_provider()

LLM_CONFIG: Dict[str, str] = {
    "provider": os.environ.get("LLM_PROVIDER", _init_provider),
    "api_key": _init_key,
    "model": os.environ.get("LLM_MODEL") or resolve_llm_model(
        os.environ.get("LLM_PROVIDER", _init_provider), _init_model
    ),
    "ollama_base_url": os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
}


def query_llm(prompt: str, system_prompt: str = "", json_mode: bool = False) -> Optional[str]:
    """HTTP dispatch to Gemini / OpenAI / Ollama. Returns None on offline/error."""
    provider = LLM_CONFIG.get("provider", "offline").lower()
    api_key = LLM_CONFIG.get("api_key", "").strip()

    if provider == "gemini" and api_key:
        model = resolve_llm_model(provider, LLM_CONFIG.get("model"))
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{model}:generateContent?key={api_key}"
        )
        parts: List[Dict[str, str]] = []
        if system_prompt:
            parts.append({"text": f"System Instructions:\n{system_prompt}\n\n"})
        parts.append({"text": prompt})
        gen_config: Dict[str, Any] = {"temperature": 0.2 if json_mode else 0.7}
        if json_mode:
            gen_config["responseMimeType"] = "application/json"
        body = {"contents": [{"parts": parts}], "generationConfig": gen_config}
        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(body).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=12) as response:
                res_data = json.loads(response.read().decode("utf-8"))
                candidates = res_data.get("candidates", [])
                if candidates:
                    out_parts = candidates[0].get("content", {}).get("parts", [])
                    if out_parts:
                        return out_parts[0].get("text", "")
        except Exception as exc:  # noqa: BLE001
            print(f"[LLM] Gemini error: {exc}")
            return None

    elif provider == "openai" and api_key:
        model = resolve_llm_model(provider, LLM_CONFIG.get("model"))
        msgs: List[Dict[str, str]] = []
        if system_prompt:
            msgs.append({"role": "system", "content": system_prompt})
        msgs.append({"role": "user", "content": prompt})
        body: Dict[str, Any] = {
            "model": model,
            "messages": msgs,
            "temperature": 0.2 if json_mode else 0.7,
        }
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        try:
            req = urllib.request.Request(
                "https://api.openai.com/v1/chat/completions",
                data=json.dumps(body).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=12) as response:
                res_data = json.loads(response.read().decode("utf-8"))
                choices = res_data.get("choices", [])
                if choices:
                    return choices[0].get("message", {}).get("content", "")
        except Exception as exc:  # noqa: BLE001
            print(f"[LLM] OpenAI error: {exc}")
            return None

    elif provider == "ollama":
        model = resolve_llm_model(provider, LLM_CONFIG.get("model"))
        base_url = LLM_CONFIG.get("ollama_base_url", "http://localhost:11434")
        msgs = []
        if system_prompt:
            msgs.append({"role": "system", "content": system_prompt})
        msgs.append({"role": "user", "content": prompt})
        body = {"model": model, "messages": msgs, "stream": False}
        if json_mode:
            body["format"] = "json"
        try:
            req = urllib.request.Request(
                f"{base_url}/api/chat",
                data=json.dumps(body).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=15) as response:
                res_data = json.loads(response.read().decode("utf-8"))
                return res_data.get("message", {}).get("content", "")
        except Exception as exc:  # noqa: BLE001
            print(f"[LLM] Ollama error: {exc}")
            return None

    return None


# =====================================================================
# TOPIC 4: SHORT-TERM (checkpointer) + LONG-TERM (store) MEMORY
# =====================================================================
checkpointer = MemorySaver()
store = InMemoryStore()
ACTIVE_THREADS: set[str] = set()

DEFAULT_PROFILE = {
    "name": "Learner",
    "target_language": "Spanish",
    "level": "beginner",
}

# Phrase banks for offline practice / scenarios (useful for learners without API keys)
PHRASEBOOK: Dict[str, Dict[str, Any]] = {
    "spanish": {
        "hello": ("Hola", "hello"),
        "thanks": ("Gracias", "thank you"),
        "please": ("Por favor", "please"),
        "water": ("agua", "water"),
        "coffee": ("café", "coffee"),
        "bill": ("la cuenta", "the bill"),
        "menu": ("el menú", "the menu"),
        "train": ("el tren", "the train"),
        "ticket": ("el billete", "the ticket"),
        "friend": ("amigo / amiga", "friend"),
        "food": ("la comida", "food"),
        "today": ("hoy", "today"),
        "cafe_open": "¡Bienvenido/a al café! ¿Qué desea pedir?",
        "travel_open": "Estamos en la estación. ¿A dónde quiere viajar?",
        "meet_open": "¡Mucho gusto! ¿Cómo se llama usted?",
        "grammar": (
            "Ser vs Estar (both mean 'to be'):\n"
            "• Ser — identity, origin, time (Soy Ana. Es la una.)\n"
            "• Estar — location, feelings, temporary states (Estoy en casa. Estoy feliz.)"
        ),
    },
    "french": {
        "hello": ("Bonjour", "hello"),
        "thanks": ("Merci", "thank you"),
        "please": ("S'il vous plaît", "please"),
        "water": ("eau", "water"),
        "coffee": ("café", "coffee"),
        "bill": ("l'addition", "the bill"),
        "menu": ("la carte", "the menu"),
        "train": ("le train", "the train"),
        "ticket": ("le billet", "the ticket"),
        "friend": ("ami / amie", "friend"),
        "food": ("la nourriture / le repas", "food"),
        "today": ("aujourd'hui", "today"),
        "cafe_open": "Bienvenue au café ! Que souhaitez-vous commander ?",
        "travel_open": "Nous sommes à la gare. Où voulez-vous aller ?",
        "meet_open": "Enchanté(e) ! Comment vous appelez-vous ?",
        "grammar": (
            "Regular -er verbs (parler — to speak):\n"
            "• Je parle · Tu parles · Il/Elle parle\n"
            "• Nous parlons · Vous parlez · Ils/Elles parlent\n"
            "Spoken 'parle' and 'parles' sound the same — spelling carries the person."
        ),
    },
    "german": {
        "hello": ("Hallo / Guten Tag", "hello"),
        "thanks": ("Danke", "thank you"),
        "please": ("Bitte", "please"),
        "water": ("Wasser", "water"),
        "coffee": ("Kaffee", "coffee"),
        "bill": ("die Rechnung", "the bill"),
        "menu": ("die Speisekarte", "the menu"),
        "train": ("der Zug", "the train"),
        "ticket": ("die Fahrkarte", "the ticket"),
        "friend": ("der Freund / die Freundin", "friend"),
        "food": ("das Essen", "food"),
        "today": ("heute", "today"),
        "cafe_open": "Willkommen im Café! Was möchten Sie bestellen?",
        "travel_open": "Wir sind am Bahnhof. Wohin möchten Sie fahren?",
        "meet_open": "Freut mich! Wie heißen Sie?",
        "grammar": (
            "German nouns have gender and are capitalized:\n"
            "• der (masc.) · die (fem.) · das (neut.)\n"
            "V2 rule: the finite verb is usually in second position in main clauses."
        ),
    },
    "japanese": {
        "hello": ("こんにちは (konnichiwa)", "hello"),
        "thanks": ("ありがとう (arigatou)", "thank you"),
        "please": ("お願いします (onegaishimasu)", "please"),
        "water": ("水 (mizu)", "water"),
        "coffee": ("コーヒー (koohii)", "coffee"),
        "bill": ("お会計 (okaikei)", "the bill"),
        "menu": ("メニュー (menyuu)", "the menu"),
        "train": ("電車 (densha)", "the train"),
        "ticket": ("切符 (kippu)", "the ticket"),
        "friend": ("友達 (tomodachi)", "friend"),
        "food": ("食べ物 (tabemono)", "food"),
        "today": ("今日 (kyou)", "today"),
        "cafe_open": "いらっしゃいませ！ご注文は？ (Irasshaimase! What would you like?)",
        "travel_open": "駅です。どこへ行きますか？ (We're at the station. Where are you going?)",
        "meet_open": "はじめまして！お名前は？ (Nice to meet you! What's your name?)",
        "grammar": (
            "Japanese is SOV (subject–object–verb).\n"
            "Particles: は (topic) · を (object) · が (subject).\n"
            "Example: 私は本を読みます — I read a book."
        ),
    },
    "korean": {
        "hello": ("안녕하세요 (annyeonghaseyo)", "hello"),
        "thanks": ("감사합니다 (gamsahamnida)", "thank you"),
        "please": ("주세요 (juseyo)", "please / please give me"),
        "water": ("물 (mul)", "water"),
        "coffee": ("커피 (keopi)", "coffee"),
        "bill": ("계산서 (gyesanseo)", "the bill"),
        "menu": ("메뉴 (menyu)", "the menu"),
        "train": ("기차 (gicha)", "the train"),
        "ticket": ("표 (pyo)", "the ticket"),
        "friend": ("친구 (chingu)", "friend"),
        "food": ("음식 (eumsik)", "food"),
        "today": ("오늘 (oneul)", "today"),
        "cafe_open": "어서 오세요! 무엇을 드릴까요? (Welcome! What can I get you?)",
        "travel_open": "역이에요. 어디로 가세요? (We're at the station. Where are you going?)",
        "meet_open": "만나서 반가워요! 이름이 뭐예요? (Nice to meet you! What's your name?)",
        "grammar": (
            "Korean is SOV. Particles mark roles:\n"
            "• 은/는 topic · 이/가 subject · 을/를 object\n"
            "Example: 저는 물을 마셔요 — I drink water."
        ),
    },
}


def phrasebook_for(lang: str) -> Dict[str, Any]:
    return PHRASEBOOK.get(lang.lower(), PHRASEBOOK["spanish"])


def get_user_profile(user_id: str) -> Dict[str, Any]:
    item = store.get(("profile", user_id), "profile")
    if item and isinstance(item.value, dict):
        return {**DEFAULT_PROFILE, **item.value}
    return dict(DEFAULT_PROFILE)


def offline_extract_profile(text: str) -> Dict[str, Optional[str]]:
    detected_name = None
    name_match = re.search(
        r"\b(?:my name is|i am|i'm|call me)\s+([A-Z][a-z]+)",
        text,
        re.IGNORECASE,
    )
    blocked_names = {
        "learning",
        "practicing",
        "ready",
        "here",
        "trying",
        "a",
        "an",
        "want",
        "learner",
        "tired",
        "hungry",
        "happy",
        "sad",
        "good",
        "fine",
        "well",
        "okay",
        "ok",
        "bored",
        "excited",
        "busy",
        "late",
        "early",
        "going",
        "feeling",
        "doing",
        "studying",
        "working",
        "not",
        "very",
        "so",
        "really",
    }
    if name_match:
        cand = name_match.group(1).capitalize()
        if cand.lower() not in blocked_names:
            detected_name = cand

    detected_lang = None
    lang_match = re.search(
        r"\b(?:learning|learn|practice|practicing|speak|speaking|study|studying|"
        r"teach me|switch to|target language is)\s+"
        r"(?:to\s+)?(?:learn\s+|practice\s+|speak\s+)?"
        r"(Spanish|French|German|Italian|Japanese|Korean|Mandarin|Chinese|"
        r"Portuguese|Russian|Arabic)\b",
        text,
        re.IGNORECASE,
    )
    if not lang_match:
        lang_match = re.search(
            r"\b(Spanish|French|German|Italian|Japanese|Korean|Mandarin|Chinese|"
            r"Portuguese|Russian|Arabic)\b",
            text,
            re.IGNORECASE,
        )
    if lang_match:
        detected_lang = lang_match.group(1).capitalize()
        if detected_lang.lower() == "chinese":
            detected_lang = "Mandarin"

    detected_level = None
    level_match = re.search(
        r"\b(beginner|novice|intermediate|advanced|expert)\b",
        text,
        re.IGNORECASE,
    )
    if level_match:
        detected_level = level_match.group(1).lower()

    return {
        "detected_name": detected_name,
        "detected_language": detected_lang,
        "detected_level": detected_level,
    }


def offline_heuristic_router(text: str, messages: list[BaseMessage]) -> Dict[str, Any]:
    text_lower = text.lower()
    extraction = offline_extract_profile(text)

    # If the previous AI turn was a quiz prompt, treat this reply as a quiz answer
    # unless the user clearly asks for a different skill.
    last_ai = None
    for m in reversed(messages[:-1] if messages else []):
        if isinstance(m, AIMessage):
            last_ai = m
            break
    pending_quiz = bool(
        last_ai
        and getattr(last_ai, "additional_kwargs", {}).get("scaffolding")
    )
    explicit_other = any(
        k in text_lower
        for k in (
            "translate",
            "how do you say",
            "grammar",
            "why is it",
            "conjugat",
            "quiz me again",
            "another quiz",
            "new quiz",
            "test me again",
        )
    )

    if pending_quiz and not explicit_other:
        intent = "quiz_vocab"
    elif any(
        k in text_lower
        for k in ("quiz", "test me", "flashcard", "vocab quiz", "review words", "test my vocab")
    ):
        intent = "quiz_vocab"
    elif any(
        k in text_lower
        for k in (
            "translate",
            "how do you say",
            "what does",
            "what is",
            "meaning of",
            "in korean",
            "in japanese",
            "in french",
            "in spanish",
            "in german",
        )
    ):
        intent = "translate"
    elif any(
        k in text_lower
        for k in (
            "why is it",
            "grammar",
            "conjugat",
            "difference between",
            "subjunctive",
            "tense",
            "particle",
            "word order",
            "sentence order",
        )
    ) or re.search(r"\brule\b", text_lower):
        intent = "explain_grammar"
    else:
        intent = "practice_conversation"

    return {"intent": intent, **extraction}


def llm_classify_intent_and_profile(
    text: str,
    current_profile: Dict[str, Any],
    messages: list[BaseMessage],
) -> Dict[str, Any]:
    if LLM_CONFIG.get("provider", "offline").lower() == "offline":
        return offline_heuristic_router(text, messages)

    router_prompt = (
        f"You are the router for LinguaLoop, a language practice coach.\n"
        f"User message: \"{text}\"\n"
        f"Current Profile: {json.dumps(current_profile)}\n\n"
        f"Classify intent as exactly ONE of:\n"
        f"- practice_conversation\n- translate\n- explain_grammar\n- quiz_vocab\n\n"
        f"Also extract detected_name, detected_language, detected_level when stated.\n"
        f"Output ONLY JSON with keys: intent, detected_name, detected_language, detected_level."
    )
    llm_output = query_llm(
        router_prompt,
        system_prompt="You are a strict JSON intent classifier.",
        json_mode=True,
    )
    if llm_output:
        try:
            cleaned = re.sub(r"^```(?:json)?\s*", "", llm_output.strip(), flags=re.IGNORECASE)
            cleaned = re.sub(r"\s*```$", "", cleaned)
            parsed = json.loads(cleaned)
            valid = {"practice_conversation", "translate", "explain_grammar", "quiz_vocab"}
            intent = parsed.get("intent", "practice_conversation")
            if intent not in valid:
                intent = "practice_conversation"
            # Preserve quiz-answer routing even when LLM is live
            heuristic = offline_heuristic_router(text, messages)
            if heuristic["intent"] == "quiz_vocab" and intent != "quiz_vocab":
                # Only override when heuristic detected a pending quiz answer
                last_ai = next(
                    (m for m in reversed(messages[:-1]) if isinstance(m, AIMessage)),
                    None,
                )
                if last_ai and getattr(last_ai, "additional_kwargs", {}).get("scaffolding"):
                    intent = "quiz_vocab"
            return {
                "intent": intent,
                "detected_name": parsed.get("detected_name"),
                "detected_language": parsed.get("detected_language"),
                "detected_level": parsed.get("detected_level"),
            }
        except Exception as exc:  # noqa: BLE001
            print(f"[LLM Router] parse fallback: {exc}")

    return offline_heuristic_router(text, messages)


def apply_profile_updates(user_id: str, extraction: Dict[str, Any]) -> Dict[str, Any]:
    current = get_user_profile(user_id)
    updated = False

    if extraction.get("detected_name"):
        name_cand = str(extraction["detected_name"]).strip().capitalize()
        blocked = {
            "learning",
            "practicing",
            "ready",
            "here",
            "trying",
            "a",
            "an",
            "learner",
            "want",
            "tired",
            "hungry",
            "happy",
            "sad",
            "good",
            "fine",
            "well",
            "okay",
            "ok",
            "bored",
            "excited",
            "busy",
            "feeling",
            "doing",
            "studying",
            "working",
        }
        if name_cand.lower() not in blocked:
            current["name"] = name_cand
            updated = True

    if extraction.get("detected_language"):
        lang_cand = str(extraction["detected_language"]).strip().capitalize()
        if lang_cand.lower() == "chinese":
            lang_cand = "Mandarin"
        current["target_language"] = lang_cand
        updated = True

    if extraction.get("detected_level"):
        lvl = str(extraction["detected_level"]).strip().lower()
        if lvl in {"beginner", "novice", "intermediate", "advanced", "expert"}:
            current["level"] = lvl
            updated = True

    if updated:
        store.put(("profile", user_id), "profile", current)
    return current


def update_user_profile_if_detected(user_id: str, text: str) -> Dict[str, Any]:
    extraction = offline_extract_profile(text)
    return apply_profile_updates(user_id, extraction)


# =====================================================================
# TOPIC 2: STATE SCHEMA — default + non-default reducers
# =====================================================================
class ChatState(TypedDict):
    # TOPIC 2: default reducer — appends / dedupes chat messages
    messages: Annotated[list[BaseMessage], add_messages]
    # TOPIC 2: NON-default reducer — concatenates vocab lists turn by turn
    vocab_learned: Annotated[list[str], operator.add]
    # Last-write-wins fields (contrast with the two Annotated channels above)
    intent: Optional[str]
    trimmed_messages: Optional[list[Dict[str, Any]]]
    trimmed_count: Optional[int]
    full_count: Optional[int]
    filtered_dropped: Optional[int]
    full_token_estimate: Optional[int]
    trimmed_token_estimate: Optional[int]
    user_id: Optional[str]
    pending_quiz_word: Optional[str]


# =====================================================================
# TOPIC 3: TOKEN BUDGET & FILTERING HELPERS
# =====================================================================
# Trim still applies (Topic 3), but keep enough same-thread turns for real dialogue.
MAX_TOKENS = 1200
MAX_MESSAGES_TO_MODEL = 12


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def estimate_messages_tokens(messages: list[BaseMessage]) -> int:
    return sum(estimate_tokens(str(m.content)) for m in messages)


def serialize_messages(messages: list[BaseMessage]) -> list[Dict[str, Any]]:
    out: list[Dict[str, Any]] = []
    for m in messages:
        role = "user" if isinstance(m, HumanMessage) else "assistant"
        out.append({"role": role, "content": str(m.content)})
    return out


def hydrate_trimmed(state: ChatState) -> list[BaseMessage]:
    """Rebuild BaseMessages from the trim node's UI-facing payload.

    TOPIC 3 requirement: content nodes must call the model with trimmed history,
    not the full checkpointer message list.
    """
    raw = state.get("trimmed_messages") or []
    hydrated: list[BaseMessage] = []
    for item in raw:
        if isinstance(item, BaseMessage):
            hydrated.append(item)
            continue
        if not isinstance(item, dict):
            continue
        content = str(item.get("content", ""))
        if item.get("role") == "user":
            hydrated.append(HumanMessage(content=content))
        else:
            hydrated.append(AIMessage(content=content))
    if hydrated:
        return hydrated
    # Safety fallback — should not happen after trim_and_filter
    return list(state.get("messages") or [])


def unique_new_vocab(existing: list[str], candidates: list[str]) -> list[str]:
    """Return only vocab not already present (case-insensitive), for clean operator.add appends."""
    seen = {v.lower().strip() for v in existing}
    fresh: list[str] = []
    for item in candidates:
        key = item.lower().strip()
        if key and key not in seen:
            seen.add(key)
            fresh.append(item)
    return fresh


def parse_vocab_entry(entry: str) -> tuple[str, str]:
    """Split 'hola (hello)' → ('hola', 'hello')."""
    m = re.match(r"^(.+?)\s*\((.+)\)\s*$", entry.strip())
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return entry.strip(), ""


def score_quiz_answer(user_text: str, target_entry: str) -> tuple[bool, str]:
    term, meaning = parse_vocab_entry(target_entry)
    blob = user_text.lower()
    term_l = term.lower()
    meaning_l = meaning.lower()
    # Accept either the foreign term or the English gloss (or a clear substring)
    hits = []
    if meaning_l and (meaning_l in blob or any(w in blob for w in meaning_l.split() if len(w) > 3)):
        hits.append("meaning")
    if term_l and (term_l in blob or term_l.split("(")[0].strip() in blob):
        hits.append("term")
    # Also accept romanization / first token of term
    first = re.split(r"[\s/（(]", term_l)[0].strip()
    if first and len(first) > 2 and first in blob:
        hits.append("term")

    if hits:
        return True, (
            f"Nice work — **{target_entry}** is correct.\n\n"
            f"Want another word from your bank, or shall we keep chatting?"
        )
    hint = meaning if meaning else term
    return False, (
        f"Not quite. **{target_entry}** means **{hint}**.\n\n"
        f"Try using it once in a short sentence, or ask me to quiz you again."
    )


def prior_assistant_messages(messages: list[BaseMessage]) -> list[AIMessage]:
    return [m for m in messages[:-1] if isinstance(m, AIMessage)]


def prior_human_messages(messages: list[BaseMessage]) -> list[HumanMessage]:
    return [m for m in messages[:-1] if isinstance(m, HumanMessage)]


def history_blob(messages: list[BaseMessage], limit: int = 8) -> str:
    return " ".join(str(m.content).lower() for m in messages[-limit:])


def infer_scenario(messages: list[BaseMessage], last_lower: str) -> Optional[str]:
    """Infer active scene from this turn, or keep prior scene only for short follow-ups."""
    checks = [
        ("cafe", ("café", "cafe", "coffee", "order", "menu", "waiter", "cuenta", "addition", "rechnung", "espresso", "latte")),
        ("travel", ("travel", "train", "airport", "ticket", "station", "hotel", "billete", "billet", "fahrkarte")),
        ("meet", ("introduce", "introduction", "nice to meet", "mucho gusto", "enchanté", "freut mich")),
        ("food", ("food", "eat", "restaurant", "hungry", "lunch", "dinner", "croissant", "comida", "fromage")),
    ]
    for name, keys in checks:
        if any(k in last_lower for k in keys):
            return name

    # Clear topic shift → leave the old scene (same-thread memory still used in generic reply)
    if re.search(
        r"\b(i am|i'm|i feel|tired|my day|today was|yesterday|weather|at work|at school)\b",
        last_lower,
    ):
        return None

    # Sticky scene only for short in-scene answers (e.g. "coffee", "water please")
    prior_ai = [m for m in messages[:-1] if isinstance(m, AIMessage)]
    if prior_ai and len(last_lower.split()) <= 6:
        ai_blob = str(prior_ai[-1].content).lower()
        for name, keys in checks:
            if any(k in ai_blob for k in keys):
                return name
    return None


def clip_quote(text: str, max_len: int = 72) -> str:
    cleaned = re.sub(r"\s+", " ", text.strip())
    if len(cleaned) <= max_len:
        return cleaned
    return cleaned[: max_len - 1].rstrip() + "…"


def language_tone(level: str, lang: str) -> str:
    """Return teacher guidance depending on the learner's level."""
    level_key = (level or "beginner").lower()
    if level_key in {"beginner", "novice"}:
        return (
            f"Use a beginner-friendly bilingual style: speak mostly in English, but include 1–2 short {lang} phrases "
            "that the learner can copy. Explain clearly, do not overload with grammar jargon."
        )
    if level_key in {"intermediate"}:
        return (
            f"Use a mixed approach: explain mainly in English, but give the key idea in {lang} with a few natural examples."
        )
    return (
        f"Use mostly {lang}, but keep brief English glosses when needed for clarity."
    )


def offline_practice_reply(
    messages: list[BaseMessage],
    profile: Dict[str, Any],
    book: Dict[str, Any],
) -> tuple[str, list[str]]:
    """
    Context-aware offline practice: continues the same-thread scene and
    references what the learner already said (short-term memory via checkpointer).
    """
    user_name = profile.get("name", "Learner")
    lang = profile.get("target_language", "Spanish")
    level = profile.get("level", "beginner")
    tone = language_tone(level, lang)

    last_text = str(messages[-1].content).strip() if messages else ""
    last_lower = last_text.lower()
    prior_ai = prior_assistant_messages(messages)
    prior_humans = prior_human_messages(messages)
    has_history = bool(prior_ai or prior_humans)
    scenario = infer_scenario(messages, last_lower)

    correction = ""
    if level == "beginner" and re.search(r"\bi want\b|\bi need\b|\bcan i\b", last_lower):
        term, gloss = book["please"]
        correction = f"\n\n✏️ Tip: try wrapping requests with **{term}** ({gloss})."

    explicit_intro = bool(
        re.search(r"\b(my name is|call me|i want to learn|i'm learning|i am learning)\b", last_lower)
    )
    bare_greeting = last_lower.strip() in {
        "hi",
        "hey",
        "hello",
        "hola",
        "bonjour",
        "hallo",
        "annyeong",
        "konnichiwa",
    } or last_lower.startswith(("hi ", "hey ", "hello "))

    intro_match = re.search(r"\b(?:hola|hello|bonjour|hallo|annyeong|konnichiwa)\s*,?\s*me llamo\s+([A-Za-zÀ-ÿ]+)", last_lower)
    if not intro_match:
        intro_match = re.search(r"\bme llamo\s+([A-Za-zÀ-ÿ]+)", last_lower)

    if intro_match and len(prior_humans) >= 1:
        if level.lower() in {"beginner", "novice"}:
            reply = (
                f"Perfect — that is the correct pattern: **Hola, me llamo {user_name}.**\n\n"
                f"In English, that means: **Hello, my name is {user_name}.**\n"
                f"Now let's move one step further with a friendly follow-up: **Mucho gusto** (nice to meet you) or **¿Cómo estás?** (how are you?)\n\n"
                f"Try: **Hola, me llamo {user_name}. Mucho gusto.**"
            )
        else:
            reply = (
                f"Perfect — that is the correct pattern: **Hola, me llamo {user_name}.**\n\n"
                f"Now let's move one step further: add a friendly follow-up like **Mucho gusto** or **¿Cómo estás?**\n"
                f"Try: **Hola, me llamo {user_name}. Mucho gusto.**"
            )
        return reply, ["mucho gusto (nice to meet you)", "¿Cómo estás? (How are you?)"]

    # --- First turn / explicit re-intro only ---
    if (not has_history and (bare_greeting or explicit_intro)) or (explicit_intro and not scenario):
        term, gloss = book["hello"]
        if level.lower() in {"beginner", "novice"}:
            reply = (
                f"Hi **{user_name}**! This is the beginner-friendly way to begin.\n\n"
                f"In Spanish, you can say: **{term}**, which means **{gloss}**.\n"
                f"A simple introduction is: **Hola, me llamo {user_name}.**\n\n"
                f"Try saying that once, then we can add a follow-up like **Mucho gusto**."
            )
        else:
            reply = (
                f"{term}, **{user_name}**! Ready to practice some **{lang}** "
                f"at your **{level}** level?\n\n"
                f"Pick a scene (café, travel, introductions) or just chat — "
                f"I'll reply in {lang} with a short English gloss. How are you today?"
            )
        return reply, [f"{term} ({gloss})", f"{book['today'][0]} ({book['today'][1]})"]

    # Mid-thread greeting → acknowledge, don't restart
    if bare_greeting and has_history:
        term, gloss = book["hello"]
        topic_hint = f" Still on our {scenario} practice?" if scenario else " What should we practice next?"
        reply = f"{term} again, {user_name}!{topic_hint}"
        return reply, [f"{term} ({gloss})"]

    last_ai_text = str(prior_ai[-1].content).lower() if prior_ai else ""
    last_user_earlier = str(prior_humans[-1].content) if prior_humans else ""

    # --- Continue café roleplay ---
    if scenario == "cafe":
        coffee_t, coffee_g = book["coffee"]
        water_t, water_g = book["water"]
        bill_t, bill_g = book["bill"]
        if any(k in last_lower for k in ("coffee", "espresso", "latte")) or re.search(
            r"\b(un café|a coffee|the coffee)\b", last_lower
        ):
            reply = (
                f"Nice — one **{coffee_t}** ({coffee_g}). "
                f"Anything else — maybe **{water_t}** ({water_g})?\n"
                f"(Say your order again in {lang} if you can.)"
                f"{correction}"
            )
            return reply, [f"{coffee_t} ({coffee_g})", f"{water_t} ({water_g})"]
        if any(k in last_lower for k in ("water", "agua", "eau", "wasser", "물", "水")):
            reply = (
                f"Got it — **{water_t}** ({water_g}). "
                f"When you're ready, ask for **{bill_t}** ({bill_g}).\n"
                f"How do you say “the bill, please” in {lang}?"
                f"{correction}"
            )
            return reply, [f"{water_t} ({water_g})", f"{bill_t} ({bill_g})"]
        if any(k in last_lower for k in ("bill", "cuenta", "addition", "check", "pay", "rechnung")):
            please_t, please_g = book["please"]
            reply = (
                f"Perfect — **{bill_t}**, **{please_t}** ({bill_g}, {please_g}). "
                f"Want to order something else, or try a new scene?"
                f"{correction}"
            )
            return reply, [f"{bill_t} ({bill_g})", f"{please_t} ({please_g})"]
        if has_history and any(
            k in last_ai_text for k in ("order", "pedir", "commander", "bestellen", "注文", "드릴까요")
        ) and len(last_lower.split()) <= 10:
            reply = (
                f"Keeping your last line in mind: “{clip_quote(last_text)}”. "
                f"In {lang} you can order **{coffee_t}** ({coffee_g}) or **{water_t}** ({water_g}). "
                f"Which one do you want?"
                f"{correction}"
            )
            return reply, [f"{coffee_t} ({coffee_g})", f"{water_t} ({water_g})"]
        # Opening the café scene (or generic in-scene turn)
        reply = (
            f"{book['cafe_open']}\n"
            f"(Welcome — what would you like to order?)\n\n"
            f"Useful words: **{coffee_t}** ({coffee_g}), **{water_t}** ({water_g})."
            f"{correction}"
        )
        return reply, [
            f"{coffee_t} ({coffee_g})",
            f"{water_t} ({water_g})",
            f"{bill_t} ({bill_g})",
        ]

    # --- Continue travel roleplay ---
    if scenario == "travel":
        ticket_t, ticket_g = book["ticket"]
        train_t, train_g = book["train"]
        city = None
        for c in ("paris", "madrid", "berlin", "tokyo", "seoul", "rome", "lyon", "barcelona", "munich"):
            if c in last_lower or c in history_blob(messages):
                city = c.capitalize()
                break
        if city or any(k in last_lower for k in ("go to", "to ", "visit", "quiero", "veux", "möchte")):
            place = city or "there"
            reply = (
                f"Nice — heading to **{place}**. "
                f"Ask for **{ticket_t}** ({ticket_g}) for the **{train_t}** ({train_g}).\n"
                f"Try: “One ticket to {place}, please.” in {lang} (I’ll help)."
                f"{correction}"
            )
            return reply, [f"{ticket_t} ({ticket_g})", f"{train_t} ({train_g})"]
        if has_history:
            reply = (
                f"Still at the station with you. You said “{clip_quote(last_text)}”. "
                f"Do you need a **{ticket_t}** ({ticket_g}) or info about the **{train_t}** ({train_g})?"
                f"{correction}"
            )
            return reply, [f"{ticket_t} ({ticket_g})", f"{train_t} ({train_g})"]
        reply = (
            f"{book['travel_open']}\n"
            f"(We're at the station — where do you want to go?)\n\n"
            f"Useful words: **{ticket_t}** ({ticket_g}), **{train_t}** ({train_g})."
            f"{correction}"
        )
        return reply, [f"{ticket_t} ({ticket_g})", f"{train_t} ({train_g})"]

    # --- Continue introductions ---
    if scenario == "meet":
        friend_t, friend_g = book["friend"]
        hello_t, hello_g = book["hello"]
        if has_history:
            reply = (
                f"Building on “{clip_quote(last_user_earlier or last_text)}”: "
                f"you can introduce a **{friend_t}** ({friend_g}) with **{hello_t}** ({hello_g}). "
                f"How would you introduce yourself in {lang}?"
                f"{correction}"
            )
            return reply, [f"{friend_t} ({friend_g})", f"{hello_t} ({hello_g})"]
        reply = (
            f"{book['meet_open']}\n"
            f"(Nice to meet you — what's your name?)\n\n"
            f"Useful word: **{friend_t}** ({friend_g})."
            f"{correction}"
        )
        return reply, [f"{friend_t} ({friend_g})", f"{hello_t} ({hello_g})"]

    # --- Continue food talk ---
    if scenario == "food":
        food_t, food_g = book["food"]
        if has_history:
            reply = (
                f"You mentioned “{clip_quote(last_text)}”. "
                f"In {lang}, **{food_t}** means {food_g}. "
                f"Can you describe that dish with one short {lang} sentence?"
                f"{correction}"
            )
            return reply, [f"{food_t} ({food_g})"]
        reply = (
            f"Let's talk food in {lang}. "
            f"**{food_t}** means {food_g}. What do you like to eat?\n"
            f"(Answer in {lang} if you can — I'll help.)"
            f"{correction}"
        )
        return reply, [f"{food_t} ({food_g})"]

    # --- Generic continuation: always reference prior turn ---
    today_t, today_g = book["today"]
    hello_t, hello_g = book["hello"]
    if has_history:
        # Answer common follow-ups to "how are you / how was your day"
        if any(k in last_lower for k in ("good", "fine", "great", "ok", "okay", "tired", "well", "bien", "mal")):
            reply = (
                f"Glad you shared that, {user_name} — “{clip_quote(last_text)}”. "
                f"In {lang}, try responding about **{today_t}** ({today_g}). "
                f"What did you do {today_g}?"
                f"{correction}"
            )
            return reply, [f"{today_t} ({today_g})"]

        # If coach asked a question last turn, treat this as the answer
        if "?" in last_ai_text or "¿" in last_ai_text:
            bank_hint = ""
            reply = (
                f"Thanks — I’m keeping that in this chat: “{clip_quote(last_text)}”. "
                f"Let’s stay with it in {lang}. "
                f"Can you say that idea again using one word from what we’ve practiced?"
                f"{bank_hint}{correction}"
            )
            return reply, [f"{today_t} ({today_g})"]

        reply = (
            f"Continuing from earlier (“{clip_quote(last_user_earlier or last_ai_text[:60])}”): "
            f"you just said “{clip_quote(last_text)}”. "
            f"Here’s a natural {lang} follow-up — tell me more, or ask me to translate a phrase."
            f"{correction}"
        )
        return reply, [f"{hello_t} ({hello_g})"]

    reply = (
        f"{hello_t}, {user_name}! Ready for some {level} {lang}?\n"
        f"**{today_t}** ({today_g}) — tell me one thing about your day "
        f"(in {lang} or English), and I'll answer in {lang}."
        f"{correction}"
    )
    return reply, [f"{hello_t} ({hello_g})", f"{today_t} ({today_g})"]


# =====================================================================
# LLM INTERFACE + OFFLINE DEMO MODEL (swap point)
# =====================================================================


def call_model(
    messages: list[BaseMessage],
    system_prompt: str,
    intent: str,
    profile: Dict[str, Any],
    vocab_learned: list[str],
    pending_quiz_word: Optional[str] = None,
) -> tuple[str, list[str], Optional[str]]:
    """
    SWAP POINT FOR REAL LLM PROVIDERS
    ---------------------------------
    Replace the offline branch with e.g. ChatOpenAI / ChatGoogleGenerativeAI.
    Graph nodes always pass the *trimmed* message list into this function.

    Returns: (reply_text, new_vocab_list, pending_quiz_word_or_None)
    """
    user_name = profile.get("name", "Learner")
    lang = profile.get("target_language", "Spanish")
    lang_key = lang.lower()
    level = profile.get("level", "beginner")
    book = phrasebook_for(lang_key)

    last_user_msg = messages[-1].content if messages else ""
    last_text = str(last_user_msg).strip()
    last_lower = last_text.lower()

    # Live LLM path
    if LLM_CONFIG.get("provider", "offline").lower() != "offline" and (
        LLM_CONFIG.get("api_key") or LLM_CONFIG.get("provider") == "ollama"
    ):
        history_str = ""
        for m in messages:
            role = "User" if isinstance(m, HumanMessage) else "Assistant"
            history_str += f"{role}: {m.content}\n"

        llm_gen_prompt = (
            f"You are LinguaLoop, an encouraging language practice coach.\n"
            f"Learner: {user_name} · Language: {lang} · Level: {level}\n"
            f"Intent: {intent}\n"
            f"Word bank this thread: {', '.join(vocab_learned[-10:]) or 'empty'}\n"
            f"Pending quiz word: {pending_quiz_word or 'none'}\n\n"
            f"Trimmed conversation sent to you (SAME THREAD — you MUST use this context):\n"
            f"{history_str}\n\n"
            f"Guidelines:\n"
            f"- NEVER restart as if you just met the learner when prior turns exist. "
            f"Continue the scene, answer their last message, and refer to what they already said.\n"
            f"- NEVER repeat the user's exact sentence back to them as if it were a new assistant answer. "
            f"Acknowledge the correct pattern and move to the next mini-step.\n"
            f"- For beginner/novice learners, respond in a bilingual style: mostly in English, with short {lang} phrases and one copyable sentence. "
            f"For intermediate learners, mix English and {lang}. For advanced learners, mostly use {lang} with brief English clarification.\n"
            f"- practice: keep the response accessible, helpful, and simple; ask one follow-up tied to this thread.\n"
            f"- translate: translate + one nuance note; introduce 1–2 useful terms.\n"
            f"- explain_grammar: short rule + 2 examples.\n"
            f"- quiz_vocab: if pending quiz word is set, grade the user's answer; else ask one quiz question from the word bank.\n"
            f"End with a line: VOCAB: term (gloss), ... OR VOCAB: none"
        )
        raw = query_llm(llm_gen_prompt, system_prompt=system_prompt)
        if raw:
            extracted: list[str] = []
            vocab_matches = re.findall(
                r"^VOCAB:\s*(.+)$", raw, flags=re.MULTILINE | re.IGNORECASE
            )
            clean = raw
            if vocab_matches:
                v_str = vocab_matches[0].strip()
                if v_str.lower() != "none":
                    extracted = [
                        v.strip()
                        for v in v_str.split(",")
                        if v.strip() and "(" in v
                    ][:3]
                clean = re.sub(
                    r"\n*VOCAB:\s*.+$",
                    "",
                    raw,
                    flags=re.MULTILINE | re.IGNORECASE,
                ).strip()
            next_pending = pending_quiz_word if intent == "quiz_vocab" else None
            return clean, extracted, next_pending

    # -------------------- Offline DemoChatModel --------------------
    new_vocab: list[str] = []
    next_pending: Optional[str] = None

    if intent == "translate":
        clean_phrase = re.sub(
            r"^(?:translate|how do you say|what is|what does)\s*:?\s*",
            "",
            last_text,
            flags=re.IGNORECASE,
        ).strip(' :"\'')
        mapped = None
        for key in ("hello", "thanks", "please", "water", "coffee", "friend", "food", "ticket", "train"):
            if key in last_lower or (key == "thanks" and "thank" in last_lower):
                mapped = book[key]
                break
        if mapped:
            term, gloss = mapped
            reply = (
                f"**{clean_phrase or gloss}** → **{term}** ({gloss}) in {lang}.\n\n"
                f"Try saying it once out loud, then use it in a short sentence."
            )
            new_vocab = [f"{term} ({gloss})"]
        else:
            term, gloss = book["hello"]
            reply = (
                f"For **\"{clean_phrase}\"** in {lang}, a useful practice phrase is "
                f"**{term}** ({gloss}). Ask me for a more specific phrase if you need one."
            )
            new_vocab = [f"{term} ({gloss})"]

    elif intent == "explain_grammar":
        reply = f"Grammar tip for {user_name} ({level} {lang}):\n\n{book['grammar']}"

    elif intent == "quiz_vocab":
        if pending_quiz_word:
            ok, feedback = score_quiz_answer(last_text, pending_quiz_word)
            badge = "✅" if ok else "📘"
            reply = f"{badge} Quiz check\n\n{feedback}"
            next_pending = None
        elif vocab_learned:
            # Prefer a random earlier word so quizzes aren't always the newest entry
            target = random.choice(vocab_learned[-8:])
            term, gloss = parse_vocab_entry(target)
            reply = (
                f"🎯 Quick quiz for {user_name}\n\n"
                f"What does **{term}** mean?\n"
                f"(Or write a short sentence using it in {lang}.)"
            )
            next_pending = target
        else:
            reply = (
                f"Your word bank for this thread is empty, {user_name}. "
                f"Chat or ask for a translation first — I'll collect words, then we can quiz."
            )
            next_pending = None

    else:
        # practice_conversation — use full trimmed thread history (short-term memory)
        reply, new_vocab = offline_practice_reply(messages, profile, book)

    return reply, new_vocab, next_pending


# =====================================================================
# TOPIC 1: LANGGRAPH NODES
# =====================================================================


def router_node(state: ChatState) -> Dict[str, Any]:
    """Classify intent + update long-term profile facts from the latest human message."""
    messages = state.get("messages", [])
    if not messages:
        return {"intent": "practice_conversation"}

    user_id = state.get("user_id", "default_user")
    current_profile = get_user_profile(user_id)
    latest = messages[-1].content if hasattr(messages[-1], "content") else str(messages[-1])

    routing = llm_classify_intent_and_profile(str(latest), current_profile, messages)
    apply_profile_updates(user_id, routing)
    return {"intent": routing.get("intent", "practice_conversation")}


def trim_and_filter_node(state: ChatState) -> Dict[str, Any]:
    """
    TOPIC 3 — runs before every content/model node via the graph edge order.

    Filter rule (concrete): drop AI messages tagged additional_kwargs['scaffolding']=True
    (quiz prompts that already served their purpose) and drop empty messages.

    Trim rule: keep the newest messages within MAX_TOKENS / MAX_MESSAGES_TO_MODEL,
    always retaining the latest human message.
    """
    raw_messages = list(state.get("messages") or [])
    full_count = len(raw_messages)
    full_tokens = estimate_messages_tokens(raw_messages)

    filtered: list[BaseMessage] = []
    dropped = 0
    for msg in raw_messages:
        is_scaffolding = bool(getattr(msg, "additional_kwargs", {}).get("scaffolding", False))
        if is_scaffolding or not str(msg.content).strip():
            dropped += 1
            continue
        filtered.append(msg)

    trimmed: list[BaseMessage] = []
    current_tokens = 0
    for msg in reversed(filtered):
        msg_tokens = estimate_tokens(str(msg.content))
        if not trimmed or (
            current_tokens + msg_tokens <= MAX_TOKENS and len(trimmed) < MAX_MESSAGES_TO_MODEL
        ):
            trimmed.insert(0, msg)
            current_tokens += msg_tokens
        else:
            break

    if not any(isinstance(m, HumanMessage) for m in trimmed) and filtered:
        trimmed = [filtered[-1]]

    return {
        "trimmed_messages": serialize_messages(trimmed),
        "trimmed_count": len(trimmed),
        "full_count": full_count,
        "filtered_dropped": dropped,
        "full_token_estimate": full_tokens,
        "trimmed_token_estimate": estimate_messages_tokens(trimmed),
    }


def route_by_intent(state: ChatState) -> str:
    """TOPIC 1: conditional edge — branch by intent set in the router."""
    return state.get("intent") or "practice_conversation"


def _run_content_node(state: ChatState, intent: str, system_prompt: str) -> Dict[str, Any]:
    user_id = state.get("user_id", "default_user")
    profile = get_user_profile(user_id)
    # TOPIC 3: model sees trimmed history only
    model_messages = hydrate_trimmed(state)
    vocab_list = list(state.get("vocab_learned") or [])
    pending = state.get("pending_quiz_word")

    reply_text, new_vocab, next_pending = call_model(
        messages=model_messages,
        system_prompt=system_prompt,
        intent=intent,
        profile=profile,
        vocab_learned=vocab_list,
        pending_quiz_word=pending,
    )
    fresh = unique_new_vocab(vocab_list, new_vocab)

    kwargs: Dict[str, Any] = {"pending_quiz_word": next_pending}
    if intent == "quiz_vocab" and next_pending:
        # Tag as scaffolding so later turns can filter this quiz prompt out of model context
        ai_message = AIMessage(
            content=reply_text,
            additional_kwargs={"scaffolding": True, "is_quiz": True},
        )
    else:
        ai_message = AIMessage(content=reply_text)

    out: Dict[str, Any] = {"messages": [ai_message], **kwargs}
    if fresh and intent in {"practice_conversation", "translate"}:
        out["vocab_learned"] = fresh
    return out


def practice_conversation_node(state: ChatState) -> Dict[str, Any]:
    profile = get_user_profile(state.get("user_id", "default_user"))
    system_prompt = (
        f"You are LinguaLoop, practicing {profile['target_language']} with "
        f"{profile['name']} at {profile['level']} level."
    )
    return _run_content_node(state, "practice_conversation", system_prompt)


def translate_node(state: ChatState) -> Dict[str, Any]:
    profile = get_user_profile(state.get("user_id", "default_user"))
    system_prompt = f"You are a translation coach for {profile['target_language']}."
    return _run_content_node(state, "translate", system_prompt)


def explain_grammar_node(state: ChatState) -> Dict[str, Any]:
    profile = get_user_profile(state.get("user_id", "default_user"))
    system_prompt = f"You explain {profile['target_language']} grammar clearly and briefly."
    return _run_content_node(state, "explain_grammar", system_prompt)


def quiz_vocab_node(state: ChatState) -> Dict[str, Any]:
    """Reads vocab_learned (operator.add channel) to quiz — or grades a pending answer."""
    profile = get_user_profile(state.get("user_id", "default_user"))
    system_prompt = f"You are a vocabulary coach for {profile['target_language']}."
    return _run_content_node(state, "quiz_vocab", system_prompt)


# =====================================================================
# TOPIC 1: GRAPH BUILD + COMPILE
# =====================================================================
builder = StateGraph(ChatState)
builder.add_node("router", router_node)
builder.add_node("trim_and_filter", trim_and_filter_node)
builder.add_node("practice_conversation_node", practice_conversation_node)
builder.add_node("translate_node", translate_node)
builder.add_node("explain_grammar_node", explain_grammar_node)
builder.add_node("quiz_vocab_node", quiz_vocab_node)

builder.add_edge(START, "router")
builder.add_edge("router", "trim_and_filter")
builder.add_conditional_edges(
    "trim_and_filter",
    route_by_intent,
    {
        "practice_conversation": "practice_conversation_node",
        "translate": "translate_node",
        "explain_grammar": "explain_grammar_node",
        "quiz_vocab": "quiz_vocab_node",
    },
)
builder.add_edge("practice_conversation_node", END)
builder.add_edge("translate_node", END)
builder.add_edge("explain_grammar_node", END)
builder.add_edge("quiz_vocab_node", END)

compiled_graph = builder.compile(checkpointer=checkpointer, store=store)


def recompile_graph() -> None:
    """Used by /api/reset so short-term MemorySaver truly clears."""
    global checkpointer, compiled_graph, ACTIVE_THREADS
    checkpointer = MemorySaver()
    ACTIVE_THREADS = set()
    compiled_graph = builder.compile(checkpointer=checkpointer, store=store)


# =====================================================================
# FASTAPI
# =====================================================================
app = FastAPI(title="LinguaLoop — Language Practice Coach")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    thread_id: str
    user_id: str = "sam_user"
    message: str


class LLMConfigRequest(BaseModel):
    provider: str
    api_key: Optional[str] = None
    model: Optional[str] = None


@app.get("/")
def get_index():
    index_path = os.path.join(os.path.dirname(__file__), "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return HTMLResponse("<h1>LinguaLoop: index.html not found</h1>")


@app.get("/api/graph")
def get_graph():
    """TOPIC 1 UI: Mermaid + ASCII of the compiled StateGraph."""
    try:
        mermaid_chart = compiled_graph.get_graph().draw_mermaid()
    except Exception:  # noqa: BLE001
        mermaid_chart = (
            "graph TD;\nSTART-->router;\nrouter-->trim_and_filter;\n"
            "trim_and_filter-->practice_conversation_node;\n"
            "trim_and_filter-->translate_node;\n"
            "trim_and_filter-->explain_grammar_node;\n"
            "trim_and_filter-->quiz_vocab_node;\n"
            "practice_conversation_node-->END;\ntranslate_node-->END;\n"
            "explain_grammar_node-->END;\nquiz_vocab_node-->END;"
        )
    try:
        ascii_chart = compiled_graph.get_graph().draw_ascii()
    except Exception:  # noqa: BLE001
        ascii_chart = (
            "START -> router -> trim_and_filter -> "
            "{practice | translate | grammar | quiz} -> END"
        )
    return {"mermaid": mermaid_chart, "ascii": ascii_chart}


@app.get("/api/profile/{user_id}")
def get_profile_endpoint(user_id: str):
    return {"user_id": user_id, "profile": get_user_profile(user_id)}


@app.get("/api/threads")
def get_threads_endpoint():
    return {"threads": sorted(ACTIVE_THREADS)}


@app.get("/api/history/{thread_id}")
def get_history_endpoint(thread_id: str):
    config = {"configurable": {"thread_id": thread_id}}
    snapshot = compiled_graph.get_state(config)
    messages: list[Dict[str, Any]] = []
    vocab_learned: list[str] = []
    trim_info: Dict[str, Any] = {}

    if snapshot and snapshot.values:
        vals = snapshot.values
        for m in vals.get("messages", []):
            role = "user" if isinstance(m, HumanMessage) else "assistant"
            messages.append({"role": role, "content": m.content})
        # Deduplicate for display while state still uses operator.add concatenation
        seen = set()
        for v in vals.get("vocab_learned", []) or []:
            key = v.lower().strip()
            if key not in seen:
                seen.add(key)
                vocab_learned.append(v)
        trim_info = {
            "trimmed_count": vals.get("trimmed_count"),
            "full_count": vals.get("full_count"),
            "filtered_dropped": vals.get("filtered_dropped"),
            "full_token_estimate": vals.get("full_token_estimate"),
            "trimmed_token_estimate": vals.get("trimmed_token_estimate"),
            "intent": vals.get("intent"),
            "trimmed_messages": vals.get("trimmed_messages"),
            "pending_quiz_word": vals.get("pending_quiz_word"),
        }

    return {
        "thread_id": thread_id,
        "messages": messages,
        "vocab_learned": vocab_learned,
        "trim_info": trim_info,
    }


@app.post("/api/chat")
def chat_endpoint(req: ChatRequest):
    ACTIVE_THREADS.add(req.thread_id)
    update_user_profile_if_detected(req.user_id, req.message)
    current_profile = get_user_profile(req.user_id)

    result = compiled_graph.invoke(
        {"messages": [HumanMessage(content=req.message)], "user_id": req.user_id},
        config={"configurable": {"thread_id": req.thread_id}},
    )

    latest_reply = ""
    for m in reversed(result.get("messages", [])):
        if isinstance(m, AIMessage):
            latest_reply = m.content
            break

    # Deduped vocab for UI
    seen = set()
    vocab_ui: list[str] = []
    for v in result.get("vocab_learned", []) or []:
        key = v.lower().strip()
        if key not in seen:
            seen.add(key)
            vocab_ui.append(v)

    return {
        "thread_id": req.thread_id,
        "user_id": req.user_id,
        "reply": latest_reply,
        "intent": result.get("intent"),
        "vocab_learned": vocab_ui,
        "full_count": result.get("full_count"),
        "trimmed_count": result.get("trimmed_count"),
        "filtered_dropped": result.get("filtered_dropped"),
        "full_token_estimate": result.get("full_token_estimate"),
        "trimmed_token_estimate": result.get("trimmed_token_estimate"),
        "trimmed_messages": result.get("trimmed_messages"),
        "pending_quiz_word": result.get("pending_quiz_word"),
        "profile": current_profile,
    }


@app.post("/api/reset")
def reset_endpoint(user_id: str = "sam_user"):
    recompile_graph()
    store.put(("profile", user_id), "profile", dict(DEFAULT_PROFILE))
    return {"status": "reset", "profile": DEFAULT_PROFILE}


@app.get("/api/config")
def get_config_endpoint():
    provider = LLM_CONFIG.get("provider", "offline")
    api_key = LLM_CONFIG.get("api_key", "")
    return {
        "provider": provider,
        "model": LLM_CONFIG.get("model", ""),
        "has_api_key": bool(api_key.strip()),
    }


@app.post("/api/config")
def set_config_endpoint(req: LLMConfigRequest):
    if req.provider:
        LLM_CONFIG["provider"] = req.provider.lower()
    if req.api_key is not None:
        LLM_CONFIG["api_key"] = req.api_key.strip()
    provider = LLM_CONFIG.get("provider", "offline").lower()
    if req.model is not None:
        LLM_CONFIG["model"] = req.model.strip() or resolve_llm_model(provider, "")
    else:
        LLM_CONFIG["model"] = resolve_llm_model(provider, LLM_CONFIG.get("model"))
    return {"status": "updated", "config": get_config_endpoint()}


if __name__ == "__main__":
    print("Starting LinguaLoop at http://127.0.0.1:8000 ...")
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
