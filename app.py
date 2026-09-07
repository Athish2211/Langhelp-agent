"""
LinguaLoop — Language Learning Practice Partner built on LangGraph.
Demonstrates 4 core LangGraph topics:
  1. Graph (StateGraph, conditional routing, visualization)
  2. Reducers (Default add_messages vs. Non-default operator.add for vocab_learned)
  3. Trimming & Filtering (Scaffolding message filter + token budget trimming)
  4. Short-Term Memory (MemorySaver checkpointer) & Long-Term Memory (InMemoryStore BaseStore)
"""

import os
import re
import json
import random
import operator
import urllib.request
import urllib.error
from typing import Annotated, TypedDict, Optional, List, Dict, Any

from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
)
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore


# =====================================================================
# LLM ENGINE & MULTI-PROVIDER DISPATCH
# Supports: Google Gemini, OpenAI, Ollama (Local), & Offline Fallback
# Zero extra pip packages required (uses standard urllib.request)
# =====================================================================
def detect_initial_provider() -> tuple[str, str, str]:
    """Detects available API key from environment."""
    if os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
        return "gemini", os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or "", "gemini-1.5-flash"
    elif os.environ.get("OPENAI_API_KEY"):
        return "openai", os.environ.get("OPENAI_API_KEY") or "", "gpt-4o-mini"
    return "offline", "", "demo-model"

init_provider, init_key, init_model = detect_initial_provider()

LLM_CONFIG = {
    "provider": os.environ.get("LLM_PROVIDER", init_provider),
    "api_key": init_key,
    "model": os.environ.get("LLM_MODEL", init_model),
    "ollama_base_url": os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
}


def query_llm(prompt: str, system_prompt: str = "", json_mode: bool = False) -> Optional[str]:
    """
    Sends prompt to configured LLM (Gemini / OpenAI / Ollama).
    Returns response text or None if offline/error.
    """
    provider = LLM_CONFIG.get("provider", "offline").lower()
    api_key = LLM_CONFIG.get("api_key", "").strip()

    # 1. Google Gemini Provider
    if provider == "gemini" and api_key:
        model = LLM_CONFIG.get("model", "gemini-1.5-flash")
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
        parts = []
        if system_prompt:
            parts.append({"text": f"System Instructions:\n{system_prompt}\n\n"})
        parts.append({"text": prompt})
        
        gen_config: Dict[str, Any] = {"temperature": 0.2 if json_mode else 0.7}
        if json_mode:
            gen_config["responseMimeType"] = "application/json"

        body = {
            "contents": [{"parts": parts}],
            "generationConfig": gen_config
        }
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
                    parts = candidates[0].get("content", {}).get("parts", [])
                    if parts:
                        return parts[0].get("text", "")
        except Exception as e:
            print(f"[LLM] Gemini request error: {e}")
            return None

    # 2. OpenAI Provider
    elif provider == "openai" and api_key:
        model = LLM_CONFIG.get("model", "gpt-4o-mini")
        url = "https://api.openai.com/v1/chat/completions"
        msgs = []
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
                url,
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
        except Exception as e:
            print(f"[LLM] OpenAI request error: {e}")
            return None

    # 3. Ollama Provider (Local)
    elif provider == "ollama":
        model = LLM_CONFIG.get("model", "llama3")
        base_url = LLM_CONFIG.get("ollama_base_url", "http://localhost:11434")
        url = f"{base_url}/api/chat"
        msgs = []
        if system_prompt:
            msgs.append({"role": "system", "content": system_prompt})
        msgs.append({"role": "user", "content": prompt})

        body = {
            "model": model,
            "messages": msgs,
            "stream": False,
        }
        if json_mode:
            body["format"] = "json"

        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(body).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=15) as response:
                res_data = json.loads(response.read().decode("utf-8"))
                return res_data.get("message", {}).get("content", "")
        except Exception as e:
            print(f"[LLM] Ollama request error: {e}")
            return None

    return None


def offline_heuristic_router(text: str, current_profile: Dict[str, Any]) -> Dict[str, Any]:
    """Fallback router when offline or no API key is provided."""
    text_lower = text.lower()
    
    # 1. Intent classification
    if any(k in text_lower for k in ["quiz", "test me", "flashcard", "vocab quiz", "review words", "test my vocab"]):
        intent = "quiz_vocab"
    elif any(k in text_lower for k in ["translate", "how do you say", "what does", "what is", "meaning of", "in korean", "in japanese", "in french", "in spanish"]):
        intent = "translate"
    elif any(k in text_lower for k in ["why is it", "grammar", "rule", "conjugat", "difference between", "subjunctive", "tense", "particle", "order"]):
        intent = "explain_grammar"
    else:
        intent = "practice_conversation"

    # 2. Entity extraction
    detected_name = None
    name_match = re.search(r"\b(?:my name is|i am|i'm|call me)\s+([A-Z][a-z]+)", text, re.IGNORECASE)
    if name_match:
        cand = name_match.group(1).capitalize()
        if cand.lower() not in {"learning", "practicing", "ready", "here", "trying", "a", "an", "want", "learner"}:
            detected_name = cand

    detected_lang = None
    lang_match = re.search(
        r"\b(?:learning|learn|practice|practicing|speak|speaking|study|studying|in|to|want|teach me|switch to|target language is)\s+"
        r"(?:to\s+)?(?:learn\s+|practice\s+|speak\s+)?"
        r"(Spanish|French|German|Italian|Japanese|Korean|Mandarin|Chinese|Portuguese|Russian|Arabic)\b",
        text,
        re.IGNORECASE,
    )
    if not lang_match:
        lang_match = re.search(r"\b(Spanish|French|German|Italian|Japanese|Korean|Mandarin|Chinese|Portuguese|Russian|Arabic)\b", text, re.IGNORECASE)
    if lang_match:
        detected_lang = lang_match.group(1).capitalize()
        if detected_lang.lower() == "chinese":
            detected_lang = "Mandarin"

    detected_level = None
    level_match = re.search(r"\b(beginner|novice|intermediate|advanced|expert)\b", text, re.IGNORECASE)
    if level_match:
        detected_level = level_match.group(1).lower()

    return {
        "intent": intent,
        "detected_name": detected_name,
        "detected_language": detected_lang,
        "detected_level": detected_level,
    }


def llm_classify_intent_and_profile(text: str, current_profile: Dict[str, Any]) -> Dict[str, Any]:
    """
    Uses LLM reasoning to classify intent into practice_conversation, translate,
    explain_grammar, or quiz_vocab, and dynamically extracts user name, language, and level.
    """
    # If in offline mode, use heuristic router
    if LLM_CONFIG.get("provider", "offline").lower() == "offline":
        return offline_heuristic_router(text, current_profile)

    router_prompt = (
        f"You are the intelligent router and entity extractor for LinguaLoop, an AI language learning partner.\n"
        f"User message: \"{text}\"\n"
        f"Current Profile: {json.dumps(current_profile)}\n\n"
        f"Task 1: Classify the user intent into exactly ONE of:\n"
        f"- 'practice_conversation': General conversation, greetings, chatting about daily life, food, travel, roleplay.\n"
        f"- 'translate': Asking for a phrase or word translation into or from a target language.\n"
        f"- 'explain_grammar': Asking about grammar rules, sentence structure, conjugation, particles, word differences.\n"
        f"- 'quiz_vocab': Asking to test or quiz vocabulary ('quiz me', 'test what we learned', 'flashcards').\n\n"
        f"Task 2: Extract any declared learner facts:\n"
        f"- 'detected_name': name if user introduces themselves (e.g., 'My name is Sam', 'I am Alex')\n"
        f"- 'detected_language': target language user states they want to learn or practice (e.g., Korean, Japanese, French, Spanish, German, etc.)\n"
        f"- 'detected_level': proficiency level if stated (beginner, intermediate, advanced)\n\n"
        f"Output ONLY a JSON object with keys: 'intent', 'detected_name', 'detected_language', 'detected_level'."
    )

    llm_output = query_llm(router_prompt, system_prompt="You are a strict JSON intent classifier.", json_mode=True)
    if llm_output:
        try:
            cleaned = re.sub(r"^```(?:json)?\s*", "", llm_output.strip(), flags=re.IGNORECASE)
            cleaned = re.sub(r"\s*```$", "", cleaned)
            parsed = json.loads(cleaned)
            valid_intents = {"practice_conversation", "translate", "explain_grammar", "quiz_vocab"}
            intent = parsed.get("intent", "practice_conversation")
            if intent not in valid_intents:
                intent = "practice_conversation"
            return {
                "intent": intent,
                "detected_name": parsed.get("detected_name"),
                "detected_language": parsed.get("detected_language"),
                "detected_level": parsed.get("detected_level"),
            }
        except Exception as e:
            print(f"[LLM Router] JSON parse fallback: {e}")

    return offline_heuristic_router(text, current_profile)


# =====================================================================
# TOPIC 4: SHORT-TERM & LONG-TERM MEMORY STORES
# =====================================================================
checkpointer = MemorySaver()
store = InMemoryStore()
ACTIVE_THREADS: set[str] = set()


# =====================================================================
# TOPIC 2: STATE SCHEMA WITH DEFAULT & NON-DEFAULT REDUCERS
# =====================================================================
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    vocab_learned: Annotated[list[str], operator.add]
    intent: Optional[str]
    trimmed_messages: Optional[list[Dict[str, Any]]]
    trimmed_count: Optional[int]
    full_count: Optional[int]
    full_token_estimate: Optional[int]
    trimmed_token_estimate: Optional[int]
    user_id: Optional[str]


# =====================================================================
# TOPIC 4 (cont.): LONG-TERM PROFILE HELPERS
# =====================================================================
DEFAULT_PROFILE = {
    "name": "Learner",
    "target_language": "Spanish",
    "level": "beginner",
}


def get_user_profile(user_id: str) -> Dict[str, Any]:
    """Retrieve long-term profile from LangGraph InMemoryStore."""
    item = store.get(("profile", user_id), "profile")
    if item and isinstance(item.value, dict):
        return {**DEFAULT_PROFILE, **item.value}
    return dict(DEFAULT_PROFILE)


def update_user_profile_if_detected(user_id: str, text: str) -> Dict[str, Any]:
    """
    Extracts profile attributes (name, target language, proficiency level)
    from user messages and writes directly to LangGraph InMemoryStore under ("profile", user_id).
    """
    current_profile = get_user_profile(user_id)
    extraction = llm_classify_intent_and_profile(text, current_profile)
    updated = False

    if extraction.get("detected_name"):
        name_cand = extraction["detected_name"].strip().capitalize()
        if name_cand.lower() not in {"learning", "practicing", "ready", "here", "trying", "a", "an", "learner", "want"}:
            current_profile["name"] = name_cand
            updated = True

    if extraction.get("detected_language"):
        lang_cand = extraction["detected_language"].strip().capitalize()
        if lang_cand.lower() == "chinese":
            lang_cand = "Mandarin"
        current_profile["target_language"] = lang_cand
        updated = True

    if extraction.get("detected_level"):
        lvl = extraction["detected_level"].strip().lower()
        if lvl in {"beginner", "novice", "intermediate", "advanced", "expert"}:
            current_profile["level"] = lvl
            updated = True

    if updated:
        store.put(("profile", user_id), "profile", current_profile)

    return current_profile


# =====================================================================
# TOPIC 3: TOKEN BUDGET & FILTERING HELPERS
# =====================================================================
def estimate_tokens(text: str) -> int:
    """Heuristic token count: ~4 characters per token."""
    return max(1, len(text) // 4)


def estimate_messages_tokens(messages: list[BaseMessage]) -> int:
    return sum(estimate_tokens(str(m.content)) for m in messages)


# =====================================================================
# LLM INTERFACE & OFFLINE DEMO MODEL (SWAPPABLE)
# =====================================================================
def call_model(
    messages: list[BaseMessage],
    system_prompt: str,
    intent: str,
    profile: Dict[str, Any],
    vocab_learned: list[str],
) -> tuple[str, list[str]]:
    """
    SWAP POINT FOR REAL LLM PROVIDERS (OpenAI, Gemini, Ollama, etc.)
    -----------------------------------------------------------------
    To swap in a production model (e.g., ChatOpenAI or ChatGoogleGenerativeAI):
        1. Set your API key in environment variables (e.g. OPENAI_API_KEY).
        2. from langchain_openai import ChatOpenAI
           llm = ChatOpenAI(model="gpt-4o-mini")
           response = llm.invoke([SystemMessage(content=system_prompt)] + messages)
           return response.content, extract_vocab(response.content)

    By default, LinguaLoop uses this high-fidelity, deterministic, offline DemoChatModel.
    It produces realistic language learning dialogue, vocabulary introductions, translations,
    and quizzes without requiring any API keys or network requests.
    """
    user_name = profile.get("name", "Learner")
    lang = profile.get("target_language", "Spanish")
    lang_key = lang.lower()
    level = profile.get("level", "beginner")

    last_user_msg = messages[-1].content if messages else ""
    last_text = last_user_msg.strip()
    last_lower = last_text.lower()

    # -------------------------------------------------------------
    # 1. DYNAMIC LLM GENERATION (When an LLM provider is active)
    # -------------------------------------------------------------
    if LLM_CONFIG.get("provider", "offline").lower() != "offline" and (LLM_CONFIG.get("api_key") or LLM_CONFIG.get("provider") == "ollama"):
        history_str = ""
        for m in messages[-6:]:
            role = "User" if isinstance(m, HumanMessage) else "Assistant"
            history_str += f"{role}: {m.content}\n"

        llm_gen_prompt = (
            f"You are LinguaLoop, an encouraging and expert language learning partner.\n\n"
            f"Learner Context:\n"
            f"- Name: {user_name}\n"
            f"- Target Language: {lang}\n"
            f"- Proficiency Level: {level}\n"
            f"- Current Intent: {intent}\n"
            f"- Previously accumulated vocabulary words in this thread: {', '.join(vocab_learned[-8:]) if vocab_learned else 'None'}\n\n"
            f"Recent Conversation History:\n"
            f"{history_str}\n\n"
            f"Guidelines for this response:\n"
            f"- If intent is 'translate': Provide the translation into or from {lang}, explain nuance and context.\n"
            f"- If intent is 'explain_grammar': Explain the grammar rule clearly with examples in {lang}.\n"
            f"- If intent is 'quiz_vocab': Quiz the learner using one of the learned vocabulary words.\n"
            f"- If intent is 'practice_conversation': Converse naturally in {lang} suitable for a {level} learner, and ask a follow-up question.\n\n"
            f"At the very end of your response, on a final separate line, output any 1 to 2 key new vocabulary words introduced in this format:\n"
            f"VOCAB: term (translation), term2 (translation)\n"
            f"If no new vocabulary was introduced, output: VOCAB: none"
        )

        raw_llm_reply = query_llm(llm_gen_prompt, system_prompt=system_prompt)
        if raw_llm_reply:
            extracted_vocab = []
            vocab_matches = re.findall(r"^VOCAB:\s*(.+)$", raw_llm_reply, flags=re.MULTILINE | re.IGNORECASE)
            clean_reply = raw_llm_reply
            if vocab_matches:
                v_str = vocab_matches[0].strip()
                if v_str.lower() != "none":
                    extracted_vocab = [v.strip() for v in v_str.split(",") if v.strip() and "(" in v][:3]
                clean_reply = re.sub(r"\n*VOCAB:\s*.+$", "", raw_llm_reply, flags=re.MULTILINE | re.IGNORECASE).strip()

            return clean_reply, extracted_vocab

    # -------------------------------------------------------------
    # 2. OFFLINE / DETERMINISTIC FALLBACK MODEL
    # -------------------------------------------------------------
    clean_phrase = re.sub(r"^(?:translate|how do you say|what is|what does)\s*:?\s*", "", last_text, flags=re.IGNORECASE).strip(' :""\'')

    new_vocab: list[str] = []

    # ==========================================
    # 1. TRANSLATE INTENT
    # ==========================================
    if intent == "translate":
        if lang_key == "korean":
            if "hello" in last_lower or "greeting" in last_lower:
                reply = "In Korean, 'hello' is translated as **'안녕하세요' (Annyeonghaseyo)** in standard polite speech, or **'안녕' (Annyeong)** informally with friends. Nuance: Politeness levels reflect social distance and respect in Korean culture."
                new_vocab = ["안녕하세요 (hello)", "안녕 (hi/bye)"]
            elif "thank" in last_lower:
                reply = "In Korean, 'thank you' is **'감사합니다' (Gamsahamnida)** or **'고마워요' (Gomawoyo)**. You can reply with **'천만에요' (Cheonman-eyo - you're welcome)** or **'아니에요' (Anieyo - it's nothing/no problem)**."
                new_vocab = ["감사합니다 (thank you)", "천만에요 (you're welcome)"]
            elif "friend" in last_lower:
                reply = "In Korean, 'friend' is **'친구' (Chingu)**. Nuance: Historically 'chingu' means friends of the same birth year. For older friends, you use terms like 'hyeong/oppa' (older brother) or 'noona/eonni' (older sister)."
                new_vocab = ["친구 (friend)", "우정 (friendship)"]
            else:
                reply = f"Translation to Korean: **'{clean_phrase}'** translates as **'정말 멋져요!' (Jeongmal meotjyeoyo - That's really wonderful!)**. In Korean, verbs and adjectives conjugate at the end of the sentence."
                new_vocab = ["멋져요 (wonderful)"]

        elif lang_key == "japanese":
            if "hello" in last_lower or "greeting" in last_lower:
                reply = "In Japanese, daytime 'hello' is **'こんにちは' (Konnichiwa)**. In the morning use **'おはようございます' (Ohayou gozaimasu)**, and in the evening use **'こんばんは' (Konbanwa)**."
                new_vocab = ["こんにちは (hello)", "おはよう (good morning)"]
            elif "thank" in last_lower:
                reply = "In Japanese, 'thank you' is **'ありがとうございます' (Arigatou gozaimasu)** (polite) or **'ありがとう' (Arigatou)** (casual). Response: **'どういたしまして' (Dou itashimashite - you're welcome)**."
                new_vocab = ["ありがとう (thank you)", "どういたしまして (you're welcome)"]
            elif "friend" in last_lower:
                reply = "In Japanese, 'friend' is **'友達' (Tomodachi)**. Nuance: A very close friend can also be called **'親友' (Shin'yuu)**."
                new_vocab = ["友達 (friend)", "親友 (close friend)"]
            else:
                reply = f"Translation to Japanese: **'{clean_phrase}'** translates as **'素晴らしいですね' (Subarashii desu ne - That's wonderful!)**."
                new_vocab = ["素晴らしい (wonderful)"]

        elif lang_key == "french":
            if "hello" in last_lower or "greeting" in last_lower:
                reply = "In French, 'hello' is translated as **'Bonjour'** (formal/standard) or **'Salut'** (informal with friends). Nuance: Use 'Bonjour' during daylight hours with acquaintances and shopkeepers!"
                new_vocab = ["bonjour (hello)", "salut (hi/bye)"]
            elif "thank" in last_lower:
                reply = "In French, 'thank you' is **'Merci'** or **'Merci beaucoup'** (thank you very much). Nuance: It is polite to add 'monsieur' or 'madame' in formal settings."
                new_vocab = ["merci (thank you)", "beaucoup (very much)"]
            elif "friend" in last_lower:
                reply = "In French, 'friend' is **'ami'** (masculine) or **'amie'** (feminine). Plural: **'amis'** / **'amies'**."
                new_vocab = ["l'ami (the friend)", "l'amitié (friendship)"]
            else:
                reply = f"Translation to French: **'{clean_phrase}'** translates as **'C'est magnifique'** (That's wonderful / magnificent). In French, adjectives often agree in gender and number with the noun they modify."
                new_vocab = ["magnifique (wonderful)"]

        elif lang_key == "german":
            if "hello" in last_lower or "greeting" in last_lower:
                reply = "In German, 'hello' is **'Hallo'** (casual/friendly) or **'Guten Tag'** (good day / standard). In the morning, use **'Guten Morgen'**."
                new_vocab = ["hallo (hello)", "guten Tag (good day)"]
            elif "thank" in last_lower:
                reply = "In German, 'thank you' is **'Danke'** or **'Danke schön'**. You can politely reply with **'Bitte schön'** (you're welcome)."
                new_vocab = ["danke (thank you)", "bitte (please / you're welcome)"]
            elif "friend" in last_lower:
                reply = "In German, 'friend' is **'der Freund'** (male friend) or **'die Freundin'** (female friend)."
                new_vocab = ["der Freund (friend)", "die Freundschaft (friendship)"]
            else:
                reply = f"Translation to German: **'{clean_phrase}'** translates as **'Das ist wunderbar!'** (That is wonderful!)."
                new_vocab = ["wunderbar (wonderful)"]

        elif lang_key == "spanish":
            if "hello" in last_lower or "greeting" in last_lower:
                reply = "In Spanish, 'hello' is translated as **'Hola'**. For time-specific greetings: **'Buenos días'** (good morning) or **'Buenas tardes'** (good afternoon)."
                new_vocab = ["hola (hello)", "buenos días (good morning)"]
            elif "thank" in last_lower:
                reply = "In Spanish, 'thank you' is **'Gracias'**, and 'thank you very much' is **'Muchas gracias'**. Nuance: You can respond with 'De nada' (you're welcome)."
                new_vocab = ["gracias (thank you)", "de nada (you're welcome)"]
            elif "friend" in last_lower:
                reply = "In Spanish, 'friend' is **'amigo'** (masculine) or **'amiga'** (feminine). Diminutive: 'amiguito/a'."
                new_vocab = ["el amigo (the friend)", "la amistad (friendship)"]
            else:
                reply = f"Translation to Spanish: **'{clean_phrase}'** translates to Spanish as **'¡Qué maravilloso!'** (How wonderful!). Remember to place inverted exclamation marks at the beginning."
                new_vocab = ["maravilloso (wonderful)"]

        else:
            # Clean generic translation fallback for any other language
            reply = f"Translation to {lang}: **'{clean_phrase}'** translates to an expression of greeting and goodwill in {lang}. Pay attention to polite registers when speaking {lang}!"
            new_vocab = [f"phrase in {lang}"]

    # ==========================================
    # 2. EXPLAIN GRAMMAR INTENT
    # ==========================================
    elif intent == "explain_grammar":
        if lang_key == "korean":
            reply = (
                f"Grammar tip for {user_name} ({level} Korean):\n\n"
                f"Unlike English Subject-Verb-Object (SVO) order, Korean strictly uses **SOV (Subject - Object - Verb)**:\n"
                f"• English: 'I (S) drink (V) water (O)'\n"
                f"• Korean: '저는 (I) 물을 (water) 마셔요 (drink)'\n\n"
                f"Key grammatical particles attached to nouns:\n"
                f"• **-은 / -는**: Topic marker (marks what the sentence is about)\n"
                f"• **-이 / -가**: Subject marker (identifies who performs the action)\n"
                f"• **-을 / -를**: Object marker (marks the direct receiver of action)\n"
                f"The verb ALWAYS comes at the very end of the sentence in Korean!"
            )
        elif lang_key == "japanese":
            reply = (
                f"Grammar tip for {user_name} ({level} Japanese):\n\n"
                f"Japanese sentences follow **SOV (Subject - Object - Verb)** structure and rely on post-position particles:\n"
                f"• **は (wa)**: Marks the topic of conversation\n"
                f"• **を (o)**: Marks the direct object\n"
                f"• **が (ga)**: Marks the specific subject\n\n"
                f"Example: 私は本を読みます (Watashi wa hon o yomimasu) = 'I read a book'. The verb always ends the clause!"
            )
        elif lang_key == "french":
            reply = (
                f"Grammar tip for {user_name} ({level} French):\n\n"
                f"In French, verbs conjugate according to the subject pronoun. For regular '-er' verbs like *parler* (to speak):\n"
                f"- Je parle (I speak)\n"
                f"- Tu parles (You speak - informal)\n"
                f"- Nous parlons (We speak)\n"
                f"Notice that 'parle' and 'parles' sound identical in spoken French even though their spellings differ!"
            )
        elif lang_key == "german":
            reply = (
                f"Grammar tip for {user_name} ({level} German):\n\n"
                f"Three essential German grammar rules:\n"
                f"1. **Noun Genders**: All German nouns are capitalized and have gender: **der** (masculine), **die** (feminine), or **das** (neuter).\n"
                f"2. **Verb Second (V2)**: In main declarative clauses, the finite verb must occupy the second syntactic position!\n"
                f"3. **Cases**: Articles change depending on Nominative, Accusative, Dative, and Genitive cases."
            )
        elif lang_key == "spanish":
            reply = (
                f"Grammar tip for {user_name} ({level} Spanish):\n\n"
                f"A key concept in Spanish is the difference between **Ser** and **Estar** (both mean 'to be'):\n"
                f"• **Ser**: Used for permanent traits, identity, nationality, time (e.g., *Soy {user_name}*, *Es la una*).\n"
                f"• **Estar**: Used for temporary states, emotions, locations (e.g., *Estoy feliz*, *Estoy en casa*)."
            )
        else:
            reply = (
                f"Grammar tip for {user_name} ({level} {lang}):\n\n"
                f"When studying {lang} at a {level} level, focus on core sentence structure, basic verb conjugation, and noun agreement. Daily immersion and formulating simple sentences will accelerate your mastery!"
            )

    # ==========================================
    # 3. QUIZ VOCAB INTENT
    # ==========================================
    elif intent == "quiz_vocab":
        if vocab_learned:
            target_word = vocab_learned[-1]
            reply = (
                f"🎯 **Vocab Quiz Time, {user_name}!**\n\n"
                f"Earlier in our conversation, we accumulated this word into your vocabulary list:\n"
                f"👉 **\"{target_word}\"**\n\n"
                f"Can you tell me what it means or use it in a short sentence in {lang}?"
            )
        else:
            reply = (
                f"We don't have any accumulated vocabulary words in this thread yet, {user_name}! "
                f"Let's practice a few sentences or translate some phrases in {lang} first, and I will record the new words for you."
            )

    # ==========================================
    # 4. PRACTICE CONVERSATION INTENT
    # ==========================================
    else:
        # Check if user stated their name, language, or started a greeting
        is_intro = any(
            k in last_lower
            for k in [
                "name is", "i am", "i'm", "call me",
                "want to learn", "learning", "learn", "study", "practice",
                "hello", "hi", "hey", "hola", "bonjour", "annyeong", "konnichiwa"
            ]
        )

        if is_intro:
            if lang_key == "korean":
                reply = f"안녕하세요, {user_name}! 만나서 반가워요 (Nice to meet you). Welcome to our Korean practice ({level}). 오늘 어떤 것을 연습하고 싶으세요? (What would you like to practice today?)"
                new_vocab = ["안녕하세요 (hello)", "만나서 반가워요 (nice to meet you)"]
            elif lang_key == "japanese":
                reply = f"はじめまして、{user_name}さん！(Nice to meet you!) 日本語の練習へようこそ ({level})。何について話しましょうか？ (What shall we talk about today?)"
                new_vocab = ["はじめまして (nice to meet you)", "ようこそ (welcome)"]
            elif lang_key == "french":
                reply = f"Enchanté, {user_name}! C'est un plaisir de faire votre connaissance. Bienvenue dans notre cours de français ({level}). Prêt à commencer notre pratique?"
                new_vocab = ["enchanté (nice to meet you)", "bienvenue (welcome)"]
            elif lang_key == "german":
                reply = f"Guten Tag, {user_name}! Freut mich, dich kennenzulernen. Willkommen zu unserer Deutsch-Übung ({level}). Was möchtest du heute üben?"
                new_vocab = ["guten Tag (hello)", "willkommen (welcome)"]
            elif lang_key == "spanish":
                reply = f"¡Mucho gusto, {user_name}! Qué alegría conocerte. Bienvenido a nuestra práctica de español ({level}). ¿Qué te gustaría practicar hoy?"
                new_vocab = ["mucho gusto (pleased to meet you)", "bienvenido (welcome)"]
            else:
                reply = f"Welcome {user_name}! It is wonderful to meet you. Ready to practice {lang} at your {level} level? What would you like to practice today?"
                new_vocab = [f"welcome ({lang})"]

        elif "food" in last_lower or "eat" in last_lower or any(f in last_lower for f in ["croissant", "taco", "kimchi", "bibimbap", "sushi", "ramen", "pizza"]):
            if lang_key == "korean":
                reply = f"한국 음식은 정말 맛있어요! 비빔밥, 김치, 불고기 좋아하시나요? {user_name}, 가장 좋아하는 한국 음식이 뭐예요? (Korean food is delicious! What is your favorite food?)"
                new_vocab = ["맛있어요 (delicious)", "음식 (food)"]
            elif lang_key == "japanese":
                reply = f"日本食は本当に美味しいですね！ラーメンやすしは好きですか？ {user_name}さんの好きな食べ物は何ですか？ (Japanese food is delicious! What is your favorite?)"
                new_vocab = ["美味しい (delicious)", "食べ物 (food)"]
            elif lang_key == "french":
                reply = f"Ah, la gastronomie française! J'adore les croissants chauds et le fromage. {user_name}, quel est votre plat préféré?"
                new_vocab = ["le fromage (cheese)", "le petit déjeuner (breakfast)"]
            elif lang_key == "german":
                reply = f"Deutsches Essen ist wunderbar! Magst du Brezeln, Schnitzel oder frisches Brot? Was ist dein Lieblingsessen, {user_name}?"
                new_vocab = ["das Essen (food)", "lecker (delicious)"]
            elif lang_key == "spanish":
                reply = f"¡Excelente tema, {user_name}! La comida es fantástica. A mí me encantan los tacos y las tapas. ¿Cuál es tu comida favorita?"
                new_vocab = ["la comida (food)", "las tapas (appetizers/snacks)"]
            else:
                reply = f"Food is a great topic to practice in {lang}! {user_name}, what dishes do you enjoy eating when practicing {lang}?"
                new_vocab = [f"delicious ({lang})"]

        elif "travel" in last_lower or "city" in last_lower or any(c in last_lower for c in ["seoul", "tokyo", "paris", "berlin", "madrid", "rome"]):
            if lang_key == "korean":
                reply = f"한국 여행은 정말 멋져요! 서울, 부산, 제주도 중 어디에 가보고 싶으신가요, {user_name}? (Traveling to Korea is wonderful! Which city would you like to visit?)"
                new_vocab = ["여행 (travel/trip)", "도시 (city)"]
            elif lang_key == "japanese":
                reply = f"日本への旅行は素晴らしいですね！東京や京都、大阪に行ってみたいですか？ (Traveling to Japan is wonderful! Would you like to visit Tokyo or Kyoto?)"
                new_vocab = ["旅行 (travel/trip)", "街 (city/town)"]
            elif lang_key == "french":
                reply = f"J'adore voyager! Aimeriez-vous visiter Paris, Lyon ou la Côte d'Azur, {user_name}?"
                new_vocab = ["le voyage (trip)", "la ville (city)"]
            elif lang_key == "german":
                reply = f"Reisen in Deutschland ist toll! Möchtest du Berlin, München oder Hamburg besuchen, {user_name}?"
                new_vocab = ["die Reise (trip)", "die Stadt (city)"]
            elif lang_key == "spanish":
                reply = f"¡Me encanta viajar! ¿Qué ciudades de España o Latinoamérica te gustaría visitar, {user_name}?"
                new_vocab = ["el viaje (trip/journey)", "la ciudad (city)"]
            else:
                reply = f"Traveling is an inspiring way to practice {lang}! What cities or destinations in {lang}-speaking regions would you love to explore, {user_name}?"
                new_vocab = [f"travel ({lang})"]

        else:
            # Default conversation turn
            if lang_key == "korean":
                reply = f"안녕하세요 {user_name}! Ready to practice some Korean at your {level} level? 오늘 하루 어떠셨어요? (How was your day today?)"
                new_vocab = ["오늘 (today)", "행운을 빌어요 (good luck)"]
            elif lang_key == "japanese":
                reply = f"こんにちは、{user_name}さん！Ready to practice some Japanese at your {level} level? 今日は何をしましたか？ (What did you do today?)"
                new_vocab = ["今日 (today)", "頑張って (do your best)"]
            elif lang_key == "french":
                reply = f"Bonjour {user_name}! Ready to practice some French at your {level} level? Comment se passe votre journée aujourd'hui? (How is your day going today?)"
                new_vocab = ["la journée (the day)", "aujourd'hui (today)"]
            elif lang_key == "german":
                reply = f"Hallo {user_name}! Ready to practice some German at your {level} level? Wie geht es dir heute? (How are you today?)"
                new_vocab = ["wie geht's? (how are you?)", "viel Glück (good luck)"]
            elif lang_key == "spanish":
                reply = f"¡Hola {user_name}! Ready to practice some Spanish at your {level} level? ¿Cómo estás hoy? (How are you today?)"
                new_vocab = ["¿cómo estás? (how are you?)", "buena suerte (good luck)"]
            else:
                reply = f"Hello {user_name}! Ready to practice some {lang} at your {level} level? What would you like to talk about today?"
                new_vocab = [f"practice ({lang})"]

    return reply, new_vocab


# =====================================================================
# TOPIC 1: LANGGRAPH NODES
# =====================================================================

def router_node(state: ChatState) -> Dict[str, Any]:
    """
    DYNAMIC LLM ROUTER NODE:
    Uses LLM classification (with heuristic fallback) to determine intent:
      - practice_conversation
      - translate
      - explain_grammar
      - quiz_vocab
    Also dynamically extracts and updates profile attributes (name, language, level).
    """
    messages = state.get("messages", [])
    if not messages:
        return {"intent": "practice_conversation"}

    user_id = state.get("user_id", "default_user")
    current_profile = get_user_profile(user_id)
    latest_msg = messages[-1].content if hasattr(messages[-1], "content") else str(messages[-1])

    # Dynamic LLM routing & entity classification
    routing_result = llm_classify_intent_and_profile(str(latest_msg), current_profile)

    # Persist any newly detected profile fields into long-term store
    updated = False
    if routing_result.get("detected_name"):
        cand_name = routing_result["detected_name"].strip().capitalize()
        if cand_name.lower() not in {"learning", "practicing", "ready", "trying", "a", "an", "learner", "here", "want"}:
            current_profile["name"] = cand_name
            updated = True

    if routing_result.get("detected_language"):
        cand_lang = routing_result["detected_language"].strip().capitalize()
        if cand_lang.lower() == "chinese":
            cand_lang = "Mandarin"
        current_profile["target_language"] = cand_lang
        updated = True

    if routing_result.get("detected_level"):
        lvl = routing_result["detected_level"].strip().lower()
        if lvl in {"beginner", "novice", "intermediate", "advanced", "expert"}:
            current_profile["level"] = lvl
            updated = True

    if updated:
        store.put(("profile", user_id), "profile", current_profile)

    return {"intent": routing_result.get("intent", "practice_conversation")}


def trim_and_filter_node(state: ChatState) -> Dict[str, Any]:
    """
    TOPIC 3: Trimming & Filtering
    1. FILTER: Drops messages that are internal bookkeeping/scaffolding or empty.
    2. TRIM: Enforces a configurable MAX_TOKENS / message budget, keeping the
       most recent context while ensuring the latest human message is preserved.
    3. Records full_count, trimmed_count, and token estimates for visible UI readout.
    """
    raw_messages = state.get("messages", [])
    full_count = len(raw_messages)
    full_tokens = estimate_messages_tokens(raw_messages)

    # Concrete Filter Rule: Filter out bookkeeping messages flagged as scaffolding
    filtered_messages: list[BaseMessage] = []
    for msg in raw_messages:
        # Check if tagged as scaffolding or empty bookkeeping
        is_scaffolding = getattr(msg, "additional_kwargs", {}).get("scaffolding", False)
        if not is_scaffolding and str(msg.content).strip():
            filtered_messages.append(msg)

    # Token Budget Trim: Keep most recent messages up to MAX_TOKENS (e.g. ~400 tokens / 4 messages)
    MAX_TOKENS = 400
    trimmed: list[BaseMessage] = []
    current_tokens = 0

    # Walk backwards from most recent to oldest
    for msg in reversed(filtered_messages):
        msg_tokens = estimate_tokens(str(msg.content))
        if len(trimmed) == 0 or (current_tokens + msg_tokens <= MAX_TOKENS and len(trimmed) < 4):
            trimmed.insert(0, msg)
            current_tokens += msg_tokens
        else:
            break

    # Guarantee at least the latest human message is preserved
    if not any(isinstance(m, HumanMessage) for m in trimmed) and filtered_messages:
        trimmed.append(filtered_messages[-1])

    trimmed_count = len(trimmed)
    trimmed_tokens = estimate_messages_tokens(trimmed)

    # Serialize trimmed messages for UI inspectability
    serialized_trimmed = [
        {"role": "user" if isinstance(m, HumanMessage) else "assistant", "content": m.content}
        for m in trimmed
    ]

    return {
        "trimmed_messages": serialized_trimmed,
        "trimmed_count": trimmed_count,
        "full_count": full_count,
        "full_token_estimate": full_tokens,
        "trimmed_token_estimate": trimmed_tokens,
    }


def route_by_intent(state: ChatState) -> str:
    """TOPIC 1: Conditional edge function reading state['intent']."""
    return state.get("intent") or "practice_conversation"


def practice_conversation_node(state: ChatState) -> Dict[str, Any]:
    """Generates conversational practice and appends introduced vocab."""
    user_id = state.get("user_id", "default_user")
    profile = get_user_profile(user_id)
    messages = state.get("messages", [])
    vocab_list = state.get("vocab_learned", [])

    system_prompt = (
        f"You are LinguaLoop, a supportive conversational partner practicing {profile['target_language']} "
        f"with {profile['name']} at a {profile['level']} level."
    )

    reply_text, new_vocab = call_model(
        messages=messages,
        system_prompt=system_prompt,
        intent="practice_conversation",
        profile=profile,
        vocab_learned=vocab_list,
    )

    ai_message = AIMessage(content=reply_text)
    # Return both new message and newly learned vocabulary (concatenated by operator.add)
    return {
        "messages": [ai_message],
        "vocab_learned": new_vocab,
    }


def translate_node(state: ChatState) -> Dict[str, Any]:
    """Translates phrase, explains nuance, and appends key terms to vocab_learned."""
    user_id = state.get("user_id", "default_user")
    profile = get_user_profile(user_id)
    messages = state.get("messages", [])
    vocab_list = state.get("vocab_learned", [])

    system_prompt = f"You are a translation assistant for {profile['target_language']}."
    reply_text, new_vocab = call_model(
        messages=messages,
        system_prompt=system_prompt,
        intent="translate",
        profile=profile,
        vocab_learned=vocab_list,
    )

    ai_message = AIMessage(content=reply_text)
    return {
        "messages": [ai_message],
        "vocab_learned": new_vocab,
    }


def explain_grammar_node(state: ChatState) -> Dict[str, Any]:
    """Explains grammar concepts concisely."""
    user_id = state.get("user_id", "default_user")
    profile = get_user_profile(user_id)
    messages = state.get("messages", [])
    vocab_list = state.get("vocab_learned", [])

    system_prompt = f"You are a language teacher explaining {profile['target_language']} grammar."
    reply_text, _ = call_model(
        messages=messages,
        system_prompt=system_prompt,
        intent="explain_grammar",
        profile=profile,
        vocab_learned=vocab_list,
    )

    ai_message = AIMessage(content=reply_text)
    return {
        "messages": [ai_message],
    }


def quiz_vocab_node(state: ChatState) -> Dict[str, Any]:
    """
    Pulls from accumulated state['vocab_learned'] reducer channel to formulate quiz questions.
    """
    user_id = state.get("user_id", "default_user")
    profile = get_user_profile(user_id)
    messages = state.get("messages", [])
    vocab_list = state.get("vocab_learned", [])

    system_prompt = f"You are a vocabulary coach for {profile['target_language']}."
    reply_text, _ = call_model(
        messages=messages,
        system_prompt=system_prompt,
        intent="quiz_vocab",
        profile=profile,
        vocab_learned=vocab_list,
    )

    # Example of tagging a bookkeeping message with scaffolding metadata
    ai_message = AIMessage(
        content=reply_text,
        additional_kwargs={"is_quiz": True}
    )
    return {
        "messages": [ai_message],
    }


# =====================================================================
# TOPIC 1 (cont.): GRAPH CONSTRUCTION & COMPILATION
# =====================================================================
builder = StateGraph(ChatState)

# 1. Add nodes
builder.add_node("router", router_node)
builder.add_node("trim_and_filter", trim_and_filter_node)
builder.add_node("practice_conversation_node", practice_conversation_node)
builder.add_node("translate_node", translate_node)
builder.add_node("explain_grammar_node", explain_grammar_node)
builder.add_node("quiz_vocab_node", quiz_vocab_node)

# 2. Add edges
builder.add_edge(START, "router")
builder.add_edge("router", "trim_and_filter")

# Conditional edge based on detected intent
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

# Compile with both checkpointer (short-term) and store (long-term)
compiled_graph = builder.compile(checkpointer=checkpointer, store=store)


# =====================================================================
# FASTAPI APPLICATION & REST ENDPOINTS
# =====================================================================
app = FastAPI(title="LinguaLoop — LangGraph Language Learning Demo")

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


@app.get("/")
def get_index():
    """Serves the single-page demo UI."""
    index_path = os.path.join(os.path.dirname(__file__), "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return HTMLResponse("<h1>LinguaLoop: index.html not found</h1>")


@app.get("/api/graph")
def get_graph():
    """
    TOPIC 1 UI Requirement:
    Returns the compiled StateGraph structure in both Mermaid and ASCII representations.
    """
    try:
        mermaid_chart = compiled_graph.get_graph().draw_mermaid()
    except Exception as e:
        mermaid_chart = f"graph TD;\nSTART-->router;\nrouter-->trim_and_filter;\ntrim_and_filter-->practice_conversation_node;\ntrim_and_filter-->translate_node;\ntrim_and_filter-->explain_grammar_node;\ntrim_and_filter-->quiz_vocab_node;\npractice_conversation_node-->END;\ntranslate_node-->END;\nexplain_grammar_node-->END;\nquiz_vocab_node-->END;"

    try:
        ascii_chart = compiled_graph.get_graph().draw_ascii()
    except Exception:
        ascii_chart = (
            "                    +-----------+\n"
            "                    | __start__ |\n"
            "                    +-----------+\n"
            "                          |\n"
            "                          v\n"
            "                    +-----------+\n"
            "                    |  router   |\n"
            "                    +-----------+\n"
            "                          |\n"
            "                          v\n"
            "               +---------------------+\n"
            "               |   trim_and_filter   |\n"
            "               +---------------------+\n"
            "                /      |      |      \\\n"
            "    practice_conv  translate grammar quiz_vocab\n"
            "             /         |      |         \\\n"
            "            v          v      v          v\n"
            "     +------------+ +-----+ +-------+ +------------+\n"
            "     |  practice  | |trans| |explain| | quiz_vocab |\n"
            "     +------------+ +-----+ +-------+ +------------+\n"
            "            \\          |      |         /\n"
            "             +-------->+<-----+<-------+\n"
            "                       |\n"
            "                       v\n"
            "                   +---------+\n"
            "                   | __end__ |\n"
            "                   +---------+"
        )

    return {
        "mermaid": mermaid_chart,
        "ascii": ascii_chart,
    }


@app.get("/api/profile/{user_id}")
def get_profile_endpoint(user_id: str):
    """TOPIC 4: Returns the user's cross-thread long-term profile from BaseStore."""
    profile = get_user_profile(user_id)
    return {"user_id": user_id, "profile": profile}


@app.get("/api/threads")
def get_threads_endpoint():
    """Returns all active short-term memory thread IDs."""
    return {"threads": sorted(list(ACTIVE_THREADS))}


@app.get("/api/history/{thread_id}")
def get_history_endpoint(thread_id: str):
    """
    TOPIC 2 & 4: Returns short-term state for a given thread_id.
    Demonstrates state isolation across different threads.
    """
    config = {"configurable": {"thread_id": thread_id}}
    state_snapshot = compiled_graph.get_state(config)

    messages = []
    vocab_learned = []
    trim_info = {}

    if state_snapshot and state_snapshot.values:
        vals = state_snapshot.values
        raw_msgs = vals.get("messages", [])
        for m in raw_msgs:
            role = "user" if isinstance(m, HumanMessage) else "assistant"
            messages.append({"role": role, "content": m.content})
        vocab_learned = vals.get("vocab_learned", [])
        trim_info = {
            "trimmed_count": vals.get("trimmed_count"),
            "full_count": vals.get("full_count"),
            "full_token_estimate": vals.get("full_token_estimate"),
            "trimmed_token_estimate": vals.get("trimmed_token_estimate"),
            "intent": vals.get("intent"),
        }

    return {
        "thread_id": thread_id,
        "messages": messages,
        "vocab_learned": vocab_learned,
        "trim_info": trim_info,
    }


@app.post("/api/chat")
def chat_endpoint(req: ChatRequest):
    """
    Executes one turn of the compiled LangGraph StateGraph.
    1. Updates long-term profile if user mentioned name/language/level.
    2. Invokes graph with short-term checkpointer config {"configurable": {"thread_id": req.thread_id}}.
    3. Returns assistant response, updated vocab_learned list, trim stats, and profile.
    """
    ACTIVE_THREADS.add(req.thread_id)

    # TOPIC 4: Extract and store long-term profile facts across threads
    update_user_profile_if_detected(req.user_id, req.message)
    current_profile = get_user_profile(req.user_id)

    # Prepare input for StateGraph
    user_msg = HumanMessage(content=req.message)
    input_state = {
        "messages": [user_msg],
        "user_id": req.user_id,
    }
    config = {"configurable": {"thread_id": req.thread_id}}

    # Invoke graph execution
    result = compiled_graph.invoke(input_state, config=config)

    # Extract latest assistant message
    messages = result.get("messages", [])
    latest_reply = ""
    for m in reversed(messages):
        if isinstance(m, AIMessage):
            latest_reply = m.content
            break

    vocab_list = result.get("vocab_learned", [])

    return {
        "thread_id": req.thread_id,
        "user_id": req.user_id,
        "reply": latest_reply,
        "intent": result.get("intent"),
        "vocab_learned": vocab_list,
        "full_count": result.get("full_count"),
        "trimmed_count": result.get("trimmed_count"),
        "full_token_estimate": result.get("full_token_estimate"),
        "trimmed_token_estimate": result.get("trimmed_token_estimate"),
        "profile": current_profile,
    }


@app.post("/api/reset")
def reset_endpoint(user_id: str = "sam_user"):
    """Reset store and threads for clean demo run."""
    global ACTIVE_THREADS
    ACTIVE_THREADS.clear()
    store.put(("profile", user_id), "profile", dict(DEFAULT_PROFILE))
    return {"status": "reset", "profile": DEFAULT_PROFILE}


# =====================================================================
# LLM PROVIDER CONFIGURATION ENDPOINTS
# =====================================================================
class LLMConfigRequest(BaseModel):
    provider: str
    api_key: Optional[str] = None
    model: Optional[str] = None


@app.get("/api/config")
def get_config_endpoint():
    """Returns current active LLM configuration."""
    provider = LLM_CONFIG.get("provider", "offline")
    api_key = LLM_CONFIG.get("api_key", "")
    return {
        "provider": provider,
        "model": LLM_CONFIG.get("model", ""),
        "has_api_key": bool(api_key.strip()),
    }


@app.post("/api/config")
def set_config_endpoint(req: LLMConfigRequest):
    """Updates active LLM provider, API key, and model in real-time."""
    if req.provider:
        LLM_CONFIG["provider"] = req.provider.lower()
    if req.api_key is not None:
        LLM_CONFIG["api_key"] = req.api_key.strip()
    if req.model:
        LLM_CONFIG["model"] = req.model.strip()
    return {"status": "updated", "config": get_config_endpoint()}


if __name__ == "__main__":
    print("Starting LinguaLoop server at http://127.0.0.1:8000 ...")
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
