# LinguaLoop — Language Practice Coach (LangGraph Demo)

A small chat app that helps anyone practice a foreign language — and visibly demonstrates four LangGraph mechanics graders look for:

1. **Graph** — typed `StateGraph`, nodes, conditional edge, Mermaid/ASCII in the UI  
2. **Reducers** — `add_messages` + `operator.add` on `vocab_learned`, checkpointer + thread switcher  
3. **Trimming & filtering** — applied *before* every model call; UI shows before/after counts  
4. **Memory** — short-term (`MemorySaver` / thread) vs long-term (`InMemoryStore` / profile)

**Useful for learners:** scenario starters (café, travel, introductions), translations with nuance, grammar tips, and quizzes that **score** answers from your thread’s word bank. Your name / language / level follow you into every new conversation.

Files:
- [`app.py`](app.py) — FastAPI + LangGraph graph, reducers, trim, memory, offline demo model  
- [`index.html`](index.html) — practice UI with thread switcher, profile, word bank, graph tab  

---

## Quickstart

```bash
pip install -r requirements.txt
python app.py
```

Open **http://127.0.0.1:8000**

No API keys required (offline demo model). Optional: Gemini / OpenAI / Ollama via the **LLM** button or env vars (`GEMINI_API_KEY`, `OPENAI_API_KEY`, `LLM_PROVIDER`, `OLLAMA_BASE_URL`).

> For real persistence, swap `MemorySaver` → `SqliteSaver` / `PostgresSaver` and keep the same `thread_id` config.

---

## Architecture (graded topics)

| Topic | Where | What the UI shows |
|---|---|---|
| **1 Graph** | `StateGraph(ChatState)` · `route_by_intent` conditional edge | **StateGraph** tab (Mermaid + ASCII) |
| **2 Reducers** | `messages` + `vocab_learned` (`operator.add`) · `MemorySaver` | Thread dropdown / **+ New** · Word bank grows per thread |
| **3 Trim/filter** | `trim_and_filter` node; content nodes call `hydrate_trimmed(state)` | Under each reply: `sent to model: N of M msgs (~tok/~tok)` · filter drops quiz scaffolding |
| **4 Memory** | Checkpointer = short-term · `InMemoryStore` profile = long-term | **Long-term profile** survives new threads; word bank resets |

Flow: `START → router → trim_and_filter → {practice | translate | grammar | quiz} → END`

---

## Demo script (reproducible)

Use the sidebar **Grading demo** buttons or type manually:

1. **Thread A — profile**  
   `My name is Sam and I'm learning French at beginner level.`  
   → Profile panel updates; reply greets Sam in French.

2. **Thread A — word bank**  
   `I love food and croissants.` / or tap **Café order**  
   → Word bank grows via `operator.add`.

3. **New thread — long-term recall**  
   Click **+ New**, then: `Hi, let's practice.`  
   → Bot recalls **Sam** + **French** with an empty chat and empty word bank.

4. **Back to Thread A — quiz**  
   Switch to `thread-A` → `Quiz me on our vocabulary!`  
   → Quiz uses that thread’s bank; your next message is graded.

---

## LLM swap point

In `app.py`, `call_model()` is the only place that talks to a model. Graph nodes always pass the **trimmed** history. Set provider via UI or:

```bash
set LLM_PROVIDER=gemini
set GEMINI_API_KEY=your_key
```
