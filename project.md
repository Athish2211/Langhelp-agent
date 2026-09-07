# Project: LanguageHelp — Language Learning Practice Partner

A small chat application, built on LangGraph, that helps a user practice a foreign language. It exists primarily as a teaching demo for four LangGraph mechanics: **graphs**, **reducer-based state**, **trimming/filtering**, and **short + long-term memory**. Every one of those four mechanics must be *visibly demonstrated in the UI*, not just implemented invisibly in the backend.

---

## 1. Domain & Product Concept

**LinguaLoop** is a chat partner for practicing a target language (e.g. Spanish, French, Japanese — pick one as the default, but the profile should store whichever the user states).

The bot supports four kinds of interactions, routed by detected intent:

1. **Practice conversation** — free-form back-and-forth in the target language at the user's level.
2. **Translate** — user gives a phrase (in either language), bot translates it and briefly explains any nuance.
3. **Explain grammar** — user asks "why is it X and not Y", bot gives a short grammar explanation.
4. **Quiz vocab** — bot quizzes the user on vocabulary that has come up earlier in the conversation (this is what justifies the extra reducer channel — see below).

Every time a new vocabulary word or phrase is introduced (by the bot, in a `practice_conversation` or `translate` turn), it gets appended to a running `vocab_learned` list in the graph state. The `quiz_vocab` node pulls from this accumulated list to generate quiz questions — so the app has an organic, non-contrived reason for a second reducer channel beyond `add_messages`.

---

## 2. Tech Stack

- **Backend:** Python, LangGraph (`langgraph`), LangChain core message types (`HumanMessage`, `AIMessage`, `SystemMessage`)
- **LLM:** Use LangGraph's/LangChain's offline `DemoChatModel` (or an equivalent trivial rule-based/template model) as the default so the project runs with zero API keys and zero cost. Structure the LLM call behind a single `call_model()` function/interface so a real provider (OpenAI, Gemini, local Ollama model) can be swapped in via one config value or env var without touching graph logic. Document this swap point clearly in code comments and in the README.
- **Checkpointer:** `langgraph.checkpoint.memory.MemorySaver` (in-memory is fine for a demo; note in README that `SqliteSaver`/`PostgresSaver` would be used for real persistence)
- **Long-term store:** `langgraph.store.memory.InMemoryStore` (LangGraph's `BaseStore` interface), namespaced by user, NOT by thread
- **Backend server:** FastAPI (simple REST + one endpoint that returns graph visualization data)
- **Frontend:** A single-page app — plain HTML/CSS/JS is sufficient (no build step required), OR a small React app if preferred. Keep it lightweight; this is a demo UI, not a production product.
- **Graph visualization:** Use LangGraph's built-in `graph.get_graph().draw_mermaid()` (or `draw_ascii()`) to generate a Mermaid diagram string server-side, and render it in the frontend (e.g. via the `mermaid.js` CDN script) or simply display the ASCII art in a `<pre>` block if Mermaid rendering is too heavy. Either is acceptable — ASCII in a `<pre>` block is the fastest path to a working demo.

---

## 3. LangGraph Design (Topic 1 — Graph)

### 3.1 State Schema

```python
from typing import Annotated, TypedDict, Optional
from langgraph.graph.message import add_messages
import operator

class ChatState(TypedDict):
    messages: Annotated[list, add_messages]        # default reducer: full chat history
    vocab_learned: Annotated[list[str], operator.add]  # NON-default reducer: accumulates vocab across turns
    intent: Optional[str]                            # set by router node, read by downstream nodes
    trimmed_messages: Optional[list]                  # what was actually sent to the model this turn (for UI display)
    trimmed_count: Optional[int]                      # len(trimmed_messages) — for UI display
    full_count: Optional[int]                         # len(messages) before trimming — for UI display
```

Note: `intent`, `trimmed_messages`, `trimmed_count`, `full_count` use the **default "last write wins" reducer** (no `Annotated` needed) since only one node writes them per turn — this is a deliberate contrast to `messages` and `vocab_learned`, and should be called out in code comments/README so the "at least one non-default reducer" requirement is unambiguous.

### 3.2 Nodes (minimum two, this project uses five)

| Node | Responsibility |
|---|---|
| `router` | Classifies the latest human message into one of: `practice_conversation`, `translate`, `explain_grammar`, `quiz_vocab`. Sets `state["intent"]`. Does **not** call the LLM for the main reply — a cheap keyword/rule-based classifier is fine and keeps the demo deterministic and free to run (document this explicitly — it's fine for a classifier to be simple; grading is on graph mechanics, not classifier sophistication). |
| `trim_and_filter` | Runs before any node that calls the model. Applies the token-budget trim and message filtering (see Topic 3 below). Writes `trimmed_messages`, `trimmed_count`, `full_count` into state. |
| `practice_conversation_node` | Calls the model with trimmed history + a system prompt steering it to converse in the target language at the user's level. May append newly-introduced vocab to `vocab_learned`. |
| `translate_node` | Calls the model to translate the user's phrase and explain nuance. May append vocab. |
| `explain_grammar_node` | Calls the model to explain a grammar point. |
| `quiz_vocab_node` | Does **not** call the model for content generation (or optionally does, for phrasing) — instead reads `state["vocab_learned"]`, picks items, and asks the user to translate/use them. Demonstrates reading from the accumulated reducer channel. |

(Five content nodes plus the router and the trim step is more than the "at least two nodes" minimum — this gives the conditional edge something real to branch over.)

### 3.3 Edges

- `START → router` (unconditional)
- `router → trim_and_filter` (unconditional)
- `trim_and_filter → {practice_conversation_node | translate_node | explain_grammar_node | quiz_vocab_node}` — **this is the required conditional edge**, implemented via `add_conditional_edges(trim_and_filter, route_by_intent, {...})` where `route_by_intent` reads `state["intent"]` and returns the corresponding node name.
- Each content node `→ END`

### 3.4 Compilation

```python
graph = StateGraph(ChatState)
# ... add_node calls ...
graph.add_edge(START, "router")
graph.add_edge("router", "trim_and_filter")
graph.add_conditional_edges("trim_and_filter", route_by_intent, {
    "practice_conversation": "practice_conversation_node",
    "translate": "translate_node",
    "explain_grammar": "explain_grammar_node",
    "quiz_vocab": "quiz_vocab_node",
})
graph.add_edge("practice_conversation_node", END)
graph.add_edge("translate_node", END)
graph.add_edge("explain_grammar_node", END)
graph.add_edge("quiz_vocab_node", END)

compiled = graph.compile(checkpointer=checkpointer, store=store)
```

### 3.5 UI requirement for this topic

- A dedicated "Graph" tab/panel in the frontend that calls a backend endpoint (e.g. `GET /graph`) returning the Mermaid or ASCII representation of `compiled.get_graph()`, and renders it. Label it clearly: "This is the compiled LangGraph structure for LinguaLoop."

---

## 4. State Management with Reducers (Topic 2)

- `messages`: default `add_messages` reducer — appends new messages, handles de-duplication by message ID.
- `vocab_learned`: `operator.add` reducer — every node that introduces new vocabulary returns `{"vocab_learned": ["nueva palabra", "otra palabra"]}` (a list) rather than overwriting, and LangGraph concatenates it onto the existing list automatically. **This must be demonstrated in the UI**, e.g. a persistent sidebar "Vocab Learned" list that visibly grows turn by turn without ever shrinking or resetting mid-thread.

### 4.1 Checkpointer + thread_id

- Use `MemorySaver()` as the checkpointer, passed to `.compile(checkpointer=...)`.
- Every invocation passes `config={"configurable": {"thread_id": <id>}}`.
- **Frontend requirement:** a thread switcher UI element (dropdown or tab bar) with at least "New Thread" and a list of existing thread IDs. Switching threads must:
  - Load that thread's message history (via `compiled.get_state(config)` or by replaying stored messages) into the chat window.
  - Load that thread's `vocab_learned` sidebar list.
  - Prove isolation: **populate two threads with different conversations (e.g. Thread A practicing Spanish greetings, Thread B practicing food vocabulary) and show that switching between them shows completely different message histories and different `vocab_learned` lists.** This should be a manual, demoable step — e.g. include a short "demo script" section in the README describing exactly this sequence.

---

## 5. Trimming & Filtering (Topic 3)

Before any content node calls the model, the `trim_and_filter` node must:

1. **Filter**: drop any messages that are pure system/internal bookkeeping (define at least one concrete filter rule — e.g. drop messages tagged as `quiz_vocab` scaffolding once answered, or drop consecutive duplicate AI acknowledgments). Keep it simple but real — one clear, explainable filter rule is enough.
2. **Trim to a token budget**: use LangChain's `trim_messages` utility (or a hand-rolled token counter if avoiding the dependency) with a configurable `MAX_TOKENS` budget (e.g. 500 tokens), keeping the most recent messages that fit, and always keeping the most recent human message.
3. Record both counts into state: `full_count = len(messages)` (before) and `trimmed_count = len(trimmed_messages)` (after), plus ideally an approximate token count before/after (`full_token_estimate`, `trimmed_token_estimate` — a simple `len(text) // 4` heuristic is fine if not using a real tokenizer).

### 5.1 UI requirement for this topic

Every chat turn's response must be accompanied by a small, visible readout, e.g.:

> Sent to model: **6 of 14** messages (~180 of ~640 tokens)

This can render as a small caption under each bot message or in a collapsible "debug" panel. It must update per-turn and must be clearly visible without opening dev tools — this is a graded requirement, not a nice-to-have.

---

## 6. Short-Term & Long-Term Memory (Topic 4)

### 6.1 Short-term memory
This is the checkpointer from Topic 2 — the running `messages` and `vocab_learned` state, scoped to a `thread_id`. Already covered above; no additional design needed, just make sure it's described as "short-term memory" in the README/UI so the mapping to the assignment's terminology is explicit.

### 6.2 Long-term memory
Use LangGraph's `BaseStore` (`InMemoryStore`) with a small **profile schema**, namespaced by a stable `user_id` (NOT `thread_id` — this is the whole point):

```python
profile_schema = {
    "name": Optional[str],
    "target_language": Optional[str],
    "level": Optional[str]   # e.g. "beginner" | "intermediate" | "advanced"
}
```

- Store key: `store.put(("profile", user_id), "profile", {...})`
- A node (or a small preprocessing step before the router) checks whether the latest human message states the user's name, target language, or level (simple keyword/regex detection is fine — e.g. "my name is X", "I want to learn Y", "I'm a beginner") and if so, calls `store.put(...)` to persist it.
- At the start of every graph invocation (regardless of thread), fetch the profile via `store.get(("profile", user_id), "profile")` and inject it into the system prompt for whichever content node runs (e.g. "You are speaking with Athish, who is learning French at a beginner level.").

### 6.3 UI requirement for this topic — the core demo moment

This is the single most important thing to make undeniable in the UI:

1. In **Thread A**, the user says something like "My name is Sam and I'm learning French." The bot acknowledges it.
2. The user clicks **"New Thread"**, creating **Thread B** with an empty message history.
3. In Thread B, the user says something generic like "Hi, let's practice." The bot's reply must reference the remembered profile (e.g. "Hi Sam! Ready to practice some French?") **even though Thread B has zero prior messages.**
4. The frontend should have a small **"Long-Term Profile"** panel (separate from the per-thread chat and separate from the per-thread vocab list) showing the current stored profile fields (`name`, `target_language`, `level`), so the user can see it does not reset when threads are switched or created — in contrast to the per-thread `vocab_learned` panel, which is empty in every new thread.

Include this exact sequence as a "Demo Script" section in the README so it's trivially reproducible for grading.

---

## 7. Frontend Layout (suggested)

A single page with these regions:

1. **Top bar**: thread switcher dropdown + "New Thread" button + current `thread_id` displayed.
2. **Left sidebar**: 
   - "Vocab Learned (this thread)" — list, grows via `operator.add`, resets to empty on new thread.
   - "Long-Term Profile (all threads)" — name / target language / level, persists across threads.
3. **Center**: chat window (standard message bubbles, human vs. AI).
4. **Under each AI message**: small caption showing trim/filter stats for that turn (e.g. "6/14 messages, ~180/~640 tokens sent to model") and the detected `intent` for that turn (e.g. "routed to: quiz_vocab").
5. **A "Graph" tab or modal**: renders the compiled LangGraph structure (Mermaid or ASCII).

---

## 8. Suggested File Structure

```
lingualoop/
├── backend/
│   ├── main.py                # FastAPI app, REST endpoints
│   ├── graph.py                # StateGraph definition, nodes, edges, compile()
│   ├── state.py                 # ChatState TypedDict + reducers
│   ├── memory.py                 # checkpointer + store setup, profile helpers
│   ├── trimming.py                # trim_and_filter logic
│   ├── router.py                   # intent classifier
│   └── llm.py                       # call_model() wrapper — swap DemoChatModel for real provider here
├── frontend/
│   ├── index.html
│   ├── app.js
│   └── style.css
├── README.md                    # setup instructions + the Demo Script from section 6.3
└── requirements.txt
```

---

## 9. Acceptance Checklist (map directly to grading sheet)

- [ ] Compiled `StateGraph` with typed `ChatState`, ≥2 nodes, ≥1 conditional edge (router by intent)
- [ ] Graph structure visibly rendered in the UI (Mermaid or ASCII)
- [ ] `messages` channel uses `add_messages`; `vocab_learned` channel uses `operator.add` (non-default reducer)
- [ ] Checkpointer (`MemorySaver`) wired with `thread_id`; UI lets user create/switch threads
- [ ] Two threads demonstrably hold separate histories and separate vocab lists
- [ ] Trimming (token budget) + filtering (one concrete rule) applied before every model call
- [ ] UI visibly shows messages/tokens sent to model vs. total, per turn
- [ ] Short-term memory = checkpointer (labeled as such)
- [ ] Long-term memory = `BaseStore` with profile schema, namespaced by `user_id` not `thread_id`
- [ ] Demo: fact stated in Thread A is recalled, unprompted, in brand-new Thread B
- [ ] README includes a reproducible step-by-step demo script for the above

---

## 10. Notes for the Implementing Agent (Antigravity)

- Prioritize correctness and visibility of the four mechanics over polish — a plain, slightly ugly UI that clearly displays all required signals is worth more here than a beautiful UI that hides them.
- Keep the LLM swappable but default to a zero-dependency offline model so the whole thing runs immediately with no API keys.
- Favor small, readable node functions over cleverness — this codebase's job is to be legible as a teaching example, since it will likely be read line-by-line.
- Comment the code at the exact points that map to each of the four topics (e.g. `# TOPIC 2: non-default reducer` above the `vocab_learned` field) so a reader can trace the mapping from spec to code quickly.
