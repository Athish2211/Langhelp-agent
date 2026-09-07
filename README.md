# LinguaLoop — Language Learning Partner (LangGraph Demo)

LinguaLoop is an interactive foreign language practice partner built on **LangGraph**. It serves as an end-to-end demonstration of four essential LangGraph mechanics:
1. **Graphs & Intent-Based Routing**
2. **Reducer-Based State Management** (`add_messages` + `operator.add`)
3. **Message Trimming & Bookkeeping Filtering**
4. **Short-Term Memory (`MemorySaver`) vs. Long-Term Memory (`InMemoryStore`)**

The application is structured into a minimal two-file codebase:
- [`app.py`](app.py) — Complete FastAPI server + LangGraph StateGraph, nodes, reducers, stores, and swappable model logic.
- [`index.html`](index.html) — Single-page UI with live memory inspectors, thread isolation switcher, telemetry readout, and Mermaid diagram visualizer.

---

## 🚀 Quickstart

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Start the Server
```bash
python app.py
```
Open your browser to: **[http://127.0.0.1:8000](http://127.0.0.1:8000)**

> [!NOTE]
> **Zero API Keys Required**: LinguaLoop runs out-of-the-box using an offline, deterministic, high-fidelity demo language model. It incurs zero cost and requires no API keys or internet connection.

---

## 🧩 Architecture & The 4 LangGraph Mechanics

### Topic 1: Graph Architecture & Conditional Routing
- **File & Function**: [`app.py`](app.py) (`builder = StateGraph(ChatState)`)
- **Nodes**:
  - `router`: Classifies input into `practice_conversation`, `translate`, `explain_grammar`, or `quiz_vocab`.
  - `trim_and_filter`: Enforces token budget and filters scaffolding.
  - `practice_conversation_node`: Dynamic dialogue in the target language.
  - `translate_node`: Translations with cultural nuance and vocabulary capture.
  - `explain_grammar_node`: Concise grammatical breakdowns.
  - `quiz_vocab_node`: Vocabulary quiz formulated from accumulated state.
- **Conditional Edge**: `trim_and_filter` routes to the appropriate content node using `route_by_intent`.
- **UI Visibility**: Dedicated **StateGraph Visualizer** tab renders the live graph structure via Mermaid.js or ASCII.

### Topic 2: State Management with Reducers
- **File & Function**: [`app.py`](app.py) (`ChatState`)
- **Default Reducer (`add_messages`)**: `messages: Annotated[list[BaseMessage], add_messages]` automatically handles message deduplication and chronological ordering.
- **Non-Default Reducer (`operator.add`)**: `vocab_learned: Annotated[list[str], operator.add]` automatically concatenates new vocabulary introduced by content nodes across turns without overwriting previous entries.
- **UI Visibility**: The **Vocab Learned** card in the left sidebar dynamically grows turn-by-turn.

### Topic 3: Message Trimming & Filtering
- **File & Function**: [`app.py`](app.py) (`trim_and_filter_node`)
- **Filter Rule**: Drops internal bookkeeping and scaffolding messages before passing history to the model.
- **Token Budget**: Trims to the most recent context within the token budget (preserving the latest user message).
- **UI Visibility**: Every bot response displays a per-turn telemetry chip:
  > *Model received: 4 of 6 msgs (~80 of ~190 tok) | Routed: practice_conversation*

### Topic 4: Short-Term vs. Long-Term Memory
- **Short-Term Memory (`MemorySaver`)**:
  - Scoped to `thread_id` via checkpointer.
  - Isolated between threads (Thread A's chat history and vocab are not visible in Thread B).
- **Long-Term Memory (`InMemoryStore`)**:
  - Scoped globally to `("profile", user_id)`.
  - Extracts the user's name, target language, and proficiency level.
  - Persists across all threads.
- **UI Visibility**: The **Long-Term Profile** card stays intact across threads, while the **Vocab Learned** list and chat history reset.

---

## 🎬 Reproducible Demo Script (Grading Sequence)

You can run this sequence manually or by clicking the **Demo Script Actions** buttons in the left sidebar:

1. **Step 1: Set Long-Term Profile in Thread A**
   - Message: `"My name is Sam and I'm learning French at beginner level."`
   - *Result*: The bot greets Sam in French. The **Long-Term Profile** card updates to `Name: Sam`, `Language: French`, `Level: beginner`. The **Vocab Learned** card adds initial greeting words.
2. **Step 2: Accumulate Vocab in Thread A**
   - Message: `"I love food and croissants."`
   - *Result*: The bot introduces food vocabulary (`le fromage`, `le petit déjeuner`). The **Vocab Learned** card grows to 4 words via `operator.add`.
3. **Step 3: New Thread Isolation Check**
   - Click **"+ New Thread"** (creating `Thread B`). Notice that chat history and the **Vocab Learned** list are empty.
   - Message: `"Hi, let's practice."`
   - *Result*: The bot immediately replies:
     > *"Bonjour Sam! Ready to practice some French at your beginner level?..."*
     **Unprompted recall of name and language from long-term store in a brand new thread with zero prior history!**
4. **Step 4: Vocab Quiz from Accumulated Reducer**
   - Switch back to `thread-A` using the dropdown.
   - Message: `"Quiz me on our vocabulary!"`
   - *Result*: The bot routes to `quiz_vocab` and generates a quiz question using the words accumulated in `thread-A` (`"le petit déjeuner"`).

---

## 🔌 Swapping to a Live LLM (OpenAI / Gemini / Ollama)

In [`app.py`](app.py), locate the `call_model()` function:
```python
# To swap in OpenAI:
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4o-mini")
# pass messages & system_prompt directly to llm.invoke()
```
No changes to graph logic or state schemas are required.
