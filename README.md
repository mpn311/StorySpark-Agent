# 🎭 StorySpark-Agent

An interactive AI-powered story generation application that creates scene-wise stories using your own custom characters.

Built with **LangGraph**, **ChromaDB**, **NVIDIA LLMs**, and **Streamlit**.

---

## 🚀 Features

* **Custom Character Database:** Add, edit, delete, and reuse your own characters. Stored in **ChromaDB** with semantic search via **NVIDIA embeddings**.
* **Scene-by-Scene Story Generation:** The story is generated in multiple scenes. Each scene uses the user's prompt plus dynamically **retrieved characters**.
* **LangGraph Workflow:** Handles the stateful process of character retrieval → scene generation → and the interactive **feedback loop**.
* **Regenerate & Rewrite Scenes:** Users can regenerate scenes or rewrite them with custom instructions (tone, detail, mood, simplicity, etc.).

---

## 🔄 Workflow

The following diagram illustrates the core LangGraph-managed process for generating and refining each scene of the story.

```text
                              Start
                                │
                                ▼
                    ┌─────────────────────────┐
                    │ 📝 User Enters Story     │
                    │    Prompt + (Title)      │
                    └───────────┬─────────────┘
                                │
                                ▼
                    ┌─────────────────────────┐
                    │ 🧑 Character Retrieval    │
                    └───────────┬─────────────┘
                                │
                                ▼
                    ┌─────────────────────────┐ 
                    │ ✍️ Scene Generation     ◀───────────────────────────────┐
                    │   (LLM Generates Scene) │                                │
                    └───────────┬─────────────┘                                │
                                │                                              │
                                ▼                                              │
                    ┌─────────────────────────┐                                │
                    │ 👁️ Display Scene                                        │
                    │       to User                                            │
                    └───────────┬─────────────┘                                │
                                │                                              │
                                ▼                                              │
        ┌────────────────────────────────────────────────────────────────────────────┐
        │                          🔍 DECISION BLOCK                                                 
        ├────────────────────────────────────────────────────────────────────────────┤
        │ ✔ Accept  → Proceed to         ───────────►    Generation next Scene               
        ├────────────────────────────────────────────────────────────────────────────┤
        │ ❌ Reject  → Re-generate Scene  ─────────────► (Back to Scene Generation) 
        ├────────────────────────────────────────────────────────────────────────────┤
        │ ✏️ Custom → Apply user's changes ───────────►  (Back to Scene Generation) 
        └────────────────────────────────────────────────────────────────────────────┘
                                │
                                │ 
                                ▼
                    ┌─────────────────────────┐
                    │ 📚 Story Assembly       |
                    │ • Combine all scenes     |
                    │ • Add title/formatting   │
                    └───────────┬─────────────┘
                                │
                                ▼
                    ┌─────────────────────────┐
                    │ 💾 Download / Export     
                    │   story.txt / story.pdf  
                    └───────────┬─────────────┘
                                │
                                ▼
                               END  '''

## 📘 Example: How to Create a Story
### Enter a Story Prompt
Once your characters are added, enter a simple story prompt like this:

> **A strange dark cloud appears above Dholakpur, and people suddenly start losing their strength.  
> Bheem, Chutki, Raju, and Jaggu must find the reason behind this power-draining cloud before it spreads to the whole village.  
> Meanwhile, Kalia tries to prove he is the strongest, but his plans only create more trouble.  
> Can the team stop the danger before sunset?**



## 🧠 Tech Stack

| Component | Technology |
|----------|------------|
| Story generation | NVIDIA LLM (Llama-3.1-8B-Instruct) |
| Character embeddings | NVIDIA NV-Embed-v1 |
| Vector DB | ChromaDB |
| Orchestration | LangGraph |
| UI | Streamlit |
| Environment | Python 3.10+ |

---
