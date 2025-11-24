# 🎭 StorySpark-Agent  
An interactive AI-powered story generation app that creates scene-wise stories using your own custom characters.  
Built with **LangGraph**, **ChromaDB**, **NVIDIA LLMs**, and **Streamlit**.

---

## 🚀 Features

- **Custom Characters Database**  
  Add, edit, delete, and reuse your own characters.  
  Stored in ChromaDB with semantic search via NVIDIA embeddings.

- **Scene-by-Scene Story Generation**  
  Story is generated in multiple scenes.  
  Each scene uses retrieved characters + story prompt.

- **LangGraph Workflow**  
  Manages retrieval → generation → feedback loop for consistent storytelling.

- **Regenerate & Rewrite Scenes**  
  Modify scenes with custom instructions (tone, detail, emotion, etc.).

## Workflow
                ┌─────────────────────────┐
                │  User Enters Story       │
                │  Prompt + (Title)        │
                └───────────┬─────────────┘
                            │
                            ▼
                ┌─────────────────────────┐
                │  Character Retrieval     │
                │  • Embed prompt          │
                │  • Query ChromaDB        │
                │  • Return top matches    │
                └───────────┬─────────────┘
                            │
                            ▼
                ┌─────────────────────────┐
                │  Scene Generation        │
                │  NVIDIA LLM creates      │
                │  Scene N using:          │
                │   - Prompt               │
                │   - Retrieved characters │
                └───────────┬─────────────┘
                            │
                            ▼
                ┌─────────────────────────┐
                │ Display Scene to User    │
                └───────────┬─────────────┘
                            │
        ┌───────────────Yes ▼───────────┐
        │         ┌──────────────────┐   │
        │         │  Accept Scene?   │   │
        │         └───────┬─────────┘   │
        │                 │No            │
        │                 ▼              │
        │      ┌────────────────────┐    │
        │      │ Regenerate Scene   │◄───┘
        │      │ (LLM rewrite)      │
        │      └─────────┬──────────┘
        │                │
        └────────────────┘
                            │Yes
                            ▼
                ┌─────────────────────────┐
                │ More Scenes to Create?  │
                └───────────┬─────────────┘
                        No   │   Yes
                            │
   ┌───────────────────────▼──────────────────────┐
   │  Generate Next Scene (SceneNumber + 1)       │
   └───────────────────────┬──────────────────────┘
                           │
                           ▼
                (Loop back to Scene Generation)


                           ▼ No
                ┌─────────────────────────┐
                │  Story Assembly          │
                │  • Combine all scenes   │
                │  • Add title            │
                └───────────┬─────────────┘
                            │
                            ▼
                ┌─────────────────────────┐
                │ Download / Export Story  │
                │  story.txt               │
                └───────────┬─────────────┘
                            │
                            ▼
                           END



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
