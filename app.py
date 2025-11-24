import sys
import types
import os
from typing import List, Dict, Any

from dotenv import load_dotenv

# Torch shim for Windows
if "torch.classes" not in sys.modules:
    sys.modules["torch.classes"] = types.ModuleType("torch.classes")

# Load .env
load_dotenv()

import streamlit as st

# MUST BE FIRST STREAMLIT COMMAND
st.set_page_config(layout="wide", page_title="Simple English Story Builder")

import chromadb
from chromadb.config import Settings

from langgraph.graph import StateGraph, END
from pydantic import BaseModel
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings


# =============================================================
# CONFIG
# =============================================================

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "")
LLM_MODEL = "meta/llama-3.1-8b-instruct"
EMBED_MODEL = "nvidia/nv-embed-v1"
MAX_SCENES = 3

TEMPERATURE = 0.7
MAX_TOKENS = 200
TOP_P = 0.9


# =============================================================
# NVIDIA CLIENTS
# =============================================================

@st.cache_resource
def get_llm():
    if not NVIDIA_API_KEY:
        return None
    return ChatNVIDIA(
        model=LLM_MODEL,
        api_key=NVIDIA_API_KEY,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        top_p=TOP_P,
    )


@st.cache_resource
def get_embeddings():
    if not NVIDIA_API_KEY:
        return None
    return NVIDIAEmbeddings(model=EMBED_MODEL, api_key=NVIDIA_API_KEY)


LLM = get_llm()
EMB = get_embeddings()


# =============================================================
# FIXED CHROMA DB (STREAMLIT CLOUD SAFE)
# =============================================================

def get_chroma_path():
    """
    Streamlit Cloud does NOT allow writing to root folders.
    This function ensures the DB always goes into a writable path.
    """
    base = "./.streamlit/chroma_db"
    os.makedirs(base, exist_ok=True)
    return base


@st.cache_resource
def init_chroma():
    """
    Always initializes a fully persistent DB in a writable directory.
    Works on Streamlit Cloud, local machine, containers, everywhere.
    """
    db_path = get_chroma_path()

    client = chromadb.PersistentClient(
        path=db_path,
        settings=Settings(anonymized_telemetry=False)
    )

    coll = client.get_or_create_collection(
        "characters",
        metadata={"hnsw:space": "cosine"}
    )

    return client, coll


CHROMA_CLIENT, COLLECTION = init_chroma()


# =============================================================
# EMBEDDING HELPERS
# =============================================================

@st.cache_data(ttl=300)
def embed_texts_cached(text: str):
    return EMB.embed_documents([text])[0]


def add_or_update_character(name: str, description: str):
    vec = embed_texts_cached(description)

    # Remove previous entry if exists
    try:
        COLLECTION.delete(ids=[name])
    except:
        pass

    COLLECTION.add(
        ids=[name],
        documents=[description],
        embeddings=[vec],
        metadatas=[{"name": name}],
    )

    st.cache_data.clear()


def delete_character(name: str):
    try:
        COLLECTION.delete(ids=[name])
        st.cache_data.clear()
    except:
        pass


@st.cache_data(ttl=60)
def list_character_names():
    try:
        data = COLLECTION.get()
        return data.get("ids", [])
    except:
        return []


@st.cache_data(ttl=60)
def get_character_description(name: str):
    try:
        data = COLLECTION.get(ids=[name])
        return data.get("documents", [""])[0]
    except:
        return ""


def search_characters(query: str, top_k: int = 3):
    try:
        vec = embed_texts_cached(query)
        res = COLLECTION.query(query_embeddings=[vec], n_results=top_k)

        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]

        return [
            {"name": metas[i]["name"], "description": docs[i]}
            for i in range(len(docs))
        ]
    except:
        return []


# =============================================================
# LANGGRAPH STATE
# =============================================================

class LGState(BaseModel):
    prompt: str = ""
    retrieved: str = ""
    scene: str = ""
    feedback: str = ""
    scene_number: int = 1


# =============================================================
# GRAPH NODES
# =============================================================

def node_retrieve(state: LGState) -> LGState:
    chars = search_characters(state.prompt, top_k=3)
    state.retrieved = "\n".join([f"- {c['name']}: {c['description']}" for c in chars])
    return state


def make_scene_prompt(scene_number, prompt, characters):
    return f"""Write Scene {scene_number} in simple English (120–180 words).

Characters: {characters if characters else "Create new characters as needed"}

Story: {prompt}

Use simple clear sentences.
"""


def node_generate_scene(state: LGState) -> LGState:
    if LLM is None:
        state.scene = "ERROR: LLM not initialized"
        return state

    prompt_text = make_scene_prompt(
        state.scene_number,
        state.prompt,
        state.retrieved,
    )

    try:
        resp = LLM.invoke(prompt_text)
        state.scene = resp.content if hasattr(resp, "content") else str(resp)
    except Exception as e:
        state.scene = f"[Scene generation error: {e}]"

    return state


# =============================================================
# BUILD GRAPH
# =============================================================

graph = StateGraph(LGState)
graph.add_node("retrieve", node_retrieve)
graph.add_node("generate_scene", node_generate_scene)

graph.set_entry_point("retrieve")
graph.add_edge("retrieve", "generate_scene")
graph.add_edge("generate_scene", END)

workflow = graph.compile()


# =============================================================
# STREAMLIT UI
# =============================================================

st.title("🎭 StorySpark Agent")

if not NVIDIA_API_KEY:
    st.error("⚠️ NVIDIA_API_KEY missing in .env file!")
    st.stop()

# -------------------------------------------------------------
# SESSION
# -------------------------------------------------------------

if "lg_state" not in st.session_state:
    st.session_state["lg_state"] = None

if "scenes" not in st.session_state:
    st.session_state["scenes"] = {}

if "edit_char" not in st.session_state:
    st.session_state["edit_char"] = None


# =============================================================
# SIDEBAR – CHARACTER DB
# =============================================================

with st.sidebar:
    st.header("📚 Character Database")

    names = list_character_names()

    if names:
        st.caption(f"📊 Total: {len(names)} characters")
        st.markdown("---")

        selected = st.selectbox("Select character:", [""] + names, key="char_select")

        if selected:
            desc = get_character_description(selected)

            st.markdown(f"**{selected}**")
            st.text_area("", desc, height=120, disabled=True)

            col1, col2 = st.columns(2)

            if col1.button("✏️ Edit", key=f"edit_{selected}"):
                st.session_state["edit_char"] = {"name": selected, "desc": desc}
                st.rerun()

            if col2.button("🗑 Delete", key=f"del_{selected}"):
                delete_character(selected)
                st.success(f"Deleted {selected}")
                st.rerun()

    else:
        st.info("No characters yet. Add your first character below!")

    st.markdown("---")

    if st.session_state["edit_char"]:
        st.subheader("✏️ Edit Character")
        edit_data = st.session_state["edit_char"]

        st.text_input("Name:", edit_data["name"], disabled=True)
        new_desc = st.text_area("Description:", edit_data["desc"], height=150)

        col1, col2 = st.columns(2)

        if col1.button("💾 Save"):
            add_or_update_character(edit_data["name"], new_desc)
            st.session_state["edit_char"] = None
            st.rerun()

        if col2.button("❌ Cancel"):
            st.session_state["edit_char"] = None
            st.rerun()

    else:
        st.subheader("➕ Add New Character")

        cname = st.text_input("Name:")
        cdesc = st.text_area("Description:", height=120)

        if st.button("💾 Save Character", type="primary"):
            if cname and cdesc:
                add_or_update_character(cname, cdesc)
                st.rerun()
            else:
                st.error("Both name and description required.")


# =============================================================
# MAIN STORY BUILDER
# =============================================================

st.header("📝 Build Stories with Your Characters")

story_title = st.text_input("Story Title (optional):")
prompt_text = st.text_area("Story Prompt:", height=120)

if st.button("🚀 Generate Scene 1", type="primary"):
    if not prompt_text.strip():
        st.error("Enter a story prompt first.")
    else:
        with st.spinner("Generating scene 1..."):
            init_state = LGState(prompt=prompt_text.strip(), scene_number=1)
            result = workflow.invoke(init_state)
            rd = dict(result)

            st.session_state["lg_state"] = rd
            st.session_state["scenes"] = {1: rd["scene"]}

            st.rerun()

# --------------------------------------------------------------
# SHOW CURRENT SCENE
# --------------------------------------------------------------

if st.session_state["lg_state"]:
    s = st.session_state["lg_state"]
    sn = s["scene_number"]

    st.markdown("---")
    st.subheader(f"📘 Scene {sn}")

    if story_title:
        st.markdown(f"### **{story_title}**")

    st.markdown(s["scene"])

    st.markdown("---")

    col1, col2 = st.columns(2)

    # CONTINUE
    if col1.button("✅ Accept & Continue", type="primary"):
        if sn >= MAX_SCENES:
            st.info("Story complete!")
        else:
            next_state = LGState(
                prompt=s["prompt"],
                retrieved=s["retrieved"],
                scene_number=sn + 1
            )

            with st.spinner(f"Generating scene {sn+1}..."):
                result = workflow.invoke(next_state)
                rd = dict(result)

                st.session_state["lg_state"] = rd
                st.session_state["scenes"][sn + 1] = rd["scene"]
                st.rerun()

    # REGENERATE
    if col2.button("🔄 Regenerate Scene"):
        current_state = LGState(
            prompt=s["prompt"],
            retrieved=s["retrieved"],
            scene_number=sn
        )

        with st.spinner("Regenerating..."):
            result = workflow.invoke(current_state)
            rd = dict(result)

            st.session_state["lg_state"] = rd
            st.session_state["scenes"][sn] = rd["scene"]
            st.rerun()

    # CUSTOM CHANGES
    with st.expander("✏️ Make Custom Changes"):
        change = st.text_area("Describe changes:")

        if st.button("Apply Changes", type="primary"):
            rewrite_prompt = f"""Rewrite this scene with these changes:
{change}

Original:
{s["scene"]}

Rewritten scene:
"""

            with st.spinner("Rewriting..."):
                resp = LLM.invoke(rewrite_prompt)
                new_scene = resp.content if hasattr(resp, "content") else str(resp)

                new_state = {
                    "prompt": s["prompt"],
                    "retrieved": s["retrieved"],
                    "scene": new_scene,
                    "scene_number": sn,
                }

                st.session_state["lg_state"] = new_state
                st.session_state["scenes"][sn] = new_scene
                st.rerun()


# =============================================================
# COMPLETE STORY
# =============================================================

if len(st.session_state["scenes"]) > 1:
    st.markdown("---")
    st.subheader("📖 Complete Story")

    for k in sorted(st.session_state["scenes"].keys()):
        with st.expander(f"Scene {k}"):
            st.write(st.session_state["scenes"][k])

    if len(st.session_state["scenes"]) == MAX_SCENES:
        st.markdown("---")
        full_story = "\n\n---\n\n".join(
            [f"Scene {k}\n\n{st.session_state['scenes'][k]}" for k in sorted(st.session_state["scenes"].keys())]
        )

        if story_title:
            full_story = f"{story_title}\n\n{full_story}"

        st.download_button(
            "📥 Download Complete Story",
            full_story,
            file_name="story.txt",
            mime="text/plain",
        )
