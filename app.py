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

# MUST BE FIRST STREAMLIT COMMAND - DO NOT MOVE
st.set_page_config(layout="wide", page_title="Simple English Story Builder")

import chromadb
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

# Performance optimizations
TEMPERATURE = 0.7
MAX_TOKENS = 200
TOP_P = 0.9


# =============================================================
# NVIDIA CLIENTS - CACHED
# =============================================================

@st.cache_resource
def get_llm():
    if not NVIDIA_API_KEY:
        return None
    try:
        return ChatNVIDIA(
            model=LLM_MODEL, 
            api_key=NVIDIA_API_KEY,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            top_p=TOP_P
        )
    except Exception as e:
        st.error(f"LLM init error: {e}")
        return None


@st.cache_resource
def get_embeddings():
    if not NVIDIA_API_KEY:
        return None
    try:
        return NVIDIAEmbeddings(model=EMBED_MODEL, api_key=NVIDIA_API_KEY)
    except Exception as e:
        st.error(f"Embeddings init error: {e}")
        return None


LLM = get_llm()
EMB = get_embeddings()


# =============================================================
# CHROMA DB
# =============================================================

@st.cache_resource
def init_chroma():
    try:
        client = chromadb.PersistentClient(path="./chroma_db")
    except Exception:
        client = chromadb.Client()
    
    coll = client.get_or_create_collection("characters")
    return client, coll


CHROMA_CLIENT, COLLECTION = init_chroma()


@st.cache_data(ttl=300)
def embed_texts_cached(text: str):
    if EMB is None:
        raise RuntimeError("Embeddings not initialized. Set NVIDIA_API_KEY in .env")
    return EMB.embed_documents([text])[0]


def add_or_update_character(name: str, description: str):
    vec = embed_texts_cached(description)
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

        output = []
        for i, d in enumerate(docs):
            m = metas[i]
            output.append({"name": m["name"], "description": d})
        return output
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
    try:
        chars = search_characters(state.prompt, top_k=3)
        state.retrieved = "\n".join([f"- {c['name']}: {c['description']}" for c in chars])
    except:
        state.retrieved = ""
    return state


def make_scene_prompt(scene_number, prompt, characters):
    return f"""Write Scene {scene_number} in simple English (120-180 words).

Characters: {characters if characters else "Create new characters as needed"}

Story: {prompt}

Write clearly with short sentences."""


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

# Session Setup
if "lg_state" not in st.session_state:
    st.session_state["lg_state"] = None

if "scenes" not in st.session_state:
    st.session_state["scenes"] = {}

if "edit_char" not in st.session_state:
    st.session_state["edit_char"] = None


# =============================================================
# SIDEBAR - CHARACTER MANAGEMENT
# =============================================================

with st.sidebar:
    st.header("📚 Character Database")
    
    names = list_character_names()
    
    if names:
        st.caption(f"📊 Total: {len(names)} characters")
        st.markdown("---")
        
        # Character selection
        selected = st.selectbox(
            "Select character:",
            [""] + names,
            key="char_select"
        )
        
        if selected:
            desc = get_character_description(selected)
            
            st.markdown(f"**{selected}**")
            st.text_area(
                "Description:",
                desc,
                height=120,
                disabled=True,
                key=f"view_{selected}",
                label_visibility="collapsed"
            )
            
            col1, col2 = st.columns(2)
            
            if col1.button("✏️ Edit", key=f"edit_{selected}", use_container_width=True):
                st.session_state["edit_char"] = {"name": selected, "desc": desc}
                st.rerun()
            
            if col2.button("🗑 Delete", key=f"del_{selected}", use_container_width=True):
                delete_character(selected)
                st.success(f"Deleted {selected}")
                st.rerun()
    else:
        st.info("No characters yet. Add your first character below!")
    
    st.markdown("---")
    
    # Add/Edit Form
    if st.session_state["edit_char"]:
        st.subheader("✏️ Edit Character")
        edit_data = st.session_state["edit_char"]
        
        st.text_input(
            "Name:",
            edit_data["name"],
            disabled=True,
            key="edit_name_display"
        )
        
        new_desc = st.text_area(
            "Description:",
            edit_data["desc"],
            height=150,
            key="edit_desc"
        )
        
        col1, col2 = st.columns(2)
        
        if col1.button("💾 Save", use_container_width=True):
            add_or_update_character(edit_data["name"], new_desc)
            st.session_state["edit_char"] = None
            st.success("✅ Updated!")
            st.rerun()
        
        if col2.button("❌ Cancel", use_container_width=True):
            st.session_state["edit_char"] = None
            st.rerun()
            
    else:
        st.subheader("➕ Add New Character")
        
        cname = st.text_input("Name:", key="new_cname")
        cdesc = st.text_area("Description:", key="new_cdesc", height=120)
        
        if st.button("💾 Save Character", use_container_width=True, type="primary"):
            if cname and cdesc:
                add_or_update_character(cname, cdesc)
                st.success("✅ Character saved!")
                st.rerun()
            else:
                st.error("Both name and description required.")


# =============================================================
# MAIN AREA - STORY BUILDER
# =============================================================

st.header("📝 Build Stories with Your Own Characters")

story_title = st.text_input("Story Title (optional):", key="story_title")
prompt_text = st.text_area("Story Prompt:", height=120, key="story_prompt", 
                           placeholder="Enter your story idea here...")

if st.button("🚀 Generate Scene 1", type="primary"):
    if not prompt_text.strip():
        st.error("Please enter a story prompt.")
    else:
        with st.spinner("✨ Generating scene 1..."):
            init_state = LGState(
                prompt=prompt_text.strip(),
                scene_number=1
            )
            try:
                result = workflow.invoke(init_state)
                result_dict = dict(result)

                st.session_state["lg_state"] = result_dict
                st.session_state["scenes"] = {1: result_dict["scene"]}

                st.success("✅ Scene 1 ready!")
                st.rerun()
            except Exception as e:
                st.error(f"Error: {str(e)}")

# Show current scene
if st.session_state["lg_state"]:
    s = st.session_state["lg_state"]
    sn = s["scene_number"]

    st.markdown("---")
    st.subheader(f"📘 Scene {sn}")
    
    if story_title:
        st.markdown(f"### **{story_title}**")
    
    # Scene content in a nice container
    with st.container():
        st.markdown(s["scene"])


    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("✅ Accept & Continue", use_container_width=True, type="primary"):
            if sn >= MAX_SCENES:
                st.info("✔ Story complete! All 3 scenes generated.")
            else:
                with st.spinner(f"✨ Generating scene {sn+1}..."):
                    next_state = LGState(
                        prompt=s["prompt"],
                        retrieved=s["retrieved"],
                        scene_number=sn + 1
                    )
                    try:
                        result2 = workflow.invoke(next_state)
                        result2_dict = dict(result2)

                        st.session_state["lg_state"] = result2_dict
                        st.session_state["scenes"][sn + 1] = result2_dict["scene"]

                        st.success(f"✅ Scene {sn+1} ready!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
    
    with col2:
        if st.button("🔄 Regenerate Scene", use_container_width=True):
            with st.spinner("✨ Regenerating..."):
                current_state = LGState(
                    prompt=s["prompt"],
                    retrieved=s["retrieved"],
                    scene_number=sn
                )
                try:
                    result = workflow.invoke(current_state)
                    result_dict = dict(result)

                    st.session_state["lg_state"] = result_dict
                    st.session_state["scenes"][sn] = result_dict["scene"]

                    st.success("✅ Scene regenerated!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    
    # Custom changes expander
    with st.expander("✏️ Make Custom Changes"):
        change_instructions = st.text_area(
            "Describe changes:",
            placeholder="e.g., Make it more emotional, add action, change the tone...",
            height=80
        )
        
        if st.button("Apply Changes", type="primary"):
            if not change_instructions.strip():
                st.error("Please describe what to change.")
            else:
                with st.spinner("✨ Rewriting scene..."):
                    rewrite_prompt = f"""Rewrite this scene in simple English with these changes:
{change_instructions}

Original scene:
{s["scene"]}

Rewritten scene:"""
                    try:
                        resp = LLM.invoke(rewrite_prompt)
                        new_scene = resp.content if hasattr(resp, "content") else str(resp)

                        new_state = {
                            "prompt": s["prompt"],
                            "retrieved": s["retrieved"],
                            "scene": new_scene,
                            "scene_number": sn,
                            "feedback": "",
                        }

                        st.session_state["lg_state"] = new_state
                        st.session_state["scenes"][sn] = new_scene
                        st.success("✅ Scene updated!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {str(e)}")

# Full Story Display
if len(st.session_state["scenes"]) > 1:
    st.markdown("---")
    st.subheader("📖 Complete Story")
    
    for k in sorted(st.session_state["scenes"].keys()):
        with st.expander(f"Scene {k}", expanded=(k == sn if st.session_state["lg_state"] else True)):
            st.write(st.session_state["scenes"][k])
    
    # Export option
    if len(st.session_state["scenes"]) == MAX_SCENES:
        st.markdown("---")
        
        full_story = "\n\n---\n\n".join([
            f"Scene {k}\n\n{st.session_state['scenes'][k]}" 
            for k in sorted(st.session_state["scenes"].keys())
        ])
        
        if story_title:
            full_story = f"{story_title}\n\n{full_story}"
        
        st.download_button(
            "📥 Download Complete Story",
            full_story,
            file_name="story.txt",
            mime="text/plain",
            use_container_width=True,
            type="primary"
        )