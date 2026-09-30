"""
Laboratorio NLP + IA generativa (Groq)
--------------------------------------
Web (Streamlit):   streamlit run app.py
Jupyter Notebook:  from app import *   (las funciones del núcleo no dependen de Streamlit)

Contenido
  1. Esquemas de tokenización  (palabras, caracteres, BPE, WordPiece)
  2. Esquemas de embedding      (BoW, TF-IDF, GloVe/Word2Vec, Sentence-Transformers, GPT-2)
  3. Similitud coseno entre frases
  4. Chat generativo con la API de Groq (API compatible con OpenAI)
"""
from __future__ import annotations

import os
import re
from functools import lru_cache
from typing import Callable, Dict, Iterator, List

import numpy as np

# =====================================================================
# 1. TOKENIZACIÓN
# =====================================================================
def tok_words(text: str) -> List[str]:
    """Palabras y signos de puntuación, en minúscula."""
    return re.findall(r"\w+|[^\w\s]", text.lower())


def tok_chars(text: str) -> List[str]:
    """Caracteres individuales."""
    return list(text)


@lru_cache(maxsize=None)
def _hf_tokenizer(name: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(name)


def tok_bpe(text: str) -> List[str]:
    """Byte-Pair Encoding (el de GPT-2)."""
    return _hf_tokenizer("gpt2").tokenize(text)


def tok_wordpiece(text: str) -> List[str]:
    """WordPiece (el de BERT, sin mayúsculas)."""
    return _hf_tokenizer("bert-base-uncased").tokenize(text)


TOKENIZERS: Dict[str, Callable[[str], List[str]]] = {
    "Palabras": tok_words,
    "Caracteres": tok_chars,
    "BPE (GPT-2)": tok_bpe,
    "WordPiece (BERT)": tok_wordpiece,
}

_HF_NAMES = {"BPE (GPT-2)": "gpt2", "WordPiece (BERT)": "bert-base-uncased"}


def tokenize(text: str, scheme: str) -> List[dict]:
    """Devuelve [{'token':..., 'id':...}]. El id sólo existe en BPE / WordPiece."""
    tokens = TOKENIZERS[scheme](text)
    if scheme in _HF_NAMES:
        ids = _hf_tokenizer(_HF_NAMES[scheme]).convert_tokens_to_ids(tokens)
    else:
        ids = [None] * len(tokens)
    return [{"token": t, "id": i} for t, i in zip(tokens, ids)]


# =====================================================================
# 2. EMBEDDINGS (cada función recibe una lista de frases -> matriz n x d)
# =====================================================================
def emb_bow(sentences: List[str], tokenizer: Callable = tok_words) -> np.ndarray:
    from sklearn.feature_extraction.text import CountVectorizer

    v = CountVectorizer(tokenizer=tokenizer, lowercase=False, token_pattern=None)
    return v.fit_transform(sentences).toarray().astype(float)


def emb_tfidf(sentences: List[str], tokenizer: Callable = tok_words) -> np.ndarray:
    from sklearn.feature_extraction.text import TfidfVectorizer

    v = TfidfVectorizer(tokenizer=tokenizer, lowercase=False, token_pattern=None)
    return v.fit_transform(sentences).toarray()


@lru_cache(maxsize=None)
def _keyed_vectors(name: str):
    import gensim.downloader as api

    return api.load(name)


def emb_wordvec(sentences: List[str], model_name: str = "glove-wiki-gigaword-50") -> np.ndarray:
    """Promedio de vectores de palabra (GloVe / Word2Vec). Ignora palabras fuera de vocabulario."""
    kv = _keyed_vectors(model_name)
    out = []
    for s in sentences:
        vecs = [kv[w] for w in tok_words(s) if w in kv]
        out.append(np.mean(vecs, axis=0) if vecs else np.zeros(kv.vector_size))
    return np.array(out)


@lru_cache(maxsize=None)
def _sbert(name: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(name)


def emb_sbert(
    sentences: List[str],
    model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
) -> np.ndarray:
    return np.asarray(_sbert(model_name).encode(sentences))


@lru_cache(maxsize=None)
def _gpt2_model():
    from transformers import GPT2Model, GPT2TokenizerFast

    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    model = GPT2Model.from_pretrained("gpt2").eval()
    return tok, model


def emb_gpt2(sentences: List[str]) -> np.ndarray:
    """Promedio de los hidden states de la última capa de GPT-2."""
    import torch

    tok, model = _gpt2_model()
    out = []
    with torch.no_grad():
        for s in sentences:
            enc = tok(s, return_tensors="pt", truncation=True, max_length=512)
            out.append(model(**enc).last_hidden_state[0].mean(dim=0).numpy())
    return np.array(out)


# Registro: nombre -> f(frases, tokenizador). Sólo BoW y TF-IDF usan el tokenizador elegido.
EMBEDDINGS: Dict[str, Callable[[List[str], Callable], np.ndarray]] = {
    "Bag of Words": lambda s, tok: emb_bow(s, tok),
    "TF-IDF": lambda s, tok: emb_tfidf(s, tok),
    "GloVe (50d, inglés)": lambda s, tok: emb_wordvec(s, "glove-wiki-gigaword-50"),
    "Word2Vec (Google News 300d, ~1.6 GB)": lambda s, tok: emb_wordvec(s, "word2vec-google-news-300"),
    "Sentence-Transformers (multilingüe)": lambda s, tok: emb_sbert(s),
    "GPT-2 (promedio hidden states)": lambda s, tok: emb_gpt2(s),
}


# =====================================================================
# 3. SIMILITUD COSENO
# =====================================================================
def cosine_matrix(X: np.ndarray) -> np.ndarray:
    """cos(a, b) = a·b / (||a|| ||b||) para todas las parejas de filas."""
    X = np.asarray(X, dtype=float)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    Xn = X / norms
    return Xn @ Xn.T


def compare_sentences(sentences: List[str], embedding: str, tokenizer: str = "Palabras"):
    """Devuelve un DataFrame (n x n) con la similitud coseno entre frases."""
    import pandas as pd

    X = EMBEDDINGS[embedding](sentences, TOKENIZERS[tokenizer])
    labels = [f"F{i + 1}" for i in range(len(sentences))]
    return pd.DataFrame(cosine_matrix(X), index=labels, columns=labels)


# =====================================================================
# 4. IA GENERATIVA (Groq, API compatible con OpenAI)
# =====================================================================
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_MODEL = "openai/gpt-oss-120b"  # modelo tipo GPT; otros: llama-3.3-70b-versatile, llama-3.1-8b-instant, openai/gpt-oss-20b
DEFAULT_SYSTEM = "Eres un asistente útil. Responde en el idioma del usuario."


def groq_stream(
    messages: List[dict],
    api_key: str | None = None,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.7,
) -> Iterator[str]:
    """Genera la respuesta por fragmentos. `messages` = [{'role': 'user'|'assistant'|'system', 'content': ...}]"""
    from openai import OpenAI

    key = api_key or os.getenv("GROQ_API_KEY")
    if not key:
        raise ValueError("Falta la API key de Groq (variable GROQ_API_KEY o campo en la barra lateral).")
    client = OpenAI(api_key=key, base_url=GROQ_BASE_URL)
    stream = client.chat.completions.create(
        model=model, messages=messages, temperature=temperature, stream=True
    )
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content


def groq_chat(user_text: str, history: List[dict] | None = None, system: str = DEFAULT_SYSTEM, **kw) -> str:
    """Versión sin streaming, cómoda para notebooks."""
    msgs = [{"role": "system", "content": system}] + (history or []) + [{"role": "user", "content": user_text}]
    return "".join(groq_stream(msgs, **kw))


# =====================================================================
# INTERFAZ STREAMLIT
# =====================================================================
def main() -> None:
    import streamlit as st

    st.set_page_config(page_title="Laboratorio NLP + Groq", page_icon="🧠", layout="wide")
    st.title("🧠 Laboratorio NLP + IA generativa")

    with st.sidebar:
        st.header("Configuración de Groq")
        api_key = st.text_input("API key de Groq", type="password", value=os.getenv("GROQ_API_KEY", ""))
        model = st.text_input("Modelo", value=DEFAULT_MODEL)
        temperature = st.slider("Temperatura", 0.0, 2.0, 0.7, 0.1)
        system_prompt = st.text_area("Prompt de sistema", value=DEFAULT_SYSTEM)
        if st.button("Limpiar chat"):
            st.session_state["messages"] = []

    tab_tok, tab_emb, tab_chat = st.tabs(["🔤 Tokenización", "📐 Embeddings y coseno", "💬 Chat generativo"])

    # ---- Tokenización ------------------------------------------------
    with tab_tok:
        text = st.text_area("Texto", "La inteligencia artificial está transformando el análisis actuarial.", key="tok_text")
        schemes = st.multiselect("Esquemas", list(TOKENIZERS), default=list(TOKENIZERS))
        if st.button("Tokenizar"):
            import pandas as pd

            cols = st.columns(max(len(schemes), 1))
            for col, sch in zip(cols, schemes):
                with col, st.spinner(f"{sch}..."):
                    rows = tokenize(text, sch)
                    st.subheader(sch)
                    st.metric("Nº de tokens", len(rows))
                    st.dataframe(pd.DataFrame(rows), width="stretch")

    # ---- Embeddings + coseno ----------------------------------------
    with tab_emb:
        st.caption("Una frase por línea. El esquema de tokenización sólo aplica a Bag of Words y TF-IDF; "
                   "los demás modelos usan su propio tokenizador.")
        raw = st.text_area(
            "Frases a comparar",
            "El seguro de salud cubre hospitalización.\n"
            "La póliza médica incluye internación en clínica.\n"
            "El equipo ganó el partido de fútbol anoche.",
            height=140,
        )
        embs = st.multiselect("Esquemas de embedding", list(EMBEDDINGS), default=["TF-IDF", "Sentence-Transformers (multilingüe)"])
        tok_name = st.selectbox("Tokenización (BoW / TF-IDF)", list(TOKENIZERS))
        if st.button("Calcular similitud coseno"):
            sentences = [s.strip() for s in raw.splitlines() if s.strip()]
            if len(sentences) < 2:
                st.warning("Escribe al menos dos frases.")
            else:
                for name in embs:
                    with st.expander(name, expanded=True), st.spinner(f"Calculando {name}..."):
                        try:
                            df = compare_sentences(sentences, name, tok_name)
                            st.dataframe(df.style.background_gradient(cmap="Blues", vmin=-1, vmax=1).format("{:.3f}"))
                        except Exception as e:  # p. ej. sin internet para descargar el modelo
                            st.error(f"{type(e).__name__}: {e}")
                st.markdown("**Frases:** " + " | ".join(f"F{i + 1}: {s}" for i, s in enumerate(sentences)))

    # ---- Chat --------------------------------------------------------
    with tab_chat:
        st.session_state.setdefault("messages", [])
        for m in st.session_state["messages"]:
            with st.chat_message(m["role"]):
                st.markdown(m["content"])

        if prompt := st.chat_input("Escribe tu mensaje"):
            st.session_state["messages"].append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                try:
                    msgs = [{"role": "system", "content": system_prompt}] + st.session_state["messages"]
                    reply = st.write_stream(groq_stream(msgs, api_key, model, temperature))
                    st.session_state["messages"].append({"role": "assistant", "content": reply})
                except Exception as e:
                    st.error(f"{type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
