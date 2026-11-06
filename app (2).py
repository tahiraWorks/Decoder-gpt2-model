import html
import pandas as pd
import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="Monolingual GPT-2 Generator", page_icon="✍️", layout="wide")

GPT_ID = "TahiraAbbas/gpt2-wikitext"
GPT_URL = f"https://huggingface.co/{GPT_ID}"

st.markdown("""
<style>
.block-container {padding-top: 1.2rem; max-width: 1100px;}
.hero {background: linear-gradient(120deg, #1c1917 0%, #b45309 60%, #f59e0b 130%);
    padding: 2rem 2.2rem; border-radius: 22px; color: #ffffff; margin-bottom: 1rem;
    box-shadow: 0 12px 32px rgba(180,83,9,0.35);}
.hero h1 {margin: 0; font-size: 2.3rem; color: #ffffff;}
.hero p {margin: 0.4rem 0 0 0; font-size: 1.05rem; color: #fef3c7;}
.pill {display: inline-block; padding: 0.25rem 0.8rem; margin: 0.8rem 0.4rem 0 0;
    border-radius: 999px; background: rgba(255,255,255,0.18); font-size: 0.85rem; color: #ffffff;}
.mcard {border-radius: 16px; padding: 1.1rem 1.3rem; border: 1px solid rgba(128,128,128,0.25);
    background: rgba(128,128,128,0.06); border-top: 6px solid #f59e0b; line-height: 1.7;}
.gen {border-left: 5px solid #f59e0b; border-radius: 12px; padding: 1rem 1.2rem;
    background: rgba(245,158,11,0.09); line-height: 1.65; margin-bottom: 0.7rem;}
.gen .prompt {background: rgba(245,158,11,0.3); padding: 0 4px; border-radius: 4px; font-weight: 700;}
</style>
""", unsafe_allow_html=True)

@st.cache_resource(max_entries=1)
def load_model():
    return pipeline("text-generation", model=GPT_ID)

st.session_state.setdefault("gpt_prompt", "Artificial intelligence will")
st.session_state.setdefault("gpt_history", [])

with st.sidebar:
    st.header("Model")
    st.markdown(f"""
**✍️ GPT-2** (decoder-only)

- Base: `gpt2` (English only)
- **Dataset:** WikiText-2 (Wikipedia)
- [Open on Hugging Face]({GPT_URL})
""")
    st.caption("The model runs on CPU, so the first request can take a while.")
    if st.button("Clear history", use_container_width=True):
        st.session_state["gpt_history"] = []
        st.rerun()

st.markdown("""
<div class="hero">
  <h1>✍️ Monolingual GPT-2: Text Generation</h1>
  <p>A decoder-only Transformer that predicts the next token and continues your text.</p>
  <span class="pill">Decoder-only</span><span class="pill">Monolingual (English)</span>
  <span class="pill">WikiText-2 dataset</span><span class="pill">Hugging Face</span>
</div>
""", unsafe_allow_html=True)

tab_run, tab_info = st.tabs(["✍️ Generate", "📚 Model & Dataset"])

with tab_run:
    st.caption("Fine-tuned on WikiText-2. GPT-2 is monolingual: it works with English only.")
    st.write("**Try a prompt:**")
    pr = st.columns(4)
    prompts = ["Artificial intelligence will", "The history of", "Once upon a time", "In the future,"]
    for col, p in zip(pr, prompts):
        if col.button(p, key="pr_" + p, use_container_width=True):
            st.session_state["gpt_prompt"] = p

    prompt = st.text_input("Start of a sentence", key="gpt_prompt")
    c1, c2, c3, c4 = st.columns(4)
    max_new = c1.slider("New tokens", 10, 100, 40)
    temp = c2.slider("Temperature", 0.1, 1.5, 0.8)
    top_k = c3.slider("Top-k", 1, 100, 50)
    n_out = c4.slider("Variations", 1, 3, 2)
    st.caption("Higher temperature = more creative. Lower = more predictable.")

    if st.button("Generate", type="primary"):
        if not prompt.strip():
            st.warning("Please enter a prompt.")
        else:
            with st.spinner("Loading GPT-2 and generating..."):
                gen = load_model()
                outs = gen(prompt, max_new_tokens=max_new, do_sample=True, temperature=temp,
                           top_k=top_k, num_return_sequences=n_out, pad_token_id=50256)
            st.markdown("#### Generated text")
            for i, o in enumerate(outs, 1):
                full = o["generated_text"]
                cont = full[len(prompt):] if full.startswith(prompt) else full
                st.markdown(
                    f"<div class='gen'><b>Version {i}</b><br>"
                    f"<span class='prompt'>{html.escape(prompt)}</span>{html.escape(cont)}</div>",
                    unsafe_allow_html=True)
                st.session_state["gpt_history"].insert(0, {"Prompt": prompt, "Generated": full})
            st.info("Press Generate again: the output changes because GPT-2 samples the next token.")

    if st.session_state["gpt_history"]:
        st.markdown("#### History")
        st.dataframe(pd.DataFrame(st.session_state["gpt_history"][:10]), use_container_width=True, hide_index=True)

with tab_info:
    st.markdown(f"""
<div class="mcard">
<h3>✍️ GPT-2</h3>
<b>Architecture:</b> decoder-only (causal attention)<br>
<b>Base model:</b> gpt2 (English only)<br>
<b>Fine-tuned model:</b> <a href="{GPT_URL}" target="_blank">{GPT_ID}</a><br>
<b>Task:</b> text generation<br>
<b>Dataset:</b> WikiText-2 (Wikipedia articles)<br>
<b>Why this dataset:</b> text generation learns from plain text and needs no labels.<br>
<b>Pre-training:</b> next-token prediction<br>
<b>Best for:</b> text generation, completion
</div>""", unsafe_allow_html=True)
    st.success("The decoder generates text one token at a time, using only the previous tokens.")
