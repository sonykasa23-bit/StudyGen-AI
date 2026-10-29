
import os
import streamlit as st
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER_PATH = os.path.join(
    os.path.dirname(__file__),
    "model_files"
)

st.set_page_config(
    page_title="StudyGen AI",
    page_icon="📚",
    layout="centered"
)

st.title("📚 StudyGen AI")
st.caption("Your AI assistant for college studies")
st.write(
    "Ask questions about AIML, Python, DBMS, operating systems, "
    "software engineering, and other subjects."
)

if not os.path.isfile(
    os.path.join(ADAPTER_PATH, "adapter_config.json")
) or not os.path.isfile(
    os.path.join(ADAPTER_PATH, "adapter_model.safetensors")
):
    st.error(
        "Model adapter files are missing. Check the model_files folder."
    )
    st.stop()


@st.cache_resource
def load_chatbot():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    if torch.cuda.is_available():
        config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        )

        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            quantization_config=config,
            device_map="auto"
        )
    else:
        # CPU fallback: full precision can require substantial RAM.
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        )

    chat_model = PeftModel.from_pretrained(
        base_model,
        ADAPTER_PATH
    )
    chat_model.eval()

    return tokenizer, chat_model


try:
    with st.spinner("Loading StudyGen AI..."):
        tokenizer, model = load_chatbot()
except Exception as exc:
    st.error("Could not load the model.")
    st.exception(exc)
    st.stop()


def generate_answer(question):
    messages = [
        {
            "role": "system",
            "content": (
                "You are StudyGen AI, a college study assistant. "
                "Explain academic concepts in simple English. "
                "Use numbered steps and examples when useful. "
                "For exam questions, give a clear, structured answer. "
                "Do not invent facts when unsure."
            )
        },
        {"role": "user", "content": question}
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    )

    device = next(model.parameters()).device
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=300,
            do_sample=False,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    new_tokens = output[0][inputs["input_ids"].shape[1]:]

    return tokenizer.decode(
        new_tokens,
        skip_special_tokens=True
    ).strip()


if "history" not in st.session_state:
    st.session_state.history = []

for message in st.session_state.history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input("Ask a college study question...")

if question:
    st.session_state.history.append(
        {"role": "user", "content": question}
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                answer = generate_answer(question)
                st.markdown(answer)
            except Exception as exc:
                answer = "Sorry, I couldn't generate an answer."
                st.error(str(exc))

    st.session_state.history.append(
        {"role": "assistant", "content": answer}
    )
