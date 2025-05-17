import streamlit as st
from chatbot_generate import generate_response
st.set_page_config(page_title="TinyLLaMA Chat", page_icon="🦙")
st.title("🦙 Customer ChatBot")

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_user_index" not in st.session_state:
    st.session_state.last_user_index = -1

# Sidebar with hint buttons stacked vertically
st.sidebar.markdown("### 💡 Hint Buttons")
hints = [
    "What does your premium plan include?",
    "Do you offer free trials?",
    "Can I upgrade my plan later?"
]

for hint in hints:
    if st.sidebar.button(hint):
        st.session_state.chat_history.append({"role": "user", "content": hint})

# Main chat area
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

def process_new_user_message():
    last_user_msg_indices = [i for i, m in enumerate(st.session_state.chat_history) if m["role"] == "user"]
    if not last_user_msg_indices:
        return
    last_index = last_user_msg_indices[-1]

    if last_index > st.session_state.last_user_index:
        user_msg = st.session_state.chat_history[last_index]["content"]
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                output = user_msg  # Replace with your model call
            st.markdown(output)
        st.session_state.chat_history.append({"role": "assistant", "content": output})
        st.session_state.last_user_index = last_index

process_new_user_message()

if prompt := st.chat_input("Type your message here..."):
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    process_new_user_message()
