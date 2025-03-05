import streamlit as st

def clear_text():
    st.session_state["input"] = ""

def question_and_answer_app():
    st.title("Question and Answer App")

    question = st.text_input("Enter your question:",  key="input")
    answer_placeholder = st.empty()

    col1, col2 = st.columns(2)

    if col1.button("Submit"):
        reversed_question = question[::-1]
        answer_placeholder.text("Answer: " + reversed_question)

    if col2.button("Clear", on_click=clear_text):
        answer_placeholder.empty()

if __name__ == "__main__":
    question_and_answer_app()