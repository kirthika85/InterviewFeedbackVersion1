import streamlit as st
import openai
import os
import pandas as pd
import matplotlib.pyplot as plt
from langchain.llms import OpenAI
from langchain.agents import initialize_agent, Tool
from langchain.prompts import PromptTemplate
from langchain.chains.conversation.memory import ConversationBufferMemory

# Streamlit Title
st.title("Interview Feedback AI Agent")

# Sidebar: OpenAI API Key
with st.sidebar:
    st.header("Settings")
    openai_api_key = st.text_input("Enter your OpenAI API Key", type="password")

if not openai_api_key:
    st.warning("Please enter your OpenAI API key to proceed.")
else:
    # Initialize OpenAI Model
    llm = OpenAI(openai_api_key=openai_api_key, temperature=0.7, model="gpt-4")

    # Conversation Memory
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # Tool: Transcribe Audio
    def transcribe_audio(file_path):
        try:
            with open(file_path, "rb") as audio:
                response = openai.audio.transcriptions.create(
                        model="whisper-1",
                        file=audio
                )
            return response.text
        except Exception as e:
            return f"Error in audio transcription: {e}"

    # Tool: Generate Feedback
    def generate_feedback(interview_text, job_description, company_name):
        prompt = f"""
        You are an expert interviewer and career coach. Analyze the candidate's interview performance for the position at {company_name}.
        The interview text provided is as follows:

        {interview_text}

        Evaluate the following criteria on a scale of 0 to 100:
        1. Alignment with job description ({job_description})
        2. Clarity of communication and confidence.
        3. Strength of their responses to key questions.
        4. Areas of improvement with actionable advice.
        5. Overall assessment and a final score out of 100.

        Format the output as:
        - Alignment Score: [Score/100]
        - Clarity Score: [Score/100]
        - Strength Score: [Score/100]
        - Overall Score: [Score/100]
        - Areas of Improvement: [Explanation]
        """
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2000,
            )
            return response["choices"][0]["message"]["content"]
        except Exception as e:
            return f"Error generating feedback: {e}"

    # Tools for the Agent
    tools = [
        Tool(
            name="Transcribe Audio",
            func=transcribe_audio,
            description="Converts uploaded audio files into text.",
        ),
        Tool(
            name="Generate Feedback",
            func=lambda inputs: generate_feedback(
                inputs["interview_text"], inputs["job_description"], inputs["company_name"]
            ),
            description="Analyzes interview transcription and provides detailed feedback.",
        ),
    ]

    # Initialize the Agent
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent="zero-shot-react-description",
        memory=memory,
        verbose=True,
    )

    # Streamlit: Inputs
    with st.expander("Provide Job Details", expanded=True):
        job_description = st.text_area("Enter the Job Description", height=200)
        company_name = st.text_input("Enter the Company Name")

    with st.expander("Upload Interview Recording", expanded=True):
        uploaded_audio = st.file_uploader("Upload your audio file (mp3, wav, etc.)", type=["mp3", "wav", "ogg"])

    if st.button("Start Analysis"):
        if not all([openai_api_key, uploaded_audio, job_description, company_name]):
            st.warning("Please complete all the required fields.")
        else:
            # Save uploaded audio file
            audio_file_path = "uploaded_audio.mp3"
            with open(audio_file_path, "wb") as f:
                f.write(uploaded_audio.read())

            # Transcription
            with st.spinner("Transcribing audio..."):
                transcription_result = transcribe_audio(audio_file_path)
                if "Error" in transcription_result:
                    st.error(transcription_result)
                    os.remove(audio_file_path)
                    st.stop()

            # Feedback Generation
            with st.spinner("Generating feedback..."):
                feedback_result = generate_feedback(
                    transcription_result, job_description, company_name
                )
                if "Error" in feedback_result:
                    st.error(feedback_result)
                    os.remove(audio_file_path)
                    st.stop()

            # Display Results
            st.success("Analysis completed!")
            st.subheader("Feedback")
            st.write(feedback_result)

            os.remove(audio_file_path)
