import streamlit as st
import openai
import os
import pandas as pd
import matplotlib.pyplot as plt
from langchain_community import OpenAI
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
    def transcribe_audio_tool(file_path):
        try:
            with open(file_path, "rb") as audio:
                response = openai.audio.transcriptions.create(model="whisper-1", file=audio)
            return response.text
        except Exception as e:
            return f"Error in audio transcription: {e}"

    # Tool: Generate Feedback
    def generate_feedback_tool(interview_text, job_description, company_name):
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
            response = openai.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2000,
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating feedback: {e}"

    # Tools for the Agent
    tools = [
        Tool(
            name="Transcribe Audio",
            func=lambda q: transcribe_audio_tool(q) if isinstance(q, str) else "Invalid input.",
            description="Converts uploaded audio files into text.",
        ),
        Tool(
            name="Generate Feedback",
            func=lambda q: generate_feedback_tool(q['interview_text'], q['job_description'], q['company_name']),
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

            # Use the Agent for Transcription
            with st.spinner("Transcribing audio..."):
                transcription_result = agent.run({"input": audio_file_path})
                st.success("Transcription completed!")

            # Check transcription validity
            if "Error" in transcription_result:
                st.error(transcription_result)
            else:
                st.info("Generating feedback...")
                feedback_request = {
                    "interview_text": transcription_result,
                    "job_description": job_description,
                    "company_name": company_name,
                }

                # Use the Agent for Feedback Generation
                feedback_result = agent.run({"input": feedback_request})
                st.success("Feedback generated!")

                # Display Results
                tab1, tab2 = st.tabs(["Feedback", "Score Analysis"])

                with tab1:
                    st.subheader("Interview Feedback")
                    st.write(feedback_result)

                with tab2:
                    # Extract Scores and Plot
                    st.subheader("Score Breakdown")
                    scores = {}
                    try:
                        for criterion in ["Alignment", "Clarity", "Strength", "Overall"]:
                            score_line = next((line for line in feedback_result.split("\n") if criterion in line), None)
                            if score_line:
                                score = int(score_line.split(":")[1].split("/")[0].strip())
                                scores[criterion] = score
                            else:
                                scores[criterion] = None

                        # Display Metrics
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Alignment Score", f"{scores.get('Alignment', 'N/A')}/100")
                            st.metric("Clarity Score", f"{scores.get('Clarity', 'N/A')}/100")
                        with col2:
                            st.metric("Strength Score", f"{scores.get('Strength', 'N/A')}/100")
                            st.metric("Overall Score", f"{scores.get('Overall', 'N/A')}/100")

                        # Plot Pie Chart
                        if all(scores[criterion] is not None for criterion in scores):
                            fig, ax = plt.subplots()
                            ax.pie(
                                list(scores.values()),
                                labels=scores.keys(),
                                autopct='%1.1f%%',
                                startangle=90,
                                colors=['#66b3ff', '#99ff99', '#ffcc99', '#ff9999'],
                            )
                            ax.axis('equal')
                            st.pyplot(fig)
                        else:
                            st.warning("Some scores are missing. Pie chart visualization is not possible.")
                    except Exception as e:
                        st.error(f"Could not extract scores: {e}")
