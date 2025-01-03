import time
import streamlit as st
import openai
import matplotlib.pyplot as plt
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, Tool
from langchain.chains.conversation.memory import ConversationBufferMemory
import re

# Streamlit Title
st.title("Interview Feedback Generator")

# Sidebar: OpenAI API Key
with st.sidebar:
    st.header("Settings")
    openai_api_key = st.text_input("Enter your OpenAI API Key", type="password")

if not openai_api_key:
    st.warning("Please enter your OpenAI API key to proceed.")
else:
    openai.api_key = openai_api_key
    # Initialize OpenAI Model
    llm = ChatOpenAI(openai_api_key=openai_api_key, temperature=0.7, model="gpt-4")

    # Conversation Memory
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # Tool: Transcribe Audio
    def transcribe_audio(file_object):
        try:
            response = openai.audio.transcriptions.create(
                model="whisper-1",
                file=file_object,
            )
            return response.text
        except Exception as e:
            return f"Error in audio transcription: {e}"

    # Tool: Classify Text as Interview
    def is_interview(text):
        prompt = f"""
        You are a classifier. Analyze the following text and determine if it is likely from an interview conversation. 
        Respond with "Yes" if it is an interview, and "No" otherwise.

        Text:
        {text}
        """
        try:
            response = openai.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10,
            )
            return response.choices[0].message.content.strip().lower() == "yes"
        except Exception as e:
            return False

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
            func=transcribe_audio,
            description="Converts an audio file into text transcription. Accepts a file object.",
        ),
        Tool(
            name="Classify Text",
            func=is_interview,
            description="Determines if the transcribed text represents an interview conversation.",
        ),
        Tool(
            name="Generate Feedback",
            func=lambda text: generate_feedback(*text.split('|')),  # Keep lambda for parsing input
            description=(
                "Generate feedback for interview transcription. "
                "Input must be a string with the format: 'interview_text|job_description|company_name'."
            ),
        ),
    ]

    # Initialize the Agent with zero-shot-react-description
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent_type="zero-shot-react-description",
        memory=memory,
        verbose=True,
    )

    # Streamlit: Inputs
    with st.expander("Provide Job Details", expanded=True):
        job_description = st.text_area("Enter the Job Description", height=200)
        company_name = st.text_input("Enter the Company Name")

    with st.expander("Upload Interview Recording", expanded=True):
        uploaded_audio = st.file_uploader("Upload your audio file (mp3, wav, etc.)", type=["mp3", "wav", "ogg"])

    # Start Analysis Section
    if st.button("Start Analysis"):
        if not uploaded_audio:
            st.warning("Please upload an audio file.")
        else:
            # Log the upload
            st.write("Audio file uploaded successfully.")

            # Define the query for the agent
            input_data = f"""
            Analyze the uploaded audio file for interview feedback:
            1. Transcribe the provided audio file (File: {uploaded_audio.name}).
            2. Determine if the transcription represents an interview conversation.
            3. If it is an interview, generate detailed feedback based on the job description:
               - Job Description: {job_description}
               - Company Name: {company_name}
            4. Provide feedback and scores, or indicate if it is not an interview.
            """

            # Run the agent with the uploaded file object
            result = agent.run(input=input_data)

            # Display the agent result
            st.write("Agent Result:", result)

            # Display results in tabs
            tab1, tab2 = st.tabs(["Feedback Analysis", "Score Analysis"])

            with tab1:
                st.subheader("Feedback Analysis")
                st.write(result)

                # Extract and display scores
                scores = {}
                score_pattern = re.compile(r"(\w+)\s*Score:\s*(\d+)\s*/\s*100")
                matches = score_pattern.findall(result)

                if matches:
                    scores = {match[0]: int(match[1]) for match in matches}
                    for criterion, score in scores.items():
                        st.write(f"{criterion} Score: {score}/100")

            with tab2:
                st.subheader("Score Analysis")

                if scores:
                    # Ensure no score is zero before plotting
                    if all(value > 0 for value in scores.values()):
                        fig, ax = plt.subplots()
                        ax.pie(
                            scores.values(),
                            labels=scores.keys(),
                            autopct='%1.1f%%',
                            startangle=90,
                            colors=['#66b3ff', '#99ff99', '#ffcc99', '#ff9999'],
                        )
                        ax.axis('equal')
                        st.pyplot(fig)
                    else:
                        st.warning("Some scores are missing or zero. Pie chart visualization is not possible.")
                else:
                    st.warning("No scores were detected in the feedback.")
