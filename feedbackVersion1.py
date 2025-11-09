import io
import re

import matplotlib.pyplot as plt
import openai
import streamlit as st
from langchain.agents import create_tool_calling_agent
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import Tool
from langgraph.executors import AgentExecutor

# ---------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------
st.title("Interview Feedback Generator")

with st.sidebar:
    st.header("Settings")
    openai_api_key = st.text_input("Enter your OpenAI API Key", type="password")

if not openai_api_key:
    st.warning("Please enter your OpenAI API key to proceed.")
    st.stop()

openai.api_key = openai_api_key

# ---------------------------------------------------------------------
# LangChain + OpenAI setup
# ---------------------------------------------------------------------
llm = ChatOpenAI(openai_api_key=openai_api_key, temperature=0.7, model="gpt-4")

prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            content=(
                "You are an interview-feedback assistant. "
                "Use the available tools to transcribe audio, decide if a conversation "
                "is an interview, and generate structured feedback with scores."
            )
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

# ---------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------
def transcribe_audio(file_object):
    try:
        if isinstance(file_object, io.BytesIO):
            file_bytes = file_object.getvalue()
        else:
            file_bytes = file_object.read()

        file_io = io.BytesIO(file_bytes)
        file_io.name = "audio.mp3"

        response = openai.audio.transcriptions.create(
            model="whisper-1",
            file=file_io,
        )
        st.write("Transcription successful.")
        return response.text
    except Exception as e:
        st.write(f"Error in audio transcription: {e}")
        return f"Error in audio transcription: {e}"


def tool_transcribe(_: str) -> str:
    file_obj = st.session_state.get("uploaded_audio")
    if file_obj is None:
        return "Error: No audio file uploaded."
    return transcribe_audio(file_obj)


def is_interview(text: str) -> str:
    prompt_local = f"""
    You are a classifier. Analyze the following text and determine if it is likely
    from an interview conversation. Respond with "Yes" if it is an interview, and "No" otherwise.

    Text:
    {text}
    """
    try:
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt_local}],
            max_tokens=10,
        )
        result = response.choices[0].message.content.strip().lower()
        return "yes" if result == "yes" else "no"
    except Exception as e:
        st.write(f"Error in classification: {e}")
        return "error"


def tool_generate_feedback(user_input: str) -> str:
    try:
        interview_text, job_desc, company = user_input.split("|", maxsplit=2)
    except ValueError:
        return (
            "Invalid input format. Expected "
            "'interview_text|job_description|company_name'."
        )

    st.write("Generating feedback based on the interview text...")
    prompt_local = f"""
    You are an expert interviewer and career coach. Analyze the candidate's interview performance for the position at {company}.
    The interview text provided is as follows:

    {interview_text}

    Evaluate the following criteria on a scale of 0 to 100:
    1. Alignment with job description ({job_desc})
    2. Clarity of communication and confidence.
    3. Strength of responses to key questions.
    4. Technical competence
    5. Cultural fit with {company}
    6. Overall assessment

    For each criterion, provide detailed qualitative feedback without mentioning scores. Use this structure:
    - Alignment: [Feedback]
    - Clarity: [Feedback]
    - Strength: [Feedback]
    - Technical Competence: [Feedback]
    - Cultural Fit: [Feedback]
    - Overall: [Feedback]

    Areas of Improvement:[Provide actionable advice for improving performance in each criterion that scored below 70.]

    Key Strengths:[Summarize the standout qualities of the candidate.]

    Provide Scores for the criteria in this structure:
    - Alignment Score: [Score/100]
    - Clarity Score: [Score/100]
    - Strength Score: [Score/100]
    - Technical Competence Score: [Score/100]
    - Cultural Fit Score: [Score/100]
    - Overall Score: [Score/100]

    Ensure that the 'Scores:' section always appears immediately after the 'Key Strengths:' section, maintaining this exact structure and order.
    """
    try:
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt_local}],
            max_tokens=3000,
        )
        st.write("Feedback generation successful.")
        return response.choices[0].message.content
    except Exception as e:
        st.write(f"Error generating feedback: {e}")
        return f"Error generating feedback: {e}"

# ---------------------------------------------------------------------
# LangChain tools + agent executor
# ---------------------------------------------------------------------
tools = [
    Tool(
        name="Transcribe Audio",
        func=tool_transcribe,
        description="Transcribes the uploaded audio file into text.",
    ),
    Tool(
        name="Classify Text",
        func=is_interview,
        description="Returns 'yes' or 'no' indicating if the text looks like an interview.",
    ),
    Tool(
        name="Generate Feedback",
        func=tool_generate_feedback,
        description=(
            "Generate feedback for interview text. "
            "Input format: 'interview_text|job_description|company_name'."
        ),
    ),
]

agent = create_tool_calling_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# ---------------------------------------------------------------------
# Streamlit inputs
# ---------------------------------------------------------------------
with st.expander("Provide Job Details", expanded=True):
    job_description = st.text_area("Enter the Job Description", height=200)
    company_name = st.text_input("Enter the Company Name")

if "uploaded_audio" not in st.session_state:
    st.session_state.uploaded_audio = None

with st.expander("Upload Interview Recording", expanded=True):
    uploaded_audio = st.file_uploader(
        "Upload your audio file (mp3, wav, etc.)", type=["mp3", "wav", "ogg"]
    )
    if uploaded_audio:
        st.session_state.uploaded_audio = uploaded_audio

# ---------------------------------------------------------------------
# Agent run button
# ---------------------------------------------------------------------
if st.button("Start Analysis"):
    if st.session_state.uploaded_audio is None:
        st.warning("Please upload an audio file.")
    else:
        st.write("Audio file uploaded successfully.")

        input_data = f"""
Analyze the uploaded audio file for interview feedback:
1. Transcribe the provided audio file (File: {uploaded_audio.name}).
2. Determine if the transcription represents an interview conversation.
3. If it is an interview, generate detailed feedback based on the job description:
   - Job Description: {job_description}
   - Company Name: {company_name}
4. Provide feedback and scores, or indicate if it is not an interview.
"""
        st.write("Running agent to process the input data...")

        result = executor.invoke(
            {
                "messages": [
                    HumanMessage(content=input_data),
                ]
            }
        )
        st.write("Agent processing complete.")
        agent_output = result["output"]

        tab1, tab2 = st.tabs(["📋 Feedback Analysis", "📊 Score Analysis"])

        # Feedback tab
        with tab1:
            st.subheader("Detailed Feedback Analysis")
            st.markdown("Here is a detailed breakdown of the feedback:")

            feedback_pattern = r"(?s)(.*?Key Strengths:.*?\n)"
            feedback_match = re.search(feedback_pattern, agent_output)
            feedback_content = (
                feedback_match.group(1).strip()
                if feedback_match
                else agent_output
            )
            st.write(feedback_content)

        # Score tab
        with tab2:
            st.subheader("Score Analysis")

            scores = {}
            score_pattern = re.compile(
                r"(\w+(?:\s+\w+)?)\s*Score:\s*(\d+)\s*/\s*100"
            )
            matches = score_pattern.findall(agent_output)

            if matches:
                scores = {match[0]: int(match[1]) for match in matches}
                st.table(
                    {
                        "Criteria": list(scores.keys()),
                        "Score/100": list(scores.values()),
                    }
                )

                if all(value > 0 for value in scores.values()):
                    st.write("Visualizing score distribution...")
                    fig, ax = plt.subplots(figsize=(6, 6))
                    wedges, texts, autotexts = ax.pie(
                        scores.values(),
                        labels=scores.keys(),
                        autopct="%1.1f%%",
                        startangle=90,
                        colors=[
                            "#66b3ff",
                            "#99ff99",
                            "#ffcc99",
                            "#ff9999",
                        ],
                        textprops={"fontsize": 10},
                    )
                    ax.axis("equal")
                    ax.set_title("Performance Score Distribution", fontsize=14)

                    for autotext in autotexts:
                        autotext.set_color("white")
                        autotext.set_fontsize(10)

                    st.pyplot(fig)
                else:
                    st.warning(
                        "Some scores are missing or zero. Pie chart visualization is not possible."
                    )
            else:
                st.warning("No scores were detected in the feedback.")
