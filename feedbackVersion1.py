import streamlit as st
import openai
import os
import pandas as pd  # For preparing data for the bar chart

# Streamlit App Title
st.title("Interview Feedback Generator")

# Sidebar: OpenAI API Key
st.sidebar.header("Settings")
openai_api_key = st.sidebar.text_input("Enter your OpenAI API Key", type="password")

if openai_api_key:
    openai.api_key = openai_api_key
else:
    st.warning("Please enter your OpenAI API key in the sidebar to proceed.")

# Inputs for Job Description and Company
st.subheader("Job Details")
job_description = st.text_area("Enter the Job Description", height=200)
company_name = st.text_input("Enter the Company Name")

# Upload Interview Recording
st.subheader("Upload Interview Recording")
uploaded_audio = st.file_uploader("Upload your audio file (mp3, wav, etc.)", type=["mp3", "wav", "ogg"])

# Function: Transcribe Audio
def transcribe_audio(file_path):
    with open(file_path, "rb") as audio:
        response = openai.audio.transcriptions.create(
            model="whisper-1",
            file=audio
        )
    return response.text

# Function: Generate Feedback
def generate_feedback(interview_text, job_description, company_name):
    prompt = f"""
    You are an expert interviewer and career coach. Analyze the candidate's interview performance for the position at {company_name}. 
    Evaluate the following criteria on a scale of 0 to 100:
    1. Alignment of the candidate's response with the job description: ({job_description})
    2. Clarity of communication and confidence.
    3. Strength of their responses to key questions.
    4. Areas of improvement with actionable advice.
    5. Overall assessment and a final score out of 100.

    For each criterion, provide a detailed explanation and the individual score in the following format:
    - Alignment Score: [Score/100]
    - Clarity Score: [Score/100]
    - Strength Score: [Score/100]
    - Overall Score: [Score/100]
    - Areas of Improvement: [Explanation]
    """
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are an expert at analyzing interviews and providing thoughtful feedback."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=1500
    )
    return response.choices[0].message.content

# Main Workflow
if uploaded_audio and job_description and company_name and openai_api_key:
    try:
        # Save uploaded audio to a file
        audio_file_path = "uploaded_audio.mp3"
        with open(audio_file_path, "wb") as f:
            f.write(uploaded_audio.read())

        # Transcribe the audio
        st.info("Transcribing audio...")
        transcribed_text = transcribe_audio(audio_file_path)
        st.success("Transcription completed!")

        # Display transcribed text
        st.subheader("Transcribed Interview Text")
        st.text_area("Interview Text", transcribed_text, height=200)

        # Generate feedback
        st.info("Generating feedback based on the interview and job description...")
        feedback = generate_feedback(transcribed_text, job_description, company_name)
        st.success("Feedback generated!")

        # Display feedback
        st.subheader("Interview Feedback")
        st.write(feedback)

        # Extract individual scores and display as metrics
        try:
            alignment_score = int(feedback.split("Alignment Score:")[1].split("/")[0].strip())
            clarity_score = int(feedback.split("Clarity Score:")[1].split("/")[0].strip())
            strength_score = int(feedback.split("Strength Score:")[1].split("/")[0].strip())
            overall_score = int(feedback.split("Overall Score:")[1].split("/")[0].strip())

            # Display scores as metrics
            st.metric("Alignment Score", f"{alignment_score}/100")
            st.metric("Clarity Score", f"{clarity_score}/100")
            st.metric("Strength Score", f"{strength_score}/100")
            st.metric("Overall Score", f"{overall_score}/100")

            # Prepare data for the bar chart
            scores = {
                "Alignment": alignment_score,
                "Clarity": clarity_score,
                "Strength": strength_score,
                "Overall": overall_score,
            }

            # Display bar chart
            st.subheader("Score Breakdown")
            st.bar_chart(pd.DataFrame(scores.values(), index=scores.keys(), columns=["Score"]))

            # Gamification based on overall score
            if overall_score > 80:
                st.balloons()
                st.success("Great Job! You're a strong candidate.")
            elif overall_score > 50:
                st.info("Good effort! Keep improving.")
            else:
                st.warning("Needs Improvement. Focus on the provided feedback.")

        except Exception as e:
            st.warning("Score not available in feedback. Please check the feedback format.")

        # Cleanup
        os.remove(audio_file_path)

    except Exception as e:
        st.error(f"Error occurred: {e}")

elif not (job_description and company_name):
    st.warning("Please enter the job description and company name.")

elif not openai_api_key:
    st.warning("Please enter your OpenAI API key.")

else:
    st.info("Upload an audio file to get started.")
