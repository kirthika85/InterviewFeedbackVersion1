import streamlit as st
import openai
import os

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
    max_input_length = 3000  # Adjust this based on your token limits
    truncated_interview_text = interview_text[:max_input_length]

    prompt = f"""
    You are an expert interviewer and career coach. Analyze the candidate's interview performance for the position at {company_name}. 
    Consider the following:
    1. How well does the candidate's response align with the job description? ({job_description})
    2. Clarity of communication and confidence.
    3. Strength of their responses to key questions.
    4. Areas of improvement with actionable advice.
    5. Overall assessment and a score out of 100.

    Interview Transcript:
    {interview_text}

    Provide detailed feedback and a gamified score.
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

        # Display feedback and gamified score
        st.subheader("Interview Feedback and Score")
        st.write(feedback)

        # Gamification
        try:
            score = int(feedback.split()[-1])  # Extract score if included at the end
        except ValueError:
            score = None
        
        if score:
            st.metric("Your Score", f"{score}/100")
            if score > 80:
                st.balloons()
                st.success("Great Job! You're a strong candidate.")
            elif score > 50:
                st.info("Good effort! Keep improving.")
            else:
                st.warning("Needs Improvement. Focus on the provided feedback.")
        else:
            st.warning("Score not available in feedback.")

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
