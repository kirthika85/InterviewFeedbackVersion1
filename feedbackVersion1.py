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
    try: 
        with open(file_path, "rb") as audio:
            response = openai.audio.transcriptions.create(
                model="whisper-1",
                file=audio
            )
        return response.text
    except Exception as e:
        st.error(f"Error in audio transcription: {e}")
        return None

# Function: Generate Feedback
def generate_feedback(interview_text, job_description, company_name):
    prompt = f"""
    You are an expert interviewer and career coach. Analyze the candidate's interview performance for the position at {company_name}. 
    The interview text provided is as follows:

    {interview_text}

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
    try:
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "system", "content": "You are an expert at analyzing interviews and providing thoughtful feedback."},
                      {"role": "user", "content": prompt}],
            max_tokens=2000
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error generating feedback: {e}")
        return None

# Function to check if the text seems to be an interview
def is_interview(text):
    # Define keywords that are common in interviews
    interview_keywords = ["interview", "job", "role", "position", "hiring", "candidate", "interviewed","question"]
    text_lower = text.lower()

    # Check if any of the keywords are found in the transcribed text
    for keyword in interview_keywords:
        if keyword in text_lower:
            return True
    return False
    
# Main Workflow
if uploaded_audio and job_description and company_name and openai_api_key:
    # Step 1: Save uploaded audio file
    audio_file_path = "uploaded_audio.mp3"
    with open(audio_file_path, "wb") as f:
        f.write(uploaded_audio.read())
    
    # Step 2: Transcribe audio
    st.info("Transcribing audio...")
    transcribed_text = transcribe_audio(audio_file_path)
    if not transcribed_text:
        st.warning("Transcription failed. Please check your audio file.")
        os.remove(audio_file_path)
        st.stop()
    st.success("Transcription completed!")

    if is_interview(transcribed_text):
       # Display transcribed text
        st.subheader("Transcribed Interview Text")
        st.text_area("Interview Text", transcribed_text, height=300)
    
        # Step 3: Generate feedback
        st.info("Generating feedback based on the interview and job description...")
        feedback = generate_feedback(transcribed_text, job_description, company_name)
        if not feedback:
            st.warning("Feedback generation failed. Please try again.")
            os.remove(audio_file_path)
            st.stop()
            st.success("Feedback generated!")
    
        # Display feedback
        st.subheader("Interview Feedback")
        st.write(feedback)
    
        # Step 4: Extract and display scores
        scores = {}
        try:
            for criterion in ["Alignment", "Clarity", "Strength", "Overall"]:
                score_line = next(line for line in feedback.split("\n") if criterion in line)
                score = int(score_line.split(":")[1].split("/")[0].strip())
                scores[criterion] = score
        
            # Display metrics
            st.metric("Alignment Score", f"{scores.get('Alignment', 'N/A')}/100")
            st.metric("Clarity Score", f"{scores.get('Clarity', 'N/A')}/100")
            st.metric("Strength Score", f"{scores.get('Strength', 'N/A')}/100")
            st.metric("Overall Score", f"{scores.get('Overall', 'N/A')}/100")

            # Display bar chart
            st.subheader("Score Breakdown")
            st.bar_chart(pd.DataFrame(scores.values(), index=scores.keys(), columns=["Score"]))
        
            # Gamification
            if scores["Overall"] > 80:
                st.balloons()
                st.success("Great Job! You're a strong candidate.")
            elif scores["Overall"] > 50:
                st.info("Good effort! Keep improving.")
            else:
                st.warning("Needs Improvement. Focus on the provided feedback.")
    
        except Exception as e:
            st.warning("Could not extract scores from feedback. Please check the feedback format.")
            st.error(f"Error: {e}")
    else:
        st.warning("The uploaded audio doesn't seem to be an interview. Please upload a relevant interview file.")

    # Cleanup
    os.remove(audio_file_path)
    
elif not (job_description and company_name):
    st.warning("Please enter the job description and company name.")
    
elif not openai_api_key:
    st.warning("Please enter your OpenAI API key.")
    
else:
    st.info("Upload an audio file to get started.")
