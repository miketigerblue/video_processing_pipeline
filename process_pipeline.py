import os
import shutil
import logging
import subprocess
from datetime import datetime
import whisper
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook  # Import ProgressHook

from langchain.chains import LLMChain
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate

# Setup logging
logging.basicConfig(filename="pipeline.log", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Directories
INPUT_DIR = "input_videos"
OUTPUT_DIR = "output_transcripts"
SENTIMENT_DIR = "output_sentiment"

os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(SENTIMENT_DIR, exist_ok=True)

# Initialize Whisper model for transcription
logging.info("Loading Whisper model for transcription.")
whisper_model = whisper.load_model("base")  # Adjust model size as needed

# Initialize PyAnnote model for speaker diarization with authentication
logging.info("Loading PyAnnote model for speaker diarization.")
diarization_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")

def transcribe_video_with_diarization(video_path):
    # Convert video to audio in .wav format with a universal approach
    audio_path = os.path.splitext(video_path)[0] + ".wav"  # Ensures the file name works for both .mp4 and .m4v
    try:
        # Extract audio using ffmpeg with -y flag to auto-overwrite
        subprocess.run(f'ffmpeg -y -i "{video_path}" -ac 1 -ar 16000 "{audio_path}"', shell=True, check=True)
    except subprocess.CalledProcessError:
        logging.error(f"Failed to extract audio from {video_path}.")
        return ""

    # Step 1: Diarize the audio to detect different speakers
    logging.info(f"Diarizing audio: {audio_path}")
        # Adding progress hook
    with ProgressHook() as hook:
        diarization = diarization_pipeline(audio_path, hook=hook)
   
    
    # Step 2: Transcribe each segment separately with Whisper and label speakers
    transcription = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        start, end = turn.start, turn.end
        # Extract and transcribe the segment
        segment_audio_path = f"{audio_path}_{speaker}.wav"
        os.system(f'ffmpeg -y -i "{audio_path}" -ss {start} -to {end} -c copy "{segment_audio_path}"')
        
        logging.info(f"Transcribing segment for Speaker {speaker} from {start} to {end}.")
        result = whisper_model.transcribe(segment_audio_path)
        transcription.append(f'Speaker {speaker}: {result["text"]}')
        
        # Clean up temporary segment file
        os.remove(segment_audio_path)
    
    # Clean up main audio file
    os.remove(audio_path)
    
    # Combine the transcriptions for all segments
    return "\n".join(transcription)

def analyze_sentiment_and_themes(text, api_key):
    logging.info("Initializing OLAMA model for sentiment and theme extraction.")
    template = PromptTemplate.from_template("Analyze the following text for sentiment and themes: {text}")
    
    chain = LLMChain(llm=OpenAI(openai_api_key=api_key), prompt=template)
    
    max_chunk_length = 3000
    chunks = [text[i:i + max_chunk_length] for i in range(0, len(text), max_chunk_length)]

    results = []
    for chunk in chunks:
        logging.info("Analyzing chunk for sentiment and themes.")
        output = chain.run(text=chunk)
        results.append(output)

    return "\n".join(results)

def save_results(video_name, transcript, sentiment_analysis):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    transcript_path = os.path.join(OUTPUT_DIR, f"{video_name}_transcript_{timestamp}.txt")
    sentiment_path = os.path.join(SENTIMENT_DIR, f"{video_name}_sentiment_{timestamp}.txt")

    with open(transcript_path, "w") as f:
        f.write(transcript)

    with open(sentiment_path, "w") as f:
        f.write(sentiment_analysis)

    logging.info(f"Saved transcript to {transcript_path}")
    logging.info(f"Saved sentiment analysis to {sentiment_path}")

def process_pipeline():
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("The OpenAI API key must be set in the environment variable OPENAI_API_KEY.")

    for video_file in os.listdir(INPUT_DIR):
        if video_file.endswith((".mp4", ".m4v")):
            video_path = os.path.join(INPUT_DIR, video_file)
            video_name = os.path.splitext(video_file)[0]
            logging.info(f"Starting processing pipeline for {video_file}")

            # Stage 1: Transcription with Speaker Diarization
            transcript = transcribe_video_with_diarization(video_path)
            logging.info(f"Transcription and speaker diarization completed for {video_file}")

            # Stage 2: Sentiment and Theme Extraction
            sentiment_analysis = analyze_sentiment_and_themes(transcript, api_key)
            logging.info(f"Sentiment analysis completed for {video_file}")

            # Save results
            save_results(video_name, transcript, sentiment_analysis)
            logging.info(f"Processing pipeline completed for {video_file}")

            # Move processed file
            processed_dir = os.path.join("processed_videos", video_name)
            os.makedirs(processed_dir, exist_ok=True)
            shutil.move(video_path, processed_dir)

if __name__ == "__main__":
    process_pipeline()
    logging.info("All files in input bucket processed.")
