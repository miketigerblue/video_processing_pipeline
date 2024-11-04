
# Video Processing Pipeline

A powerful video processing pipeline that performs speaker diarization, transcription, sentiment analysis, and thematic extraction on video files. This pipeline is built to handle .mp4 and .m4v video formats, extract audio, and process it through advanced AI models for transcription and analysis. Suitable for creating structured summaries, podcasts, or insights from conversations captured on video.

## Features

* Speaker Diarization: Detects and labels different speakers within the audio using the pyannote.audio pipeline.
* Transcription: Transcribes spoken content using OpenAI’s whisper model, providing accurate and detailed transcriptions.
* Sentiment & Theme Analysis: Extracts sentiment and thematic elements from transcriptions using an OpenAI language model.
* Automated File Management: Processes video files in bulk, moves processed files, and organizes transcripts and sentiment analysis outputs.
* Compatibility: Supports both .mp4 and .m4v video files, with universal handling of audio extraction.


## Project Structure


video_processing_pipeline/
├── input_videos/          # Directory for input video files
├── output_transcripts/     # Directory for transcription outputs
├── output_sentiment/       # Directory for sentiment analysis outputs
├── processed_videos/       # Directory for processed video files
├── venv/                   # Virtual environment (not included in repo)
├── pipeline.py             # Main pipeline script
├── README.md               # Project documentation
└── requirements.txt        # Python dependencies


## Getting Started

### Prerequisites

* Python 3.11
* FFmpeg: Required for audio extraction from video files.
* PyTorch: Required for whisper and pyannote models.
* OpenAI API Key: Required for sentiment and thematic analysis.
* HuggingFace Auth Code: Required for model access


## Installation

Clone the repository:

`git clone https://github.com/yourusername/video_processing_pipeline.git`
`cd video_processing_pipeline`

### Set up a virtual environment:

`python3 -m venv venv`
`source venv/bin/activate`

### Install dependencies:

`pip install -r requirements.txt`

### Install FFmpeg (if not installed):

### For MacOS using Homebrew
`brew install ffmpeg`

### For Ubuntu

`sudo apt update && sudo apt install ffmpeg`

### Set up the OpenAI API key in your environment:

`export OPENAI_API_KEY="your_openai_api_key"`

## Required Models

This pipeline requires specific AI models for transcription, diarization, and sentiment analysis. Here’s how to set them up:

Whisper Model: The model will be automatically downloaded the first time the script is run. Choose the model size (e.g., base, small, large) according to your needs.
PyAnnote Speaker Diarization: Authenticate with the pre-trained pyannote/speaker-diarization model. You may need to create a Hugging Face account to access this model.

## Configuration

### Directories:

Place all video files you want to process in the input_videos directory.
Processed files will be moved to processed_videos after completion.
Transcriptions and sentiment analyses are saved in output_transcripts and output_sentiment respectively.


## Usage

### Run the pipeline by executing:


`python process_pipeline.py`


The script will:

Convert each video file in input_videos to an audio file.
Perform speaker diarization to identify individual speakers.
Transcribe each audio segment and associate it with the identified speaker.
Analyze the transcription for sentiment and thematic elements.
Save the transcript and sentiment analysis in the output_transcripts and output_sentiment directories, respectively.
Move processed video files to the processed_videos directory.

Example Output

For a video file example_video.mp4, the following files will be generated:

output_transcripts/example_video_transcript_YYYYMMDD_HHMMSS.txt: The transcription with speaker labels.
output_sentiment/example_video_sentiment_YYYYMMDD_HHMMSS.txt: Sentiment and theme analysis of the transcription.

Notes

Overwriting: The script is designed to automatically overwrite existing .wav files in case of re-processing.
Compatibility: Supports .mp4 and .m4v formats; additional formats may require minor modifications.
Error Handling: Errors during transcription or analysis are logged in pipeline.log for troubleshooting.
Troubleshooting

Model Compatibility Issues: If you encounter compatibility warnings related to PyTorch or model versions, ensure you’re using compatible versions or consult the model documentation for version requirements.

API Key: Ensure your OpenAI API key is valid and stored in the environment variable OPENAI_API_KEY.
FFmpeg Installation: If FFmpeg is not recognized, confirm that it’s installed and available in your system’s PATH.
Contributing

Contributions are welcome! Please fork the repository and create a pull request with your changes. Make sure to update documentation where applicable.

License

This project is licensed under the MIT License. See the LICENSE file for more details.

