# Speech-to-Text Web Application

## Overview

This project is a real-time Speech-to-Text web application developed as part of the Procucev Enterprises technical assignment.

The application records audio from the user's microphone, streams it to the FastAPI backend through WebSocket, converts speech into text using OpenAI Whisper, stores the transcription in a SQLite database, and displays both the live transcription and transcription history.

---

## Features

* Real-time speech-to-text conversion
* Supports Hindi, Marathi, English, and Auto language mode
* Live transcription using WebSocket
* Audio recording from browser microphone
* Stores transcriptions in SQLite database
* Displays transcription history
* Modern responsive UI
* Waveform visualization while recording

---

## Tech Stack

### Frontend

* HTML
* CSS
* JavaScript

### Backend

* Python
* FastAPI
* WebSocket

### AI Model

* OpenAI Whisper

### Database

* SQLite
* SQLAlchemy

---

## Project Structure

```
speech_to_text/
│
├── database.py
├── main.py
├── models.py
├── transcriber.py
├── requirements.txt
│
└── templates/
    └── index.html
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/sadashiv15/speech-to-text.git
```

Move into the project folder:

```bash
cd speech-to-text
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the application:

```bash
uvicorn main:app --reload
```

Open your browser:

```
http://127.0.0.1:8000
```

---

## Working

1. Open the application.
2. Select a language.
3. Click the microphone button.
4. Speak into the microphone.
5. Audio is streamed to the FastAPI server.
6. Whisper converts speech into text.
7. The transcription is displayed on the screen.
8. The transcription is saved to the SQLite database.
9. Previous transcriptions are available in the history section.

---

## API Endpoints

### GET `/`

Loads the application.

### GET `/config`

Returns application configuration.

### GET `/transcriptions`

Returns all saved transcription history.

### WebSocket `/ws`

Receives audio chunks and returns transcribed text.

---

## Future Improvements

* Better Hindi and Marathi speech accuracy
* User authentication
* Export transcription as PDF or TXT
* Search transcription history
* Speaker identification
* Docker deployment

---

## Repository

https://github.com/sadashiv15/speech-to-text

---

## Author

Sadashiv Karhale
