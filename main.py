import asyncio
import pipeline
import speech_recognition as sr
import os, time              
import whisper
import io
import keyboard 
import dotenv
import subprocess
from typing import Optional

dotenv.load_dotenv("rag/.env")
os.environ['GRPC_VERBOSITY'] = 'ERROR' # Suppress gRPC warnings

async def listen_for_commands(patient_id_input: str):

    recognizer = sr.Recognizer()

    # ➋ Load Whisper **once**, not inside the loop
    model = whisper.load_model("base")     

    #print("Listening for audio…")
    try:
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source, duration=1)

            while True:
                print(" Press ‘s’ to speak (10 s). Press ‘q’ to quit. Press 'h' for hazard detection")
                key = keyboard.read_key()
                if key.lower() == 'q': # 'q' on Mac keyboard
                    print("Quitting…")
                    break
                if key.lower() == 's': # 's' on Mac keyboard

                    print("Capturing audio segment…")

                    # ➌ Record up to 10 s
                    audio = recognizer.listen(source, timeout=None, phrase_time_limit=20)
                    wav_audio = audio.get_wav_data()
                    print(f"Captured audio size: {len(wav_audio)} bytes")

                    # ➍ SAVE the clip so Whisper can read it
                    fname = f"captured.wav"
                    with open(fname, "wb") as f:
                        f.write(wav_audio)

                    # ➎ Transcribe
                    
                    result = model.transcribe(fname)
                    print("Transcription:", result["text"])

                    # Send to your downstream pipeline
                    processor = pipeline.Processor()
                    await processor.process_command(result["text"], patient_id=patient_id_input)

                    await asyncio.sleep(0.1)
                if key.lower() == 'h': # 'h' on Mac keyboard
                     subprocess.run(["python", "hazard_detection.py", "--model", "yolov8n.pt"])
                    
    except KeyboardInterrupt:
        print("\nStopping voice command listener…")

if __name__ == "__main__":
    # --- PROMPT FOR USER ID ---
    print("\n--- Fortif.ai Voice Command Listener ---")
    
    # Get the patient ID from the user
    patient_id_input = input("Please enter the Patient ID for this session (e.g., patient_123): ")
    
    if not patient_id_input:
        print("❌ Patient ID cannot be empty. Exiting.")
    else:
        print(f"✅ Session Patient ID set to: {patient_id_input}")
        
        # Pass the patient ID to the asynchronous listener function
        asyncio.run(listen_for_commands(patient_id_input))