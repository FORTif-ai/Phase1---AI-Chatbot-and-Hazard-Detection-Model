import asyncio
import pipeline
import speech_recognition as sr
import os, time              
import whisper
import io
import keyboard 
import dotenv
import subprocess

dotenv.load_dotenv("keys.env")

async def listen_for_commands():

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
                if key.lower() == "q":
                    print("Quitting…")
                    break
                if key.lower() == "s":

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
                    await processor.process_command(result["text"])

                    await asyncio.sleep(0.1)
                if key.lower() == "h":
                     subprocess.run(["python", "hazard_detection.py", "--model", "yolov8n.pt"])
                    
    except KeyboardInterrupt:
        print("\nStopping voice command listener…")
                




if __name__ == "__main__":
    asyncio.run(listen_for_commands())
