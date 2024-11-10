import asyncio
import websockets
import json
import os
from openai import OpenAI
from dotenv import load_dotenv
import tempfile
import wave
import io
import time

load_dotenv()

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

class AudioServer:
    def __init__(self):
        self.sample_rate = 24000  # Updated for TTS compatibility
        self.sample_width = 2
        self.channels = 1
        self.conversation_history = []

    async def process_audio(self, websocket):
        print("New client connected!")
        try:
            async for message in websocket:
                start_time = time.time()
                
                # Create temporary WAV file
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
                    temp_wav.write(message)
                    temp_wav.flush()
                    
                    # Transcribe audio
                    with open(temp_wav.name, 'rb') as audio_file:
                        try:
                            transcription = await self.transcribe_audio(audio_file)
                            if transcription and transcription.strip():
                                print(f"Transcribed: {transcription}")
                                
                                # Get AI response
                                ai_response = await self.get_ai_response(transcription)
                                print(f"AI Response: {ai_response}")
                                
                                if ai_response:
                                    # Convert to speech
                                    audio_response = await self.text_to_speech(ai_response)
                                    if audio_response:
                                        print("Sending audio response...")
                                        await websocket.send(audio_response)
                                        print("Response sent!")
                        except Exception as e:
                            print(f"Processing error: {e}")
                        finally:
                            os.unlink(temp_wav.name)

        except websockets.exceptions.ConnectionClosed:
            print("Client disconnected")
        except Exception as e:
            print(f"Error in process_audio: {e}")

    async def transcribe_audio(self, audio_file):
        try:
            response = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="text"
            )
            return response
        except Exception as e:
            print(f"Transcription error: {e}")
            return None

    async def get_ai_response(self, text):
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful voice assistant. Keep your responses concise and natural."},
                    {"role": "user", "content": text}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"AI response error: {e}")
            return None

    async def text_to_speech(self, text):
        try:
            response = client.audio.speech.create(
                model="tts-1-hd",  # Using HD model for better quality
                voice="nova",  # More natural voice
                input=text,
                speed=1.1,  # Slightly faster speech
                response_format="mp3"
            )
            return response.content
        except Exception as e:
            print(f"Text-to-speech error: {e}")
            return None

    async def start_server(self):
        print("Starting server on ws://localhost:8765")
        server = await websockets.serve(self.process_audio, "localhost", 8765)
        await server.wait_closed()

if __name__ == "__main__":
    server = AudioServer()
    asyncio.run(server.start_server()) 