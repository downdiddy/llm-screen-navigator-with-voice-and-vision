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
        self.sample_rate = 24000
        self.sample_width = 2
        self.channels = 1
        self.conversation_history = []
        self.system_prompt = """You are a highly intelligent and helpful voice assistant. 
        Your responses should be:
        - Natural and conversational
        - Accurate and informative
        - Concise but complete
        - Contextually aware of previous conversation
        
        If you don't understand or need clarification, ask for it politely.
        If a question is cut off or unclear, ask for clarification."""

    async def transcribe_audio(self, audio_file):
        """Improved transcription using Whisper with better parameters"""
        try:
            response = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="text",
                temperature=0.2,  # Lower temperature for more accurate transcription
                language="en",    # Specify language
                prompt="This is a natural conversation with an AI assistant. The speech may include technical terms and questions."
            )
            return response
        except Exception as e:
            print(f"Transcription error: {e}")
            return None

    async def get_ai_response(self, text):
        """Get AI response using GPT-4 with conversation history"""
        try:
            # Add the new message to conversation history
            self.conversation_history.append({"role": "user", "content": text})
            
            # Prepare messages with context
            messages = [
                {"role": "system", "content": self.system_prompt},
                # Include last few messages for context
                *self.conversation_history[-4:]  # Keep last 4 messages
            ]
            
            response = client.chat.completions.create(
                model="gpt-4-turbo-preview",  # Latest GPT-4 model
                messages=messages,
                temperature=0.7,
                max_tokens=150,   # Keep responses concise
                presence_penalty=0.6,  # Encourage varied responses
                frequency_penalty=0.3,  # Reduce repetition
                top_p=0.9
            )
            
            ai_response = response.choices[0].message.content
            # Add AI response to history
            self.conversation_history.append({"role": "assistant", "content": ai_response})
            
            return ai_response
        except Exception as e:
            print(f"AI response error: {e}")
            return None

    async def text_to_speech(self, text):
        """Convert text to speech with improved settings"""
        try:
            response = client.audio.speech.create(
                model="tts-1-hd",
                voice="nova",
                input=text,
                speed=1.0,  # Normal speed for better clarity
                response_format="mp3"
            )
            return response.content
        except Exception as e:
            print(f"Text-to-speech error: {e}")
            return None

    async def process_audio(self, websocket):
        print("New client connected!")
        try:
            async for message in websocket:
                start_time = time.time()
                
                # Process audio in memory when possible
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
                    temp_wav.write(message)
                    temp_wav.flush()
                    
                    with open(temp_wav.name, 'rb') as audio_file:
                        try:
                            transcription = await self.transcribe_audio(audio_file)
                            if transcription and transcription.strip():
                                print(f"Transcribed: {transcription}")
                                
                                ai_response = await self.get_ai_response(transcription)
                                print(f"AI Response: {ai_response}")
                                
                                if ai_response:
                                    audio_response = await self.text_to_speech(ai_response)
                                    if audio_response:
                                        print("Sending audio response...")
                                        await websocket.send(audio_response)
                                        print("Response sent!")
                                        print(f"Total processing time: {time.time() - start_time:.2f}s")
                        except Exception as e:
                            print(f"Processing error: {e}")
                        finally:
                            os.unlink(temp_wav.name)

        except websockets.exceptions.ConnectionClosed:
            print("Client disconnected")
        except Exception as e:
            print(f"Error in process_audio: {e}")

    async def start_server(self):
        print("Starting server on ws://localhost:8765")
        server = await websockets.serve(self.process_audio, "localhost", 8765)
        await server.wait_closed()

if __name__ == "__main__":
    server = AudioServer()
    asyncio.run(server.start_server())