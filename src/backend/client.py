import asyncio
import websockets
import pyaudio
import wave
import threading
import queue
import numpy as np
import time
import io
import soundfile as sf

class AudioClient:
    def __init__(self):
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 16000
        self.CHUNK = 1024
        self.SILENCE_THRESHOLD = 1500  # Adjusted for better sensitivity
        self.SILENCE_DURATION = 0.8  # Reduced for faster response
        
        self.p = pyaudio.PyAudio()
        self.frames = []
        self.is_recording = False
        self.last_sound_time = time.time()

    def is_silent(self, data):
        """Check if the audio chunk is silence."""
        try:
            audio_data = np.frombuffer(data, dtype=np.int16)
            return np.max(np.abs(audio_data)) < self.SILENCE_THRESHOLD
        except Exception as e:
            print(f"Error in silence detection: {e}")
            return True

    def create_wav_buffer(self, audio_data):
        """Create a WAV file in memory."""
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, 'wb') as wav:
            wav.setnchannels(self.CHANNELS)
            wav.setsampwidth(self.p.get_sample_size(self.FORMAT))
            wav.setframerate(self.RATE)
            wav.writeframes(b''.join(audio_data))
        return wav_buffer.getvalue()

    async def process_audio(self, websocket):
        """Process audio and handle silence detection."""
        stream = self.p.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.RATE,
            input=True,
            frames_per_buffer=self.CHUNK
        )

        print("Listening... (Speak naturally, pause to process)")
        self.is_recording = True
        recording_started = False

        try:
            while self.is_recording:
                data = stream.read(self.CHUNK, exception_on_overflow=False)
                
                if not self.is_silent(data):
                    if not recording_started:
                        print("Speech detected...")
                        recording_started = True
                    self.frames.append(data)
                    self.last_sound_time = time.time()
                elif self.frames and recording_started:
                    current_time = time.time()
                    if current_time - self.last_sound_time > self.SILENCE_DURATION:
                        print("Processing speech...")
                        wav_data = self.create_wav_buffer(self.frames)
                        await websocket.send(wav_data)
                        
                        try:
                            response = await websocket.recv()
                            print("Playing response...")
                            self.play_audio(response)
                        except Exception as e:
                            print(f"Error receiving/playing response: {e}")
                        
                        self.frames = []
                        recording_started = False
                        print("\nListening... (Speak naturally, pause to process)")

                await asyncio.sleep(0.01)

        except Exception as e:
            print(f"Error in audio processing: {e}")
        finally:
            stream.stop_stream()
            stream.close()

    def play_audio(self, audio_data):
        """Play audio response from the server."""
        try:
            with io.BytesIO(audio_data) as audio_io:
                data, samplerate = sf.read(audio_io)
            
            audio_array = data.astype(np.float32)
            
            stream = self.p.open(
                format=pyaudio.paFloat32,
                channels=1,
                rate=24000,  # TTS sample rate
                output=True
            )
            
            chunk_size = 1024
            for i in range(0, len(audio_array), chunk_size):
                chunk = audio_array[i:i + chunk_size]
                stream.write(chunk.tobytes())
            
        except Exception as e:
            print(f"Error playing audio: {e}")
        finally:
            stream.stop_stream()
            stream.close()

    async def connect_websocket(self):
        """Connect to WebSocket server and handle audio processing."""
        uri = "ws://localhost:8765"
        while True:
            try:
                async with websockets.connect(uri) as websocket:
                    print("Connected to server!")
                    await self.process_audio(websocket)
            except websockets.exceptions.ConnectionClosed:
                print("Connection lost. Reconnecting...")
                await asyncio.sleep(1)
            except Exception as e:
                print(f"Connection error: {e}")
                await asyncio.sleep(1)

    def cleanup(self):
        """Clean up PyAudio resources."""
        self.is_recording = False
        self.p.terminate()

if __name__ == "__main__":
    client = AudioClient()
    try:
        asyncio.run(client.connect_websocket())
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        client.cleanup() 