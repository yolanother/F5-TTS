import requests
import threading
import tempfile
import os
import time
import sys

def stream_audio_to_file(url, files, data, output_file_path):
    with requests.post(url, files=files, data=data, stream=True) as response:
        if response.status_code != 200:
            print(f"Request failed with status code {response.status_code}", file=sys.stderr)
            print(response.text, file=sys.stderr)
            return False

        # Get audio properties from headers
        sample_rate = int(response.headers.get('X-Sample-Rate', 24000))
        audio_format = response.headers.get('X-Format', 'PCM16')
        channels = 1  # Assuming mono audio

        if audio_format != 'PCM16':
            print(f"Unsupported audio format: {audio_format}", file=sys.stderr)
            return False

        # Prepare WAV file headers
        with open(output_file_path, 'wb') as f:
            write_wav_header(f, channels, sample_rate, bits_per_sample=16)
            print("Downloading and saving audio to temporary file...")
            for chunk in response.iter_content(chunk_size=4096):
                if chunk:
                    f.write(chunk)

        return True

def write_wav_header(f, num_channels, sample_rate, bits_per_sample):
    # Calculate file parameters
    byte_rate = sample_rate * num_channels * bits_per_sample // 8
    block_align = num_channels * bits_per_sample // 8

    # Prepare the header with placeholder for data length
    f.write(b'RIFF')
    f.write((0).to_bytes(4, 'little'))  # Placeholder for file size
    f.write(b'WAVE')

    # fmt subchunk
    f.write(b'fmt ')
    f.write((16).to_bytes(4, 'little'))  # Subchunk1Size (16 for PCM)
    f.write((1).to_bytes(2, 'little'))   # AudioFormat (1 for PCM)
    f.write((num_channels).to_bytes(2, 'little'))
    f.write((sample_rate).to_bytes(4, 'little'))
    f.write((byte_rate).to_bytes(4, 'little'))
    f.write((block_align).to_bytes(2, 'little'))
    f.write((bits_per_sample).to_bytes(2, 'little'))

    # data subchunk
    f.write(b'data')
    f.write((0).to_bytes(4, 'little'))  # Placeholder for data size

def update_wav_header(file_path):
    with open(file_path, 'r+b') as f:
        f.seek(0, os.SEEK_END)
        file_size = f.tell()
        data_size = file_size - 44  # Subtract header size

        # Update file size
        f.seek(4)
        f.write((file_size - 8).to_bytes(4, 'little'))

        # Update data chunk size
        f.seek(40)
        f.write((data_size).to_bytes(4, 'little'))

def play_audio_file(file_path):
    # Use the default Windows media player to play the file
    os.startfile(file_path)

if __name__ == "__main__":
    url = "http://127.0.0.1:8000/tts"
    # Prepare form data
    data = {
        'ref_text': '',
        'gen_text': 'I would like to tell you a story about a man. A man with the power. The power of voodoo! Who has voodoo? You do! Do what? Remind me of the babe. What babe? The babe with the power. What power? The power of voodoo. Who do? You do! Do what? Remind me of the babe.',
        'exp_name': 'F5-TTS',
        'remove_silence': 'false',
        'cross_fade_duration': '1'
    }

    # Open the reference audio file
    ref_audio_file_path = 'reference_audio.mp3'  # Replace with your file path
    with open(ref_audio_file_path, 'rb') as f:
        files = {'ref_audio': ('ref_audio.mp3', f, 'audio/mpeg')}

        # Create a temporary file to save the audio
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_wav_file:
            temp_wav_file_path = temp_wav_file.name

        # Start streaming audio to file in a separate thread
        download_thread = threading.Thread(target=stream_audio_to_file, args=(url, files, data, temp_wav_file_path))
        download_thread.start()

        # Wait a bit to accumulate some audio data
        time.sleep(5)  # Adjust as needed

        # Start playing the audio file
        print("Starting playback...")
        play_audio_file(temp_wav_file_path)

        # Wait for the download to finish
        download_thread.join()

        # Update the WAV header with correct sizes
        update_wav_header(temp_wav_file_path)

        print("Playback finished. Press Enter to exit.")
        input()

        # Clean up the temporary file
        os.remove(temp_wav_file_path)
