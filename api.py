import re
import torch
import torchaudio
import numpy as np
import tempfile
from einops import rearrange
from vocos import Vocos
from pydub import AudioSegment, silence
from model import CFM, UNetT, DiT, MMDiT
from cached_path import cached_path
from model.utils import (
    load_checkpoint,
    get_tokenizer,
    convert_char_to_pinyin,
    save_spectrogram,
)
from transformers import pipeline
import soundfile as sf
import click
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from starlette.concurrency import run_in_threadpool
import uvicorn
from io import BytesIO
import os

app = FastAPI()

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)

print(f"Using {device} device")

# --------------------- Settings -------------------- #

target_sample_rate = 24000
n_mel_channels = 100
hop_length = 256
target_rms = 0.1
nfe_step = 32  # 16, 32
cfg_strength = 2.0
ode_method = "euler"
sway_sampling_coef = -1.0
speed = 1.0
fix_duration = None


# Load models on startup
@app.on_event("startup")
def load_models():
    global pipe, vocos, F5TTS_ema_model, E2TTS_ema_model, speed
    # Load the ASR pipeline
    pipe = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-large-v3-turbo",
        torch_dtype=torch.float16,
        device=device,
    )
    # Load vocoder
    vocos = Vocos.from_pretrained("charactr/vocos-mel-24khz")

    def load_model(repo_name, exp_name, model_cls, model_cfg, ckpt_step):
        ckpt_path = str(cached_path(f"hf://SWivid/{repo_name}/{exp_name}/model_{ckpt_step}.safetensors"))
        vocab_char_map, vocab_size = get_tokenizer("Emilia_ZH_EN", "pinyin")
        model = CFM(
            transformer=model_cls(
                **model_cfg, text_num_embeds=vocab_size, mel_dim=n_mel_channels
            ),
            mel_spec_kwargs=dict(
                target_sample_rate=target_sample_rate,
                n_mel_channels=n_mel_channels,
                hop_length=hop_length,
            ),
            odeint_kwargs=dict(
                method=ode_method,
            ),
            vocab_char_map=vocab_char_map,
        ).to(device)

        model = load_checkpoint(model, ckpt_path, device, use_ema=True)

        return model

    # Load models
    F5TTS_model_cfg = dict(
        dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4
    )
    E2TTS_model_cfg = dict(dim=1024, depth=24, heads=16, ff_mult=4)

    F5TTS_ema_model = load_model(
        "F5-TTS", "F5TTS_Base", DiT, F5TTS_model_cfg, 1200000
    )
    E2TTS_ema_model = load_model(
        "E2-TTS", "E2TTS_Base", UNetT, E2TTS_model_cfg, 1200000
    )


def chunk_text(text, max_chars=135):
    """
    Splits the input text into chunks, each with a maximum number of characters.
    """
    chunks = []
    current_chunk = ""
    # Split the text into sentences based on punctuation followed by whitespace
    sentences = re.split(r'(?<=[;:,.!?])\s+|(?<=[；：，。！？])', text)

    for sentence in sentences:
        if len(current_chunk.encode('utf-8')) + len(sentence.encode('utf-8')) <= max_chars:
            current_chunk += sentence + " " if sentence and len(sentence[-1].encode('utf-8')) == 1 else sentence
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + " " if sentence and len(sentence[-1].encode('utf-8')) == 1 else sentence

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


def infer_batch(ref_audio, ref_text, gen_text_batches, exp_name, remove_silence, cross_fade_duration=0.15):
    if exp_name == "F5-TTS":
        ema_model = F5TTS_ema_model
    elif exp_name == "E2-TTS":
        ema_model = E2TTS_ema_model

    audio, sr = ref_audio
    if audio.shape[0] > 1:
        audio = torch.mean(audio, dim=0, keepdim=True)

    rms = torch.sqrt(torch.mean(torch.square(audio)))
    if rms < target_rms:
        audio = audio * target_rms / rms
    if sr != target_sample_rate:
        resampler = torchaudio.transforms.Resample(sr, target_sample_rate)
        audio = resampler(audio)
    audio = audio.to(device)

    generated_waves = []
    spectrograms = []

    for i, gen_text in enumerate(gen_text_batches):
        # Prepare the text
        if len(ref_text[-1].encode('utf-8')) == 1:
            ref_text = ref_text + " "
        text_list = [ref_text + gen_text]
        final_text_list = convert_char_to_pinyin(text_list)

        # Calculate duration
        ref_audio_len = audio.shape[-1] // hop_length
        zh_pause_punc = r"。，、；：？！"
        ref_text_len = len(ref_text.encode('utf-8')) + 3 * len(re.findall(zh_pause_punc, ref_text))
        gen_text_len = len(gen_text.encode('utf-8')) + 3 * len(re.findall(zh_pause_punc, gen_text))
        duration = ref_audio_len + int(ref_audio_len / ref_text_len * gen_text_len / speed)

        # Inference
        with torch.inference_mode():
            generated, _ = ema_model.sample(
                cond=audio,
                text=final_text_list,
                duration=duration,
                steps=nfe_step,
                cfg_strength=cfg_strength,
                sway_sampling_coef=sway_sampling_coef,
            )

        generated = generated[:, ref_audio_len:, :]
        generated_mel_spec = rearrange(generated, "1 n d -> 1 d n")
        generated_wave = vocos.decode(generated_mel_spec.cpu())
        if rms < target_rms:
            generated_wave = generated_wave * rms / target_rms

        # Convert to numpy
        generated_wave = generated_wave.squeeze().cpu().numpy()

        generated_waves.append(generated_wave)
        spectrograms.append(generated_mel_spec[0].cpu().numpy())

    # Combine all generated waves with cross-fading
    if cross_fade_duration <= 0:
        # Simply concatenate
        final_wave = np.concatenate(generated_waves)
    else:
        final_wave = generated_waves[0]
        for i in range(1, len(generated_waves)):
            prev_wave = final_wave
            next_wave = generated_waves[i]

            # Calculate cross-fade samples
            cross_fade_samples = int(cross_fade_duration * target_sample_rate)
            cross_fade_samples = min(cross_fade_samples, len(prev_wave), len(next_wave))

            if cross_fade_samples <= 0:
                # No overlap possible, concatenate
                final_wave = np.concatenate([prev_wave, next_wave])
                continue

            # Overlapping parts
            prev_overlap = prev_wave[-cross_fade_samples:]
            next_overlap = next_wave[:cross_fade_samples]

            # Fade out and fade in
            fade_out = np.linspace(1, 0, cross_fade_samples)
            fade_in = np.linspace(0, 1, cross_fade_samples)

            # Cross-faded overlap
            cross_faded_overlap = prev_overlap * fade_out + next_overlap * fade_in

            # Combine
            new_wave = np.concatenate([
                prev_wave[:-cross_fade_samples],
                cross_faded_overlap,
                next_wave[cross_fade_samples:]
            ])

            final_wave = new_wave

    # Remove silence
    if remove_silence:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            sf.write(f.name, final_wave, target_sample_rate)
            aseg = AudioSegment.from_file(f.name)
            non_silent_segs = silence.split_on_silence(aseg, min_silence_len=1000, silence_thresh=-50, keep_silence=500)
            non_silent_wave = AudioSegment.silent(duration=0)
            for non_silent_seg in non_silent_segs:
                non_silent_wave += non_silent_seg
            aseg = non_silent_wave
            aseg.export(f.name, format="wav")
            final_wave, _ = torchaudio.load(f.name)
        final_wave = final_wave.squeeze().cpu().numpy()

    # Create a combined spectrogram
    combined_spectrogram = np.concatenate(spectrograms, axis=1)

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_spectrogram:
        spectrogram_path = tmp_spectrogram.name
        save_spectrogram(combined_spectrogram, spectrogram_path)

    return (target_sample_rate, final_wave), spectrogram_path


def infer(ref_audio_orig, ref_text, gen_text, exp_name, remove_silence, cross_fade_duration=0.15):
    print(gen_text)

    print("Converting audio...")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        aseg = AudioSegment.from_file(ref_audio_orig)

        non_silent_segs = silence.split_on_silence(
            aseg, min_silence_len=1000, silence_thresh=-50, keep_silence=1000
        )
        non_silent_wave = AudioSegment.silent(duration=0)
        for non_silent_seg in non_silent_segs:
            non_silent_wave += non_silent_seg
        aseg = non_silent_wave

        audio_duration = len(aseg)
        if audio_duration > 15000:
            print("Audio is over 15s, clipping to only first 15s.")
            aseg = aseg[:15000]
        aseg.export(f.name, format="wav")
        ref_audio = f.name

    if not ref_text.strip():
        print("No reference text provided, transcribing reference audio...")
        ref_text = pipe(
            ref_audio,
            chunk_length_s=30,
            batch_size=128,
            generate_kwargs={"task": "transcribe"},
            return_timestamps=False,
        )["text"].strip()
        print("Finished transcription")
    else:
        print("Using custom reference text...")

    # Ensure it ends with ". "
    if not ref_text.endswith(". "):
        if ref_text.endswith("."):
            ref_text += " "
        else:
            ref_text += ". "

    audio, sr = torchaudio.load(ref_audio)

    # Split gen_text into batches
    max_chars = int(len(ref_text.encode('utf-8')) / (audio.shape[-1] / sr) * (25 - audio.shape[-1] / sr))
    gen_text_batches = chunk_text(gen_text, max_chars=max_chars)
    print('ref_text', ref_text)
    for i, batch_text in enumerate(gen_text_batches):
        print(f'gen_text {i}', batch_text)

    print(f"Generating audio using {exp_name} in {len(gen_text_batches)} batches")
    return infer_batch((audio, sr), ref_text, gen_text_batches, exp_name, remove_silence, cross_fade_duration)


def pcm16_audio_generator(audio_data, chunk_size=1024):
    """Generator that yields PCM16 audio data in chunks."""
    # Ensure audio_data is in int16 format
    if audio_data.dtype != np.int16:
        audio_data = (audio_data * np.iinfo(np.int16).max).astype(np.int16)
    total_length = len(audio_data)
    for start in range(0, total_length, chunk_size):
        end = min(start + chunk_size, total_length)
        yield audio_data[start:end].tobytes()


@app.post("/tts")
async def tts(ref_audio: UploadFile = File(...),
              ref_text: str = Form(""),
              gen_text: str = Form(...),
              exp_name: str = Form("F5-TTS"),
              remove_silence: bool = Form(False),
              cross_fade_duration: float = Form(0.15),
              stream: bool = Form(True)):  # Added 'stream' parameter
    # Save the uploaded audio file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        temp_audio_path = f.name
        content = await ref_audio.read()
        f.write(content)

    # Call the infer function in a thread pool to avoid blocking
    audio_output, spectrogram_path = await run_in_threadpool(
        infer, temp_audio_path, ref_text, gen_text, exp_name, remove_silence, cross_fade_duration
    )

    # Remove temporary files
    os.remove(temp_audio_path)
    os.remove(spectrogram_path)

    sample_rate, audio_data = audio_output

    if stream:
        # Create the PCM16 audio generator
        audio_iter = pcm16_audio_generator(audio_data)

        # Return the audio as a StreamingResponse
        headers = {
            "Content-Type": "audio/L16",
            "Content-Disposition": "inline",
            "X-Sample-Rate": str(sample_rate),
            "X-Format": "PCM16",
        }
        return StreamingResponse(audio_iter, media_type="audio/L16", headers=headers)
    else:
        # Save the audio data to a BytesIO object as a WAV file
        audio_bytes_io = BytesIO()
        # Convert numpy array to torch tensor
        audio_tensor = torch.from_numpy(audio_data).unsqueeze(0)
        # Ensure audio_data is float32 tensor
        if audio_tensor.dtype != torch.float32:
            audio_tensor = audio_tensor.float()
        torchaudio.save(audio_bytes_io, audio_tensor, sample_rate, format='wav')
        audio_bytes_io.seek(0)
        # Return the audio as a StreamingResponse
        headers = {
            "Content-Type": "audio/wav",
            "Content-Disposition": "attachment; filename=generated_audio.wav",
        }
        return StreamingResponse(audio_bytes_io, media_type="audio/wav", headers=headers)


# ... [rest of the code remains the same for other endpoints] ...

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
