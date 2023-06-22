#!/usr/bin/env python3

import os
from lid import Model, KaldiRecognizer
import wave
import json

files: list[str] = ["Black.Mirror.S06E01.GERMAN.DL.1080p.WEB.h264-SAUERKRAUT.mkv"]

ROOT_FOLDER: str = "/media/totto/Totto_1"

for file in files:
    file_path = os.path.join(ROOT_FOLDER, file)
    wf = wave.open(file_path, "rb")
    if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getcomptype() != "NONE":
        print("Audio file must be WAV format mono PCM.")
        exit(1)

    model = Model("lid-model")
    rec = KaldiRecognizer(model, wf.getframerate())
    data: bytes = wf.readframes(-1)
    rec.AcceptWaveform(data)

    results = rec.Result()
    result = max(json.loads(results), key=lambda ev: ev["score"])
    # print(json.loads(results))

    print("identified language: ", result["language"])


# ffmpeg -i video.mp4 -acodec pcm_s16le -ac 2 audio.wav
