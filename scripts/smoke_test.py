#!/usr/bin/env python3
"""Exercise the pipeline with synthetic media in an isolated temporary project."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import wave

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    with tempfile.TemporaryDirectory(prefix='contour-smoke-') as directory:
        stage = Path(directory)
        for name in ('src', 'scripts'):
            shutil.copytree(ROOT / name, stage / name,
                            ignore=shutil.ignore_patterns('__pycache__'))
        (stage / 'reference').mkdir()
        config = json.loads((ROOT / 'project.json').read_text())
        config.update(width=96, height=64, fps=24, frame_count=12,
                      start_frame=0, output_name='synthetic')
        (stage / 'project.json').write_text(json.dumps(config))
        # The test tone and every image are generated here from scratch.
        samples = (np.sin(np.arange(24000) * (2 * np.pi * 440 / 48000)) * 2000).astype('<i2')
        tone = stage / 'tone.wav'
        with wave.open(str(tone), 'wb') as audio:
            audio.setnchannels(1)
            audio.setsampwidth(2)
            audio.setframerate(48000)
            audio.writeframes(samples.tobytes())
        frames = []
        for i in range(12):
            frame = np.full((64, 96, 3), 245, np.uint8)
            cv2.circle(frame, (18 + i * 4, 32), 13,
                       (15, 15, 15) if i < 6 else (170, 65, 20), -1, cv2.LINE_AA)
            cv2.circle(frame, (18 + i * 4, 32), 5, (245, 245, 245), -1)
            frames.append(frame)
        reference = stage / config['reference']
        subprocess.run([
            'ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pixel_format', 'bgr24',
            '-video_size', '96x64', '-framerate', '24', '-i', 'pipe:0',
            '-i', str(tone), '-c:v', 'ffv1', '-pix_fmt', 'bgr0',
            '-c:a', 'pcm_s24le', '-t', '0.5', '-f', 'matroska', str(reference)
        ], input=b''.join(frame.tobytes() for frame in frames), check=True)
        def run(script, *args):
            subprocess.run([sys.executable, str(stage / 'scripts' / script), *args],
                           cwd=stage, check=True)
        run('pipeline.py', 'extract', '--workers', '1')
        run('pipeline.py', 'render')
        run('pipeline.py', 'compare')
        run('check_audio.py')
        run('make_review.py')
        run('verify_isolation.py')
        run('pipeline.py', 'compare')
        print('PASS: synthetic grayscale/color frames, audio, review, and isolated rendering.')


if __name__ == '__main__':
    main()
