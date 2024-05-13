#!/usr/bin/env python3

import os
import time
import numpy as np
from argparse import ArgumentParser
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

do_scan = True

def process_fft_data(output_file, start, end, bandwidth, fftsize, frames):
    sdr_start = int(start + bandwidth / 2)
    sdr_end = int(end - bandwidth / 2)
    num_steps = int((sdr_end - sdr_start) / bandwidth) + 1
    all_fft_results = []

    with open(output_file, "rb") as file:
        all_data = np.fromfile(file, dtype=np.float32)
    
    all_fft_results = []
    for step in range(num_steps):
        start_index = fftsize * frames * step
        end_index = start_index + fftsize * frames
        if end_index > len(all_data):
            print("Error: Not enough data for step", step)
            break
        
        fft_mat = all_data[start_index:end_index].reshape(-1, fftsize)
        mean_fft = np.mean(fft_mat, axis=0)
        all_fft_results.extend(mean_fft)

    return all_fft_results

def MHz_to_Hz(value):
    return int(float(value) * 1e6)

def main():
    global args, infos, do_scan, last_fft, freq, last_rows, frequencies
    parser = ArgumentParser(description="FFT Replay Tool")
    parser.add_argument("--dir", type=str, help="FFT dir")
    parser.add_argument("--cutoff", type=float, default=False, help="Cutoff frequency for peaks (default: 2 standard deviations above mean)")
    parser.add_argument("--width", type=int, default=5, help="Minimum width for peak detection")
    parser.add_argument("--distance", type=int, default=30, help="Minimum distance between peaks")
    parser.add_argument("--sleep", type=float, default=1.0, help="Sleep time between FFTs")
    args = parser.parse_args()

    plt.ion()

    fig, ax = plt.subplots()
    fig.set_size_inches(12, 5)
    line, = ax.plot([0], [0], label='Spectrum')
    scatter, = ax.plot([0], [0], 'x', color='red', label='Detected Peaks')
    ax.set_title('Peak Detection')
    ax.set_xlabel('Frequency (MHz)')
    ax.set_ylabel('Amplitude')
    ax.legend()

    files = os.listdir(args.dir)
    files.sort(key=lambda x: os.path.getctime(os.path.join(args.dir, x)))

    for file in files:
        (start, end, bandwidth, fftsize, frames, ts) = file.split('.')[0].split('_')
        start = int(start)
        end = int(end)
        bandwidth = int(bandwidth)
        fftsize = int(fftsize)
        frames = int(frames)
        ts = int(ts)

        ax.set_xlim([start, end])
        print(f"Processing {file} ({start} - {end} MHz, FFT size {fftsize}, frames {frames}, ts {ts})")

        path = os.path.join(args.dir, file)
        rows = process_fft_data(path, start, end, bandwidth, fftsize, frames)

        frequencies = np.linspace(start, end, len(rows))
        line.set_data(frequencies, rows)

        cutoff = args.cutoff
        if not args.cutoff:
            cutoff = np.mean(rows) + np.std(rows) * 2

        peaks, properties = find_peaks(rows, height=cutoff, width=args.width, distance=args.distance)
        peak_freqs = np.linspace(start, end, len(rows))[peaks]
        peak_heights = properties['peak_heights']

        for i in range(len(peak_freqs)):
            print(f"Peak at {peak_freqs[i]:.2f} MHz, height {peak_heights[i]:.2f}")

        scatter.set_data(peak_freqs, peak_heights)
        ax.set_ylim([min(rows), max(rows) + 10])
        ax.relim()
        fig.canvas.draw()
        fig.canvas.flush_events()
        time.sleep(args.sleep)
if __name__ == '__main__':
    main()
