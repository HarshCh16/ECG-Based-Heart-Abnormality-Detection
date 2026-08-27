import serial
import numpy as np
import pandas as pd
import joblib
import time
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import csv
from datetime import datetime

from features import bandpass_filter  # reuse your existing filter

# --- Setup ---
model = joblib.load('ecg_model.pkl')
timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
csv_file = open(f'live_readings_{timestamp_str}.csv', 'w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['timestamp', 'rr_interval', 'heart_rate', 'hrv', 'amplitude', 'qrs_width', 'prediction'])
plt.ion()  # interactive mode, lets the plot update live
fig, ax = plt.subplots(figsize=(10, 4))
line_plot, = ax.plot([], [], color='black', linewidth=0.8)
peak_scatter = ax.scatter([], [], color='red', zorder=5)
ax.set_xlabel('Samples')
ax.set_ylabel('Amplitude')
ax.set_title('Live ECG Signal')
ser = serial.Serial('COM9', 115200)  # match your Arduino's port + baud rate
fs = 312  # approximate sampling rate based on your delay(1) loop — adjust if needed

buffer = []
window_size = fs * 5  # 5 seconds of data per analysis window

print("Starting live ECG monitoring... (Ctrl+C to stop)")

while True:
    try:
        line = ser.readline().decode('utf-8').strip()

        if line == '!':
            print("Lead off — check electrode contact")
            continue

        if line.isdigit():
            buffer.append(int(line))

        if len(buffer) >= window_size:
            signal_chunk = np.array(buffer[-window_size:], dtype=float)

            # --- Filter ---
            filtered = bandpass_filter(signal_chunk, fs=fs)

            # --- Detect R-peaks ---
            peaks, _ = find_peaks(filtered, distance=fs*0.4, height=np.mean(filtered)+np.std(filtered))

            if len(peaks) < 3:
                print("Not enough beats detected yet, waiting...")
                buffer = buffer[-int(fs*1):]  # keep a little, slide forward
                continue

            # --- Extract features per beat (skip first, need previous peak for RR) ---
            rr_intervals = []
            for i in range(1, len(peaks)):
                rr = (peaks[i] - peaks[i-1]) / fs
                if 0.3 < rr < 2.0:  # sanity range: 30-200 bpm
                    rr_intervals.append(rr)

            if len(rr_intervals) < 2:
                buffer = buffer[-int(fs*1):]
                continue

            latest_rr = rr_intervals[-1]
            heart_rate = 60 / latest_rr
            hrv = np.std(rr_intervals[-5:]) if len(rr_intervals) > 1 else 0.0

            last_peak = peaks[-1]
            half_win = int(0.1 * fs)
            start = max(0, last_peak - half_win)
            end = min(len(filtered), last_peak + half_win)
            segment = filtered[start:end]
            amplitude = filtered[last_peak]

            if len(segment) > 0 and amplitude != 0:
                threshold = amplitude * 0.5
                above = np.where(np.abs(segment) > np.abs(threshold))[0]
                qrs_width = (above[-1] - above[0]) / fs if len(above) > 1 else 0.0
            else:
                qrs_width = 0.0

            # --- Update live plot ---
            line_plot.set_data(range(len(filtered)), filtered)
            ax.set_xlim(0, len(filtered))
            ax.set_ylim(filtered.min() - 50, filtered.max() + 50)

            if len(peaks) > 0:
                peak_scatter.set_offsets(np.column_stack((peaks, filtered[peaks])))

            plt.pause(0.01)  # tiny pause lets the plot actually redraw

            # --- Build feature row and predict ---
            row = pd.DataFrame([{
                'rr_interval': latest_rr,
                'heart_rate': heart_rate,
                'hrv': hrv,
                'amplitude': amplitude,
                'qrs_width': qrs_width
            }])

            prediction = model.predict(row)[0]
            csv_writer.writerow([
                datetime.now().isoformat(),
                latest_rr, heart_rate, hrv, amplitude, qrs_width, prediction
            ])
            csv_file.flush()
            print(f"HR: {heart_rate:.0f} bpm | Prediction: {prediction}")

            # slide window forward, keep some overlap
            buffer = buffer[-int(fs*2):]

    except KeyboardInterrupt:
        print("Stopped.")
        csv_file.close()
        break
    except Exception as e:
        print(f"Error: {e}")
        continue