import serial
import numpy as np
import pandas as pd
import joblib
import csv
from datetime import datetime
from collections import deque
from scipy.signal import find_peaks, butter, lfilter, lfilter_zi
import matplotlib.pyplot as plt

from features import bandpass_filter

# --- Setup ---
model = joblib.load('ecg_model.pkl')
ser = serial.Serial('COM9', 115200, timeout=1)
fs = 300  # replace with your actual measured sampling rate

buffer = []
window_size = fs * 5  # 5-second window for prediction

# --- CSV logging setup ---
timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
csv_file = open(f'live_readings_{timestamp_str}.csv', 'w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['timestamp', 'rr_interval', 'heart_rate', 'hrv', 'amplitude', 'qrs_width', 'prediction'])

# --- Live scrolling plot setup ---
plot_window_sec = 4
plot_buffer = deque(maxlen=fs * plot_window_sec)
plot_counter = 0

# --- Causal filter setup (for stable live plot, separate from batch bandpass_filter used in predictions) ---
b_coef, a_coef = butter(2, [0.5 / (0.5 * fs), 40 / (0.5 * fs)], btype='band')
zi = lfilter_zi(b_coef, a_coef) * 0  # initial filter state, persists across samples

plt.ion()
fig, ax = plt.subplots(figsize=(10, 4))
line_plot, = ax.plot([], [], color='black', linewidth=0.8)
ax.set_xlabel('Samples')
ax.set_ylabel('Amplitude')
ax.set_title('Live ECG Signal')
ax.set_ylim(-150, 400)  # adjust based on your actual signal range

print("Starting live ECG monitoring... (Ctrl+C to stop)")

while True:
    try:
        line = ser.readline().decode('utf-8', errors='ignore').strip()

        if line == '!':
            print("Lead off — check electrode contact")
            continue

        if line.isdigit():
            value = int(line)
            buffer.append(value)

            # --- Causal filtering: one sample in, one filtered sample out, state persists ---
            filtered_value, zi = lfilter(b_coef, a_coef, [value], zi=zi)
            plot_buffer.append(filtered_value[0])
            plot_counter += 1

            # --- Smooth scrolling plot update ---
            if plot_counter % 5 == 0 and len(plot_buffer) > 10:
                line_plot.set_data(range(len(plot_buffer)), list(plot_buffer))
                ax.set_xlim(0, len(plot_buffer))
                plt.pause(0.001)

        # --- Prediction logic (unchanged, uses separate buffer + batch filtfilt for accuracy) ---
        if len(buffer) >= window_size:
            signal_chunk = np.array(buffer[-window_size:], dtype=float)
            filtered = bandpass_filter(signal_chunk, fs=fs)

            peaks, _ = find_peaks(filtered, distance=fs*0.4, height=np.mean(filtered)+np.std(filtered))

            if len(peaks) < 3:
                buffer = buffer[-int(fs*1):]
                continue

            rr_intervals = []
            for i in range(1, len(peaks)):
                rr = (peaks[i] - peaks[i-1]) / fs
                if 0.3 < rr < 2.0:
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

            row = pd.DataFrame([{
                'rr_interval': latest_rr,
                'heart_rate': heart_rate,
                'hrv': hrv,
                'amplitude': amplitude,
                'qrs_width': qrs_width
            }])

            prediction = model.predict(row)[0]
            print(f"HR: {heart_rate:.0f} bpm | Prediction: {prediction}")

            csv_writer.writerow([
                datetime.now().isoformat(),
                latest_rr, heart_rate, hrv, amplitude, qrs_width, prediction
            ])
            csv_file.flush()

            buffer = buffer[-int(fs*2):]

    except KeyboardInterrupt:
        print("Stopped.")
        csv_file.close()
        break
    except Exception as e:
        print(f"Error: {e}")
        continue