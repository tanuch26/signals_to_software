import os
import gc
import glob
import pickle
import zipfile
import requests
import urllib.request  # <--- NEW: Allows us to download files directly!
import numpy as np
import scipy.ndimage
import scipy.signal
import matplotlib.pyplot as plt
import librosa
from collections import Counter
import streamlit as st

# --- Phase 1: Application Setup and Backend Integration ---

st.set_page_config(page_title="EE200 Song Identifier", layout="wide")

@st.cache_resource
def load_database():
    db_name = "song_db.pkl"
    base_dir = os.path.dirname(os.path.abspath(__file__))
    zip_name = os.path.join(base_dir,"song_db.zip")
   
    # 1. LOCAL TESTING: Try to load the .pkl normally (for when you run on your laptop)
    for root, dirs, files in os.walk(base_dir):
        if db_name in files:
            filepath = os.path.join(root, db_name)
            try:
                with open(filepath, 'rb') as f:
                    data = pickle.load(f)
                    if isinstance(data, dict): 
                        return data, []
            except Exception as e:
                return {}, [str(e)]# Ignore errors locally, move to the cloud download step

    # 2. CLOUD DEPLOYMENT: Download the real zip directly from your GitHub Release
    # ---> PASTE YOUR LINK BETWEEN THE QUOTES BELOW <---
    url = "https://github.com/Orthodox112/Song-Identifier/releases/download/v1.0/song_db.zip" 
        
    try:
        # This downloads the file to the Streamlit server
        r = requests.get(url)
        r.raise_for_status()
        with open(zip_name,"wb") as f:
            f.write(r.content)
    except Exception as e:
        return {}, [
    f"URL: {url}",
    f"Exists: {os.path.exists(zip_name)}",
    f"Download Error: {repr(e)}"
    ]

    # 3. Read the database directly from the downloaded ZIP in Memory
    try:
        with zipfile.ZipFile(zip_name, 'r') as zip_ref:
            for file_inside_zip in zip_ref.namelist():
                if file_inside_zip.endswith('.pkl'):
                    with zip_ref.open(file_inside_zip) as f:
                        data = pickle.load(f)
                        if isinstance(data, dict):
                            return data, []
    except Exception as e:
        return {}, [f"Extraction Error: {type(e).__name__} - {e}"]
    
    return {}, ["Failed to find a valid dictionary inside the downloaded zip."]

# Safely initialize the global database
# song_db, debug_info = load_database()
# if not isinstance(song_db, dict):
#     song_db = {}
# --- Core Functions from Q3.ipynb ---

def compute_spectrogram(audio_path=None, y=None, sr=None, window_length=2048, hop_length=512, max_freq=3500):
    """
    Loads an audio file and computes its STFT spectrogram.
    Returns only frequency bins from 0 Hz to max_freq.
    """
    if y is None:
        y, sr = librosa.load(audio_path, sr=None, mono=True)

    # STFT
    stft_matrix = librosa.stft(
        y,
        n_fft=window_length,
        hop_length=hop_length,
        window='hann'
    )

    # Magnitude spectrogram
    S_mag = np.abs(stft_matrix)

    # Frequency bins
    freq_bins = librosa.fft_frequencies(sr=sr, n_fft=window_length)

    # Keep only frequencies <= max_freq
    freq_mask = freq_bins <= max_freq

    S_mag = S_mag[freq_mask, :]
    freq_bins = freq_bins[freq_mask]

    # Convert to dB scale
    S_db = librosa.amplitude_to_db(S_mag, ref=np.max)

    # Time bins
    time_bins = librosa.frames_to_time(
        np.arange(S_db.shape[1]),
        sr=sr,
        hop_length=hop_length
    )

    return S_db, freq_bins, time_bins, y, sr

def extract_peaks(spectrogram, freq_bins, time_bins, amp_threshold=-15, neighborhood_size=(30, 30)):
    """
    Finds local maxima in the spectrogram that stand out from their neighborhood.
    Returns a list of (time_idx, freq_idx) tuples.
    """
    # Create a boolean mask of local maxima
    local_max = scipy.ndimage.maximum_filter(spectrogram, size=neighborhood_size) == spectrogram
    
    # Filter out quiet peaks (background noise) using amplitude threshold
    background = (spectrogram == 0)
    eroded_background = scipy.ndimage.binary_erosion(background, structure=np.ones((1, 1)), border_value=1)
    
    # Apply thresholds
    peaks_mask = local_max ^ eroded_background
    
    # Get peak coordinates
    freq_idx, time_idx = np.where(peaks_mask & (spectrogram > amp_threshold))
    
    # Sort peaks chronologically for sequential hashing
    sort_idx = np.argsort(time_idx)
    peaks = list(zip(time_idx[sort_idx], freq_idx[sort_idx]))
    
    return peaks

def generate_hashes(peaks, fan_out=15, time_delta_min=1, time_delta_max=100):
    """
    Pairs each peak with its 'fan_out' neighbors to create a combinatorial hash.
    Returns a list of (hash_string, time_offset_of_first_peak).
    """
    hashes = []
    num_peaks = len(peaks)
    
    for i in range(num_peaks):
        t1, f1 = peaks[i]
        
        # Look ahead up to 'fan_out' peaks
        for j in range(1, fan_out + 1):
            if i + j < num_peaks:
                t2, f2 = peaks[i + j]
                delta_t = t2 - t1
                
                # Enforce bounds on the time gap between paired peaks
                if time_delta_min <= delta_t <= time_delta_max:
                    hash_str = f"{f1}|{f2}|{delta_t}"
                    hashes.append((hash_str, t1))
    return hashes

def generate_hashes_single(peaks):
    """Single peak baseline hash (just the frequency)."""
    return [(str(f), t) for t, f in peaks]

def identify_song(y, sr, database, use_pairs=True):
    """
    Identifies a song from an audio array by matching hashes against the database.
    """
    S_db, f_bins, t_bins, _, _ = compute_spectrogram(y=y, sr=sr)
    peaks = extract_peaks(S_db, f_bins, t_bins)
    
    if use_pairs:
        hashes = generate_hashes(peaks)
    else:
        hashes = generate_hashes_single(peaks)
        
    matches = []
    for hash_val, t_query in hashes:
        if hash_val in database:
            for song_name, t_db in database[hash_val]:
                delta_t = t_db - t_query
                matches.append((song_name, delta_t))
                
    if not matches:
        return None, {}

    # Accumulate matches to find the song/offset with the most hits
    match_counts = Counter(matches)
    best_match, max_score = match_counts.most_common(1)[0]
    best_song = best_match[0]
    
    winning_offsets = [offset for (song, offset) in matches if song == best_song]
    
    return best_song, winning_offsets

def get_hash_pairs(peaks, fanout=5, max_pairs=300):
    pairs = []
    for i in range(len(peaks)):
        t1, f1 = peaks[i]
        for j in range(i+1, min(i+fanout+1, len(peaks))):
            t2, f2 = peaks[j]
            pairs.append(((t1, f1), (t2, f2)))
            if len(pairs) >= max_pairs:
                return pairs
    return pairs

def plot_constellation(S_db, f_bins, t_bins, peaks):
    """Plots just the spectrogram and the constellation of peaks."""
    fig, ax = plt.subplots(figsize=(8, 4))
    plt.style.use('dark_background')
    ax.imshow(S_db, aspect='auto', origin='lower', cmap='magma', alpha=1, extent=[t_bins[0], t_bins[-1], f_bins[0], f_bins[-1]])
    
    if peaks:
        t_idx, f_idx = zip(*peaks)
        ax.scatter(t_bins[list(t_idx)], f_bins[list(f_idx)], s=15, marker='o', facecolors='none', edgecolors='cyan')
    
    ax.set_title("Constellation of Peaks")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    plt.tight_layout()
    return fig

def plot_hash_pairs(S_db, f_bins, t_bins, peaks):
    """Plots the target zones and hash pairs."""
    fig, ax = plt.subplots(figsize=(8, 4))
    plt.style.use('dark_background')
    ax.imshow(S_db, aspect='auto', origin='lower', cmap='magma', alpha=1, extent=[t_bins[0], t_bins[-1], f_bins[0], f_bins[-1]])
    
    if peaks:
        t_idx, f_idx = zip(*peaks)
        ax.scatter(t_bins[list(t_idx)], f_bins[list(f_idx)], s=10, color='cyan')
        
        hash_pairs = get_hash_pairs(peaks)
        for (t1_idx, f1_idx), (t2_idx, f2_idx) in hash_pairs:
            ax.plot([t_bins[t1_idx], t_bins[t2_idx]], [f_bins[f1_idx], f_bins[f2_idx]], color='yellow', alpha=0.3, linewidth=0.8)

    ax.set_title("Fingerprint Hash Pairs")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    plt.tight_layout()
    return fig

def plot_histogram(offsets, prediction):
    """Plots the time offset histogram."""
    fig, ax = plt.subplots(figsize=(8, 4))
    plt.style.use('dark_background')
    
    if offsets:
        ax.hist(offsets, bins=50, color='purple', edgecolor='white')
        ax.set_title(f"Offset Alignment for '{prediction}'")
        ax.set_xlabel("Time Offset Delta")
        ax.set_ylabel("Frequency of Match")
    else:
        ax.text(0.5, 0.5, "No Matches Found", ha='center', va='center', color='white')
        ax.set_title("Offset Histogram")
        
    plt.tight_layout()
    return fig

# Main function structure
def main():

    st.title("EE200 Song Identifier")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    with st.spinner("Loading database..."):
        song_db, debug_info = load_database()

    if not song_db:
        st.error("Database could not be loaded.")

        for err in debug_info:
            st.code(err)

        st.write("Current directory:")
        st.code(os.getcwd())

        st.write("Files:")
        st.write(os.listdir("."))

        st.stop()
    # --- Phase 2: User Interface Structure ---
    # Create the three tabs as required by the project specifications
    tab_library, tab_identify, tab_batch = st.tabs(["LIBRARY", "IDENTIFY", "BATCH"])
    
    # --- Phase 3: Building the 'LIBRARY' Tab ---

    with tab_library:
        st.header("Indexed Song Library")
        
        if not song_db:
            st.error("🚨 Database failed to load! Here is exactly why:")
            
            # Print the exact exceptions that caused the crash
            for err in debug_info:
                st.warning(err)
                
            st.write("---")
            st.write("**Files currently on the server:**")
            server_files = []
            base_dir = os.path.dirname(os.path.abspath(__file__))
            for root, dirs, files in os.walk(base_dir):
                for file in files:
                    if ".git" not in root and "__pycache__" not in root:
                        server_files.append(os.path.join(root, file).replace(base_dir, ""))
            st.write(server_files)
        else:
            # Calculate the total number of hashes for each song in the database
            song_hash_counts = {}
            for hash_val, occurrences in song_db.items():
                for song_name, _ in occurrences:
                    song_hash_counts[song_name] = song_hash_counts.get(song_name, 0) + 1
            
            cols = st.columns(3) 
            col_idx = 0
            for song_name, count in sorted(song_hash_counts.items()):
                with cols[col_idx % 3]:
                    with st.container(border=True):
                        st.subheader(song_name)
                        st.write(f"**{count:,}** hashes")
                        
                        image_path = os.path.join(base_dir, "fingerprints", f"{song_name}.png")
                        if os.path.exists(image_path):
                            st.image(image_path, use_container_width=True)
                        else:
                            st.caption("Visual footprint unavailable.")
                col_idx += 1
                
    # --- Phase 4: Building the 'IDENTIFY' Tab --
    with tab_identify:
        st.header("Audio Fingerprinting Engine")
        
        uploaded_file = st.file_uploader("Upload query audio file (.wav, .mp3)", type=['mp3', 'wav'])
        
        if uploaded_file is not None:
            # 1. Top Section: Audio Player and Status
            col_audio, col_status = st.columns([1, 1])
            with col_audio:
                st.audio(uploaded_file)
            
            with st.spinner('Extracting audio features and matching hashes...'):
                try:
                    # Run the pipeline manually here so we have access to the intermediate variables
                    y, sr = librosa.load(uploaded_file, sr=None, mono=True)
                    S_db, f_bins, t_bins, _, _ = compute_spectrogram(y=y, sr=sr)
                    peaks = extract_peaks(S_db, f_bins, t_bins)
                    
                    # Identify the song
                    prediction, offsets = identify_song(y, sr, song_db)
                    
                    with col_status:
                        if prediction:
                            st.success(f"Match Found: **{prediction}**")
                        else:
                            st.error("No match found in the database.")
                    
                    st.divider()
                    
                    # 2. Metrics Dashboard
                    st.subheader("Identification Statistics")
                    m1, m2, m3 = st.columns(3)
                    
                    m1.metric(label="Peaks Extracted", value=len(peaks))
                    
                    if prediction and offsets:
                        # Find the peak of the histogram (the winning offset)
                        most_common_offset = Counter(offsets).most_common(1)[0]
                        m2.metric(label="Total Hash Matches", value=len(offsets))
                        m3.metric(label="Winning Offset (Delta)", value=f"{most_common_offset[0]} frames", delta=f"{most_common_offset[1]} hits")
                    else:
                        m2.metric(label="Total Hash Matches", value=0)
                        m3.metric(label="Winning Offset", value="N/A")
                        
                    st.divider()

                    # 3. Visual Pipeline Layout (Side-by-Side Plots)
                    st.subheader("Signal Processing Pipeline")
                    
                    col_plot1, col_plot2 = st.columns(2)
                    
                    with col_plot1:
                        fig_const = plot_constellation(S_db, f_bins, t_bins, peaks)
                        st.pyplot(fig_const)
                        plt.close(fig_const)
                        
                        fig_hist = plot_histogram(offsets, prediction)
                        st.pyplot(fig_hist)
                        plt.close(fig_hist)
                        
                    with col_plot2:
                        fig_pairs = plot_hash_pairs(S_db, f_bins, t_bins, peaks)
                        st.pyplot(fig_pairs)
                        plt.close(fig_pairs)
                        
                        # You can add a 4th plot here if needed, or leave it empty for a clean layout
                        
                    gc.collect() # Clean up memory
                    
                except Exception as e:
                    st.error(f"An error occurred during processing: {e}")
        
    # --- Phase 5: Building the 'BATCH' Tab ---
    with tab_batch:
        st.header("Batch Processing & Autograder Export")
        st.write("Upload multiple query clips to generate the `results.csv` file.")
        
        # 1. Multi-File Upload
        uploaded_files = st.file_uploader("Upload query audio files", type=['mp3', 'wav'], accept_multiple_files=True)
        
        if uploaded_files:
            # Button to trigger the batch process
            if st.button("Run Batch Processing"):
                results = []
                
                # Create a progress bar so you know it hasn't frozen
                progress_bar = st.progress(0, text="Initializing batch process...")
                
                # 2. Batch Processing Loop
                for i, file in enumerate(uploaded_files):
                    progress_bar.progress((i) / len(uploaded_files), text=f"Processing {file.name}...")
                    
                    try:
                        # Load and process each file
                        y, sr = librosa.load(file, sr=None, mono=True)
                        prediction, _ = identify_song(y, sr, song_db)
                        
                        # Handle cases where no match is found
                        pred_label = prediction if prediction else "No_Match"
                        
                        results.append({"filename": file.name, "prediction": pred_label})
                        
                    except Exception as e:
                        st.error(f"Error processing {file.name}: {e}")
                        results.append({"filename": file.name, "prediction": "Error"})
                
                # Finish progress bar
                progress_bar.progress(1.0, text="Batch processing complete!")
                st.success("All files processed successfully!")
                
                # Display the results on screen
                st.table(results)
                
                # 3. Format the Output for the Autograder
                # Construct the CSV string manually to avoid needing extra libraries like pandas
                csv_content = "filename,prediction\n"
                for r in results:
                    csv_content += f"{r['filename']},{r['prediction']}\n"
                
                # 4. Download Button
                st.download_button(
                    label="Download results.csv",
                    data=csv_content,
                    file_name="results.csv",
                    mime="text/csv",
                    type="primary"
                )

if __name__ == "__main__":
    main()
