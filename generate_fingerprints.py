import os
import glob
import gc
import librosa
import matplotlib.pyplot as plt

# Import your core pipeline functions from your app.py file
from app import compute_spectrogram, extract_peaks

def generate_all_fingerprints(song_folder):
    """
    Iterates through all .mp3 files in the target folder, computes their 
    constellation plots, and saves them as .png images for the Streamlit UI.
    """
    output_folder = "fingerprints"
    
    # Create the fingerprints directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Find all mp3 files in the specified directory
    songs = glob.glob(os.path.join(song_folder, "*.mp3"))
    
    if not songs:
        print(f"No .mp3 files found in '{song_folder}'. Please check the path.")
        return

    print(f"Found {len(songs)} songs. Starting fingerprint generation...")

    for i, song_path in enumerate(songs):
        # Extract the song name without the .mp3 extension
        song_name = os.path.basename(song_path).replace(".mp3", "")
        output_path = os.path.join(output_folder, f"{song_name}.png")
        
        # Skip if the image already exists (useful if the script gets interrupted)
        if os.path.exists(output_path):
            print(f"[{i+1}/{len(songs)}] Skipping '{song_name}' (Image already exists)")
            continue
            
        print(f"[{i+1}/{len(songs)}] Processing '{song_name}'...")
        
        try:
            # 1. Load Audio
            y, sr = librosa.load(song_path, sr=None, mono=True)
            
            # 2. Extract Features
            S_db, f_bins, t_bins, _, _ = compute_spectrogram(y=y, sr=sr)
            peaks = extract_peaks(S_db, f_bins, t_bins)
            
            # 3. Create the Plot
            plt.style.use('dark_background')
            # Using a smaller figure size optimized for UI thumbnails
            fig, ax = plt.subplots(figsize=(5, 3)) 
            
            # Plot Spectrogram
            ax.imshow(S_db, aspect='auto', origin='lower', cmap='magma', alpha=0, 
                      extent=[t_bins[0], t_bins[-1], f_bins[0], f_bins[-1]])
            
            # Plot Peaks
            if peaks:
                t_idx, f_idx = zip(*peaks)
                ax.scatter(t_bins[list(t_idx)], f_bins[list(f_idx)], 
                           s=15, marker='o', facecolors='none', edgecolors='cyan')
            
            # --- NEW ADDITION ---
            # Mask out the frequencies below 300 Hz by cropping the viewable area
            ax.set_ylim(bottom=300, top=f_bins[-1])
            # --------------------

            # Turn off axes for a clean, borderless thumbnail
            ax.axis('off') 
            plt.tight_layout(pad=0)
            
            # 4. Save and free up memory
            plt.savefig(output_path, dpi=150, bbox_inches='tight', pad_inches=0)
            plt.close(fig)
            gc.collect()
            
        except Exception as e:
            print(f"Error processing '{song_name}': {e}")
            
    print("\nSuccess! All fingerprints have been generated and saved to the 'fingerprints' folder.")

if __name__ == "__main__":
    # ---> UPDATE THIS PATH to point to your actual directory containing the MP3s <---
    target_song_folder = "songs" 
    
    generate_all_fingerprints(target_song_folder)