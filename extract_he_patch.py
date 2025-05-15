import os
import pandas as pd
import openslide
from PIL import Image
from tqdm import tqdm
import multiprocessing as mp
from functools import partial

def process_slide(file, label_dir, he_dir, output_dir, patch_size):
    label_df = pd.read_csv(os.path.join(label_dir, file))
    he_id = file.split('.')[0]
    he_path = os.path.join(he_dir, f'{he_id}.svs')

    if os.path.exists(he_path):
        print(f"Processing {he_id}")

        # Open the slide
        slide = openslide.OpenSlide(he_path)

        # Create a subdirectory for this slide's patches
        slide_output_dir = os.path.join(output_dir, he_id)
        os.makedirs(slide_output_dir, exist_ok=True)

        # Iterate through each row in the DataFrame
        for index, row in label_df.iterrows():
            x = int(row['x'])
            y = int(row['y'])

            # Extract the patch
            patch = slide.read_region((x, y), 0, (patch_size, patch_size))

            # Convert to RGB (remove alpha channel)
            patch = patch.convert('RGB')

            # Save the patch as PNG
            patch_filename = f"{row['slide_index']}.png"
            patch_path = os.path.join(slide_output_dir, patch_filename)
            patch.save(patch_path)

        # Close the slide
        slide.close()

    else:
        print(f"H&E image not found for {he_id}")

def main():
    label_dir = ''
    he_dir = ''
    output_dir = ''

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Get all the csv files in the label_dir
    label_files = [x for x in os.listdir(label_dir) if x.endswith('.csv')]

    # Define patch size (adjust as needed)
    patch_size = 224

    # Set up multiprocessing
    num_processes = 8
    pool = mp.Pool(processes=num_processes)

    # Create a partial function with fixed arguments
    process_slide_partial = partial(process_slide, label_dir=label_dir, he_dir=he_dir, output_dir=output_dir, patch_size=patch_size)

    # Process slides in parallel with progress bar
    for _ in tqdm(pool.imap_unordered(process_slide_partial, label_files), total=len(label_files), desc="Processing slides"):
        pass

    # Close the pool
    pool.close()
    pool.join()

    print("Processing complete.")

if __name__ == "__main__":
    main()