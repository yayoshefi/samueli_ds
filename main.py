import os
import pandas as pd
import glob
import json
import openslide

def parser():
    import argparse
    parser = argparse.ArgumentParser(description='Processing SVS file.')
    parser.add_argument('--dataset', "-d", type=str, default='single_sample',
                         help='Dataset name', choices=os.listdir("data/datasets"))
    args = parser.parse_args()
    return args


def temp_svs(wsi_path):
    slide = openslide.OpenSlide(wsi_path)

    print("#"*50)
    # Get basic properties
    print(f"Dimensions: {slide.dimensions}")  # Full-resolution size (width, height)
    print(f"Levels: {slide.level_count}")  # Number of resolution levels
    print(f"Level Dimensions: {slide.level_dimensions}")  # Dimensions of each pyramid level
    print(f"Level Downsamples: {slide.level_downsamples}")  # Downsample factor for each level
    print(f"Magnification: {slide.properties.get('aperio.AppMag')}X")  # Objective magnification

    # Extract a thumbnail
    thumbnail = slide.get_thumbnail((500, 500))
    thumbnail.show()  # Opens a small preview

    # Extract a region (x, y, width, height)
    region = slide.read_region((1000, 1000), level=0, size=(512, 512))  # Reads at highest resolution
    region.show()
    print()


if __name__ ==  "__main__":
    args = parser()
    dataset_name = args.dataset

    print(f"Parsing Dataset: {dataset_name}")

    for f in glob.glob(f"data/datasets/{dataset_name}/metadata*"):
        df = pd.read_json(f)
        df_entities = pd.json_normalize(df['associated_entities'].explode())
        df = df.drop(columns=['associated_entities']).join(df_entities)

        print(f"Metadata file: {f} - {df.shape[0]} rows (total size: {df['file_size'].sum()/1024**3:.2f} GB)")
        
        for row in df.itertuples():
            wsi_path = f"data/raw/{row.file_name}"
            exp_file_size = row.file_size
            exp_md5 = row.md5sum
            # actual
            file_stat = os.stat(wsi_path)
            act_file_size = file_stat.st_size

            success = exp_file_size == act_file_size
            print(f"File: {row.file_name} - size: {exp_file_size/1024**3:.2f} GB (downloaded succes: {success})")
            # print(f"Expected file size: {exp_file_size}, Actual file size: {act_file_size}")

            if success:
                temp_svs(wsi_path)
        