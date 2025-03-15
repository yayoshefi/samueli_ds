import os
import pandas as pd
import numpy as np
import random
import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
from torchvision import transforms
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

from python_wsi_preproc import slide, tiles
from python_wsi_preproc.tiles import TileSummary, Tile
from balanced_tile_sampler import BalancedTilehSampler

class TileDataset(Dataset):    
    def __init__(self, slides_df, tiles_df, transform=None):
        self.slides_df = slides_df
        self.tiles_df = tiles_df
        self.transform = transform

        # commons
        self.tiles_args = [
            'tile_summary', 'slide_num', 'np_scaled_tile', 'tile_num',
            'r', 'c', 'r_s', 'r_e', 'c_s', 'c_e', 'o_r_s', 'o_r_e', 'o_c_s', 'o_c_e',
            't_p', 'color_factor', 's_and_v_factor','quantity_factor', 'score'
        ]
            

    @classmethod
    def generate_dataset(cls, csv_file, transform=None) -> 'TileDataset':
        data = pd.read_csv(csv_file)
        tile_summaries = cls._create_tile_summaries(data)
        
        tiles_df = pd.DataFrame()
        tile_summaries_records = []

        for _, row in data.iterrows():
            slide_num = row['slide_num']
            ts = tile_summaries[slide_num]
            tile_sumary_record = dict(
                slide_num=ts.slide_num,
                orig_w=ts.orig_w, orig_h=ts.orig_h,
                orig_tile_w=ts.orig_tile_w, orig_tile_h=ts.orig_tile_h,
                scaled_w=ts.scaled_w, scaled_h=ts.scaled_h,
                scaled_tile_w=ts.scaled_tile_w, scaled_tile_h=ts.scaled_tile_h,
                tissue_percentage=ts.tissue_percentage,
                num_col_tiles=ts.num_col_tiles, num_row_tiles=ts.num_row_tiles,
                # extended attributes
                count=ts.count, high=ts.high, medium=ts.medium, low=ts.low, none=ts.none,
            )
            tile_summaries_records.append(tile_sumary_record)

            tmp_tiles = pd.DataFrame.from_records([tile.__dict__ for tile in ts.tiles])
            tmp_tiles["tile_summary"] = None
            # tmp_tiles.rename(columns={'tissue_percentage': 't_p'}, inplace=True)
            tiles_df = pd.concat([tiles_df, tmp_tiles], ignore_index=True)
            
        tiles_summary_df = pd.DataFrame.from_records(tile_summaries_records)
        slides_df = data.merge(tiles_summary_df, on="slide_num", how="left")

        return cls(slides_df, tiles_df, transform)
    
    @staticmethod
    def _create_tile_summaries(data):
        tile_summaries = {}
        for _, row in data.iterrows():
            slide_num = row['slide_num']
            if slide_num not in tile_summaries:
                tile_sum = tiles.summary_and_tiles(slide_num, display=False, save_data=False, save_top_tiles=False)
                # remove empty tiles
                tile_sum.tiles = [tile for tile in tile_sum.tiles if tile.tissue_percentage > 0]
                tile_summaries[slide_num] = tile_sum

        return tile_summaries
    
    def dump_dataset(self, dst_dir: str):
        """ save the dataset to disk in a format of 2 pickles: tiles.pkl and tiles_summary.pkl

        Args:
            dst_dir (str): directory path to dump the files
        """
        self.tiles_df.to_pickle(os.path.join(dst_dir, 'tiles.pkl'))
        self.slides_df.to_pickle(os.path.join(dst_dir, 'slides.pkl'))

    @classmethod
    def load_dataset(cls, src_dir: str, transform=None) -> 'TileDataset':
        tiles_df = pd.read_pickle(os.path.join(src_dir, 'tiles.pkl'))
        slides_df = pd.read_pickle(os.path.join(src_dir, 'slides.pkl'))

        return cls(slides_df, tiles_df, transform)

    def __len__(self):
        return len(self.tiles_df)

    def __getitem__(self, index):
        tile_srs = self.tiles_df.iloc[index]
        tile_srs = tile_srs.rename({'tissue_percentage': 't_p'})
        tile = Tile(**tile_srs[self.tiles_args])
        image = tile.get_np_tile()
        label = self.slides_df[self.slides_df.slide_num == tile.slide_num].tag.values[0]
        
        data = dict(slide_num=tile.slide_num, tile_num=tile.tile_num, label=label)
        data.update(dict(tissue_percent=tile.tissue_percentage, score=float(tile.score)))  # Main scores
        data.update(dict(color_factor=float(tile.color_factor), s_and_v_factor=tile.s_and_v_factor, quantity_factor=tile.quantity_factor))
        
        if self.transform:
            image = self.transform(image=image)['image']
        return image, label, data
    
        
def collate_fn(batch):
    images, labels, data = zip(*batch)
    images = torch.stack(images)
    labels = torch.tensor(labels).unsqueeze(1)
    return images, labels, data

# Transforms
# ===========
train_transform = A.Compose([
    A.RandomCrop(448, 448),  # Min size of actual tiles
    A.Resize(224, 224),
    A.HorizontalFlip(),
    A.VerticalFlip(),
    A.RandomRotate90(),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])
test_transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

# Build DataLoader
# ================
def create_dataloader(src_dir, batch_size=32, num_workers=4, is_train=True):
    transform=train_transform if is_train else test_transform
    dataset = TileDataset.load_dataset(src_dir=src_dir, transform=transform)
    if is_train:
        kwds = dict(batch_sampler=BalancedTilehSampler(dataset, batch_size))
    else:
        kwds = dict(batch_size=32, shuffle=False)
    dataloader = DataLoader(dataset, num_workers=num_workers, collate_fn=collate_fn,**kwds)
    return dataloader


if __name__ == "__main__":
    import time
    # Test the dataloader
    tic = time.time()
    all_csv = 'data/datasets/tupac_16/data_full.csv'
    test_csv = 'data/datasets/tupac16/data_test.csv'
    
    # dataset = TileDataset.generate_dataset(csv_file=test_csv, transform=train_transform)
    dataset = TileDataset.load_dataset(src_dir="data/datasets/tupac16/test", transform=test_transform)

    dataloader = DataLoader(dataset, batch_size=32, num_workers=0, shuffle=True, collate_fn=collate_fn)
    toc = time.time()
    print(f"Time to create dataloader: {toc-tic:.2f} seconds")

    # Dataset Stats:
    tmp = dataset.tiles_df.assign(
            width=lambda tile: tile.o_c_e - tile.o_c_s,
            height=lambda tile: tile.o_r_e - tile.o_r_s,
    )
    stats_df = tmp.groupby("slide_num").min(["width", "height"])
    print(stats_df)
    
    for i, batch in enumerate(dataloader):
        print(f"Batch {i}: {len(batch)} entities, and {len(batch[0])} images")
        print(batch[0].shape, batch[1].shape)
        if i == 2:
            break
