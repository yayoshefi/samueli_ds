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

class InfiniteTileDataset(IterableDataset):
    
    def __init__(self, data, tile_summaries, transform=None):
        self.data = data
        self.tile_summaries = tile_summaries
        self.transform = transform
        self.data['num_of_high'] = self.data['slide_num'].apply(lambda x: self.tile_summaries[x].high)
        self.data['num_of_medium'] = self.data['slide_num'].apply(lambda x: self.tile_summaries[x].medium)
        self.data['num_of_low'] = self.data['slide_num'].apply(lambda x: self.tile_summaries[x].low)

    @classmethod
    def generate_dataset(cls, csv_file, transform=None) -> 'InfiniteTileDataset':
        data = pd.read_csv(csv_file)
        tile_summaries = cls._create_tile_summaries(data)
        return cls(data, tile_summaries, transform)
    
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
        tiles_df = pd.DataFrame()
        tile_summaries_records = []

        for _, row in self.data.iterrows():
            slide_num = row['slide_num']
            ts = self.tile_summaries[slide_num]
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
            tmp_tiles.rename(columns={'tissue_percentage': 't_p'}, inplace=True)
            tiles_df = pd.concat([tiles_df, tmp_tiles], ignore_index=True)
            
        tiles_summary_df = pd.DataFrame.from_records(tile_summaries_records)
        
        tiles_df.to_pickle(os.path.join(dst_dir, 'tiles.pkl'))
        tiles_summary_df.to_pickle(os.path.join(dst_dir, 'tiles_summary.pkl'))

    @classmethod
    def load_dataset(cls, csv_file:str, src_dir: str, transform=None) -> 'InfiniteTileDataset':
        data = pd.read_csv(csv_file)

        tiles_df = pd.read_pickle(os.path.join(src_dir, 'tiles.pkl'))
        tiles_summary_df = pd.read_pickle(os.path.join(src_dir, 'tiles_summary.pkl'))

        tile_summaries = {}
        for _, row in tiles_summary_df.iterrows():
            ts = TileSummary(row['slide_num'], row['orig_w'], row['orig_h'], row['orig_tile_w'], row['orig_tile_h'],
                             row['scaled_w'], row['scaled_h'], row['scaled_tile_w'], row['scaled_tile_h'],
                             row['tissue_percentage'], row['num_col_tiles'], row['num_row_tiles'])
            ts.count = row['count']
            ts.high, ts.medium, ts.low, ts.none = row['high'], row['medium'], row['low'], row['none']

            slide_tile_df = tiles_df[tiles_df['slide_num'] == ts.slide_num]
            tiles = [Tile(**tile) for _, tile in slide_tile_df.drop(columns=["rank"]).iterrows()]
            for tile in tiles:  # fill ranks (which are not part of the constructor)
                tile.rank = slide_tile_df[slide_tile_df.tile_num==tile.tile_num]['rank'].values[0]
                tile.tile_summary = ts
            ts.tiles = tiles
            tile_summaries[ts.slide_num] = ts
        return cls(data, tile_summaries, transform)


    # TODO: replace the iteratable dataset with a regular dataset and then change the __get_item__ / iter to make sure we take even number of pos, negative
    def __iter__(self):
        while True:
            for _, row in self.data.iterrows():
                slide_num = row['slide_num']
                label = row['tag']
                tile_summary = self.tile_summaries[slide_num]
                tile = random.choice(tile_summary.tiles)
                image = tile.get_np_tile()
                
                data = dict(slide_num=slide_num, tile_row=tile.r, tile_col=tile.c, label=label)
                data.update(dict(tissue_percent=tile.tissue_percentage, score=float(tile.score)))  # Main scores
                data.update(dict(color_factor=float(tile.color_factor), s_and_v_factor=tile.s_and_v_factor, quantity_factor=tile.quantity_factor))

                if self.transform:
                    image = self.transform(image=image)['image']
                yield image, label, data

        
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
def create_dataloader(csv_file, batch_size=32, num_workers=4, is_train=True):
    transform=train_transform if is_train else test_transform
    dataset = InfiniteTileDataset(csv_file, transform=transform)
    dataloader = DataLoader(
        dataset, batch_size=batch_size,
        num_workers=num_workers,
        # shuffle=True if is_train else False,
        collate_fn=collate_fn
        )
    return dataloader


if __name__ == "__main__":
    import time
    # Test the dataloader
    tic = time.time()
    dataloader = create_dataloader('data/datasets/tupac_16_100/data.csv', batch_size=32, num_workers=0, is_train=True)
    toc = time.time()
    print(f"Time to create dataloader: {toc-tic:.2f} seconds")

    # Dataset Stats:
    df = dataloader.dataset.data
    min_width, min_height = tiles.ROW_TILE_SIZE, tiles.COL_TILE_SIZE
    for row in df.itertuples():
        tile_sum = dataloader.dataset.tile_summaries[row.slide_num]
        for tile in tile_sum.tiles:
            width = tile.o_c_e - tile.o_c_s
            min_width = min(min_width, width)
            height = tile.o_r_e - tile.o_r_s
            min_height = min(min_height, height)
        df.loc[row.Index, 'min_width'] = min_width
        df.loc[row.Index, 'min_height'] = min_height
    print(df)

    for i, batch in enumerate(dataloader):
        print(f"Batch {i}: {len(batch)} entities, and {len(batch[0])} images")
        print(batch[0].shape, batch[1].shape)
        if i == 2:
            break
