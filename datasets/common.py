import collections
import glob
import json
import os
import random

import numpy as np
import torch
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Dataset, Sampler
from tqdm import tqdm


class SubsetSampler(Sampler):
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (i for i in self.indices)

    def __len__(self):
        return len(self.indices)


class ImageFolderWithPaths(datasets.ImageFolder):
    def __init__(self, path, transform, flip_label_prob=0.0):
        super().__init__(path, transform)
        self.flip_label_prob = flip_label_prob
        if self.flip_label_prob > 0:
            print(f"Flipping labels with probability {self.flip_label_prob}")
            num_classes = len(self.classes)
            for i in range(len(self.samples)):
                if random.random() < self.flip_label_prob:
                    new_label = random.randint(0, num_classes - 1)
                    self.samples[i] = (self.samples[i][0], new_label)

    def __getitem__(self, index):
        image, label = super(ImageFolderWithPaths, self).__getitem__(index)
        return {"images": image, "labels": label, "image_paths": self.samples[index][0]}


def maybe_dictionarize(batch):
    if isinstance(batch, dict):
        return batch

    if len(batch) == 2:
        batch = {"images": batch[0], "labels": batch[1]}
    elif len(batch) == 3:
        batch = {"images": batch[0], "labels": batch[1], "metadata": batch[2]}
    else:
        raise ValueError(f"Unexpected number of elements: {len(batch)}")

    return batch


def get_features_helper(image_encoder, dataloader, device):
    all_data = collections.defaultdict(list)

    image_encoder = image_encoder.to(device)
    image_encoder = torch.nn.DataParallel(
        image_encoder, device_ids=[x for x in range(torch.cuda.device_count())]
    )
    image_encoder.eval()

    with torch.no_grad():
        for batch in tqdm(dataloader):
            batch = maybe_dictionarize(batch)
            features = image_encoder(batch["images"].cuda())

            all_data["features"].append(features.cpu())

            for key, val in batch.items():
                if key == "images":
                    continue
                if hasattr(val, "cpu"):
                    val = val.cpu()
                    all_data[key].append(val)
                else:
                    all_data[key].extend(val)

    for key, val in all_data.items():
        if torch.is_tensor(val[0]):
            all_data[key] = torch.cat(val).numpy()

    return all_data


def get_features(is_train, image_encoder, dataset, device):
    split = "train" if is_train else "val"
    dname = type(dataset).__name__
    if image_encoder.cache_dir is not None:
        cache_dir = f"{image_encoder.cache_dir}/{dname}/{split}"
        cached_files = glob.glob(f"{cache_dir}/*")
    if image_encoder.cache_dir is not None and len(cached_files) > 0:
        print(f"Getting features from {cache_dir}")
        data = {}
        for cached_file in cached_files:
            name = os.path.splitext(os.path.basename(cached_file))[0]
            data[name] = torch.load(cached_file)
    else:
        print(f"Did not find cached features at {cache_dir}. Building from scratch.")
        loader = dataset.train_loader if is_train else dataset.test_loader
        data = get_features_helper(image_encoder, loader, device)
        if image_encoder.cache_dir is None:
            print("Not caching because no cache directory was passed.")
        else:
            os.makedirs(cache_dir, exist_ok=True)
            print(f"Caching data at {cache_dir}")
            for name, val in data.items():
                torch.save(val, f"{cache_dir}/{name}.pt")
    return data


class FeatureDataset(Dataset):
    def __init__(self, is_train, image_encoder, dataset, device):
        self.data = get_features(is_train, image_encoder, dataset, device)

    def __len__(self):
        return len(self.data["features"])

    def __getitem__(self, idx):
        data = {k: v[idx] for k, v in self.data.items()}
        data["features"] = torch.from_numpy(data["features"]).float()
        return data


def _extract_targets(ds):
    """Return list of integer class labels.

    Supported fast paths:
    - FeatureDataset (cached numpy array of labels)
    - Objects exposing .targets or .labels (torchvision datasets)
    - torch.utils.data.Subset (recursive)
    Fallback: iterate.
    """
    if isinstance(ds, FeatureDataset):
        labels = ds.data.get("labels")
        if torch.is_tensor(labels):
            labels = labels.tolist()
        return list(labels)
    if hasattr(ds, "targets"):
        t = ds.targets
        if torch.is_tensor(t):
            t = t.tolist()
        return list(t)
    if hasattr(ds, "labels"):
        t = ds.labels
        if torch.is_tensor(t):
            t = t.tolist()
        return list(t)
    if hasattr(ds, "dataset") and hasattr(ds, "indices"):
        base = _extract_targets(ds.dataset)
        return [base[i] for i in ds.indices]
    return [
        (ds[i][1] if isinstance(ds[i], (list, tuple)) else ds[i]["labels"])  # type: ignore[index]
        for i in range(len(ds))
    ]


def _build_undersampled_subset(ds, seed=None):
    """Return a torch.utils.data.Subset with equal number of samples per class (no replacement).
    Chooses min class count, slices each class list, shuffles combined indices.
    """
    labels = _extract_targets(ds)
    rng = random.Random(seed)
    per_class = {}
    for idx, y in enumerate(labels):
        per_class.setdefault(int(y), []).append(idx)
    # Shuffle each class list
    for lst in per_class.values():
        rng.shuffle(lst)
    counts_before = {c: len(idxs) for c, idxs in per_class.items()}
    min_count = min(counts_before.values())
    balanced_indices = []
    for lst in per_class.values():
        balanced_indices.extend(lst[:min_count])
    rng.shuffle(balanced_indices)

    # Compact logging
    def _fmt_counts(d):
        items = sorted(d.items(), key=lambda x: x[0])
        if len(items) <= 12:
            return "{" + ", ".join(f"{k}:{v}" for k, v in items) + "}"
        head = ", ".join(f"{k}:{v}" for k, v in items[:5])
        tail = ", ".join(f"{k}:{v}" for k, v in items[-5:])
        return "{" + head + ", ... , " + tail + "}"

    print(
        f"[undersample] classes={len(per_class)} min_count={min_count} "
        f"orig_total={len(labels)} new_total={len(balanced_indices)}"
    )
    print(f"[undersample] orig_counts={_fmt_counts(counts_before)}")
    print(f"[undersample] new_count_per_class={min_count}")

    return torch.utils.data.Subset(ds, balanced_indices)


def get_dataloader(dataset, is_train, args, image_encoder=None):
    undersample = is_train and getattr(args, "undersample", False)
    seed = getattr(args, "seed", None)

    if image_encoder is not None:
        base_ds = FeatureDataset(is_train, image_encoder, dataset, args.device)
        if undersample:
            base_ds = _build_undersampled_subset(base_ds, seed=seed)
        return DataLoader(base_ds, batch_size=args.batch_size, shuffle=is_train)

    if not is_train:
        return dataset.test_loader

    if undersample:
        balanced_subset = _build_undersampled_subset(dataset.train_dataset, seed=seed)
        return DataLoader(balanced_subset, batch_size=args.batch_size, shuffle=True)

    return dataset.train_loader
