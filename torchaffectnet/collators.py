import torch

from typing import Any, Dict


def collate_fn(examples):
    imgs, targets = zip(*examples)
    pixel_values = torch.stack(imgs)
    targets = torch.stack(targets)
    return {'pixel_values': pixel_values, 'labels': targets}


def contrastive_collate_fn(examples):
    imgs, targets = zip(*examples)
    imgs1, imgs2 = zip(*imgs)
    imgs1 = torch.stack(imgs1)
    imgs2 = torch.stack(imgs2)
    pixel_values = torch.cat([imgs1, imgs2])
    targets = torch.stack(targets)
    return {'pixel_values': pixel_values, 'labels': targets}


class Collator:
    def __init__(self) -> None:
        pass

    def __call__(self, examples) -> Any:
        return self.collate_fn(examples)

    def collate_fn(self, examples) -> Dict[str, Any]:
        imgs, targets = zip(*examples)
        pixel_values = torch.stack(imgs)
        targets = torch.stack(targets)
        return {'pixel_values': pixel_values, 'labels': targets}


class ContrastiveCollator(Collator):
    def __init__(self, return_labels: bool = False) -> None:
        super().__init__()
        self.return_labels = return_labels

    def collate_fn(self, examples) -> Dict[str, Any]:
        if self.return_labels:
            imgs, targets = zip(*examples)
            targets = torch.stack(targets)
        else:
            imgs = examples
        imgs1, imgs2 = zip(*imgs)
        imgs1 = torch.stack(imgs1)
        imgs2 = torch.stack(imgs2)
        pixel_values = torch.cat([imgs1, imgs2])
        if self.return_labels:
            return {'pixel_values': pixel_values, 'labels': targets}
        else:
            return {'pixel_values': pixel_values}
