from torch.utils import data


def get_loader(dataset, batch_size, shuffle, workers):
    loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers)
    return loader
