import h5py
import torch


class HDF5Dataset(torch.utils.data.Dataset):

    def __init__(self, file_path, dataset_name, label_name='label', transform=None):
        self.file_path = file_path
        self.dataset_name = dataset_name
        self.label_name = label_name
        self.transform = transform
        self.dataset = None
        self.label = None
        self.file = None
        with h5py.File(self.file_path, 'r') as file:
            self.dataset_len = len(file[self.dataset_name])

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, index):
        if self.file is None:
            self.file = h5py.File(self.file_path, 'r')

        x_in = self.file[self.dataset_name][index]
        label_in = self.file[self.label_name][index]
        if self.transform is not None:
            x_in = self.transform(x_in)
        return x_in, label_in

    def close(self):
        if self.file is not None:
            self.file.close()
            self.file = None

    def __del__(self):
        self.close()

