import torch
import numpy as np

class Dataset(torch.utils.data.Dataset):
    def __init__(self,
        path,                   # path of the dataset.
    ):
        super(Dataset, self).__init__()
        self.path = path
        print(self.path)
        with open(path, 'r') as f:
            self.texts = f.readlines()

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = self.texts[index]
        out = {}
        out['id'], out['caption'], out['code'] = text.split('\t')[1:]
        out['code'] = out['code'].strip()
        # Str to int
        out['code'] = torch.tensor([int(x) for x in out['code'].split(' ')]).unsqueeze(0)
        return out

if __name__ == '__main__':
    dataset = Dataset(path='GenerativeResults/vq_diffusion/custom_data_code.txt')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    for batch in dataloader:
        breakpoint()
        print(batch)