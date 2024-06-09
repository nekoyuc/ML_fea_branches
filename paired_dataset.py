import torchvision
from torch.utils.data import Dataset
from PIL import Image
import os

class PairedImageDataset(Dataset):
    def __init__(self, image_dir, max_images, image_size, device, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.max_images = max_images
        self.image_size = image_size
        self.device = device
        if (transform is None):
            self.transform = torchvision.transforms.Compose([
                torchvision.transforms.Resize([image_size, image_size]),
                torchvision.transforms.ToTensor(),
            ])
        self.image_pairs = self._find_pairs()

    def _find_pairs(self):
      image_pairs = [(filename, filename[:-8] + "_OUT.png") 
                for filename in os.listdir(self.image_dir) 
                if filename.endswith("_INP.png")][:self.max_images]
      return image_pairs

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        inp_name, out_name = self.image_pairs[idx]
        inp_path = os.path.join(self.image_dir, inp_name)
        out_path = os.path.join(self.image_dir, out_name)

        inp_image = Image.open(inp_path)
        out_image = Image.open(out_path)
        

        if self.transform:
            inp_image = self.transform(inp_image)
            out_image = self.transform(out_image)
        #print(inp_image.max())
      #  inp_img = torch.log(inp_image+1e-4)
       # out_img = torch.log(out_image+1e-4)
        #inp_img = inp_image - inp_img.min()
        #inp_img = inp_img / inp_img.max()
        #out_img = out_image - out_img.min()
        #out_img = out_img / out_img.max()

        out_image = out_image*2.0 -1.0
        inp_image = inp_image*2.0 -1.0

        return inp_image.to(self.device), out_image.to(self.device)