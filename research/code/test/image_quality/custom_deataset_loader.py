from PIL import Image, ImageFile
import tqdm
import glob
ImageFile.LOAD_TRUNCATED_IMAGES = True

class DatasetLoader(Dataset):

    def __init__(self, X, y, input_transform=None, label_transform=None):

        self.data = X
        self.labels = y
        self.input_transform = input_transform
        self.label_transform = label_transform

    @staticmethod
    def load_dataset(data_dir: str):
        logger.debug(f"load_dataset: Loading dataset from {data_dir}")

        inputs_dir = f'{data_dir}/inputs'
        labels_dir = f'{data_dir}/labels'

        inputs = []
        for image_path in tqdm(glob.glob(inputs_dir + '/*')):
            image = Image.open(image_path)
            inputs.append(image)

        labels = []
        for image_path in tqdm(glob.glob(labels_dir + '/*')):
            label = Image.open(image_path).convert('L')
            labels.append(label)

        return inputs, labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        if self.input_transform is not None:
            data = self.input_transform(data)

        label = self.labels[idx]
        if self.label_transform is not None:
            label = self.label_transform(label)
        return data, label
