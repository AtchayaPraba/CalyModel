import yaml
import sys
import warnings
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
import os

from finetune.regression.biomasters_datamodule import BioMastersDataModule
from finetune.regression.biomasters_model import BioMastersClassifier

class Prediction:
    def __init__(self, config_path):
        self.config = self.read_config(config_path)
        self.model = None
        self.metadata = None
    
    def read_config(self, config_path):
        """
        Reads the YAML configuration file and returns the configuration as a dictionary.
        """
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config

    def load_model(self):
        """
        Loads the model using paths specified in the configuration.
        """
        biomasters_checkpoint_path = self.config['model']['biomasters_checkpoint_path']
        clay_checkpoint_path = self.config['model']['clay_checkpoint_path']
        metadata_path = self.config['metadata_path']
        
        self.model = BioMastersClassifier.load_from_checkpoint(
            checkpoint_path=biomasters_checkpoint_path,
            metadata_path=metadata_path,
            ckpt_path=clay_checkpoint_path,
        )
        self.model.eval()
        print("Model loaded successfully !")

    def prepare_data(self):
        """
        Prepares the data using paths and parameters specified in the configuration.
        """
        train_chip_dir = self.config['data']['train_chip_dir']
        train_label_dir = self.config['data']['train_label_dir']
        val_chip_dir = self.config['data']['val_chip_dir']
        val_label_dir = self.config['data']['val_label_dir']
        metadata_path = self.config['metadata_path']
        batch_size = self.config['batch_size']
        num_workers = self.config['num_workers']

        dm = BioMastersDataModule(
            train_chip_dir=train_chip_dir,
            train_label_dir=train_label_dir,
            val_chip_dir=val_chip_dir,
            val_label_dir=val_label_dir,
            metadata_path=metadata_path,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        dm.setup(stage="fit")
        val_dl = iter(dm.val_dataloader())
        batch = next(val_dl)
        self.metadata = dm.metadata
        print("Data prepared successfully !")
        return batch

    def run_prediction(self, batch):
        """
        Runs prediction on a given batch of data using the loaded model.
        """
        if self.model is None:
            raise ValueError("Model is not loaded. Please load the model first.")
        
        with torch.no_grad():
            outputs = self.model(batch)
        outputs = F.interpolate(
            outputs, size=(256, 256), mode="bilinear", align_corners=False
        )
        print("Prediction completed")
        return outputs

    def denormalize_images(self, normalized_images, means, stds):
        """
        Denormalizes the normalized images using the provided means and stds.
        """
        means = np.array(means).reshape(1, -1, 1, 1)
        stds = np.array(stds).reshape(1, -1, 1, 1)
        denormalized_images = normalized_images * stds + means
        return denormalized_images

    def post_process(self, batch, outputs, metadata):
        """
        Post-processes the batch, outputs, and metadata to produce images, labels, and predictions.
        """
        labels = batch["label"].detach().cpu().numpy()
        pixels = batch["pixels"].detach().cpu().numpy()
        outputs = outputs.detach().cpu().numpy()

        means = list(metadata["sentinel-2-l2a"].bands.mean.values())
        stds = list(metadata["sentinel-2-l2a"].bands.std.values())
        norm_pixels = self.denormalize_images(pixels, means, stds)

        # Rearrange the image channels
        images = rearrange(norm_pixels[:, :3, :, :], "b c h w -> b h w c")

        # Clipping the values for labels, outputs, and images
        labels = np.clip(labels.squeeze(axis=1), 0, 400)
        outputs = np.clip(outputs.squeeze(axis=1), 0, 400)
        images = np.clip(images / 2000, 0, 1)

        return images, labels, outputs
    
    def save_results(self, images, labels, outputs, output_folder):
        """
        Saves images, labels, and outputs as .npz files in the desired output folder.
        """
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        # Define the file paths
        images_path = os.path.join(output_folder, 'images.npz')
        labels_path = os.path.join(output_folder, 'labels.npz')
        outputs_path = os.path.join(output_folder, 'outputs.npz')
        
        # Save the numpy arrays as .npz files
        np.savez(images_path, images=images)
        np.savez(labels_path, labels=labels)
        np.savez(outputs_path, outputs=outputs)
        
        print(f"Saved images to {images_path}")
        print(f"Saved labels to {labels_path}")
        print(f"Saved outputs to {outputs_path}")


    def run(self):
        """
        Runs the entire prediction pipeline: loading model, preparing data, running prediction, post-processing, and saving results.
        """
        self.load_model()
        batch = self.prepare_data()
        predictions = self.run_prediction(batch)
        images, labels, outputs = self.post_process(batch, predictions, self.metadata)
        output_folder = self.config.get('output_folder', './output')  # Default to './output' if not specified
        self.save_results(images, labels, outputs, output_folder)
        return images, labels, outputs

# Example usage:
if __name__ == "__main__":
    # Pass the path of the config file
    config_file = 'configs/biomasters_inference.yaml'
    predictor = Prediction(config_file)
    images, labels, outputs = predictor.run()
    # You can now work with 'result', which contains the denormalized predictions.
