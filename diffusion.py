from dataclasses import dataclass
from datasets import load_dataset
import sys
from torchvision import transforms
import torch
import torch.nn as nn
from diffusers import UNet2DModel, UNet2DConditionModel
from diffusers import DDPMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers import DDPMPipeline
from diffusers.utils import make_image_grid
import os
from accelerate import Accelerator
from huggingface_hub import create_repo, upload_folder
from tqdm.auto import tqdm
from pathlib import Path
import os
from accelerate import Accelerator
import torch.nn.functional as F
from paired_dataset import PairedImageDataset
from autoencoder_model import ConvAutoencoder
from conditional_ddpm_pipeline import ConditionalDDPMPipeline

device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")

def transform(examples):
    images = [preprocess(image.convert("RGB")) for image in examples["image"]]
    return {"images": images}

TRAIN_BATCH_SIZE = 16

@dataclass
class TrainingConfig:
    image_size = 128  # the generated image resolution
    train_batch_size = TRAIN_BATCH_SIZE
    eval_batch_size = 16  # how many images to sample during evaluation
    num_epochs = 50
    gradient_accumulation_steps = 1
    learning_rate = 1e-4
    lr_warmup_steps = 500
    save_image_epochs = 1
    save_model_epochs = 10
    # TODO(racarr) Reenable 16 bit float later
    mixed_precision = "no"  # `no` for float32, `fp16` for automatic mixed precision
    output_dir = "ddpm-yucfea-128"  # the model name locally and on the HF Hub

    push_to_hub = False  # whether to upload the saved model to the HF Hub
    overwrite_output_dir = True  # overwrite the old model when re-running the notebook
    seed = 0


config = TrainingConfig()

model = UNet2DConditionModel(
    sample_size=config.image_size,
    in_channels=3,
    out_channels=3,
    layers_per_block=1,
    block_out_channels=(64, 128, 128, 256, 256, 512),
    down_block_types=(
        "DownBlock2D",
        "DownBlock2D",
        "DownBlock2D",
        "DownBlock2D",
        "AttnDownBlock2D",
        "DownBlock2D",
    ),
    up_block_types=(
        "UpBlock2D",
        "AttnUpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
    ),
    cross_attention_dim=288,  # Adjust based on your autoencoder's latent space
)

model = model.to(device)
noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

if len(sys.argv) < 3:
    print("Usage: python train_diffusion.py <dataset_dir> <autoencoder_weights_file>")
    sys.exit(1)

dataset_dir = sys.argv[1]
dataset = PairedImageDataset(dataset_dir, max_images=10, image_size=config.image_size, device=device)
train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.train_batch_size, shuffle=True)

autoencoder_weights_file = sys.argv[2]
autoencoder = ConvAutoencoder()
autoencoder.to(device)

autoencoder.load_state_dict(torch.load(autoencoder_weights_file))
for param in autoencoder.parameters():
    param.requires_grad = False

optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=config.lr_warmup_steps,
    num_training_steps=(len(train_dataloader) * config.num_epochs),
)


def evaluate(config, epoch, pipeline, dataloader, encoder):
    # Sample some images from random noise (this is the backward diffusion process).
    # The default pipeline output type is `List[PIL.Image]`


    condition = None
    inputs = None
    for inp, _ in dataloader:
        inputs = inp
        condition = encoder.encoder(inp).reshape(inp.shape[0], -1).unsqueeze(1)
        break
    images = pipeline(
        batch_size=TRAIN_BATCH_SIZE, #config.eval_batch_size,
        generator=torch.Generator(device='cpu').manual_seed(config.seed), # Use a separate torch generator to avoid rewinding the random state of the main training loop
        condition=condition,
        num_inference_steps=30,
    ).images

    # Make a grid out of the images
    image_grid = make_image_grid(images, rows=4, cols=4)
    input_image_grid = make_image_grid(inputs, rows=4, cols=4)

    # Save the images
    test_dir = os.path.join(config.output_dir, "samples")
    os.makedirs(test_dir, exist_ok=True)
    image_grid.save(f"{test_dir}/{epoch:04d}.png")
    input_image_grid.save(f"{test_dir}/{epoch:04d}_input.png")

def train_loop(config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler):
    # Initialize accelerator and tensorboard logging
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with="tensorboard",
        project_dir=os.path.join(config.output_dir, "logs"),
    )
    if accelerator.is_main_process:
        if config.output_dir is not None:
            os.makedirs(config.output_dir, exist_ok=True)
        if config.push_to_hub:
            repo_id = create_repo(
                repo_id=config.hub_model_id or Path(config.output_dir).name, exist_ok=True
            ).repo_id
        accelerator.init_trackers("train_example")

    # Prepare everything
    # There is no specific order to remember, you just need to unpack the
    # objects in the same order you gave them to the prepare method.
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    global_step = 0

    # Now you train the model
    for epoch in range(config.num_epochs):
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(train_dataloader):
            src_images, clean_images = batch
            condition = None
            with torch.no_grad():
                condition = autoencoder.encoder(src_images)
            condition = condition.reshape(condition.shape[0], -1)
            condition = condition.unsqueeze(1)
    


            # Sample noise to add to the images
            noise = torch.randn(clean_images.shape, device=clean_images.device)
            bs = clean_images.shape[0]

            # Sample a random timestep for each image
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bs,), device=clean_images.device,
                dtype=torch.int64
            )

            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            with accelerator.accumulate(model):
                # Predict the noise residual
                noise_pred = model(noisy_images, timesteps, return_dict=False, encoder_hidden_states=condition)[0]
                loss = F.mse_loss(noise_pred, noise)
                accelerator.backward(loss)

                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1

        # After each epoch you optionally sample some demo images with evaluate() and save the model
        if accelerator.is_main_process:
            pipeline = ConditionalDDPMPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_scheduler)

            if (epoch + 1) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:
                evaluate(config, epoch, pipeline, train_dataloader, autoencoder)

            if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
                if config.push_to_hub:
                    upload_folder(
                        repo_id=repo_id,
                        folder_path=config.output_dir,
                        commit_message=f"Epoch {epoch}",
                        ignore_patterns=["step_*", "epoch_*"],
                    )
                else:
                    model.save_pretrained(config.output_dir)

args = (config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler)
#accelerator = Accelerator()
#accelerator.launch(train_loop, args, num_processes=1)
accelerator = Accelerator()
train_loop(*args)