from autoencoder_model import ConvAutoencoder
import sys
import torchvision
from PIL import Image
from conditional_ddpm_pipeline import ConditionalDDPMPipeline
import torch

image_size = 128

device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")

diffusion_model_dir = sys.argv[1]
pipe = ConditionalDDPMPipeline.from_pretrained(diffusion_model_dir, use_safetensors=True)
pipe = pipe.to(device)
autoencoder_weights_file = sys.argv[2]
autoencoder = ConvAutoencoder()
autoencoder = autoencoder.to(device)
autoencoder.load_state_dict(torch.load(autoencoder_weights_file))

condition_image_path = sys.argv[3]
condition_image = Image.open(condition_image_path)
transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize([image_size, image_size]),
            torchvision.transforms.ToTensor(),
])
condition_image = transform(condition_image).to(device)
condition_image = autoencoder.encoder(condition_image.unsqueeze(0))
condition_image = condition_image.reshape(-1).unsqueeze(0).unsqueeze(0)
print(condition_image.shape)
images = pipe(
        batch_size=1, #config.eval_batch_size,
        #generator=torch.Generator(device='cpu').manual_seed(config.seed), # Use a separate torch generator to avoid rewinding the random state of the main training loop
        condition=condition_image,
        num_inference_steps=30,
).images

image = images[0]
image.save(f"{condition_image_path[:-4]}_ddpm_generated_image.png")

