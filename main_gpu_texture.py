import torch
import torch.nn as nn
import clip
import trimesh
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF

# PyTorch3D imports
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    PointLights,
    AmbientLights,
    TexturesVertex,
    look_at_view_transform
)

import os

from google.colab import drive
drive.mount('/content/drive')

# Ensure output directory exists
output_dir = "outputs"
if os.path.exists(output_dir):
    for f in os.listdir(output_dir):
        full_path = os.path.join(output_dir, f)
        if os.path.isfile(full_path):
            os.remove(full_path)
else:
    os.makedirs(output_dir)

# ------------------------------- Device -------------------------------

device = torch.device("cuda")


# ------------------------------- Load CLIP -------------------------------

clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
clip_model.eval()

text_prompt = "a 3D render of Iron Man armor, red metallic chest plate, gold legs and arms, superhero suit"

text_tokens = clip.tokenize([text_prompt]).to(device)
with torch.no_grad():
    text_feat = clip_model.encode_text(text_tokens)
    text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
    

# ------------------------------- Load Mesh -------------------------------

mesh_input = trimesh.load("male_human.obj")

#center and scale to unit size
mesh_input.vertices -= mesh_input.centroid
scale = 1.0 / (mesh_input.bounds[1][1] - mesh_input.bounds[0][1])
mesh_input.vertices *= scale

verts = torch.tensor(mesh_input.vertices, dtype=torch.float32, device=device)
faces = torch.tensor(mesh_input.faces, dtype=torch.int64, device=device)


# ------------------------------- Fourier Encoding -------------------------------

class FourierEncoding(nn.Module):
    def __init__(self, num_freqs=6):
        super().__init__()
        # Frequencies: 2^0 * pi, 2^1 * pi, ..., 2^(num_freqs-1) * pi
        # Registered as a buffer so it moves to the correct device with .to(device)
        self.register_buffer(
            'freqs',
            2.0 ** torch.arange(num_freqs).float() * torch.pi
        )

    def forward(self, x):
        # x: [N, 3]
        x_freq = x.unsqueeze(-1) * self.freqs   # [N, 3, num_freqs]
        x_freq = x_freq.reshape(x.shape[0], -1) # [N, 3 * num_freqs]
        #concatenate raw coords + sin + cos encodings
        return torch.cat([x, torch.sin(x_freq), torch.cos(x_freq)], dim=-1)
        #output dim: 3 + 2 * 3 * num_freqs = 39 (for num_freqs=6)


# ------------------------------- Colour MLP -------------------------------

class ColourMLP(nn.Module):
    def __init__(self, num_freqs=6):
        super().__init__()
        self.enc = FourierEncoding(num_freqs)
        in_dim = 3 + 2 * 3 * num_freqs
        
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 3),
            nn.Sigmoid()  #outputs RGB in [0, 1]
        )

    def forward(self, verts):
        encoded = self.enc(verts)
        return self.net(encoded)  #returns per-vertex RGB colours

mlp = ColourMLP(num_freqs=6).to(device)
optimiser = torch.optim.Adam(mlp.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimiser, step_size=200, gamma=0.5)


# ------------------------------- Renderer -------------------------------

#can render the mesh from any viewpoint
# - elev controls the vertical angle of the camera, azim controls the horizontal angle
def get_renderer(elev=0, azim=0):
    R, T = look_at_view_transform(2.0, elev, azim)
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T, fov=30)
    raster_settings = RasterizationSettings(
        image_size=512,
        blur_radius=0.0,
        faces_per_pixel=1,
    )
    
    #ambient light for flat lighting, no shadows
    lights = AmbientLights(device=device)
    
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras, 
            raster_settings=raster_settings),
        shader=SoftPhongShader(
            device=device, 
            cameras=cameras, 
            lights=lights)
    )
    return renderer


# ------------------------------- Optimisation Loop -------------------------------

num_steps = 800
eps = 1e-8
viewpoints = [(20, 0), (20, 90), (20, 180), (20, 270), (60, 45), (-10, 45), (90, 0)]

def colour_smoothness_loss(verts_rgb, faces):
        v0 = verts_rgb[faces[:, 0]]
        v1 = verts_rgb[faces[:, 1]]
        v2 = verts_rgb[faces[:, 2]]
        return ((v0 - v1).pow(2) + (v1 - v2).pow(2) + (v0 - v2).pow(2)).mean()

for step in range(num_steps):

    optimiser.zero_grad()

    #get per-vertex colours from MLP
    verts_rgb = mlp(verts)
    textures = TexturesVertex(verts_features=verts_rgb.unsqueeze(0))

    mesh_obj = Meshes(
        verts=[verts],
        faces=[faces],
        textures=textures
    )

    #CLIP loss across multiple viewpoints
    clip_loss = 0
    for elev, azim in viewpoints:
        r = get_renderer(elev, azim)
        images = r(mesh_obj)
        image = images[0, ..., :3].permute(2, 0, 1).unsqueeze(0)
        image = TF.resize(image, [224, 224])
        
        #normalise using CLIP's mean and std
        image = TF.normalize(image, 
            mean=[0.48145466, 0.4578275,  0.40821073],
            std= [0.26862954, 0.26130258, 0.27577711])

        img_feat = clip_model.encode_image(image)
        img_feat = img_feat / (img_feat.norm(dim=-1, keepdim=True) + eps)
        clip_loss += 1 - torch.cosine_similarity(img_feat, text_feat)
    clip_loss /= len(viewpoints)

    #colour smooth loss to suppress noisy colour variation
    colour_smooth_loss = colour_smoothness_loss(verts_rgb, faces)
    
    for step in range(num_steps):
        optimiser.zero_grad()
        verts_rgb = mlp(verts)
    
        #colour smoothness weight: start high, decay to let CLIP sharpen details
        colour_smooth_weight = 1.0 * (0.1 ** (step / num_steps))  # 1.0 -> 0.1
    
    loss = clip_loss + colour_smooth_weight * colour_smooth_loss
    loss.backward()
    torch.nn.utils.clip_grad_norm_(mlp.parameters(), max_norm=1.0)
    optimiser.step()
    scheduler.step()

    if step % 50 == 0:
        rendered = get_renderer(0, 0)(mesh_obj)[0, ..., :3].detach().cpu().numpy()
        rendered = (rendered * 255).astype(np.uint8)
        Image.fromarray(rendered).save(os.path.join(output_dir, f"render_{step}.png"))

    if step % 20 == 0:
        print(f"Step {step} | Loss: {loss.item():.4f}")

print("Optimisation complete.")


# ------------------------------- Save Final Mesh -------------------------------

final_verts = verts.detach().cpu().numpy()
final_faces = faces.detach().cpu().numpy()
final_colours = mlp(verts).detach().cpu().numpy()

final_mesh = trimesh.Trimesh(
    vertices=final_verts,
    faces=final_faces,
    vertex_colors=(final_colours * 255).astype(np.uint8)
)
final_mesh.export(os.path.join(output_dir, "final_result.obj"))
print("Saved final_result.obj")