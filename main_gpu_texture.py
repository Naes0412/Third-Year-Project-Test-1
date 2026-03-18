import torch
import torch.nn as nn
import clip
import trimesh
import numpy as np
from PIL import Image

# PyTorch3D imports
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    PointLights,
    TexturesVertex,
    look_at_view_transform
)

import os

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

text_prompt = "Iron Man metallic red and gold suit"

text_tokens = clip.tokenize([text_prompt]).to(device)
with torch.no_grad():
    text_feat = clip_model.encode_text(text_tokens)
    text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
    

# ------------------------------- Load Mesh -------------------------------

mesh_input = trimesh.load("male_human.obj")

if isinstance(mesh_input, trimesh.Scene):
    mesh_input = trimesh.util.concatenate(mesh_input.dump())

# Center and scale to unit size
mesh_input.vertices -= mesh_input.centroid
scale = 1.0 / (mesh_input.bounds[1][1] - mesh_input.bounds[0][1])
mesh_input.vertices *= scale

verts = torch.tensor(mesh_input.vertices, dtype=torch.float32, device=device)
faces = torch.tensor(mesh_input.faces, dtype=torch.int64, device=device)


# ------------------------------- Colour MLP -------------------------------

class ColourMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
            nn.Sigmoid()  # outputs RGB in [0, 1]
        )

    def forward(self, verts):
        return self.net(verts)  # returns per-vertex RGB colours

mlp = ColourMLP().to(device)
optimiser = torch.optim.Adam(mlp.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimiser, step_size=200, gamma=0.5)


# ------------------------------- Renderer -------------------------------

def get_renderer(elev=20, azim=45):
    R, T = look_at_view_transform(3.0, elev, azim)
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T, fov=60)
    raster_settings = RasterizationSettings(
        image_size=224,
        blur_radius=0.0,
        faces_per_pixel=1,
    )
    lights = PointLights(device=device, location=[[2.0, 2.0, -2.0]])
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
        shader=SoftPhongShader(device=device, cameras=cameras, lights=lights)
    )
    return renderer


# ------------------------------- Optimisation Loop -------------------------------

num_steps = 800
eps = 1e-8
viewpoints = [(20, 0), (20, 90), (20, 180), (20, 270), (60, 45), (-10, 45), (90, 0)]

for step in range(num_steps):

    optimiser.zero_grad()

    # get per-vertex colours from MLP
    verts_rgb = mlp(verts).unsqueeze(0)  # shape [1, num_verts, 3]
    textures = TexturesVertex(verts_features=verts_rgb)

    mesh_obj = Meshes(
        verts=[verts],
        faces=[faces],
        textures=textures
    )

    # CLIP loss across multiple viewpoints
    clip_loss = 0
    for elev, azim in viewpoints:
        r = get_renderer(elev, azim)
        images = r(mesh_obj)
        image = images[0, ..., :3].permute(2, 0, 1).unsqueeze(0)
        img_feat = clip_model.encode_image(image)
        img_feat = img_feat / (img_feat.norm(dim=-1, keepdim=True) + eps)
        clip_loss += 1 - torch.cosine_similarity(img_feat, text_feat)
    clip_loss /= len(viewpoints)

    loss = clip_loss
    loss.backward()
    torch.nn.utils.clip_grad_norm_(mlp.parameters(), max_norm=1.0)
    optimiser.step()
    scheduler.step()

    if step % 50 == 0:
        rendered = get_renderer(20, 45)(mesh_obj)[0, ..., :3].detach().cpu().numpy()
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