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

text_prompt = "a pyramid shape"

text_tokens = clip.tokenize([text_prompt]).to(device)
with torch.no_grad():
    text_feat = clip_model.encode_text(text_tokens)
    text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)


# ------------------------------- Create Mesh -------------------------------

mesh = trimesh.creation.box(extents=(1, 1, 1))
mesh = mesh.subdivide().subdivide()

verts = torch.tensor(mesh.vertices, dtype=torch.float32, device=device)
faces = torch.tensor(mesh.faces, dtype=torch.int64, device=device)

# -------------------------------- MLP Displacement Network -------------------------------

class DisplacementMLP(nn.Module):
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
            nn.Tanh()
        )

    def forward(self, verts):
        # max displacement of 0.2 units per vertex
        return verts + 0.2 * self.net(verts)

mlp = DisplacementMLP().to(device)
optimiser = torch.optim.Adam(mlp.parameters(), lr=1e-3)

scheduler = torch.optim.lr_scheduler.StepLR(optimiser, step_size=200, gamma=0.5)


# ------------------------------- Differentiable Renderer -------------------------------

def get_renderer(elev=20, azim=45):
    R, T = look_at_view_transform(2.5, elev, azim)
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T)

    raster_settings = RasterizationSettings(
        image_size=224,
        blur_radius=0.0,
        faces_per_pixel=1,
    )

    lights = PointLights(device=device, location=[[2.0, 2.0, -2.0]])

    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings
        ),
        shader=SoftPhongShader(
            device=device,
            cameras=cameras,
            lights=lights
        )
    )

    return renderer


# ------------------------------- Optimisation Loop -------------------------------

num_steps = 800
eps = 1e-8
viewpoints = [(20, 0), (20, 90), (20, 180), (20, 270), (60, 45), (-10, 45)]

#save initial vertices for regularisation
verts_init = verts.clone()

for step in range(num_steps):

    optimiser.zero_grad()
    
    verts = mlp(verts_init)
    verts_rgb = torch.ones_like(verts)[None]
    textures = TexturesVertex(verts_features=verts_rgb)

    mesh = Meshes(
        verts=[verts],
        faces=[faces],
        textures=textures
    )

    clip_loss = 0
    
    #render the mesh from multiple viewpoints and compute the CLIP loss for each view, 
    #encouraging the mesh to match the text prompt from all angles
    for elev, azim in viewpoints:
        r = get_renderer(elev, azim)
        images = r(mesh)
        image = images[0, ..., :3].permute(2, 0, 1).unsqueeze(0)
        img_feat = clip_model.encode_image(image)
        img_feat = img_feat / (img_feat.norm(dim=-1, keepdim=True) + eps)
        clip_loss += 1 - torch.cosine_similarity(img_feat, text_feat)
        
    #average the CLIP loss across all viewpoints to get a more stable optimisation signal
    clip_loss /= len(viewpoints)
    
    #centroid_loss encourages the mesh to stay centered around the origin, preventing it from drifting too far away
    centroid = verts.mean(dim=0)
    centroid_loss = 0.01 * (centroid ** 2).sum()
    
    #displacement_loss encourages the vertices to not deviate too much from their initial positions, which helps maintain a reasonable shape and prevents extreme distortions
    displaced = mlp(verts_init)
    displacement_loss = 0.01 * ((displaced - verts_init) ** 2).mean()
    
    reg_loss = centroid_loss + displacement_loss

    loss = clip_loss + reg_loss
    loss.backward()
    torch.nn.utils.clip_grad_norm_(mlp.parameters(), max_norm=1.0)
    optimiser.step()
    
    #save renders every 50 steps to visualise optimisation progress
    if step % 50 == 0:
        rendered = get_renderer(20,45)(mesh)[0, ..., :3].detach().cpu().numpy()
        rendered = (rendered * 255).astype(np.uint8)
        Image.fromarray(rendered).save(os.path.join(output_dir, f"render_{step}.png"))

    #print the loss every 20 steps to monitor optimisation progress
    if step % 20 == 0:
        print(f"Step {step} | Loss: {loss.item():.4f}")
    
    #reduce learning rate every 200 steps to allow for finer adjustments as optimisation progresses
    scheduler.step()

print("Optimisation complete.")


# ------------------------------- Save final mesh to file -------------------------------

final_verts = verts.detach().cpu().numpy()
final_faces = faces.detach().cpu().numpy()

final_mesh = trimesh.Trimesh(vertices=final_verts, faces=final_faces)

final_mesh.export(os.path.join(output_dir, "final_result.obj"))
print("Saved final_result.obj")