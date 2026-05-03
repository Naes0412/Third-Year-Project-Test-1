# deformation.py:
# This code optimises the vertex positions of a 3D mesh to match a text prompt using CLIP.
# It uses a simple MLP with Fourier feature encoding to predict per-vertex displacements,
# and a differentiable renderer to render the mesh from multiple viewpoints for CLIP loss calculation.

import site
site.addsitedir('/content/drive/MyDrive/pytorch3d_cache')
import torch
import torch.nn as nn
import clip
import trimesh
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
import torchvision.transforms as T
#PyTorch3D imports
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
import warnings
import logging
#suppress the PyTorch3D bin size warning
logging.getLogger("pytorch3d").setLevel(logging.ERROR)

#ensure output directory exists
output_dir = "outputs_deformation"
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

clip_model, clip_preprocess = clip.load("ViT-B/16", device=device)
clip_model.eval()

base_prompt = "3D render of Iron Man armor, armored suit silhouette, chest plate protrusion, helmet"

#View-dependent prompts inspired by DreamFusion (Poole et al., 2022)
# - each viewpoint gets a direction-specific suffix appended to the base prompt
viewpoint_prompts = {
    (20, 0): f"{base_prompt}, front view, chest plate",
    (20, 90): f"{base_prompt}, side view, shoulder pauldron",
    (20, 180): f"{base_prompt}, back view, back plate",
    (20, 270): f"{base_prompt}, side view, shoulder pauldron",
    (60, 45): f"{base_prompt}, overhead view, helmet",
    (-10, 45): f"{base_prompt}, low angle view, boots",
    (90, 0): f"{base_prompt}, top down view, helmet faceplate",
}

text_feats = {}
with torch.no_grad():
    for vp, prompt in viewpoint_prompts.items():
        tokens = clip.tokenize([prompt]).to(device)
        feat = clip_model.encode_text(tokens)
        text_feats[vp] = feat / feat.norm(dim=-1, keepdim=True)
        print(f"Encoded text prompt for viewpoint {vp}: '{prompt}'")


# ------------------------------- Load Mesh -------------------------------

mesh_input = trimesh.load("male_human.obj")

#handle scene vs mesh
if isinstance(mesh_input, trimesh.Scene):
    mesh_input = trimesh.util.concatenate(mesh_input.dump())

#center and scale to unit size
mesh_input.vertices -= mesh_input.centroid
scale = 1.0 / (mesh_input.bounds[1][1] - mesh_input.bounds[0][1])
mesh_input.vertices *= scale

verts_init = torch.tensor(mesh_input.vertices, dtype=torch.float32, device=device)
faces = torch.tensor(mesh_input.faces, dtype=torch.int64, device=device)

print(f"Mesh loaded: {verts_init.shape[0]} vertices, {faces.shape[0]} faces")
print(f"Y range: {verts_init[:, 1].min().item():.3f} to {verts_init[:, 1].max().item():.3f}")
print(f"X range: {verts_init[:, 0].min().item():.3f} to {verts_init[:, 0].max().item():.3f}")


# ------------------------------- Fourier Encoding -------------------------------

class FourierEncoding(nn.Module):
    def __init__(self, num_freqs=6):
        super().__init__()
        self.register_buffer(
            'freqs',
            2.0 ** torch.arange(num_freqs).float() * torch.pi
        )

    def forward(self, x):
        # x: [N, 3]
        x_freq = x.unsqueeze(-1) * self.freqs   # [N, 3, num_freqs]
        x_freq = x_freq.reshape(x.shape[0], -1) # [N, 3 * num_freqs]
        # output dim: 3 + 2 * 3 * num_freqs = 39 (for num_freqs=6)
        return torch.cat([x, torch.sin(x_freq), torch.cos(x_freq)], dim=-1)


# ------------------------------- Displacement MLP -------------------------------

class DisplacementMLP(nn.Module):
    def __init__(self, num_freqs=6, displacement_scale=0.07):
        super().__init__()
        self.enc = FourierEncoding(num_freqs)
        self.displacement_scale = displacement_scale
        in_dim = 3 + 2 * 3 * num_freqs  # 39

        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 3),
            nn.Tanh()  # bounds displacement to [-1, 1] before scaling
        )

    def forward(self, verts):
        encoded = self.enc(verts)
        raw = self.net(encoded)
        y = verts[:, 1:2]
        torso_mask = (y.abs() < 0.3).float()
        scale = 0.03 + 0.04 * torso_mask  # allow more deformation in torso region, less in extremities
        return verts + scale * raw

mlp = DisplacementMLP(num_freqs=6).to(device)
optimiser = torch.optim.Adam(mlp.parameters(), lr=5e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimiser, step_size=400, gamma=0.6)


# ------------------------------- Augmented Crops -------------------------------

def get_augmented_crops(image, n_crops=8):
    crops = []
    _, h, w = image.shape
    min_size = min(h, w) // 2  # min crop is 50% of image
    for _ in range(n_crops):
        scale = torch.FloatTensor(1).uniform_(0.5, 1.0).item()
        size = int(min_size + scale * (min(h, w) - min_size))
        crop = T.RandomCrop(size)(image)
        crop = TF.resize(crop.unsqueeze(0), [224, 224])
        crops.append(crop)
    return torch.cat(crops, dim=0)  # [n_crops, 3, 224, 224]


# ------------------------------- Renderer -------------------------------

def get_renderer(elev=0, azim=0):
    R, T = look_at_view_transform(2.0, elev, azim)
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T, fov=30)
    raster_settings = RasterizationSettings(
        image_size=512,
        blur_radius=0.0,
        faces_per_pixel=1,
    )
    
    #point lights to prevent CLIP seeing flat white/grey blobs - adds shading and depth cues to the render
    lights = PointLights(
        device=device,
        location=[[2.0, 2.0, -2.0]],
        ambient_color=[[0.4, 0.4, 0.4]],
        diffuse_color=[[0.6, 0.6, 0.6]],
        specular_color=[[0.2, 0.2, 0.2]]
    )

    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
        shader=SoftPhongShader(device=device, cameras=cameras, lights=lights)
    )
    return renderer


# ------------------------------- Regularisation Losses -------------------------------

def laplacian_smoothness_loss(verts, faces):
    # Build adjacency: for each vertex, get mean of neighbour positions
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]  
    v2 = verts[faces[:, 2]]
    
    # Accumulate neighbour sums
    neighbour_sum = torch.zeros_like(verts)
    neighbour_count = torch.zeros(verts.shape[0], 1, device=verts.device)
    
    for i, j in [(0,1),(0,2),(1,0),(1,2),(2,0),(2,1)]:
        neighbour_sum.index_add_(0, faces[:,i], verts[faces[:,j]])
        neighbour_count.index_add_(0, faces[:,i], torch.ones(faces.shape[0],1,device=verts.device))
    
    neighbour_mean = neighbour_sum / neighbour_count.clamp(min=1)
    # Laplacian: how far each vertex is from its neighbours' mean
    return ((verts - neighbour_mean)**2).mean()
    

#penalises large displacements from the original mesh to preserve human topology
# - allows for armor-scale shape changes while preventing collapse or explosion of the mesh
def displacement_regularisation(verts, verts_init):
    diff = verts - verts_init
    #penalise vertical (Y) displacement more than lateral (X/Z)
    # Iron Man armor adds bulk outward, not height — this prevents elongation
    weights = torch.tensor([1.0, 3.0, 1.0], device=device)
    return (diff ** 2 * weights).mean()

def normal_consistency_loss(verts, faces):
    #compute per-face normals
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    normals = torch.cross(v1 - v0, v2 - v0, dim=1)
    normals = normals / (normals.norm(dim=1, keepdim=True) + 1e-8)
    
    return normals.var(dim=0).mean()


# ------------------------------- Optimisation Loop -------------------------------

num_steps = 1500
eps = 1e-8
viewpoints = [(20, 0), (20, 90), (20, 180), (20, 270), (60, 45), (-10, 45), (90, 0)]

#sanity check for Laplacian before training:
with torch.no_grad():
    test_lap = laplacian_smoothness_loss(verts_init, faces)
    print(f"Laplacian sanity check on verts_init: {test_lap.item():.8f}")

for step in range(num_steps):

    optimiser.zero_grad()

    #get displaced vertices from MLP
    verts = mlp(verts_init)

    #white vertex colours - shape only, no colour signal
    verts_rgb = torch.ones_like(verts).unsqueeze(0)
    textures = TexturesVertex(verts_features=verts_rgb)
    mesh_obj = Meshes(verts=[verts], faces=[faces], textures=textures)

    #CLIP loss across multiple viewpoints with augmented crops
    clip_loss = 0
    for elev, azim in viewpoints:
        text_feat_vp = text_feats[(elev, azim)] #get pre-encoded text feature for this viewpoint
        r = get_renderer(elev, azim)
        images = r(mesh_obj)
        image = images[0, ..., :3].permute(2, 0, 1)  # [3, 512, 512]

        crops = get_augmented_crops(image, n_crops=8)
        crops = TF.normalize(crops,
                             mean=[0.48145466, 0.4578275,  0.40821073],
                             std= [0.26862954, 0.26130258, 0.27577711])

        img_feats = clip_model.encode_image(crops)
        img_feats = img_feats / (img_feats.norm(dim=-1, keepdim=True) + eps)
        clip_loss = clip_loss + (1 - torch.cosine_similarity(img_feats, text_feat_vp.expand(8, -1))).mean()

    clip_loss = clip_loss / len(viewpoints)

    #regularisation losses to preserve human mesh topology
    lap_loss = laplacian_smoothness_loss(verts, faces)
    disp_loss = displacement_regularisation(verts, verts_init)
    norm_consist_loss = normal_consistency_loss(verts, faces)

    #centroid loss keeps mesh centered at origin
    centroid_loss = (verts.mean(dim=0) ** 2).sum()

    #displacement reg starts strong then decays to allow more deformation later
    disp_weight = 8.0 * (0.1 ** (step / num_steps))
    
    #combine - CLIP drives shape, regularisation preserves topology
    loss = clip_loss + 1.0 * lap_loss + disp_weight * disp_loss + 0.01 * centroid_loss # + 0.05 * norm_consist_loss

    loss.backward()
    max_norm = 0.25 if step < 500 else 1.0 #tighter clipping early on to prevent divergence, then relax to allow finer details
    torch.nn.utils.clip_grad_norm_(mlp.parameters(), max_norm=max_norm)
    optimiser.step()
    scheduler.step()

    if step % 50 == 0:
        rendered = get_renderer(20, 0)(mesh_obj)[0, ..., :3].detach().cpu().numpy()
        rendered = (rendered * 255).astype(np.uint8)
        Image.fromarray(rendered).save(os.path.join(output_dir, f"render_{step}.png"))

    if step % 20 == 0:
        print(f"Step {step:4d} | Loss: {loss.item():.4f} | CLIP: {clip_loss.item():.6f} | Lap: {lap_loss.item():.6f} | Disp: {disp_loss.item():.6f}")

#saving final render
rendered = get_renderer(20, 0)(mesh_obj)[0, ..., :3].detach().cpu().numpy()
rendered = (rendered * 255).astype(np.uint8)
Image.fromarray(rendered).save(os.path.join(output_dir, f"render_{num_steps}.png"))

print("Optimisation complete.")


# ------------------------------- Save Final Mesh -------------------------------

final_verts = verts.detach().cpu().numpy()
final_faces = faces.detach().cpu().numpy()

final_mesh = trimesh.Trimesh(vertices=final_verts, faces=final_faces)
final_mesh.export(os.path.join(output_dir, "final_result.obj"))
print("Saved final_result.obj")