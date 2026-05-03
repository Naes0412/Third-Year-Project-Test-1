#combined.py:
# Simultaneously optimises vertex positions (DisplacementMLP) and per-vertex colours (ColourMLP) to stylise a 
# 3D human mesh toward a text prompt using CLIP guidance.

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
import logging

#suppress PyTorch3D bin size warning
logging.getLogger("pytorch3d").setLevel(logging.ERROR)

#ensure output directory exists and is clean
output_dir = "outputs_combined"
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

#Single prompt covering both shape and colour — combined script can use colour words
# - because PointLights rendering makes colour visible to CLIP (unlike white-mesh deformation alone)
base_prompt = "a 3D render of Iron Man armor, red metallic chest plate, gold legs and arms, armored suit silhouette, superhero"

#View-dependent prompts inspired by DreamFusion (Poole et al., NeurIPS 2022)
# - direction suffixes reduce the Janus problem and give CLIP per-view shape+colour signal
viewpoint_prompts = {
    (20, 0):   f"{base_prompt}, front view, red chest plate arc reactor",
    (20, 90):  f"{base_prompt}, side view, gold shoulder pauldron",
    (20, 180): f"{base_prompt}, back view, red and gold back plate",
    (20, 270): f"{base_prompt}, side view, gold shoulder pauldron",
    (60, 45):  f"{base_prompt}, overhead view, red helmet",
    (-10, 45): f"{base_prompt}, low angle view, gold boots",
    (90, 0):   f"{base_prompt}, top down view, red helmet faceplate",
}

#precompute all text features once before training
text_feats = {}
with torch.no_grad():
    for vp, prompt in viewpoint_prompts.items():
        tokens = clip.tokenize([prompt]).to(device)
        feat = clip_model.encode_text(tokens)
        text_feats[vp] = feat / feat.norm(dim=-1, keepdim=True)
        print(f"Encoded prompt for viewpoint {vp}: '{prompt}'")

#negative prompt: push away from plain unarmoured human appearance
with torch.no_grad():
    neg_tokens = clip.tokenize(["a plain human body, smooth skin, no armor, naked"]).to(device)
    neg_feat = clip_model.encode_text(neg_tokens)
    neg_feat = neg_feat / neg_feat.norm(dim=-1, keepdim=True)
    print("Encoded negative prompt.")


# ------------------------------- Load Mesh -------------------------------

mesh_input = trimesh.load("male_human.obj")
if isinstance(mesh_input, trimesh.Scene):
    mesh_input = trimesh.util.concatenate(mesh_input.dump())

mesh_input = mesh_input.simplify_quadric_decimation(face_count=15000)
print(f"Decimated: {len(mesh_input.vertices)} vertices, {len(mesh_input.faces)} faces")

#centre and scale to unit height
mesh_input.vertices -= mesh_input.centroid
scale = 1.0 / (mesh_input.bounds[1][1] - mesh_input.bounds[0][1])
mesh_input.vertices *= scale

verts_init = torch.tensor(mesh_input.vertices, dtype=torch.float32, device=device)
faces = torch.tensor(mesh_input.faces, dtype=torch.int64, device=device)

print(f"Mesh: {verts_init.shape[0]} vertices, {faces.shape[0]} faces")
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
    def __init__(self, num_freqs=6):
        super().__init__()
        self.enc = FourierEncoding(num_freqs)
        in_dim = 3 + 2 * 3 * num_freqs  # 39

        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 3),
            nn.Tanh()
        )

    def forward(self, verts):
        encoded = self.enc(verts)
        raw = self.net(encoded)
        y = verts[:, 1:2]
        torso_mask = (y.abs() < 0.3).float()
        scale = 0.03 + 0.04 * torso_mask
        return verts + scale * raw


# ------------------------------- Colour MLP -------------------------------

class ColourMLP(nn.Module):
    def __init__(self, num_freqs=6):
        super().__init__()
        self.enc = FourierEncoding(num_freqs)
        in_dim = 3 + 2 * 3 * num_freqs  # 39

        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 3),
            nn.Sigmoid()
        )

    def forward(self, verts):
        encoded = self.enc(verts)
        return self.net(encoded)


#separate optimisers for each MLP — allows independent lr tuning if needed
displacement_mlp = DisplacementMLP(num_freqs=6).to(device)
colour_mlp = ColourMLP(num_freqs=6).to(device)

disp_optimiser = torch.optim.Adam(displacement_mlp.parameters(), lr=5e-4)
colour_optimiser = torch.optim.Adam(colour_mlp.parameters(), lr=5e-3)

disp_scheduler = torch.optim.lr_scheduler.StepLR(disp_optimiser, step_size=400, gamma=0.6)
colour_scheduler = torch.optim.lr_scheduler.StepLR(colour_optimiser, step_size=500, gamma=0.7)


# ------------------------------- Augmented Crops -------------------------------

def get_augmented_crops(image, n_crops=8):
    crops = []
    _, h, w = image.shape
    min_size = min(h, w) // 2
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
    neighbour_sum = torch.zeros_like(verts)
    neighbour_count = torch.zeros(verts.shape[0], 1, device=verts.device)
    for i, j in [(0,1),(0,2),(1,0),(1,2),(2,0),(2,1)]:
        neighbour_sum.index_add_(0, faces[:,i], verts[faces[:,j]])
        neighbour_count.index_add_(0, faces[:,i], torch.ones(faces.shape[0], 1, device=verts.device))
    neighbour_mean = neighbour_sum / neighbour_count.clamp(min=1)
    return ((verts - neighbour_mean)**2).mean()

#precompute Laplacian baseline on initial mesh for normalisation
with torch.no_grad():
    lap_baseline = laplacian_smoothness_loss(verts_init, faces).clamp(min=1e-12)
    print(f"Laplacian baseline: {lap_baseline.item():.2e}")


def displacement_regularisation(verts, verts_init):
    diff = verts - verts_init
    weights = torch.tensor([1.0, 3.0, 1.0], device=device)
    return (diff ** 2 * weights).mean()


def colour_smoothness_loss(verts_rgb, faces):
    v0 = verts_rgb[faces[:, 0]]
    v1 = verts_rgb[faces[:, 1]]
    v2 = verts_rgb[faces[:, 2]]
    return ((v0 - v1).pow(2) + (v1 - v2).pow(2) + (v0 - v2).pow(2)).mean()


def saturation_loss(verts_rgb):
    grey = torch.tensor([0.5, 0.5, 0.5], device=device)
    return -((verts_rgb - grey).pow(2).sum(dim=-1).mean())


# ------------------------------- Optimisation Loop -------------------------------

num_steps = 2000
eps = 1e-8
viewpoints = [(20, 0), (20, 90), (20, 180), (20, 270), (60, 45), (-10, 45), (90, 0)]

print(f"\nStarting combined optimisation for {num_steps} steps...")

for step in range(num_steps):

    disp_optimiser.zero_grad()
    colour_optimiser.zero_grad()

    #Step 1: get deformed vertices from DisplacementMLP
    verts = displacement_mlp(verts_init)

    #Step 2: get per-vertex colours from ColourMLP on DEFORMED vertices
    verts_rgb = colour_mlp(verts)

    textures = TexturesVertex(verts_features=verts_rgb.unsqueeze(0))
    mesh_obj = Meshes(verts=[verts], faces=[faces], textures=textures)

    #CLIP loss across viewpoints with augmented crops and negative prompt
    clip_loss = 0
    for elev, azim in viewpoints:
        text_feat_vp = text_feats[(elev, azim)]
        r = get_renderer(elev, azim)
        images = r(mesh_obj)
        image = images[0, ..., :3].permute(2, 0, 1)  # [3, 512, 512]

        #black background: background pixels set to 0 via alpha mask
        # - gives CLIP a cleaner silhouette signal than white-on-white
        alpha = images[0, ..., 3]
        image = image * alpha.unsqueeze(0)

        crops = get_augmented_crops(image, n_crops=8)
        crops = TF.normalize(crops,
                             mean=[0.48145466, 0.4578275,  0.40821073],
                             std= [0.26862954, 0.26130258, 0.27577711])

        img_feats = clip_model.encode_image(crops)
        img_feats = img_feats / (img_feats.norm(dim=-1, keepdim=True) + eps)

        pos_sim = torch.cosine_similarity(img_feats, text_feat_vp.expand(8, -1))
        neg_sim = torch.cosine_similarity(img_feats, neg_feat.expand(8, -1))
        #Maximise similarity to Iron Man prompt, minimise similarity to plain human body
        clip_loss = clip_loss + (1 - pos_sim + neg_sim).mean()

    clip_loss = clip_loss / len(viewpoints)

    #geometry regularisation
    lap_loss = laplacian_smoothness_loss(verts, faces) / lap_baseline
    disp_loss = displacement_regularisation(verts, verts_init)
    centroid_loss = (verts.mean(dim=0) ** 2).sum()

    #colour regularisation
    colour_smooth_loss = colour_smoothness_loss(verts_rgb, faces)
    sat_loss = saturation_loss(verts_rgb)

    disp_weight = 8.0 * (0.1 ** (step / num_steps))

    #Combined loss
    loss = (clip_loss
            + 1.0 * lap_loss
            + disp_weight * disp_loss
            + 0.01 * centroid_loss
            + 0.1 * colour_smooth_loss
            + 0.05 * sat_loss)

    loss.backward()

    #tighter gradient clipping early on, relax after step 500
    max_norm = 0.25 if step < 500 else 1.0
    torch.nn.utils.clip_grad_norm_(displacement_mlp.parameters(), max_norm=max_norm)
    torch.nn.utils.clip_grad_norm_(colour_mlp.parameters(), max_norm=1.0)

    disp_optimiser.step()
    colour_optimiser.step()
    disp_scheduler.step()
    colour_scheduler.step()

    if step % 50 == 0:
        rendered = get_renderer(20, 0)(mesh_obj)[0, ..., :3].detach().cpu().numpy()
        rendered = (rendered * 255).astype(np.uint8)
        Image.fromarray(rendered).save(os.path.join(output_dir, f"render_{step}.png"))

    if step % 20 == 0:
        print(f"Step {step:4d} | Loss: {loss.item():.4f} | CLIP: {clip_loss.item():.4f} | "
              f"Lap: {lap_loss.item():.4f} | Disp: {disp_loss.item():.6f} | "
              f"ColSmooth: {colour_smooth_loss.item():.6f} | Sat: {sat_loss.item():.4f}")

#final render
rendered = get_renderer(20, 0)(mesh_obj)[0, ..., :3].detach().cpu().numpy()
rendered = (rendered * 255).astype(np.uint8)
Image.fromarray(rendered).save(os.path.join(output_dir, f"render_{num_steps}.png"))
print("Optimisation complete.")


# ------------------------------- Save Final Mesh -------------------------------

final_verts = verts.detach().cpu().numpy()
final_faces = faces.detach().cpu().numpy()
final_colours = colour_mlp(verts).detach().cpu().numpy()

final_mesh = trimesh.Trimesh(
    vertices=final_verts,
    faces=final_faces,
    vertex_colors=(final_colours * 255).astype(np.uint8)
)
final_mesh.export(os.path.join(output_dir, "final_result.obj"))
print("Saved final_result.obj")