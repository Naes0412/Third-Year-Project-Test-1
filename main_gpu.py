import torch
import clip
import trimesh
import numpy as np
from PIL import Image

# -------------------------------
# Device (use GPU if available)
# -------------------------------
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -------------------------------
# Load CLIP
# -------------------------------
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
clip_model.eval()

text_prompt = "a tall, narrow box"
text_tokens = clip.tokenize([text_prompt]).to(device)
with torch.no_grad():
    text_feat = clip_model.encode_text(text_tokens)
    text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)

# -------------------------------
# Build PyTorch3D mesh
# -------------------------------
import pytorch3d
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    MeshRenderer, MeshRasterizer, SoftPhongShader,
    RasterizationSettings, FoVPerspectiveCameras,
    PointLights, TexturesVertex
)

# Create initial mesh (cube + subdivisions)
mesh_trimesh = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
mesh_trimesh = mesh_trimesh.subdivide().subdivide().subdivide()  # higher res

verts = torch.tensor(mesh_trimesh.vertices, dtype=torch.float32, device=device, requires_grad=True)
faces = torch.tensor(mesh_trimesh.faces, dtype=torch.int64, device=device)

# Vertex colors (uniform grey)
colors = torch.ones_like(verts)[None] * 0.7  # (1, V, 3)
textures = TexturesVertex(verts_features=colors.to(device))

mesh = Meshes(
    verts=[verts],
    faces=[faces],
    textures=textures
)

# -------------------------------
# PyTorch3D Renderer Setup
# -------------------------------
# Define rasterization and shading settings
raster_settings = RasterizationSettings(
    image_size=224,
    blur_radius=0.0,
    faces_per_pixel=1,
)

# Cameras with randomisable viewpoints
def get_cameras(elev, azim):
    R, T = pytorch3d.renderer.look_at_view_transform(dist=2.7, elev=elev, azim=azim)
    return FoVPerspectiveCameras(device=device, R=R, T=T)

# Simple phong lights
lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])

# One shader, rasterizer pair
renderer = MeshRenderer(
    rasterizer=MeshRasterizer(raster_settings=raster_settings),
    shader=SoftPhongShader(device=device, lights=lights)
)

# Multi‑view camera angles
camera_views = [
    (0, 0),
    (0, 90),
    (0, 180),
    (0, 270),
    (60, 45),
    (30, 135),
    (75, 225),
    (30, 315)
]

# -------------------------------
# Laplacian Matrix for smoothing
# -------------------------------
def compute_laplacian(vertices, faces_idx):
    n = vertices.shape[0]
    L = torch.zeros((n, n), device=vertices.device)
    for f in faces_idx:
        for i, j in [(0, 1), (1, 2), (2, 0)]:
            L[f[i], f[j]] = 1
            L[f[j], f[i]] = 1
    D = torch.diag(L.sum(dim=1))
    L = D - L
    return L

L = compute_laplacian(verts, mesh_trimesh.faces)

# -------------------------------
# Optimiser
# -------------------------------
optimiser = torch.optim.Adam([verts], lr=1e-3)
num_steps = 350
lambda_smooth = 2.0
eps = 1e-8

for step in range(num_steps):
    total_loss = 0.0

    # Batched rendering
    batch_meshes = mesh.extend(len(camera_views))
    cameras = get_cameras(
        elev=[v[0] for v in camera_views],
        azim=[v[1] for v in camera_views],
    )

    images = renderer(batch_meshes, cameras=cameras, lights=lights)  # (N, H, W, 4)

    for img in images:
        img_rgb = img[..., :3]
        img_pil = Image.fromarray((img_rgb.cpu().numpy() * 255).astype(np.uint8))
        img_tensor = clip_preprocess(img_pil).unsqueeze(0).to(device)
        with torch.no_grad():
            img_feat = clip_model.encode_image(img_tensor)
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True) + eps
        clip_loss = 1 - torch.cosine_similarity(img_feat, text_feat)
        total_loss = total_loss + clip_loss

    avg_loss = total_loss / len(camera_views)

    # Regularisers
    smooth_loss = lambda_smooth * torch.trace(verts.t() @ L @ verts)

    min_xyz = verts.min(dim=0).values
    max_xyz = verts.max(dim=0).values
    extents = max_xyz - min_xyz

    tallness_loss = ((extents[0] - extents[1])**2 + (extents[2] - extents[1])**2)
    volume = torch.prod(extents)
    volume_loss = (volume - 1.0).abs()
    centroid = verts.mean(dim=0)
    upright_loss = centroid[0]**2 + centroid[2]**2
    xz = torch.sqrt(verts[:, 0]**2 + verts[:, 2]**2 + eps)
    symmetry_loss = torch.var(xz)

    loss_total = avg_loss + 0.5 * tallness_loss + 0.5 * smooth_loss \
                 + 0.3 * upright_loss + 0.2 * symmetry_loss + 0.1 * volume_loss

    optimiser.zero_grad()
    loss_total.backward()
    optimiser.step()

    # Vertex clamping for stability
    with torch.no_grad():
        verts.data.clamp_(-2.0, 2.0)

    if step % 10 == 0:
        print(f"Step {step} | CLIP {avg_loss.item():.4f} | Smooth {smooth_loss.item():.4f}")

print("Optimisation done")
