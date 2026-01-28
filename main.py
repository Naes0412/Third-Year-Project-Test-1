import torch
import clip
import torchvision.transforms as T

from pytorch3d.utils import ico_sphere
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    RasterizationSettings,
    PointLights,
    TexturesVertex,
)

# --------------------------------------------------
# 0. Device (CPU for macOS)
# --------------------------------------------------
device = torch.device("cpu")

# --------------------------------------------------
# 1. Load CLIP (frozen)
# --------------------------------------------------
clip_model, _ = clip.load("ViT-B/32", device=device)
clip_model.eval()

text_prompt = "a giraffe"
text_tokens = clip.tokenize([text_prompt]).to(device)

with torch.no_grad():
    text_feat = clip_model.encode_text(text_tokens)
    text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)

# --------------------------------------------------
# 2. Create a simple mesh (sphere)
# --------------------------------------------------
mesh = ico_sphere(level=3, device=device)

verts = mesh.verts_packed().clone().detach()
faces = mesh.faces_packed()

# These are the parameters we optimize
verts.requires_grad_(True)

textures = TexturesVertex(
    verts_features=torch.ones_like(verts)[None]
)

# --------------------------------------------------
# 3. Renderer setup
# --------------------------------------------------
cameras = FoVPerspectiveCameras(device=device)
lights = PointLights(device=device, location=[[0.0, 0.0, 3.0]])

raster_settings = RasterizationSettings(
    image_size=224,
    blur_radius=0.0,
    faces_per_pixel=1,
)

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

# --------------------------------------------------
# 4. Optimizer
# --------------------------------------------------
optimizer = torch.optim.Adam([verts], lr=1e-3)

# CLIP image normalization
clip_normalize = T.Normalize(
    mean=(0.48145466, 0.4578275, 0.40821073),
    std=(0.26862954, 0.26130258, 0.27577711),
)

# --------------------------------------------------
# 5. Optimization loop
# --------------------------------------------------
num_steps = 200

for step in range(num_steps):
    mesh = Meshes(
        verts=[verts],
        faces=[faces],
        textures=textures
    )

    image = renderer(mesh)[..., :3]  # (1, H, W, 3)
    image = image.permute(0, 3, 1, 2)  # -> (1, 3, H, W)
    image = clip_normalize(image)

    img_feat = clip_model.encode_image(image)
    img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)

    loss = 1 - torch.cosine_similarity(img_feat, text_feat)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 20 == 0:
        print(f"Step {step:03d} | Loss: {loss.item():.4f}")

print("Optimization finished.")
