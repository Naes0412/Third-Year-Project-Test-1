import torch
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

# ------------------------------- Device -------------------------------

device = torch.device("cuda")


# ------------------------------- Load CLIP -------------------------------

clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
clip_model.eval()

text_prompt = "a tall, narrow object"

text_tokens = clip.tokenize([text_prompt]).to(device)
with torch.no_grad():
    text_feat = clip_model.encode_text(text_tokens)
    text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)


# ------------------------------- Create Mesh -------------------------------

mesh = trimesh.creation.box(extents=(1, 1, 1))
mesh = mesh.subdivide().subdivide()

verts = torch.tensor(mesh.vertices, dtype=torch.float32, device=device, requires_grad=True)
faces = torch.tensor(mesh.faces, dtype=torch.int64, device=device)

# White vertex colour
verts_rgb = torch.ones_like(verts)[None]
textures = TexturesVertex(verts_features=verts_rgb)

# ------------------------------- Differentiable Renderer -------------------------------

def get_renderer():
    R, T = look_at_view_transform(2.5, 20, 45)
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

renderer = get_renderer()

# ------------------------------- Optimiser -------------------------------

optimiser = torch.optim.Adam([verts], lr=1e-2)
num_steps = 300
eps = 1e-8

# ------------------------------- Optimisation Loop -------------------------------

for step in range(num_steps):

    optimiser.zero_grad()

    mesh = Meshes(
        verts=[verts],
        faces=[faces],
        textures=textures
    )

    images = renderer(mesh)
    image = images[0, ..., :3]  # RGB only
    image = image.permute(2, 0, 1).unsqueeze(0)

    img_feat = clip_model.encode_image(image)
    img_feat = img_feat / (img_feat.norm(dim=-1, keepdim=True) + eps)

    clip_loss = 1 - torch.cosine_similarity(img_feat, text_feat)

    # simple regularisation
    centroid = verts.mean(dim=0)
    reg_loss = 0.1 * (centroid ** 2).sum()

    loss = clip_loss + reg_loss
    loss.backward()
    optimiser.step()
    
    # if step % 100 == 0:
    #     final_verts = verts.detach().cpu().numpy()
    #     final_faces = faces.detach().cpu().numpy()

    #     temp_mesh = trimesh.Trimesh(
    #         vertices=final_verts,
    #         faces=final_faces
    #     )

    #     temp_mesh.export(f"step_{step}.obj")
    #     print(f"Saved step_{step}.obj")
    
    if step % 50 == 0:
        rendered = renderer(mesh)[0, ..., :3].detach().cpu().numpy()
        rendered = (rendered * 255).astype(np.uint8)
        Image.fromarray(rendered).save(f"render_{step}.png")

    if step % 20 == 0:
        print(f"Step {step} | Loss: {loss.item():.4f}")

print("Optimisation complete.")

# -------------------------------
# Save final mesh to file
# -------------------------------

final_verts = verts.detach().cpu().numpy()
final_faces = faces.detach().cpu().numpy()

final_mesh = trimesh.Trimesh(vertices=final_verts, faces=final_faces)

final_mesh.export("final_result.obj")
print("Saved final_result.obj")