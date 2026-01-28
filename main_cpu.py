import torch
import clip
import trimesh
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# -------------------------------
# 0. Device
# -------------------------------
device = torch.device("cpu")

# -------------------------------
# 1. Load CLIP
# -------------------------------
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
clip_model.eval()

text_prompt = "a pointy cone"
text_tokens = clip.tokenize([text_prompt]).to(device)
with torch.no_grad():
    text_feat = clip_model.encode_text(text_tokens)
    text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)

# -------------------------------
# 2. Create a simple mesh (icosphere)
# -------------------------------
mesh = trimesh.creation.icosphere(subdivisions=3, radius=1.0)

# Vertices as torch tensor for optimization
verts = torch.tensor(mesh.vertices, dtype=torch.float32, device=device, requires_grad=True)
faces = np.array(mesh.faces)

# -------------------------------
# 3. Simple CPU renderer
# -------------------------------
def render_mesh(verts_np, faces_np, elev=30, azim=45, image_size=224):
    """
    Render a 3D mesh to a numpy RGB image using Matplotlib.
    """
    fig = plt.figure(figsize=(3, 3), dpi=image_size // 3)
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(
        verts_np[:, 0],
        verts_np[:, 1],
        verts_np[:, 2],
        triangles=faces_np,
        color=(0.5, 0.5, 0.5),
        shade=True
    )
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    fig.canvas.draw()

    # Get RGBA buffer, convert to RGB numpy array
    buf = fig.canvas.buffer_rgba()
    img = np.asarray(buf)[:, :, :3]
    plt.close(fig)

    # Resize to 224x224 for CLIP
    img_pil = Image.fromarray(img).resize((224, 224))
    return np.array(img_pil)

# -------------------------------
# 4. Optimizer
# -------------------------------
optimizer = torch.optim.Adam([verts], lr=1e-2)
num_steps = 100

for step in range(num_steps):
    # Render mesh as image
    verts_np = verts.detach().cpu().numpy()
    image_np = render_mesh(verts_np, faces)
    
    # Convert to PIL, then to tensor for CLIP
    image = Image.fromarray(image_np)
    image_tensor = clip_preprocess(image).unsqueeze(0).to(device)

    # Compute CLIP loss
    img_feat = clip_model.encode_image(image_tensor)
    img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
    loss = 1 - torch.cosine_similarity(img_feat, text_feat)

    # Backprop
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 10 == 0:
        print(f"Step {step:03d} | Loss: {loss.item():.4f}")

print("Optimization finished!")

# -------------------------------
# 5. Visualize final mesh (Matplotlib only)
# -------------------------------
final_verts = verts.detach().cpu().numpy()

fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_trisurf(
    final_verts[:, 0],
    final_verts[:, 1],
    final_verts[:, 2],
    triangles=faces,
    color=(0.5, 0.5, 0.5),
    shade=True
)
ax.set_axis_off()
plt.show()
