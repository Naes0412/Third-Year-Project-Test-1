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

text_prompt = "tall cone"  # simpler prompt for more visible change
text_tokens = clip.tokenize([text_prompt]).to(device)
with torch.no_grad():
    text_feat = clip_model.encode_text(text_tokens)
    text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)

# -------------------------------
# 2. Create mesh (simple sphere)
# -------------------------------
mesh = trimesh.creation.icosphere(subdivisions=4, radius=1.0)
verts = torch.tensor(mesh.vertices, dtype=torch.float32, device=device, requires_grad=True)
faces = np.array(mesh.faces)

# -------------------------------
# 3. Compute Laplacian adjacency for smoothing
# -------------------------------
def compute_laplacian(vertices, faces):
    n = vertices.shape[0]
    L = torch.zeros((n, n), device=vertices.device)
    for f in faces:
        for i, j in [(0,1),(1,2),(2,0)]:
            L[f[i], f[j]] = 1
            L[f[j], f[i]] = 1
    # Degree matrix
    D = torch.diag(L.sum(dim=1))
    L = D - L
    return L

L = compute_laplacian(verts, faces)

# -------------------------------
# 4. CPU renderer
# -------------------------------
def render_mesh(verts_np, faces_np, elev=30, azim=45, image_size=224):
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
    buf = fig.canvas.buffer_rgba()
    img = np.asarray(buf)[:, :, :3]
    plt.close(fig)
    img_pil = Image.fromarray(img).resize((224, 224))
    return np.array(img_pil)

# -------------------------------
# 5. Optimizer with Laplacian smoothing
# -------------------------------
optimizer = torch.optim.Adam([verts], lr=1e-2)
num_steps = 100
lambda_smooth = 0.1  # weight for Laplacian smoothing

camera_views = [
    (30, 45),
    (30, 135),
    (30, 225),
    (30, 315)
]

for step in range(num_steps):
    verts_np = verts.detach().cpu().numpy()
    total_loss = 0.0

    # Multi-view CLIP loss
    for elev, azim in camera_views:
        image_np = render_mesh(verts_np, faces, elev=elev, azim=azim)
        image = Image.fromarray(image_np)
        image_tensor = clip_preprocess(image).unsqueeze(0).to(device)
        img_feat = clip_model.encode_image(image_tensor)
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
        loss = 1 - torch.cosine_similarity(img_feat, text_feat)
        total_loss += loss

    avg_loss = total_loss / len(camera_views)

    # Laplacian smoothing loss
    smooth_loss = lambda_smooth * torch.trace(verts.t() @ L @ verts)
    loss_total = avg_loss + smooth_loss

    optimizer.zero_grad()
    loss_total.backward()
    optimizer.step()

    if step % 10 == 0:
        print(f"Step {step:03d} | CLIP loss: {avg_loss.item():.4f} | Smooth loss: {smooth_loss.item():.4f}")

print("Optimization finished!")

# -------------------------------
# 6. Visualize final mesh
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
