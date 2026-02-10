import torch
import clip
import trimesh
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# -------------------------------
# Device
# -------------------------------
device = torch.device("cpu")


# -------------------------------
# Load CLIP
# -------------------------------
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
clip_model.eval()

text_prompt = "a tall, narrow box"
#turn text prompt into tokens that CLIP can understand
text_tokens = clip.tokenize([text_prompt]).to(device)
#encode text prompt into feature vector
with torch.no_grad():
    text_feat = clip_model.encode_text(text_tokens)
    text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
    

# -------------------------------
# Create mesh (simple cube)
# -------------------------------
mesh = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
#subdivide to increase vertex count for smoother optimisation
mesh = mesh.subdivide()
mesh = mesh.subdivide()
verts = torch.tensor(mesh.vertices, dtype=torch.float32, device=device, requires_grad=True)
faces = np.array(mesh.faces)


# -------------------------------
# Compute Laplacian adjacency for smoothing
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
# Renderer (CPU)
# -------------------------------
def render_mesh(verts_np, faces_np, elev=30, azim=45, image_size=224):
    #Simple CPU renderer using matplotlib
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
    #Resize to 224x224 for CLIP
    img_pil = Image.fromarray(img).resize((224, 224))
    return np.array(img_pil)


# -------------------------------
# Optimiser with Laplacian smoothing
# -------------------------------
#lr = learning rate for vertex updates - controls how much the mesh changes each step 
optimiser = torch.optim.Adam([verts], lr=1e-3) 
#number of optimisation steps - more steps allows for better convergence but takes longer
num_steps = 250
#weight for Laplacian smoothing
lambda_smooth = max(0.1, 10.0 / num_steps)  #decay smoothing weight over time to allow more deformation later
#epsilon to prevent division by zero in loss calculations
eps = 1e-8

#multiple camera views (elev, azim) for multi-view CLIP loss
camera_views = [
    (0, 0),
    (0, 90),
    (0, 180),
    (0, 270),
    (75, 45)
]

#optimisation loop, 
# - rendering from multiple views and applying CLIP loss + Laplacian smoothing
for step in range(num_steps):
    verts_np = verts.detach().cpu().numpy()
    total_loss = 0.0

    #multi-view CLIP loss
    # - rendering from different camera angles and comparing to text prompt
    for elev, azim in camera_views:
        image_np = render_mesh(verts_np, faces, elev=elev, azim=azim)
        image = Image.fromarray(image_np)
        image_tensor = clip_preprocess(image).unsqueeze(0).to(device)
        img_feat = clip_model.encode_image(image_tensor)
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True) + eps
        loss = 1 - torch.cosine_similarity(img_feat, text_feat)
        total_loss += loss

    avg_loss = total_loss / len(camera_views) + eps

    #laplacian smoothing loss 
    # - encourages neighboring vertices to stay close, preventing mesh distortion
    smooth_loss = lambda_smooth * torch.trace(verts.t() @ L @ verts)
    #decay smoothing weight over time to allow more deformation in later steps
    smooth_weight = min(step / 50, 1.0)
    
    #anisotropy loss to encourage one dominant axis
    #bounding box extents
    min_xyz = verts.min(dim=0).values
    max_xyz = verts.max(dim=0).values
    extents = max_xyz - min_xyz
    #encourage one dominant axis
    anisotropy_loss = -torch.max(extents)
    
    #volume loss to encourage the mesh to maintain a reasonable size
    volume = torch.prod(extents)
    volume_loss = (volume - 1.0).abs()
    
    #encourage vertical alignment
    height_alignment_loss = -extents[1]
    
    #discourage centroid shift in X/Z, encourage it to stay centered and upright
    centroid = verts.mean(dim=0)
    upright_loss = centroid[0]**2 + centroid[2]**2
    
    # radial symmetry around Y axis, discourages lopsidedness
    x, z = verts[:, 0], verts[:, 2]
    radius = torch.sqrt(x**2 + z**2 + eps)
    symmetry_loss = torch.var(radius)

    loss_total = avg_loss + (smooth_loss * smooth_weight) + 0.2 * anisotropy_loss + 0.05 * volume_loss + 0.3 * height_alignment_loss + 0.1 * upright_loss + 0.2 * symmetry_loss
    
    optimiser.zero_grad()
    loss_total.backward()
    optimiser.step()
    
    verts_prev = verts.clone().detach()
    
    with torch.no_grad():
        #clamp vertices to prevent extreme deformations
        verts[:] = torch.clamp(verts, -2.0, 2.0)
        #limit vertex movement per step to prevent instability
        delta = verts - verts_prev
        delta = torch.clamp(delta, -0.005, 0.005)
        verts.copy_(verts_prev + delta)

    #print progress every 10 steps, showing CLIP loss and smoothness loss
    if step % 10 == 0:
        print(f"Step {step:03d} | CLIP loss: {avg_loss.item():.4f} | Smooth loss: {smooth_loss.item():.4f}")


print("Optimisation done")


# -------------------------------
# Visualize final mesh
# -------------------------------
#convert final optimised vertices to numpy for visualization
final_verts = verts.detach().cpu().numpy()

#simple visualization of the final optimised mesh using matplotlib
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
