import imageio.v2 as imageio
import numpy as np
import matplotlib.pyplot as plt

# =========================
# 1. 读取菊池图
# =========================
img = imageio.imread("pattern.bmp")

if img.ndim == 3:
    img = img[:, :, 0]

img = img.astype(np.float32)
img = (img - img.min()) / (img.max() - img.min() + 1e-8)

H, W = img.shape
print("Pattern shape:", H, W)

# =========================
# 2. 几何参数（示例）
# =========================
pcx = 0.50
pcy = 0.50
pcz = 0.60

# =========================
# 3. 像素坐标 -> gnomonic坐标
# =========================
jj, ii = np.meshgrid(np.arange(W), np.arange(H))

cx = pcx * (W - 1)
cy = pcy * (H - 1)

xg = (jj - cx) / (pcz * H)
yg = -(ii - cy) / (pcz * H)

# =========================
# 4. gnomonic -> sphere
# =========================
X = xg.copy()
Y = yg.copy()
Z = np.ones_like(X)

norm = np.sqrt(X**2 + Y**2 + Z**2) + 1e-8
X /= norm
Y /= norm
Z /= norm

# =========================
# 5. mask 去背景
# =========================
mask = img > 0.05

Xp = X[mask]
Yp = Y[mask]
Zp = Z[mask]
Ip = img[mask]

print("Valid sphere points:", len(Xp))

# =========================
# 6. 保存原始图
# =========================
plt.figure(figsize=(5, 5))
plt.imshow(img, cmap="gray")
plt.title("Original Kikuchi Pattern")
plt.axis("off")
plt.tight_layout()
plt.savefig("original_kikuchi.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
plt.close()

# =========================
# 7. 生成 2D 球面参数图（最稳）
# =========================
# 把球面点转成经纬度坐标，做成“展开图”
theta = np.arctan2(Yp, Xp)          # [-pi, pi]
phi = np.arccos(np.clip(Zp, -1, 1)) # [0, pi]

# 归一化到图像网格
out_h, out_w = 800, 1600
sphere_map = np.zeros((out_h, out_w), dtype=np.float32)
count_map = np.zeros((out_h, out_w), dtype=np.float32)

u = ((theta + np.pi) / (2 * np.pi) * (out_w - 1)).astype(int)
v = (phi / np.pi * (out_h - 1)).astype(int)

for uu, vv, val in zip(u, v, Ip):
    sphere_map[vv, uu] += val
    count_map[vv, uu] += 1

valid = count_map > 0
sphere_map[valid] /= count_map[valid]

# 保存球面展开图
plt.figure(figsize=(12, 6))
plt.imshow(sphere_map, cmap="gray", origin="upper")
plt.title("Spherical corrected Kikuchi map")
plt.axis("off")
plt.tight_layout()
plt.savefig("kikuchi_spherical_map.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
plt.close()

# =========================
# 8. 再保存一个3D散点图（抽样，避免太慢）
# =========================
idx = np.arange(len(Xp))
if len(idx) > 20000:
    idx = idx[:: len(idx) // 20000]

Xp_s = Xp[idx]
Yp_s = Yp[idx]
Zp_s = Zp[idx]
Ip_s = Ip[idx]

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(
    Xp_s, Yp_s, Zp_s,
    c=Ip_s,
    cmap="gray",
    s=1,
    depthshade=False
)
ax.set_box_aspect([1, 1, 1])
ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_zlim([-1, 1])
ax.set_axis_off()
plt.tight_layout()
plt.savefig("kikuchi_sphere_patch_3d.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
plt.close()

print("Saved:")
print("  original_kikuchi.png")
print("  kikuchi_spherical_map.png")
print("  kikuchi_sphere_patch_3d.png")