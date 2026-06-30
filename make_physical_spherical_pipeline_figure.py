from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Circle, FancyArrowPatch, FancyBboxPatch, Polygon
import numpy as np


OUTPUT_DIR = Path("outputs") / "article_figures" / "spherical_calibration_pipeline"


COLORS = {
    "ink": "#1f2933",
    "muted": "#52606d",
    "light": "#f5f7fa",
    "line": "#9aa5b1",
    "detector": "#e6f4f1",
    "sphere": "#edf2ff",
    "opt": "#fff4e6",
    "residual": "#fcebea",
    "accent": "#1f7a8c",
    "accent2": "#b7791f",
    "red": "#c53030",
}


def add_box(ax, xy, wh, title, body, facecolor, edgecolor=None, title_size=10.2, body_size=8.2):
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=1.25,
        edgecolor=edgecolor or COLORS["line"],
        facecolor=facecolor,
    )
    ax.add_patch(patch)
    ax.text(
        x + 0.025,
        y + h - 0.045,
        title,
        ha="left",
        va="top",
        fontsize=title_size,
        fontweight="bold",
        color=COLORS["ink"],
    )
    if body:
        ax.text(
            x + 0.025,
            y + h - 0.105,
            body,
            ha="left",
            va="top",
            fontsize=body_size,
            color=COLORS["muted"],
            linespacing=1.28,
        )
    return patch


def arrow(ax, start, end, color=None, lw=1.25, rad=0.0):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=lw,
            color=color or COLORS["line"],
            connectionstyle=f"arc3,rad={rad}",
        )
    )


def draw_detector(ax, cx, cy, scale=1.0):
    plane = Polygon(
        [
            (cx - 0.085 * scale, cy - 0.055 * scale),
            (cx + 0.105 * scale, cy - 0.018 * scale),
            (cx + 0.085 * scale, cy + 0.070 * scale),
            (cx - 0.105 * scale, cy + 0.035 * scale),
        ],
        closed=True,
        facecolor="#d8f3dc",
        edgecolor=COLORS["accent"],
        linewidth=1.0,
    )
    ax.add_patch(plane)
    ax.add_patch(Circle((cx - 0.005 * scale, cy + 0.012 * scale), 0.010 * scale, color=COLORS["red"]))
    for offset in [-0.045, 0.0, 0.045]:
        ax.plot(
            [cx - 0.07 * scale, cx + 0.07 * scale],
            [cy + offset * scale, cy + (offset + 0.025) * scale],
            color="#74a57f",
            lw=0.8,
            alpha=0.75,
        )
    ax.text(cx - 0.012 * scale, cy - 0.085 * scale, "PC", ha="center", va="top", fontsize=7.5, color=COLORS["red"])


def draw_sphere(ax, cx, cy, r):
    ax.add_patch(Circle((cx, cy), r, facecolor="#e8edff", edgecolor="#6172b0", linewidth=1.1))
    ax.add_patch(Arc((cx, cy), 2 * r, 0.55 * r, angle=0, theta1=0, theta2=360, color="#94a3d8", lw=0.7))
    ax.add_patch(Arc((cx, cy), 0.75 * r, 2 * r, angle=0, theta1=0, theta2=360, color="#94a3d8", lw=0.7))
    ax.add_patch(Arc((cx, cy), 1.5 * r, 2 * r, angle=35, theta1=0, theta2=360, color="#94a3d8", lw=0.7, alpha=0.85))
    rng = np.random.default_rng(3)
    pts = rng.normal(size=(32, 2))
    pts /= np.linalg.norm(pts, axis=1, keepdims=True) + 1e-12
    radii = rng.uniform(0.08, 0.85, size=(32, 1)) * r
    pts = np.column_stack([cx + pts[:, 0] * radii[:, 0], cy + pts[:, 1] * radii[:, 0]])
    ax.scatter(pts[:, 0], pts[:, 1], s=7, c="#7c3aed", alpha=0.45, linewidths=0)


def draw_residual_map(ax, x, y, w, h):
    nx, ny = 42, 25
    xx, yy = np.meshgrid(np.linspace(-1, 1, nx), np.linspace(-1, 1, ny))
    data = 0.7 * np.exp(-((xx + 0.35) ** 2 + (yy - 0.15) ** 2) / 0.18)
    data -= 0.5 * np.exp(-((xx - 0.45) ** 2 + (yy + 0.2) ** 2) / 0.12)
    data += 0.12 * np.sin(5 * xx) * np.cos(3 * yy)
    ax.imshow(data, extent=(x, x + w, y, y + h), origin="lower", cmap="coolwarm", vmin=-1, vmax=1, alpha=0.94)
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.01", fill=False, edgecolor=COLORS["line"], linewidth=0.8))


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11.8, 5.3))
    fig.subplots_adjust(left=0.035, right=0.985, top=0.965, bottom=0.055)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    y_top = 0.555
    h_top = 0.275

    add_box(
        ax,
        (0.045, y_top),
        (0.200, h_top),
        "1. Pattern",
        "",
        COLORS["detector"],
        COLORS["accent"],
        title_size=9.8,
    )
    draw_detector(ax, 0.145, 0.660, 0.78)
    ax.text(
        0.070,
        0.575,
        r"$I_{\exp}\rightarrow\tilde I_{\exp},\;B_{\exp}$",
        ha="left",
        va="bottom",
        fontsize=8.5,
        color=COLORS["ink"],
    )

    add_box(
        ax,
        (0.292, y_top),
        (0.270, h_top),
        "2. Geometry",
        "",
        COLORS["sphere"],
        "#6172b0",
    )
    ax.text(
        0.318,
        0.670,
        r"$\hat r_d=\mathrm{norm}[x(u;\mathrm{PC}),y(v;\mathrm{PC}),1]$"
        "\n"
        r"$\hat r_c=\Delta R\,G\,R_{sd}(\alpha,\beta)\,\hat r_d$"
        "\n"
        r"$I_{\mathrm{pred}}(u,v)=M(\hat r_c)$",
        ha="left",
        va="center",
        fontsize=8.4,
        color=COLORS["ink"],
        linespacing=1.55,
    )

    add_box(
        ax,
        (0.605, y_top),
        (0.168, h_top),
        "3. Sphere",
        "",
        COLORS["sphere"],
        "#6172b0",
        title_size=9.9,
        body_size=8.1,
    )
    draw_sphere(ax, 0.690, 0.650, 0.063)
    ax.text(0.690, 0.575, r"$M(\hat r_c)$", ha="center", va="bottom", fontsize=9.2, color=COLORS["ink"])

    add_box(
        ax,
        (0.805, y_top),
        (0.165, h_top),
        "4. Residual",
        "",
        COLORS["residual"],
        COLORS["red"],
        title_size=9.9,
        body_size=7.9,
    )
    draw_residual_map(ax, 0.842, 0.625, 0.096, 0.078)
    ax.text(
        0.826,
        0.588,
        r"$e=\tilde I_{\exp}-\tilde I_{\mathrm{pred}}$",
        ha="left",
        va="bottom",
        fontsize=7.8,
        color=COLORS["ink"],
    )

    arrow(ax, (0.253, 0.700), (0.288, 0.700), COLORS["accent"])
    arrow(ax, (0.570, 0.700), (0.607, 0.700), "#6172b0")
    arrow(ax, (0.778, 0.700), (0.802, 0.700), COLORS["red"])

    add_box(
        ax,
        (0.090, 0.175),
        (0.450, 0.235),
        "Bounded inverse problem",
        "",
        COLORS["opt"],
        COLORS["accent2"],
        title_size=10.0,
        body_size=8.4,
    )
    ax.text(
        0.115,
        0.272,
        r"$\theta=\{\mathrm{PC},D,\alpha,\beta,\Delta R,s_r\}$"
        "\n"
        r"$\min_\theta\mathcal{L}=\mathcal{L}_{image}"
        r"+\lambda_b\mathcal{L}_{band}+\lambda_p\mathcal{L}_{prior}$",
        ha="left",
        va="center",
        fontsize=8.9,
        color=COLORS["ink"],
        linespacing=1.55,
    )

    add_box(
        ax,
        (0.590, 0.175),
        (0.315, 0.235),
        "Recovered geometry",
        "",
        COLORS["light"],
        COLORS["line"],
        title_size=10.8,
        body_size=8.4,
    )
    ax.text(
        0.610,
        0.292,
        r"$\mathrm{PC}^\ast,\;D^\ast,\;\alpha^\ast,\;\beta^\ast,\;\Delta R^\ast$"
        "\n"
        "residual map -> model adequacy",
        ha="left",
        va="center",
        fontsize=9.5,
        color=COLORS["ink"],
        linespacing=1.6,
    )

    arrow(ax, (0.895, 0.560), (0.370, 0.414), COLORS["red"], lw=1.15, rad=0.10)
    arrow(ax, (0.542, 0.292), (0.588, 0.292), COLORS["accent2"], lw=1.4)
    arrow(ax, (0.315, 0.412), (0.430, 0.562), COLORS["accent2"], lw=1.1, rad=-0.12)

    ax.text(0.510, 0.472, "iterate while residual remains structured", ha="center", va="center", fontsize=8.3, color=COLORS["muted"])

    for x, label in [(0.147, "experimental domain"), (0.430, "projection model"), (0.692, "crystal domain"), (0.895, "diagnostic domain")]:
        ax.text(x, 0.845, label, ha="center", va="bottom", fontsize=7.7, color=COLORS["muted"])

    fig.savefig(OUTPUT_DIR / "physical_spherical_calibration_pipeline.svg")
    fig.savefig(OUTPUT_DIR / "physical_spherical_calibration_pipeline.pdf")
    fig.savefig(OUTPUT_DIR / "physical_spherical_calibration_pipeline.png", dpi=320)
    plt.close(fig)

    print(f"Saved figure files to {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
