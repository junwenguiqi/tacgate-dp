"""Draw TacGate-DP as a publication-sized, fully vector figure.

Run from any directory with Python >= 3.9 and matplotlib >= 3.7:
    python figures/draw_model_architecture.py

Exports PDF (embedded TrueType fonts), SVG (editable text), and 450-dpi PNG
next to this script. The default dimensions are read from PolicyConfig using
the AST, so importing PyTorch or downloading vision weights is unnecessary.
The drawing uses English labels for inclusion in an arXiv manuscript.
"""

import ast
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch, PathPatch, Rectangle
from matplotlib.path import Path as MplPath


OUTPUT = Path(__file__).resolve().parent
RELEASE = OUTPUT.parent
WIDTH, HEIGHT = 180.0, 140.0  # millimetres; intended full-width paper figure
INK = "#26313B"
MUTED = "#54616C"
FRAME = "#CDD3D8"
BLUE, BLUE_FILL = "#3B729F", "#E7F0F8"
ORANGE, ORANGE_FILL = "#9D632F", "#FAEBDD"
GREEN, GREEN_FILL = "#52724A", "#EAF1E6"
PURPLE, PURPLE_FILL = "#765795", "#F0EAF6"


def read_defaults():
    source = RELEASE / "src/tactile_gate_policy/config.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "PolicyConfig")
    return {
        n.target.id: ast.literal_eval(n.value)
        for n in cls.body
        if isinstance(n, ast.AnnAssign) and n.value is not None
    }


def main():
    cfg = read_defaults()
    if cfg["num_modalities"] != 5 or cfg["share_vision_backbone"] or cfg["share_tactile_backbone"]:
        raise ValueError("This layout depicts five modalities with independent camera/touch encoders.")
    t, d, h = cfg["obs_horizon"], cfg["token_dim"], cfg["action_horizon"]
    m, dc, da = cfg["num_modalities"], cfg["condition_dim"], cfg["action_dim"]
    k_steps, prefix = cfg["num_inference_steps"], cfg["n_action_steps"]
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8.0,
        "mathtext.fontset": "dejavusans",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.facecolor": "white",
        "axes.unicode_minus": False,
    })
    fig = plt.figure(figsize=(WIDTH / 25.4, HEIGHT / 25.4), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, WIDTH), ylim=(HEIGHT, 0))
    ax.axis("off")
    texts = []

    def label(x, y, text, size=8.0, weight="normal", color=INK, ha="center"):
        obj = ax.text(x, y, text, fontsize=size, weight=weight, color=color,
                      ha=ha, va="center", linespacing=1.18, zorder=5)
        texts.append(obj)
        return obj

    def box(x, y, w, height, fill="white", edge=INK, lw=0.7, rounding=1.1):
        patch = FancyBboxPatch(
            (x, y), w, height,
            boxstyle=f"round,pad=0,rounding_size={rounding}",
            linewidth=lw, edgecolor=edge, facecolor=fill, zorder=2,
        )
        ax.add_patch(patch)
        return patch

    def arrow(points, color=INK, lw=0.85, dashed=False, head=1.6):
        codes = [MplPath.MOVETO] + [MplPath.LINETO] * (len(points) - 1)
        style = (0, (3.0, 2.0)) if dashed else "solid"
        ax.add_patch(PathPatch(MplPath(points, codes), facecolor="none",
                               edgecolor=color, linewidth=lw, linestyle=style,
                               capstyle="round", joinstyle="round", zorder=3))
        end, prev = points[-1], points[-2]
        dx, dy = end[0] - prev[0], end[1] - prev[1]
        length = (dx * dx + dy * dy) ** 0.5
        ux, uy = dx / length, dy / length
        tip = end
        base = (tip[0] - ux * head, tip[1] - uy * head)
        side = head * 0.4
        verts = [tip, (base[0] - uy * side, base[1] + ux * side),
                 (base[0] + uy * side, base[1] - ux * side), tip]
        ax.add_patch(PathPatch(MplPath(verts, [1, 2, 2, 79]),
                               facecolor=color, edgecolor=color, lw=0.2, zorder=4))

    def dot(x, y, color=INK, radius=0.45):
        ax.add_patch(Circle((x, y), radius, edgecolor="none", facecolor=color, zorder=4))

    # A: five independent per-frame encoders and the time/modality token grid.
    box(1, 1, 79, 73, edge=FRAME, lw=0.6, rounding=1.4)
    label(4, 6, "(a) Multimodal observation encoding", 9.1, "bold", ha="left")
    label(4, 12, rf"$T_o={t}$ synchronized historical observations", ha="left", color=MUTED)
    image_h, image_w = cfg["image_size"]
    input_names = [
        ("Global RGB", rf"$3\times {image_h}\times {image_w}$"),
        ("Wrist RGB", rf"$3\times {image_h}\times {image_w}$"),
        ("Left touch", rf"${cfg['tactile_dim'] // cfg['tactile_channels']}\times {cfg['tactile_channels']}$"),
        ("Right touch", rf"${cfg['tactile_dim'] // cfg['tactile_channels']}\times {cfg['tactile_channels']}$"),
        ("Robot state", rf"${cfg['proprio_dim']}$-D"),
    ]
    colors = [(BLUE, BLUE_FILL)] * 2 + [(ORANGE, ORANGE_FILL)] * 2 + [(GREEN, GREEN_FILL)]
    cnn_widths = "/".join(map(str, cfg["tactile_hidden_dims"]))
    gn_text = " + GN" if cfg["vision_group_norm"] else ""
    for i, ((name, shape), (edge, fill)) in enumerate(zip(input_names, colors)):
        y = 18.0 + i * 10.0
        box(4.5, y, 21.0, 8.2, fill, edge)
        label(15, y + 2.3, name, 8.0, "bold")
        label(15, y + 5.9, shape, 8.0)
        arrow([(25.5, y + 4.1), (31, y + 4.1)], color=edge)
        box(31, y, 32, 8.2, fill, edge)
        if i < 2:
            label(47, y + 2.3, "DP ResNet18" + gn_text, 8.0, "bold")
            label(47, y + 5.9, rf"Crop; $512\to {d}$", 8.0)
        elif i < 4:
            label(47, y + 2.3, rf"Conv1D ${cnn_widths}$", 8.0, "bold")
            label(47, y + 5.9, rf"Pool + proj. $\to {d}$", 8.0)
        else:
            label(47, y + 2.3, "State MLP", 8.0, "bold")
            label(47, y + 5.9, rf"${cfg['proprio_dim']}\to {d}\to {d}$", 8.0)
        arrow([(63, y + 4.1), (68, y + 4.1)], color=edge, head=1.3)
        # Rows are modalities; columns are historical time steps.
        cell_width = (10.0 - (t - 1) * 0.35) / t
        for j in range(t):
            x = 68.3 + j * (cell_width + 0.35)
            ax.add_patch(Rectangle((x, y + 0.6), cell_width, 7,
                                   facecolor=fill, edgecolor=edge, linewidth=0.6, zorder=2))
        ax.plot([78.5, 79.6], [y + 4.1, y + 4.1], color=INK, lw=0.7, zorder=3)
    label(73.3, 15.6, "Time", 8.0, color=MUTED)
    ax.plot([79.6, 79.6], [22.1, 62.1], color=INK, lw=0.8, zorder=3)
    label(42, 70.0, rf"$Z\in\mathbb{{R}}^{{{t}\times {m}\times {d}}}$  ({t * m} tokens)", 8.5)

    # B: observation-only routing and a second, independent fusion Transformer.
    box(83, 1, 96, 73, edge=FRAME, lw=0.6, rounding=1.4)
    label(87, 6, "(b) Time-modality gated tactile conditioning", 9.1, "bold", ha="left")
    box(88, 15, 47, 28, PURPLE_FILL, PURPLE)
    label(111.5, 19.1, "Time-modality router", 8.7, "bold")
    label(111.5, 23.7, "Stage B / inference", 8.0, color=PURPLE)
    label(111.5, 29.0, "Time + modality embeddings", 8.0)
    label(111.5, 34.0, f"Transformer: {cfg['router_layers']} layers, {cfg['router_heads']} heads", 8.0)
    label(111.5, 39.0, r"Linear + sigmoid $\to G$", 8.0)
    arrow([(79.6, 42.1), (85, 42.1), (85, 29), (88, 29)])
    arrow([(85, 42.1), (85, 59.6), (88, 59.6)])
    dot(85, 42.1)
    box(143, 17, 32, 22, "#F6F6F6", MUTED)
    label(159, 21.1, "Stage A only", 8.5, "bold")
    label(159, 26.5, "Random binary gates", 8.0)
    label(159, 32.7, r"$p\sim\mathcal{U}(0.35,0.95)$", 8.0)
    arrow([(159, 39), (159, 46), (128, 46), (128, 50)], color=MUTED, dashed=True)
    arrow([(111.5, 43), (111.5, 50)], color=PURPLE)
    label(105.7, 46.5, r"$G$", 8.6, color=PURPLE)
    box(88, 50, 50, 21, PURPLE_FILL, PURPLE)
    label(113, 53.8, "Soft feature gate", 8.7, "bold")
    label(113, 59.0, r"$Z_g=G\odot Z+(1-G)\odot m$", 8.6)
    label(113, 63.8, r"Learned per-modality mask $m$", 8.0)
    label(113, 68.0, rf"$G: {t}\times {m}$; all {t * m} tokens retained", 8.0)
    arrow([(138, 60.5), (143, 60.5)], color=PURPLE)
    box(143, 49, 32, 22, PURPLE_FILL, PURPLE)
    label(159, 52.8, "Condition fusion", 8.5, "bold")
    label(159, 57.0, "+ time / modality", 8.0)
    label(159, 61.2, f"Transformer ({cfg['condition_layers']}L,{cfg['condition_heads']}H)", 8.0)
    label(159, 65.2, "Mean pool + MLP", 8.0)
    label(159, 69.0, rf"$c\in\mathbb{{R}}^{{{dc}}}$", 8.3)

    # C: only the U-Net and scheduler repeat. All action states are normalized.
    box(1, 78, 178, 61, edge=FRAME, lw=0.6, rounding=1.4)
    label(5, 83.1, "(c) FiLM condition injection and action diffusion", 9.1, "bold", ha="left")
    arrow([(159, 71), (159, 87), (110, 87), (110, 94)], color=PURPLE, lw=1.05)
    label(114, 90.6, rf"$c$ fixed over {k_steps} steps", 8.0, color=PURPLE, ha="left")
    box(34, 86.5, 12, 7, "#F5F6F7", MUTED)
    label(40, 90, r"Step $k$", 8.0)
    arrow([(46, 90), (67, 90), (67, 94)])
    box(57, 94, 64, 14, BLUE_FILL, BLUE)
    label(89, 97.2, "DP conditional 1D U-Net", 8.8, "bold")
    label(89, 101.8, r"FiLM injection: $[\mathrm{emb}(k),c]$", 8.2)
    label(89, 105.7, r"$\hat\epsilon=\epsilon_\theta(A_k,k,c)$", 8.2)
    box(7, 112, 27, 13, "#F5F6F7", MUTED)
    label(20.5, 116.0, r"$A_{\mathrm{init}}\sim\mathcal{N}(0,I)$", 8.6)
    label(20.5, 121.1, rf"${h}\times {da}$ action noise", 8.0)
    arrow([(34, 118.5), (57, 118.5)])
    dot(47, 118.5)
    arrow([(47, 118.5), (47, 101), (57, 101)])
    label(41, 105.1, r"$A_k$", 8.5)
    arrow([(89, 108), (89, 113)])
    label(95.2, 110.4, r"$\hat\epsilon$", 8.4)
    box(57, 113, 64, 12, "#F4F5F7", MUTED)
    label(89, 116.9, "DDPM update (cosine schedule)", 8.2, "bold")
    label(89, 121.6, r"$(A_k,\hat\epsilon,k)\mapsto A_{\mathrm{prev}}$", 8.2)
    arrow([(121, 118.5), (143, 118.5)])
    dot(133, 118.5)
    arrow([(133, 118.5), (133, 130), (47, 130), (47, 118.5)], dashed=True, color=MUTED)
    label(90, 133.6, rf"Repeat: {k_steps} reverse DDPM steps", 8.0, color=MUTED)
    label(135.9, 114.7, "Clean", 8.0)
    box(143, 109, 32, 21, GREEN_FILL, GREEN)
    label(159, 112.6, "Action chunk", 8.7, "bold")
    label(159, 117.1, rf"$H={h},\ d_a={da}$", 8.3)
    label(159, 121.6, r"Rescale + unit $q$", 8.0)
    label(159, 126.4, f"Execute first {prefix}", 8.1, "bold")

    # Check the outer page bounds in display coordinates before exporting.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    page = fig.bbox
    for text in texts:
        bb = text.get_window_extent(renderer)
        if bb.x0 < page.x0 or bb.y0 < page.y0 or bb.x1 > page.x1 or bb.y1 > page.y1:
            raise RuntimeError(f"Text leaves the figure page: {text.get_text()}")
    stem = OUTPUT / "model_architecture"
    fig.savefig(stem.with_suffix(".pdf"), metadata={
        "Title": "TacGate-DP: Time-Modality Gated Tactile Conditioning for Diffusion Policies",
        "Subject": "TacGate-DP release model; fused tactile conditioning; default configuration; batch dimension omitted",
        "Author": "TacGate-DP contributors",
        "Keywords": "Diffusion Policy, tactile CNN, multimodal gating, architecture",
    })
    fig.savefig(stem.with_suffix(".svg"))
    fig.savefig(stem.with_suffix(".png"), dpi=450)
    plt.close(fig)
    for ext in ("pdf", "svg", "png"):
        path = stem.with_suffix("." + ext)
        print(f"{path.name}: {path.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
