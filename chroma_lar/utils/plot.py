from tqdm import tqdm
import torch
import plotly.graph_objects as go
from chroma.event import SURFACE_DETECT, NO_HIT, NAN_ABORT, SURFACE_ABSORB, BULK_ABSORB
import numpy as np

def plot_geometry(event, g=None):
    fig = go.Figure()

    if g is not None:
        solids = g.solids
        for i, solid in enumerate(tqdm(solids)):
            colors = solid.color  # (N,) in hex (uint16)
            triangles = solid.mesh.triangles  # (N, 3)
            vertices = solid.mesh.vertices  # (M, 3), M<=N

            # Convert to PyTorch tensors on GPU
            vertices_torch = torch.tensor(vertices, device="cuda", dtype=torch.float32)
            try:
                pos = torch.tensor(g.solid_displacements[i], device="cuda", dtype=torch.float32)
                rot = torch.tensor(g.solid_rotations[i], device="cuda", dtype=torch.float32)
            except:
                pos = None
                rot = None

            # Convert hex colors to RGB format for plotly
            rgb_colors = []
            for color in colors:
                # Extract RGB components from uint16 hex
                r = ((color >> 10) & 0x1F) / 31.0
                _g = ((color >> 5) & 0x1F) / 31.0
                b = (color & 0x1F) / 31.0
                rgb_colors.append(f"rgb({int(r * 255)},{int(_g * 255)},{int(b * 255)})")

            # Create mesh3d for each solid
            i_vertices = []
            j_vertices = []
            k_vertices = []

            for triangle in triangles:
                i_vertices.append(triangle[0])
                j_vertices.append(triangle[1])
                k_vertices.append(triangle[2])

            # Perform displacements using GPU
            if rot is not None:
                vertices_torch = torch.matmul(vertices_torch, -rot)
            if pos is not None:
                vertices_torch = vertices_torch + pos

            # Convert back to numpy for plotting
            vertices = vertices_torch.cpu().numpy()

            fig.add_trace(
                go.Mesh3d(
                    x=vertices[:, 0],
                    y=vertices[:, 1],
                    z=vertices[:, 2],
                    i=i_vertices,
                    j=j_vertices,
                    k=k_vertices,
                    facecolor=rgb_colors,
                    opacity=0.1,
                    name=f"Solid {i}",
                )
            )

    # Plot beginning photons
    fig.add_trace(
        go.Scatter3d(
            x=event.photons_beg.pos[:,0],
            y=event.photons_beg.pos[:,1],
            z=event.photons_beg.pos[:,2],
            mode='markers',
            marker=dict(
                size=2,
                color='blue',
                opacity=0.5
            ),
            name='Beginning Photons'
        )
    )

    # Plot ending photons with different colors based on their flags
    # Detected photons
    detected_mask = (event.photons_end.flags & SURFACE_DETECT) == SURFACE_DETECT
    if detected_mask.any():
        fig.add_trace(
            go.Scatter3d(
                x=event.photons_end.pos[detected_mask,0],
                y=event.photons_end.pos[detected_mask,1],
                z=event.photons_end.pos[detected_mask,2],
                mode='markers',
                marker=dict(
                    size=0.5,
                    color=event.photons_end.t[detected_mask],
                    opacity=0.5,
                    colorscale='Viridis',
                    colorbar=dict(
                        title="Time (ns)",
                        thickness=20,
                        len=0.75,
                        # x=1.1,
                        # y=0.5
                    )
                ),
                name='Detected Photons'
            )
        )

    # No hit photons
    nohit_mask = (event.photons_end.flags & NO_HIT) == NO_HIT
    if nohit_mask.any():
        fig.add_trace(
            go.Scatter3d(
                x=event.photons_end.pos[nohit_mask,0],
                y=event.photons_end.pos[nohit_mask,1],
                z=event.photons_end.pos[nohit_mask,2],
                mode='markers',
                marker=dict(
                    size=0.5,
                    color='blackyellow',
                    opacity=0.5
                ),
                name='No Hit Photons'
            )
        )

    # Absorbed photons (surface and bulk)
    surface_absorbed_mask = (((event.photons_end.flags & SURFACE_ABSORB) == SURFACE_ABSORB))
    if surface_absorbed_mask.any():
        fig.add_trace(
            go.Scatter3d(
                x=event.photons_end.pos[surface_absorbed_mask,0],
                y=event.photons_end.pos[surface_absorbed_mask,1],
                z=event.photons_end.pos[surface_absorbed_mask,2],
                mode='markers',
                marker=dict(
                    size=0.5,
                    color='red',
                    opacity=0.5
                ),
                name='Surface Absorbed Photons'
            )
        )

    bulk_absorbed_mask = (((event.photons_end.flags & BULK_ABSORB) == BULK_ABSORB))
    if bulk_absorbed_mask.any():
        fig.add_trace(
            go.Scatter3d(
                x=event.photons_end.pos[bulk_absorbed_mask,0],
                y=event.photons_end.pos[bulk_absorbed_mask,1],
                z=event.photons_end.pos[bulk_absorbed_mask,2],
                mode='markers',
                marker=dict(
                    size=0.5,
                    color='black',
                    opacity=0.5
                ),
                name='Bulk Absorbed Photons'
            )
        )


    # Print statistics
    print(f"Total photons: {len(event.photons_end.flags)}")
    print(f"Detected: {np.sum(detected_mask)}")
    print(f"No Hit: {np.sum(nohit_mask)}")
    print(f"Surface Absorbed: {np.sum(surface_absorbed_mask)}")
    print(f"Bulk Absorbed: {np.sum(bulk_absorbed_mask)}")
    print(f"Aborted: {np.sum((event.photons_end.flags & NAN_ABORT) == NAN_ABORT)}")


    # Update layout
    fig.update_layout(
        scene=dict(aspectmode="data", xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),    width=800,
        height=800,
        legend=dict(
            itemsizing='constant',  # Force constant size for legend items
            itemwidth=30,  # Increase width of legend items
            font=dict(size=12),  # Increase font size
            # Make legend markers much larger
            itemclick="toggleothers",
            traceorder="normal",
            bgcolor="rgba(255, 255, 255, 0.5)",
            bordercolor="rgba(255, 255, 255, 0.5)",
            borderwidth=0
        )
    )

    fig.show()