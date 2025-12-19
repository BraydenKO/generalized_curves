import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import streamlit as st
import plotly.graph_objects as go

# ==================== PAGE CONFIGURATION ====================
# Must be the very first Streamlit command
st.set_page_config(page_title="Ultimate Conics Generator", layout="wide")

# CSS to prevent page scroll jumping
# 1. Enforce a minimum height for the plotly chart container so the page doesn't collapse during calc.
# 2. Adjust padding to maximize screen real estate.
st.markdown("""
    <style>
        .stPlotlyChart {
            min-height: 700px;
        }
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
    </style>
""", unsafe_allow_html=True)

# Check for 3D dependencies
try:
    from skimage import measure
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False

# Initialize session state for foci
if 'foci' not in st.session_state:
    st.session_state.foci = []
if 'k_val' not in st.session_state:
    st.session_state.k_val = 5.0

# ==================== CORE MATHEMATICS ====================

def get_dist(dx, dy, dz, metric, p_val, angular_n, angular_a):
    """Vectorized distance calculation based on selected metric."""
    if metric == 'Euclidean':
        return np.sqrt(dx**2 + dy**2 + dz**2)
    elif metric == 'Manhattan':
        return np.abs(dx) + np.abs(dy) + np.abs(dz)
    elif metric == 'Chebyshev':
        return np.maximum(np.maximum(np.abs(dx), np.abs(dy)), np.abs(dz))
    elif metric == 'Minkowski':
        # Protect against negative bases for fractional powers
        return (np.abs(dx)**p_val + np.abs(dy)**p_val + np.abs(dz)**p_val)**(1/p_val)
    elif metric == 'Angular':
        r = np.sqrt(dx**2 + dy**2 + dz**2)
        theta = np.arctan2(dy, dx)
        return r * (1 + angular_a * np.cos(angular_n * theta))
    return 0

def compute_field_2d(foci, X, Y, metric, p_val, angular_n, angular_a):
    if not foci: return None
    field = np.zeros_like(X)
    for fx, fy, sign in foci:
        dist = get_dist(X - fx, Y - fy, 0, metric, p_val, angular_n, angular_a)
        field += sign * dist
    return field

def compute_field_3d(foci, X, Y, Z, metric, p_val, angular_n, angular_a, mode):
    if not foci: return None
    field = np.zeros_like(X)
    for fx, fy, sign in foci:
        dz = Z if mode == '3D Constant' else 0
        dist = get_dist(X - fx, Y - fy, dz, metric, p_val, angular_n, angular_a)
        field += sign * dist
    return field

def auto_constant(foci):
    if len(foci) <= 1: return 5.0
    f_arr = np.array(foci)
    dx = f_arr[:, 0, None] - f_arr[:, 0]
    dy = f_arr[:, 1, None] - f_arr[:, 1]
    dist_matrix = np.sqrt(dx**2 + dy**2)
    max_d = np.max(dist_matrix)
    if any(f[2] < 0 for f in foci):
        return max_d * 0.4 if max_d > 0 else 2.0
    return max_d * 1.2 if max_d > 0 else 5.0

# ==================== GEOMETRY GENERATORS (PLOTLY) ====================

def get_2d_contour_traces(foci, constant, metric, p_val, angular_n, angular_a, resolution):
    """Generates Plotly traces for 2D contours using matplotlib backend for calculation."""
    traces = []
    if not foci: return traces

    # Compute Field
    l = np.linspace(-10.5, 10.5, resolution)
    X, Y = np.meshgrid(l, l)
    field = compute_field_2d(foci, X, Y, metric, p_val, angular_n, angular_a)
    
    levels = sorted([constant, -constant]) if any(f[2] < 0 for f in foci) else [constant]
    colors = ['#2ca02c', '#9467bd'] # Green, Purple
    
    # Use matplotlib (headless) to calculate contour paths accurately
    try:
        temp_fig, temp_ax = plt.subplots()
        cs = temp_ax.contour(X, Y, field, levels=levels)
        all_segs = cs.allsegs
        plt.close(temp_fig)

        for i, level_segs in enumerate(all_segs):
            color = colors[i % len(colors)]
            for seg in level_segs:
                traces.append(go.Scatter(
                    x=seg[:, 0], y=seg[:, 1],
                    mode='lines',
                    line=dict(color=color, width=3),
                    hoverinfo='skip', # Important: Let clicks pass through lines
                    name=f'Level {levels[i]:.2f}'
                ))
    except Exception as e:
        print(f"Contour generation error: {e}")
        
    return traces

def get_3d_mesh_trace(foci, constant, metric, p_val, angular_n, angular_a, resolution, mode):
    """Generates a Plotly Mesh3d trace for the 3D surface."""
    if not foci or not HAS_SKIMAGE: return None, []
    
    # Range is expanded to [-20, 20] to accommodate larger K constants
    RANGE = 20.0
    l = np.linspace(-RANGE, RANGE, resolution)
    X, Y, Z = np.meshgrid(l, l, l, indexing='ij')
    field = compute_field_3d(foci, X, Y, Z, metric, p_val, angular_n, angular_a, mode)
    target = field if mode == '3D Constant' else field - Z
    
    # Determine levels
    iso_levels = sorted([constant, -constant]) if (mode=='3D Constant' and any(f[2]<0 for f in foci)) else ([constant] if mode=='3D Constant' else [0])
    
    meshes = []
    debug_logs = []

    for i, iso in enumerate(iso_levels):
        # 1. Pre-check: Does the field even cross the iso level?
        min_val, max_val = target.min(), target.max()
        if iso < min_val or iso > max_val:
            debug_logs.append(f"Skipping Iso {iso:.2f}: Value out of range [{min_val:.2f}, {max_val:.2f}]")
            continue

        try:
            # 2. Try Generating Surface
            # Try newer API first, then fallback to older 'lewiner' if needed
            if hasattr(measure, 'marching_cubes'):
                result = measure.marching_cubes(target, iso)
            elif hasattr(measure, 'marching_cubes_lewiner'):
                result = measure.marching_cubes_lewiner(target, iso)
            else:
                debug_logs.append("No valid marching_cubes function found in skimage.measure")
                continue

            # Robust unpacking
            if len(result) >= 2:
                verts = result[0]
                faces = result[1]
            else:
                continue
            
            # 3. Map Coordinates
            # Robustly map index coordinates back to spatial coordinates (-RANGE to RANGE)
            max_idx = resolution - 1
            width = 2 * RANGE
            x = (verts[:, 0] / max_idx) * width - RANGE
            y = (verts[:, 1] / max_idx) * width - RANGE
            z = (verts[:, 2] / max_idx) * width - RANGE
            
            # Match colors to 2D plot (Green for first level, Purple for second)
            mesh_color = '#2ca02c' if i == 0 else '#9467bd'
            
            meshes.append(go.Mesh3d(
                x=x, y=y, z=z,
                i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
                opacity=0.6,
                color=mesh_color,
                flatshading=True,
                name=f'Iso {iso:.2f}',
                hoverinfo='name'
            ))
        except RuntimeError as re:
            # "No surface found at the given iso value"
            debug_logs.append(f"Iso {iso:.2f}: {str(re)}")
            continue
        except Exception as e:
            debug_logs.append(f"Error on Iso {iso:.2f}: {str(e)}")
            continue
        
    return meshes, debug_logs

# ==================== STREAMLIT UI ====================

with st.sidebar:
    # Move Title to sidebar to keep main area top-aligned with chart
    st.title("Ultimate Conics Generator")
    st.info(f"Mode: **{mode if 'mode' in locals() else '2D'}** — Click on the grid to add foci points.")

    st.header("Settings")
    mode = st.radio("Mode", ['2D', '3D Constant', '3D Variable'])
    st.divider()
    metric = st.radio("Distance Metric", ['Euclidean', 'Manhattan', 'Chebyshev', 'Minkowski', 'Angular'])
    
    p_val, ang_n, ang_a = 2.0, 3, 0.3
    if metric == 'Minkowski': p_val = st.slider("P (Minkowski)", 1.0, 6.0, 2.0)
    if metric == 'Angular':
        ang_n = st.slider("Petals (n)", 1, 10, 3)
        ang_a = st.slider("Amplitude (a)", 0.1, 0.9, 0.3)
    
    st.divider()
    st.session_state.k_val = st.slider("Constant (K)", 0.1, 25.0, float(st.session_state.k_val))
    
    if st.button("🎯 Auto Constant"):
        st.session_state.k_val = auto_constant(st.session_state.foci)
        st.rerun()
    
    res = st.slider("Resolution", 100, 500, 300) if mode == '2D' else st.slider("Resolution (3D)", 20, 80, 40)
    if st.button("🗑️ Clear Foci", type="primary"):
        st.session_state.foci = []
        st.rerun()
        
    # --- MOVED FOCI LIST TO SIDEBAR TO PREVENT MAIN SCROLL JUMPING ---
    if st.session_state.foci:
        st.divider()
        st.write("### Active Foci")
        # Use a simpler layout in sidebar
        for i, (fx, fy, s) in enumerate(st.session_state.foci):
            c1, c2, c3 = st.columns([2, 1, 1])
            c1.write(f"**F{i+1}** ({fx:.1f}, {fy:.1f})")
            if c2.button("±", key=f"toggle_{i}"): 
                st.session_state.foci[i][2] *= -1
                st.rerun()
            if c3.button("x", key=f"delete_{i}"):
                st.session_state.foci.pop(i)
                st.rerun()

# ==================== MAIN GRAPH LOGIC ====================

# 1. EARLY EVENT PROCESSING (Prevents Flashing)
if "main_graph" in st.session_state:
    event = st.session_state.main_graph
    if event and "selection" in event:
        points = event["selection"].get("points", [])
        
        valid_point = None
        for p in reversed(points):
            try:
                px = float(p.get('x'))
                py = float(p.get('y'))
                valid_point = (px, py)
                break
            except (TypeError, ValueError):
                continue
        
        if valid_point:
            new_x, new_y = valid_point
            if not any(np.isclose(f[0], new_x, atol=0.15) and np.isclose(f[1], new_y, atol=0.15) for f in st.session_state.foci):
                st.session_state.foci.append([new_x, new_y, 1])


# 2. GENERATE FIGURE (Using Updated State)
fig = go.Figure()

# Setup Sensor Grid
if mode == '2D':
    grid_pts = np.arange(-10, 10.5, 0.5)
else:
    # Use a wider sensor grid for 3D visual context, but keep 0.5 step for clicking
    grid_pts = np.arange(-10, 10.5, 0.5)

gx, gy = np.meshgrid(grid_pts, grid_pts)

if mode == '2D':
    # 2D Sensor
    fig.add_trace(go.Scatter(
        x=gx.flatten(), y=gy.flatten(),
        mode='markers',
        marker=dict(color='rgba(0,0,0,0.01)', size=22, symbol='square'),
        hoverinfo='none',
        name='sensor'
    ))
    
    if st.session_state.foci:
        geom_traces = get_2d_contour_traces(st.session_state.foci, st.session_state.k_val, metric, p_val, ang_n, ang_a, res)
        for t in geom_traces:
            fig.add_trace(t)

    for i, (fx, fy, s) in enumerate(st.session_state.foci):
        color = '#1f77b4' if s > 0 else '#d62728'
        fig.add_trace(go.Scatter(
            x=[fx], y=[fy],
            mode='markers+text',
            text=[f"F{i+1}"],
            textposition="top center",
            marker=dict(color=color, size=15, line=dict(width=2, color='white')),
            name=f"F{i+1}",
            hoverinfo='text'
        ))
        
    fig.update_layout(
        xaxis=dict(range=[-10.5, 10.5], fixedrange=True, zeroline=True, gridcolor='lightgray'),
        yaxis=dict(range=[-10.5, 10.5], fixedrange=True, scaleanchor="x", zeroline=True, gridcolor='lightgray'),
        width=700, height=700,
        margin=dict(l=0,r=0,t=0,b=0),
        showlegend=False,
        plot_bgcolor='white',
        clickmode='event+select',
        dragmode='pan',
        uirevision='constant' # Keeps zoom/pan state when data updates
    )

else:
    # 3D Sensor
    fig.add_trace(go.Scatter3d(
        x=gx.flatten(), y=gy.flatten(), z=np.zeros_like(gx.flatten()),
        mode='markers',
        marker=dict(color='rgba(0,0,0,0.01)', size=10, symbol='square'),
        hoverinfo='none',
        name='sensor'
    ))
    
    # 3D Geometry Traces
    debug_info = []
    if st.session_state.foci:
        meshes, debug_info = get_3d_mesh_trace(st.session_state.foci, st.session_state.k_val, metric, p_val, ang_n, ang_a, res, mode)
        if meshes:
            for m in meshes:
                fig.add_trace(m)
        elif not HAS_SKIMAGE:
            st.sidebar.error("scikit-image is required for 3D visualization.")
        else:
            # If no meshes but we have logs, user needs to know why
            # Move warning to sidebar to avoid layout shift in main area
            st.sidebar.warning("No surface geometry found within the bounds.")

    if debug_info:
        with st.sidebar.expander("Debug: 3D Generation Logs"):
            for log in debug_info:
                st.write(f"- {log}")

    # 3D Foci
    for i, (fx, fy, s) in enumerate(st.session_state.foci):
        color = '#1f77b4' if s > 0 else '#d62728'
        if mode == '3D Constant':
            fig.add_trace(go.Scatter3d(
                x=[fx], y=[fy], z=[0],
                mode='markers+text',
                text=[f"F{i+1}"],
                marker=dict(color=color, size=8, line=dict(width=2, color='white')),
                name=f"F{i+1}"
            ))
        else:
            fig.add_trace(go.Scatter3d(
                x=[fx, fx], y=[fy, fy], z=[-10, 10],
                mode='lines',
                line=dict(color=color, dash='dash', width=4),
                name=f"F{i+1}"
            ))

    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[-20, 20]),
            yaxis=dict(range=[-20, 20]),
            zaxis=dict(range=[-20, 20]),
            aspectmode='cube'
        ),
        width=800, height=800,
        margin=dict(l=0,r=0,t=0,b=0),
        showlegend=False,
        clickmode='event+select',
        dragmode='turntable',
        uirevision='constant' # Keeps camera rotation when data updates
    )

# 3. RENDER CHART
# Using a container to reserve space and stabilize layout
with st.container():
    st.plotly_chart(
        fig, 
        use_container_width=True, 
        on_select="rerun", 
        selection_mode="points", 
        key="main_graph",
        config={'displayModeBar': True}
    )
