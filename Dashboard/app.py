import streamlit as st
import pandas as pd
import pydeck as pdk
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from itertools import cycle
from pyproj import Transformer
from scipy.spatial import Delaunay
import time
import os
from datetime import datetime

# ==============================================================================
# 0. CONFIGURATION & STATE MANAGEMENT
# ==============================================================================
st.set_page_config(layout="wide", page_title="CPT 3D Explorer")
Z_SCALE = 5

# Initialize Session State for Navigation
if 'view_mode' not in st.session_state:
    st.session_state['view_mode'] = 'map' # Options: 'map', 'deep_dive'
if 'deep_dive_target' not in st.session_state:
    st.session_state['deep_dive_target'] = None

# File Paths
FILE_VITO = "labelled_prepared_cpt.parquet"
FILE_PRED = "predicted_cpt.parquet"
FILE_DOV = "DOV_all_cpts_full_columns.parquet"
FILE_LOG = "cpt_edit_log.csv"

PRIORITY_LAYERS = [
    "quartair", "diest", "bolderberg", "sint_huibrechts_hern",
    "ursel", "asse", "wemmel", "lede", "brussel",
    "merelbeke", "kwatrecht", "mont_panisel", "aalbeke", "mons_en_pevele"
]

COLOR_MAP = {
    'quartair': [220, 220, 220, 255], 'diest': [34, 139, 34, 255],
    'bolderberg': [154, 205, 50, 255], 'sint_huibrechts_hern': [189, 183, 107, 255],
    'ursel': [47, 79, 79, 255], 'asse': [107, 142, 35, 255],
    'wemmel': [255, 165, 0, 255], 'lede': [255, 215, 0, 255],
    'brussel': [255, 255, 0, 255], 'merelbeke': [0, 0, 255, 255],
    'kwatrecht': [72, 61, 139, 255], 'mont_panisel': [210, 105, 30, 255],
    'aalbeke': [165, 42, 42, 255], 'mons_en_pevele': [139, 69, 19, 255],
    'boom': [128, 128, 0, 255], 'kortrijk': [70, 130, 180, 255],
    'tielt': [46, 139, 87, 255], 'gent': [218, 165, 32, 255],
    'maldegem': [0, 128, 128, 255], 'lillo': [240, 230, 140, 255],
    'kasterlee': [189, 183, 107, 255], 'berchem': [0, 100, 0, 255]
}
DEFAULT_COLOR = [128, 128, 128, 255]

# ==============================================================================
# 1. HELPERS
# ==============================================================================
def inject_custom_css():
    css_rules = []
    def get_text_color(rgba):
        brightness = (rgba[0] * 299 + rgba[1] * 587 + rgba[2] * 114) / 1000
        return "black" if brightness > 125 else "white"

    for layer, rgba in COLOR_MAP.items():
        bg_color = f"rgba({rgba[0]}, {rgba[1]}, {rgba[2]}, 1)"
        text_color = get_text_color(rgba)
        rule = f"""
        span[data-baseweb="tag"]:has(span[title="{layer}"]) {{
            background-color: {bg_color} !important;
            border: 1px solid rgba(0,0,0,0.1);
        }}
        span[data-baseweb="tag"]:has(span[title="{layer}"]) span {{
            color: {text_color} !important;
        }}
        """
        css_rules.append(rule)
    st.markdown(f"<style>{''.join(css_rules)}</style>", unsafe_allow_html=True)

def log_edit(action, cpt_id, details):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    df_log = pd.DataFrame([{"Timestamp": timestamp, "CPT": cpt_id, "Action": action, "Details": details}])
    header = not os.path.exists(FILE_LOG)
    df_log.to_csv(FILE_LOG, mode='a', header=header, index=False)

def save_data_by_source(full_df, source_type):
    target_file = None
    if source_type == "VITO Data":
        target_file = FILE_VITO
    elif source_type == "Predicted Data":
        target_file = FILE_PRED

    if target_file:
        with st.spinner(f"Saving {source_type} to disk..."):
            df_save = full_df[full_df['source'] == source_type].copy()
            if source_type == "Predicted Data":
                df_save['pred_label'] = df_save['formation']
            visual_cols = ['lon', 'lat', 'color', 'source_color', 'scaled_elevation', 'formation']
            df_save = df_save.drop(columns=[c for c in visual_cols if c in df_save.columns])
            df_save.to_parquet(target_file, index=False)
            st.cache_data.clear()

def find_n_nearest_by_source(target_id, full_df, source_filter, n=3):
    """New helper for the 6-panel grid"""
    target_row = full_df[full_df['sondering_id'] == target_id]
    if target_row.empty: return [None]*n
    tx, ty = target_row.iloc[0]['x'], target_row.iloc[0]['y']

    candidates = full_df[full_df['source'] == source_filter].drop_duplicates('sondering_id').copy()
    if candidates.empty: return [None]*n

    candidates['dist'] = np.sqrt((candidates['x'] - tx)**2 + (candidates['y'] - ty)**2)
    # Exclude the target itself if it exists in this source group to show neighbors
    candidates = candidates[candidates['sondering_id'] != target_id]

    nearest_ids = candidates.sort_values('dist')['sondering_id'].head(n).tolist()
    while len(nearest_ids) < n: nearest_ids.append(None)
    return nearest_ids

# ==============================================================================
# 2. DATA LOADING
# ==============================================================================
@st.cache_data
def load_and_process_data(vito_path, dov_path, pred_path):
    combined_df = pd.DataFrame()

    # A. VITO
    try:
        df_vito = pd.read_parquet(vito_path)
        if 'diepte' in df_vito.columns:
            rename_map = {'diepte_mtaw': 'depth_mtaw', 'diepte': 'depth', 'sondeernummer': 'sondering_id'}
            df_vito = df_vito.rename(columns=rename_map)
        if 'formation' not in df_vito.columns:
            layer_cols = ['diest', 'kwatrecht', 'lede', 'merelbeke', 'mons_en_pevele', 'mont_panisel', 'quartair', 'wemmel']
            existing = [c for c in layer_cols if c in df_vito.columns]
            if existing: df_vito['formation'] = df_vito[existing].idxmax(axis=1)
        df_vito['source'] = 'VITO Data'
        if 'x' not in df_vito.columns: df_vito['x'], df_vito['y'] = 0, 0
        combined_df = pd.concat([combined_df, df_vito], ignore_index=True)
    except FileNotFoundError: pass

    # B. PREDICTED
    try:
        df_pred = pd.read_parquet(pred_path)
        rename_map = {'diepte_mtaw': 'depth_mtaw', 'diepte': 'depth', 'sondeernummer': 'sondering_id'}
        df_pred = df_pred.rename(columns={k:v for k,v in rename_map.items() if k in df_pred.columns})
        if 'pred_label' in df_pred.columns: df_pred['formation'] = df_pred['pred_label']
        elif 'formation' not in df_pred.columns: df_pred['formation'] = 'unknown'
        df_pred['source'] = 'Predicted Data'
        if 'x' not in df_pred.columns: df_pred['x'], df_pred['y'] = 0, 0
        combined_df = pd.concat([combined_df, df_pred], ignore_index=True)
    except FileNotFoundError: pass

    # C. DOV
    try:
        df_dov = pd.read_parquet(dov_path)
        if 'start_sondering_mtaw' in df_dov.columns:
            df_dov['depth_mtaw'] = df_dov['start_sondering_mtaw'] - df_dov['diepte']
        else: df_dov['depth_mtaw'] = 0 - df_dov['diepte']
        dov_rename = {'sondeernummer': 'sondering_id', 'x_x': 'x', 'y_x': 'y', 'diepte': 'depth', 'lithostrat_id': 'formation'}
        df_dov = df_dov.rename(columns=dov_rename)
        df_dov['source'] = 'DOV Public'
        cols = ['sondering_id', 'x', 'y', 'depth', 'depth_mtaw', 'qc', 'fs', 'formation', 'source']
        existing = [c for c in cols if c in df_dov.columns]
        combined_df = pd.concat([combined_df, df_dov[existing]], ignore_index=True)
    except FileNotFoundError: pass

    if combined_df.empty: return pd.DataFrame()

    # D. CLEANING
    combined_df['sondering_id'] = combined_df['sondering_id'].astype(str).str.strip()
    for col in ['qc', 'fs', 'depth', 'x', 'y', 'depth_mtaw']:
        combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce')
    combined_df = combined_df.dropna(subset=['depth', 'qc'])

    # E. COORDS
    if 'lon' not in combined_df.columns:
        valid = combined_df.dropna(subset=['x', 'y'])
        if not valid.empty and valid['x'].mean() > 10000:
            transformer = Transformer.from_crs("epsg:31370", "epsg:4326", always_xy=True)
            lon, lat = transformer.transform(combined_df["x"].values, combined_df["y"].values)
            combined_df["lon"], combined_df["lat"] = lon, lat
        else: combined_df["lon"], combined_df["lat"] = 4.35, 50.85

    # F. COLORS
    combined_df['formation'] = combined_df['formation'].astype(str).str.lower().str.replace(' ', '_')
    combined_df['color'] = combined_df['formation'].apply(lambda x: COLOR_MAP.get(x, DEFAULT_COLOR))

    def get_source_color(src):
        if src == 'VITO Data': return [0, 0, 255, 255]
        if src == 'Predicted Data': return [128, 0, 128, 255]
        return [255, 0, 0, 255]
    combined_df['source_color'] = combined_df['source'].apply(get_source_color)

    return combined_df

# ==============================================================================
# 3. EDITING LOGIC
# ==============================================================================
def extract_intervals_from_cpt(df_cpt):
    df_cpt = df_cpt.sort_values('depth')
    depths = df_cpt['depth'].values
    forms = df_cpt['formation'].fillna("unknown").values
    intervals = []
    if len(depths) == 0: return pd.DataFrame()
    current_fmt, start_z = forms[0], depths[0]
    for i in range(1, len(depths)):
        if forms[i] != current_fmt:
            intervals.append({"Top (m)": start_z, "Bottom (m)": depths[i-1], "Formation": current_fmt})
            current_fmt, start_z = forms[i], depths[i]
    intervals.append({"Top (m)": start_z, "Bottom (m)": depths[-1], "Formation": current_fmt})
    return pd.DataFrame(intervals)

def apply_intervals_to_dataframe(full_df, cpt_id, interval_df):
    cpt_mask = full_df['sondering_id'] == cpt_id
    cpt_rows = full_df.loc[cpt_mask].copy()
    for _, row in interval_df.iterrows():
        mask = (cpt_rows['depth'] >= row['Top (m)']) & (cpt_rows['depth'] <= row['Bottom (m)'])
        cpt_rows.loc[mask, 'formation'] = row['Formation']
    full_df.loc[cpt_mask, 'formation'] = cpt_rows['formation']
    return full_df

def split_layer_at_depth(df, cpt_id, split_depth):
    cpt_data = df[df['sondering_id'] == cpt_id].sort_values('depth')
    intervals = extract_intervals_from_cpt(cpt_data)
    new_rows = []
    split_happened = False
    for _, row in intervals.iterrows():
        if row['Top (m)'] < split_depth < row['Bottom (m)']:
            new_rows.append({"Top (m)": row['Top (m)'], "Bottom (m)": split_depth, "Formation": row['Formation']})
            new_rows.append({"Top (m)": split_depth, "Bottom (m)": row['Bottom (m)'], "Formation": row['Formation']})
            split_happened = True
        else: new_rows.append(row.to_dict())
    if split_happened: return apply_intervals_to_dataframe(df, cpt_id, pd.DataFrame(new_rows))
    return df

def move_nearest_boundary(df, cpt_id, new_depth):
    cpt_data = df[df['sondering_id'] == cpt_id].sort_values('depth')
    intervals = extract_intervals_from_cpt(cpt_data)
    boundaries = []
    for idx, row in intervals.iterrows():
        if idx < len(intervals) - 1: boundaries.append({'val': row['Bottom (m)'], 'idx': idx})
    if not boundaries: return df
    nearest = min(boundaries, key=lambda x: abs(x['val'] - new_depth))
    if abs(nearest['val'] - new_depth) > 10.0: return df
    idx = nearest['idx']
    intervals.at[idx, 'Bottom (m)'] = new_depth
    intervals.at[idx+1, 'Top (m)'] = new_depth
    return apply_intervals_to_dataframe(df, cpt_id, intervals)

# ==============================================================================
# 4. VISUALIZATION HELPERS
# ==============================================================================
@st.cache_data
def generate_triangulated_surfaces(df, target_layers=None):
    polygons = []
    if 'formation' not in df.columns or df.empty: return pd.DataFrame()
    if not target_layers: return pd.DataFrame()
    cpt_locs = df.drop_duplicates('sondering_id')[['sondering_id', 'x', 'y', 'lon', 'lat']].reset_index(drop=True)
    if len(cpt_locs) < 3: return pd.DataFrame()
    try: tri = Delaunay(cpt_locs[['x', 'y']].values)
    except: return pd.DataFrame()
    for layer in target_layers:
        layer_data = df[df['formation'] == layer]
        if layer_data.empty: continue
        z_map = layer_data.groupby('sondering_id')['depth_mtaw'].max().to_dict()
        color = list(layer_data.iloc[0]['color']); color[3] = 150
        for simplex in tri.simplices:
            ids = [cpt_locs.iloc[i]['sondering_id'] for i in simplex]
            if all(id_ in z_map for id_ in ids):
                coords = [[cpt_locs.iloc[i]['lon'], cpt_locs.iloc[i]['lat'], z_map[ids[idx]] * Z_SCALE] for idx, i in enumerate(simplex)]
                polygons.append({'polygon': coords, 'color': color, 'layer_name': layer})
    return pd.DataFrame(polygons)

def get_optimized_map_data(df, stride=10):
    cols = ['sondering_id', 'lon', 'lat', 'depth_mtaw', 'color', 'formation', 'depth', 'source', 'source_color']
    df_map = df[cols].iloc[::stride].copy()
    df_map['depth_mtaw'] = pd.to_numeric(df_map['depth_mtaw'], errors='coerce').fillna(0)
    df_map['scaled_elevation'] = df_map['depth_mtaw'] * Z_SCALE
    return df_map

def find_nearest_cpt(target_id, df):
    cpt_locs = df.drop_duplicates('sondering_id')[['sondering_id', 'x', 'y']]
    target = cpt_locs[cpt_locs['sondering_id'] == target_id]
    if target.empty: return None, 0.0
    tx, ty = target.iloc[0]['x'], target.iloc[0]['y']
    candidates = cpt_locs[cpt_locs['sondering_id'] != target_id].copy()
    if candidates.empty: return None, 0.0
    candidates['dist'] = np.sqrt((candidates['x'] - tx)**2 + (candidates['y'] - ty)**2)
    nearest = candidates.sort_values('dist').iloc[0]
    return nearest['sondering_id'], nearest['dist']

def plot_cpt_interactive(df_cpt, title, neighbor_df=None, neighbor_title=None):
    has_neighbor = neighbor_df is not None and not neighbor_df.empty
    cols = 2 if has_neighbor else 1
    titles = [f"<b>{title}</b>", f"<b>{neighbor_title}</b>"] if has_neighbor else [f"<b>{title}</b>"]

    fig = make_subplots(rows=1, cols=cols, shared_yaxes=True, subplot_titles=titles, horizontal_spacing=0.05)

    def add_traces(df, col_idx):
        df = df.sort_values('depth')
        # QC (with Hover Text)
        fig.add_trace(go.Scatter(
            x=df['qc'], y=df['depth'], mode='lines', name='qc',
            text=df['formation'], # <--- Formation Name for Hover
            line=dict(color='black', width=1.5),
            hovertemplate="<b>%{text}</b><br>D: %{y:.2f}m<br>Qc: %{x:.2f} MPa",
            legendgroup="qc"
        ), row=1, col=col_idx)

        # FS (with Hover Text)
        xaxis_id = f"x{col_idx*2 + 1}" if col_idx == 1 else f"x{col_idx*2 + 2}"
        fig.add_trace(go.Scatter(
            x=df['fs'], y=df['depth'], mode='lines', name='fs',
            text=df['formation'], # <--- Formation Name for Hover
            line=dict(color='red', width=1.5, dash='dot'),
            hovertemplate="<b>%{text}</b><br>D: %{y:.2f}m<br>Fs: %{x:.2f} MPa",
            xaxis=xaxis_id, legendgroup="fs"
        ), row=1, col=col_idx)

        # Background Geology Blocks
        labels = df['formation'].fillna("Unknown").tolist()
        depths = df['depth'].tolist()
        if not depths: return
        start_idx = 0
        for i in range(1, len(labels)):
            if labels[i] != labels[i-1]:
                fmt = labels[start_idx]
                c_rgba = COLOR_MAP.get(fmt, DEFAULT_COLOR)
                c_str = f"rgba({c_rgba[0]},{c_rgba[1]},{c_rgba[2]},0.3)"
                fig.add_hrect(y0=depths[start_idx], y1=depths[i-1], fillcolor=c_str, layer="below", line_width=0, row=1, col=col_idx)
                start_idx = i
        # Final block
        fmt = labels[start_idx]
        c_rgba = COLOR_MAP.get(fmt, DEFAULT_COLOR)
        c_str = f"rgba({c_rgba[0]},{c_rgba[1]},{c_rgba[2]},0.3)"
        fig.add_hrect(y0=depths[start_idx], y1=depths[-1], fillcolor=c_str, layer="below", line_width=0, row=1, col=col_idx)

    add_traces(df_cpt, 1)
    if has_neighbor: add_traces(neighbor_df, 2)

    layout_args = {
        "height": 650, "showlegend": False, "template": "plotly_white", "margin": dict(l=20, r=20, t=60, b=20),
        "yaxis": dict(title="Depth (m)", autorange="reversed"),
        "xaxis": dict(title="qc (MPa)", side="bottom", range=[0, 30], showgrid=True, title_font=dict(color="black")),
        "xaxis3": dict(title="fs (MPa)", side="top", overlaying="x", anchor="y", range=[0, 1.5], showgrid=False, title_font=dict(color="red"), tickfont=dict(color="red"), matches=None),
        "clickmode": "event+select", "dragmode": "select"
    }
    if has_neighbor:
        layout_args.update({
            "yaxis2": dict(autorange="reversed", showticklabels=False),
            "xaxis2": dict(title="qc (MPa)", side="bottom", range=[0, 30], showgrid=True, title_font=dict(color="black")),
            "xaxis4": dict(title="fs (MPa)", side="top", overlaying="x2", anchor="y2", range=[0, 1.5], showgrid=False, title_font=dict(color="red"), tickfont=dict(color="red"), matches=None)
        })
    fig.update_layout(**layout_args)
    return fig

def plot_mini_cpt(df_cpt, title, config, height=400):
    """Simplified plotter for the 6-grid view with adjustable height"""
    if df_cpt is None or df_cpt.empty:
        fig = go.Figure(); fig.update_layout(title="No Data", template="plotly_white")
        return fig

    df = df_cpt.sort_values('depth')
    # Calc RF on fly if missing
    if 'rf' not in df.columns and 'qc' in df.columns and 'fs' in df.columns:
        df['rf'] = (df['fs'] / df['qc']) * 100
        df['rf'] = df['rf'].replace([np.inf, -np.inf], 0).fillna(0)

    fig = go.Figure()

    # 1. Geology
    if config['show_geo']:
        labels = df['formation'].fillna("Unknown").tolist()
        depths = df['depth'].tolist()
        if depths:
            start_idx = 0
            for i in range(1, len(labels)):
                if labels[i] != labels[i-1]:
                    fmt = labels[start_idx]
                    c_rgba = COLOR_MAP.get(fmt, DEFAULT_COLOR)
                    c_str = f"rgba({c_rgba[0]},{c_rgba[1]},{c_rgba[2]},0.3)"
                    fig.add_hrect(y0=depths[start_idx], y1=depths[i-1], fillcolor=c_str, layer="below", line_width=0)
                    start_idx = i
            # Final
            fmt = labels[start_idx]
            c_rgba = COLOR_MAP.get(fmt, DEFAULT_COLOR)
            c_str = f"rgba({c_rgba[0]},{c_rgba[1]},{c_rgba[2]},0.3)"
            fig.add_hrect(y0=depths[start_idx], y1=depths[-1], fillcolor=c_str, layer="below", line_width=0)

    # 2. Qc
    if config['show_qc']:
        fig.add_trace(go.Scatter(x=df['qc'], y=df['depth'], mode='lines', name='qc', line=dict(color='black', width=1.5)))

    # 3. Fs
    if config['show_fs']:
        fig.add_trace(go.Scatter(x=df['fs'], y=df['depth'], mode='lines', name='fs', line=dict(color='blue', width=1.5, dash='dot'), xaxis='x2'))

    # 4. Rf
    if config['show_rf']:
        fig.add_trace(go.Scatter(x=df['rf'], y=df['depth'], mode='lines', name='Rf', line=dict(color='green', width=1.5), xaxis='x3'))

    layout = {
        "title": dict(text=f"<b>{title}</b>", font=dict(size=14)),
        "template": "plotly_white", "height": height, # <--- Dynamic Height
        "margin": dict(l=10, r=10, t=40, b=10),
        "yaxis": dict(title="Depth (m)", autorange="reversed"),
        "xaxis": dict(title="qc (MPa)", side="bottom", showgrid=True, domain=[0, 1]),
        "showlegend": False
    }
    if config['show_fs']:
        layout['xaxis2'] = dict(title="fs", overlaying="x", side="top", showgrid=False, title_font=dict(color="blue"), tickfont=dict(color="blue"))
    if config['show_rf']:
        layout['xaxis3'] = dict(title="Rf (%)", overlaying="x", side="top", position=0.15, showgrid=False, title_font=dict(color="green"), tickfont=dict(color="green"))

    fig.update_layout(**layout)
    return fig

# ==============================================================================
# 5. PAGE RENDERERS
# ==============================================================================
inject_custom_css()
st.title("🇧🇪 CPT Explorer")

# Load Data (Global)
raw_df = load_and_process_data(FILE_VITO, FILE_DOV, FILE_PRED)
if raw_df.empty: st.error("No data loaded."); st.stop()

# --- STATE 1: MAP VIEW ---
if st.session_state['view_mode'] == 'map':

    # --- SIDEBAR ---
    st.sidebar.header("1. Filters")
    show_vito = st.sidebar.checkbox("Show VITO Data (Blue)", True)
    show_pred = st.sidebar.checkbox("Show Predicted Data (Purple)", True)
    show_dov = st.sidebar.checkbox("Show DOV Data (Red)", True)

    min_lat, max_lat = raw_df['lat'].min(), raw_df['lat'].max()
    min_lon, max_lon = raw_df['lon'].min(), raw_df['lon'].max()
    if min_lat == max_lat: max_lat += 0.01
    if min_lon == max_lon: max_lon += 0.01
    lat_range = st.sidebar.slider("Latitude", min_lat, max_lat, (min_lat, max_lat), 0.001)
    lon_range = st.sidebar.slider("Longitude", min_lon, max_lon, (min_lon, max_lon), 0.001)

    st.sidebar.header("2. Layers")
    show_surfaces = st.sidebar.checkbox("Show Surfaces", True)
    formations_in_data = set(raw_df['formation'].unique())
    interpolation_options = [fmt for fmt in PRIORITY_LAYERS if fmt in formations_in_data]
    sel_forms = st.sidebar.multiselect("Interpolate Layers", interpolation_options, default=interpolation_options)

    st.sidebar.header("3. Map Style")
    map_style_name = st.sidebar.selectbox("Style", ["Road", "Satellite", "Light", "Dark"])
    style_urls = {"Satellite": "mapbox://styles/mapbox/satellite-streets-v12", "Road": "mapbox://styles/mapbox/streets-v12", "Light": "mapbox://styles/mapbox/light-v11", "Dark": "mapbox://styles/mapbox/dark-v11"}

    # --- FILTERING ---
    df = raw_df.copy()
    srcs = []
    if show_vito: srcs.append("VITO Data")
    if show_pred: srcs.append("Predicted Data")
    if show_dov: srcs.append("DOV Public")
    df = df[df['source'].isin(srcs)]
    mask = (df['lat']>=lat_range[0])&(df['lat']<=lat_range[1])&(df['lon']>=lon_range[0])&(df['lon']<=lon_range[1])
    df = df[mask]

    st.caption(f"Showing **{df['sondering_id'].nunique()}** CPTs.")

    # --- MAP LAYERS ---
    layers, view = [], None
    if not df.empty:
        df_map = get_optimized_map_data(df, stride=10)
        df_tops = df_map.loc[df_map.groupby(['sondering_id', 'source'])['scaled_elevation'].idxmax()]

        if show_surfaces and sel_forms:
            poly_df = generate_triangulated_surfaces(df, target_layers=sel_forms)
            if not poly_df.empty:
                layers.append(pdk.Layer("PolygonLayer", poly_df, get_polygon="polygon", get_fill_color="color", wireframe=True, filled=True, opacity=0.4, pickable=False, id="geo_surfaces"))

        layers.append(pdk.Layer("PointCloudLayer", df_map, get_position=["lon", "lat", "scaled_elevation"], get_color="color", point_size=8, opacity=1.0, pickable=True, auto_highlight=True, id="cpt_pillars"))
        layers.append(pdk.Layer("ScatterplotLayer", df_tops, get_position=["lon", "lat", "scaled_elevation"], get_fill_color=[0,0,0,0], get_line_color="source_color", get_radius=30, line_width_min_pixels=3, stroked=True, filled=True, pickable=True, auto_highlight=True, id="cpt_rings"))
        view = pdk.ViewState(latitude=df_map['lat'].mean(), longitude=df_map['lon'].mean(), zoom=11, pitch=60)
    else:
        view = pdk.ViewState(latitude=50.85, longitude=4.35, zoom=10, pitch=60)

    deck = pdk.Deck(layers=layers, initial_view_state=view, tooltip={"html": "<b>ID:</b> {sondering_id}<br/>{source}<br/>{formation}"}, map_style=style_urls[map_style_name])

    col1, col2 = st.columns([3, 2])
    with col1:
        event = st.pydeck_chart(deck, on_select="rerun", selection_mode="single-object", height=700)

    # --- RIGHT PANEL (Interactive Analysis) ---
    with col2:
        st.subheader("Interactive Analysis")
        selected_id = None
        if event.selection and "objects" in event.selection:
            for layer_name in ["cpt_pillars", "cpt_rings"]:
                if layer_name in event.selection["objects"] and event.selection["objects"][layer_name]:
                    selected_id = event.selection["objects"][layer_name][0].get("sondering_id")
                    break

        if selected_id:
            cpt_main = raw_df[raw_df['sondering_id'] == selected_id]

            # Source selection priority
            source_type = "Unknown"
            available_sources = cpt_main['source'].unique()

            if len(available_sources) > 1:
                if "VITO Data" in available_sources and show_vito: source_type = "VITO Data"
                elif "Predicted Data" in available_sources and show_pred: source_type = "Predicted Data"
                elif "DOV Public" in available_sources and show_dov: source_type = "DOV Public"
                cpt_main = cpt_main[cpt_main['source'] == source_type]
            elif len(available_sources) == 1:
                source_type = available_sources[0]

            if not cpt_main.empty:
                st.markdown(f"### Selected: `{selected_id}` ({source_type})")

                # --- NEW BUTTON: GO TO DEEP DIVE ---
                if st.button("🔍 Go to Deep Dive Analysis", type="primary"):
                    st.session_state['deep_dive_target'] = selected_id
                    st.session_state['view_mode'] = 'deep_dive'
                    st.rerun()
                # -----------------------------------

                neighbor_id, dist = find_nearest_cpt(selected_id, df)
                cpt_neighbor = raw_df[raw_df['sondering_id'] == neighbor_id] if neighbor_id else None

                fig = plot_cpt_interactive(cpt_main, title=f"{selected_id} ({source_type})", neighbor_df=cpt_neighbor, neighbor_title=neighbor_id)
                chart_event = st.plotly_chart(fig, use_container_width=True, on_select="rerun", selection_mode="points")

                clicked_depth = None
                if chart_event and chart_event.selection["points"]:
                    clicked_depth = chart_event.selection["points"][0]["y"]
                    st.info(f"📍 **Clicked Depth:** {clicked_depth:.2f} m")

                st.divider()

                # Editing Tools (Only for VITO/Predicted)
                if source_type in ["VITO Data", "Predicted Data"]:
                    st.markdown("### 🛠️ Graph Tools")
                    if clicked_depth:
                        c1, c2 = st.columns(2)
                        with c1:
                            if st.button(f"✂️ Split at {clicked_depth:.2f}m"):
                                raw_df = split_layer_at_depth(raw_df, selected_id, clicked_depth)
                                log_edit("SPLIT", selected_id, f"Depth: {clicked_depth:.2f}")
                                save_data_by_source(raw_df, source_type)
                                st.success("Split Saved!"); st.rerun()
                        with c2:
                            if st.button(f"↔️ Move Boundary"):
                                raw_df = move_nearest_boundary(raw_df, selected_id, clicked_depth)
                                log_edit("MOVE", selected_id, f"Depth: {clicked_depth:.2f}")
                                save_data_by_source(raw_df, source_type)
                                st.success("Moved!"); st.rerun()
                    else:
                        st.caption("👆 Click the chart line to enable tools.")

                    with st.expander("Table Editor", expanded=False):
                        current_layers = extract_intervals_from_cpt(cpt_main)
                        edited = st.data_editor(current_layers, num_rows="dynamic", use_container_width=True, column_config={"Formation": st.column_config.SelectboxColumn("Formation", options=PRIORITY_LAYERS)}, key=f"ed_{selected_id}")
                        if st.button("💾 Save Table"):
                            raw_df = apply_intervals_to_dataframe(raw_df, selected_id, edited)
                            log_edit("TABLE", selected_id, "Bulk Edit")
                            save_data_by_source(raw_df, source_type)
                            st.success("Saved!"); st.rerun()
                else:
                    st.info(f"🔒 {source_type} is read-only.")
            else:
                st.error("Data empty.")
        else:
            st.info("Select a CPT on the map to start.")

# --- STATE 2: DEEP DIVE VIEW ---
elif st.session_state['view_mode'] == 'deep_dive':

    target_id = st.session_state['deep_dive_target']

    # Header & Back Button
    h1, h2 = st.columns([1, 6])
    with h1:
        if st.button("← Back to Map"):
            st.session_state['view_mode'] = 'map'
            st.rerun()
    with h2:
        st.subheader(f"Deep Dive Analysis: {target_id}")

    # Settings: Toggles AND Height Slider
    with st.expander("📊 Visualization Settings", expanded=True):
        c1, c2 = st.columns([3, 1])
        with c1:
            t1, t2, t3, t4 = st.columns(4)
            show_qc = t1.checkbox("Cone Resistance (qc)", True)
            show_fs = t2.checkbox("Sleeve Friction (fs)", True)
            show_rf = t3.checkbox("Friction Ratio (Rf)", False)
            show_geo = t4.checkbox("Geology Background", True)
        with c2:
            chart_height = st.slider("Chart Height (px)", 200, 1000, 400, step=50)

        viz_config = {'show_qc': show_qc, 'show_fs': show_fs, 'show_rf': show_rf, 'show_geo': show_geo}

    # Data Fetching
    vito_ids = find_n_nearest_by_source(target_id, raw_df, "VITO Data", n=3)
    pred_ids = find_n_nearest_by_source(target_id, raw_df, "Predicted Data", n=3)

    # --- MINI MAP FOR CONTEXT ---
    all_context_ids = [i for i in vito_ids + pred_ids if i is not None]
    if target_id not in all_context_ids: all_context_ids.append(target_id)

    if all_context_ids:
        # Filter raw_df for just these points to make a mini map
        mini_map_df = raw_df[raw_df['sondering_id'].isin(all_context_ids)].drop_duplicates('sondering_id')
        if not mini_map_df.empty:
            avg_lat = mini_map_df['lat'].mean()
            avg_lon = mini_map_df['lon'].mean()

            mini_layer = pdk.Layer(
                "ScatterplotLayer",
                mini_map_df,
                get_position=["lon", "lat"],
                get_fill_color="source_color",
                get_radius=100, # Bigger dots for mini map
                pickable=True,
                stroked=True,
                get_line_color=[0,0,0],
                line_width_min_pixels=2
            )
            mini_view = pdk.ViewState(latitude=avg_lat, longitude=avg_lon, zoom=13, pitch=0)

            st.caption("📍 Location of Selected CPTs")
            st.pydeck_chart(pdk.Deck(layers=[mini_layer], initial_view_state=mini_view, map_style="mapbox://styles/mapbox/light-v10", tooltip={"text": "{sondering_id}"}), height=200)

    # Top Row: VITO
    st.markdown("### Reference Data (VITO)")
    row1 = st.columns(3)
    for i, cid in enumerate(vito_ids):
        with row1[i]:
            if cid:
                df_local = raw_df[raw_df['sondering_id'] == cid]
                df_local = df_local[df_local['source'] == 'VITO Data']
                title = f"⭐ {cid}" if cid == target_id else f"{cid}"
                st.plotly_chart(plot_mini_cpt(df_local, title, viz_config, height=chart_height), use_container_width=True)
            else:
                st.warning("No neighbor")

    # Bottom Row: Predicted
    st.markdown("### Predicted / Unlabeled Data")
    row2 = st.columns(3)
    for i, cid in enumerate(pred_ids):
        with row2[i]:
            if cid:
                df_local = raw_df[raw_df['sondering_id'] == cid]
                df_local = df_local[df_local['source'] == 'Predicted Data']
                title = f"⭐ {cid}" if cid == target_id else f"{cid}"
                st.plotly_chart(plot_mini_cpt(df_local, title, viz_config, height=chart_height), use_container_width=True)
            else:
                st.warning("No neighbor")