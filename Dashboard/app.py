import streamlit as st
import pandas as pd
import pydeck as pdk
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pyproj import Transformer
from scipy.spatial import Delaunay
from scipy import stats
import os
from datetime import datetime
import time

# ==============================================================================
# 0. CONFIGURATION & STATE MANAGEMENT
# ==============================================================================
st.set_page_config(layout="wide", page_title="CPT 3D Explorer")
Z_SCALE = 5

# Initialize Session State
if 'view_mode' not in st.session_state:
    st.session_state['view_mode'] = 'map'
if 'deep_dive_target' not in st.session_state:
    st.session_state['deep_dive_target'] = None
if 'selected_cpt' not in st.session_state:
    st.session_state['selected_cpt'] = None
if 'dd_extra_cpts' not in st.session_state:
    st.session_state['dd_extra_cpts'] = []

# --- FILE PATHS ---
FILE_VITO = "labelled_prepared_cpt.parquet"
FILE_DOV = "DOV_all_cpts_full_columns.parquet"
FILE_SMOOTHED = "labelled_smoothed.parquet"
FILE_UNSMOOTHED = "labelled_unsmoothed.parquet"
FILE_VITERBI = "labelled_viterbi.parquet"
FILE_AMENDED = "amended_cpt.parquet"
FILE_LOG = "cpt_edit_log.csv"

# Model Mapping
MODEL_OPTIONS = {
    "Meta-Ensemble": "meta",
    "XGBoost": "xgb",
    "LSTM": "lstm",
    "Grande": "grande"
}

# --- COLOR MAPPING ---
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
PRIORITY_LAYERS = list(COLOR_MAP.keys()) + ["unknown"]

# ==============================================================================
# 1. HELPERS & AMENDMENT LOGIC
# ==============================================================================
def inject_custom_css():
    """Injects CSS to color the multiselect tags based on COLOR_MAP"""
    css_rules = []
    css_rules.append(".block-container { padding-top: 1rem; padding-bottom: 1rem; }")
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

def load_amended_data(target_id):
    if not os.path.exists(FILE_AMENDED): return None
    try:
        df_amended = pd.read_parquet(FILE_AMENDED)
        df_subset = df_amended[df_amended['sondering_id'] == str(target_id)].copy()
        if not df_subset.empty:
            df_subset['source'] = 'Amended Data'
            return df_subset
    except: pass
    return None

def save_amendment(df_cpt, cpt_id):
    df_cpt = df_cpt.copy()
    df_cpt['sondering_id'] = str(cpt_id)
    df_cpt['source'] = 'Amended Data'
    for col in ['x', 'y', 'depth_mtaw', 'lon', 'lat']:
        if col not in df_cpt.columns: df_cpt[col] = 0

    if os.path.exists(FILE_AMENDED):
        try:
            df_existing = pd.read_parquet(FILE_AMENDED)
            df_existing = df_existing[df_existing['sondering_id'] != str(cpt_id)]
            df_final = pd.concat([df_existing, df_cpt], ignore_index=True)
        except: df_final = df_cpt
    else:
        df_final = df_cpt
    df_final.to_parquet(FILE_AMENDED, index=False)

def find_n_nearest_by_source(target_id, full_df, source_filter, n=3):
    target_row = full_df[full_df['sondering_id'] == target_id]
    if target_row.empty: return [None]*n
    tx, ty = target_row.iloc[0]['x'], target_row.iloc[0]['y']
    candidates = full_df[full_df['source'] == source_filter].drop_duplicates('sondering_id').copy()
    if candidates.empty: return [None]*n
    candidates['dist'] = np.sqrt((candidates['x'] - tx)**2 + (candidates['y'] - ty)**2)
    candidates = candidates[candidates['sondering_id'] != target_id]
    nearest = candidates.sort_values('dist').head(n)
    # Return list of tuples (id, dist) for better UI
    return list(zip(nearest['sondering_id'], nearest['dist']))

# ==============================================================================
# 2. DATA LOADING
# ==============================================================================
@st.cache_data
def load_and_process_data(vito_path, dov_path, smoothed_path):
    combined_df = pd.DataFrame()
    # A. VITO
    try:
        df_vito = pd.read_parquet(vito_path)
        rename_map = {'diepte_mtaw': 'depth_mtaw', 'diepte': 'depth', 'sondeernummer': 'sondering_id'}
        df_vito = df_vito.rename(columns={k:v for k,v in rename_map.items() if k in df_vito.columns})
        if 'formation' not in df_vito.columns:
            layer_cols = [c for c in PRIORITY_LAYERS if c in df_vito.columns]
            if layer_cols: df_vito['formation'] = df_vito[layer_cols].idxmax(axis=1)
            else: df_vito['formation'] = 'unknown'
        df_vito['source'] = 'VITO Data'
        if 'x' not in df_vito.columns: df_vito['x'], df_vito['y'] = 0, 0
        combined_df = pd.concat([combined_df, df_vito], ignore_index=True)
    except: pass

    # B. PREDICTED
    try:
        df_pred = pd.read_parquet(smoothed_path)
        rename_map = {'diepte_mtaw': 'depth_mtaw', 'diepte': 'depth', 'sondeernummer': 'sondering_id'}
        df_pred = df_pred.rename(columns={k:v for k,v in rename_map.items() if k in df_pred.columns})
        if 'meta_top1_label_smooth' in df_pred.columns: df_pred['formation'] = df_pred['meta_top1_label_smooth']
        elif 'xgb_top1_label_smooth' in df_pred.columns: df_pred['formation'] = df_pred['xgb_top1_label_smooth']
        else: df_pred['formation'] = 'unknown'
        df_pred['source'] = 'Predicted Data'
        if 'x' not in df_pred.columns: df_pred['x'], df_pred['y'] = 0, 0
        combined_df = pd.concat([combined_df, df_pred], ignore_index=True)
    except: pass

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
    except: pass

    if combined_df.empty: return pd.DataFrame()
    combined_df['sondering_id'] = combined_df['sondering_id'].astype(str).str.strip()
    combined_df = combined_df.dropna(subset=['depth'])

    if 'lon' not in combined_df.columns:
        valid = combined_df.dropna(subset=['x', 'y'])
        if not valid.empty and valid['x'].mean() > 10000:
            transformer = Transformer.from_crs("epsg:31370", "epsg:4326", always_xy=True)
            lon, lat = transformer.transform(combined_df["x"].values, combined_df["y"].values)
            combined_df["lon"], combined_df["lat"] = lon, lat
        else: combined_df["lon"], combined_df["lat"] = 4.35, 50.85

    combined_df['formation'] = combined_df['formation'].astype(str).str.lower().str.replace(' ', '_')
    combined_df['color'] = combined_df['formation'].apply(lambda x: COLOR_MAP.get(x, DEFAULT_COLOR))

    def get_source_color(src):
        if src == 'VITO Data': return [0, 0, 255, 255]
        if src == 'Predicted Data': return [128, 0, 128, 255]
        return [255, 0, 0, 255]
    combined_df['source_color'] = combined_df['source'].apply(get_source_color)

    return combined_df

def load_deep_dive_data(target_id, mode='smoothed'):
    target_file = None
    if mode == 'unsmoothed': target_file = FILE_UNSMOOTHED
    elif mode == 'smoothed': target_file = FILE_SMOOTHED
    elif mode == 'viterbi': target_file = FILE_VITERBI

    if not target_file or not os.path.exists(target_file): return pd.DataFrame()
    try:
        df_full = pd.read_parquet(target_file)
        rename_map = {'diepte_mtaw': 'depth_mtaw', 'diepte': 'depth', 'sondeernummer': 'sondering_id', 'Predicted_Class_Viterbi': 'viterbi_label'}
        df_full = df_full.rename(columns={k:v for k,v in rename_map.items() if k in df_full.columns})
        df_full['sondering_id'] = df_full['sondering_id'].astype(str).str.strip()
        subset = df_full[df_full['sondering_id'] == str(target_id)].copy()
        subset['depth'] = subset['depth'].round(2)
        subset['source'] = 'Predicted Data'
        return subset
    except: return pd.DataFrame()

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
    is_subset = len(full_df['sondering_id'].unique()) == 1
    cpt_rows = full_df if is_subset else full_df.loc[full_df['sondering_id'] == cpt_id].copy()
    for _, row in interval_df.iterrows():
        mask = (cpt_rows['depth'] >= row['Top (m)']) & (cpt_rows['depth'] <= row['Bottom (m)'])
        cpt_rows.loc[mask, 'formation'] = row['Formation']
    if is_subset: return cpt_rows
    full_df.loc[full_df['sondering_id'] == cpt_id, 'formation'] = cpt_rows['formation']
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
def apply_rolling_smoothing(series, window=5):
    if len(series) < window: return series
    return series.rolling(window=window, center=True).apply(lambda x: stats.mode(x, keepdims=False)[0], raw=True).fillna(method='bfill').fillna(method='ffill').astype(int)

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

def prepare_prediction_data(df_in, model_prefix, mode):
    df = df_in.copy()
    label_col = "unknown"
    need_runtime_smoothing = False

    if mode == 'viterbi':
        if 'viterbi_label' in df.columns: label_col = 'viterbi_label'
        elif 'Predicted_Class_Viterbi' in df.columns: label_col = 'Predicted_Class_Viterbi'
    elif mode == 'smoothed':
        candidate = f"{model_prefix}_top1_label_smooth"
        if candidate in df.columns: label_col = candidate
        else:
            candidate_raw = f"{model_prefix}_top1_label"
            if candidate_raw in df.columns: label_col = candidate_raw; need_runtime_smoothing = True
    else: # Unsmoothed
        candidate = f"{model_prefix}_top1_label"
        if candidate in df.columns: label_col = candidate

    if label_col in df.columns:
        if need_runtime_smoothing:
            df[label_col] = df[label_col].fillna("Unknown")
            codes, uniques = pd.factorize(df[label_col])
            smooth_codes = apply_rolling_smoothing(pd.Series(codes), window=7)
            df['formation_display'] = uniques[smooth_codes.astype(int)]
        else: df['formation_display'] = df[label_col]
        df['formation'] = df['formation_display'].astype(str).str.lower().str.replace(' ', '_')
    else: df['formation'] = 'unknown'; df['formation_display'] = 'Unknown'

    prob_prefix = 'meta' if mode == 'viterbi' else model_prefix
    p1, l1 = (f"{prob_prefix}_top1_prob", f"{prob_prefix}_top1_label")
    if mode == 'viterbi' and 'Final_Top1_Prob' in df.columns: p1, l1 = ('Final_Top1_Prob', 'Predicted_Class_Viterbi')

    def fmt_prob(row):
        try:
            txt = f"<b>{mode.upper()}</b><br>"
            if l1 in row and p1 in row: txt += f"{row[l1]}: {row[p1]:.2f}"
            return txt
        except: return "No Prob Data"

    if l1 in df.columns: df['hover_text'] = df.apply(fmt_prob, axis=1)
    else: df['hover_text'] = df['formation_display']
    return df

def plot_cpt_interactive(df_cpt, title, height=650, neighbor_df=None, neighbor_title=None):
    has_neighbor = neighbor_df is not None and not neighbor_df.empty
    cols = 2 if has_neighbor else 1
    titles = [f"<b>{title}</b>", f"<b>{neighbor_title}</b>"] if has_neighbor else [f"<b>{title}</b>"]

    fig = make_subplots(rows=1, cols=cols, shared_yaxes=True, subplot_titles=titles, horizontal_spacing=0.05)

    def add_traces(df, col_idx):
        df = df.sort_values('depth')
        hover_src = df['hover_text'] if 'hover_text' in df.columns else df['formation']

        # 1. INVISIBLE HOVER LAYER
        fig.add_trace(go.Scatter(
            x=df['qc'], y=df['depth'], mode='lines', opacity=0,
            text=hover_src, hoverinfo='text+y',
            hovertemplate="<b>Depth: %{y:.2f}m</b><br>%{text}<extra></extra>",
            showlegend=False
        ), row=1, col=col_idx)

        # 2. QC Line
        fig.add_trace(go.Scatter(
            x=df['qc'], y=df['depth'], mode='lines', name='qc',
            line=dict(color='black', width=1.5), hoverinfo='skip', legendgroup="qc"
        ), row=1, col=col_idx)

        # 3. FS Line
        xaxis_id = f"x{col_idx*2 + 1}" if col_idx == 1 else f"x{col_idx*2 + 2}"
        fig.add_trace(go.Scatter(
            x=df['fs'], y=df['depth'], mode='lines', name='fs',
            line=dict(color='red', width=1.5, dash='dot'), hoverinfo='skip',
            xaxis=xaxis_id, legendgroup="fs"
        ), row=1, col=col_idx)

        # 4. Background Geology Blocks
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
        fmt = labels[start_idx]
        c_rgba = COLOR_MAP.get(fmt, DEFAULT_COLOR)
        c_str = f"rgba({c_rgba[0]},{c_rgba[1]},{c_rgba[2]},0.3)"
        fig.add_hrect(y0=depths[start_idx], y1=depths[-1], fillcolor=c_str, layer="below", line_width=0, row=1, col=col_idx)

    add_traces(df_cpt, 1)
    if has_neighbor: add_traces(neighbor_df, 2)

    layout_args = {
        "height": height, "showlegend": False, "template": "plotly_white", "margin": dict(l=20, r=20, t=60, b=20),
        "yaxis": dict(title="Depth (m)", autorange="reversed"),
        "xaxis": dict(title="qc (MPa)", side="bottom", range=[0, 30], showgrid=True),
        "xaxis3": dict(title="fs (MPa)", side="top", overlaying="x", anchor="y", range=[0, 1.5], showgrid=False, title_font=dict(color="red"), tickfont=dict(color="red")),
        "clickmode": "event+select", "dragmode": "select", "hovermode": "y unified"
    }
    if has_neighbor:
        layout_args.update({
            "yaxis2": dict(autorange="reversed", showticklabels=False),
            "xaxis2": dict(title="qc (MPa)", side="bottom", range=[0, 30], showgrid=True),
            "xaxis4": dict(title="fs (MPa)", side="top", overlaying="x2", anchor="y2", range=[0, 1.5], showgrid=False, title_font=dict(color="red"), tickfont=dict(color="red"))
        })
    fig.update_layout(**layout_args)
    return fig

def plot_mini_cpt(df_cpt, title, config, height=400):
    if df_cpt is None or df_cpt.empty: return go.Figure(layout=dict(title="No Data"))
    df = df_cpt.sort_values('depth')
    hover_src = df['hover_text'] if 'hover_text' in df.columns else df['formation']

    fig = go.Figure()
    if config['show_geo']:
        labels, depths = df['formation'].fillna("Unknown").tolist(), df['depth'].tolist()
        if depths:
            start_idx = 0
            for i in range(1, len(labels)):
                if labels[i] != labels[i-1]:
                    c_rgba = COLOR_MAP.get(labels[start_idx], DEFAULT_COLOR)
                    fig.add_hrect(y0=depths[start_idx], y1=depths[i-1], fillcolor=f"rgba({c_rgba[0]},{c_rgba[1]},{c_rgba[2]},0.3)", layer="below", line_width=0)
                    start_idx = i
            c_rgba = COLOR_MAP.get(labels[start_idx], DEFAULT_COLOR)
            fig.add_hrect(y0=depths[start_idx], y1=depths[-1], fillcolor=f"rgba({c_rgba[0]},{c_rgba[1]},{c_rgba[2]},0.3)", layer="below", line_width=0)

    fig.add_trace(go.Scatter(x=df['qc'], y=df['depth'], mode='lines', opacity=0, text=hover_src, hovertemplate="%{text}<extra></extra>"))
    if config['show_qc']: fig.add_trace(go.Scatter(x=df['qc'], y=df['depth'], mode='lines', line=dict(color='black', width=1), hoverinfo='skip'))
    if config['show_fs']: fig.add_trace(go.Scatter(x=df['fs'], y=df['depth'], mode='lines', line=dict(color='blue', dash='dot', width=1), xaxis='x2', hoverinfo='skip'))

    layout = {"title": title, "height": height, "yaxis": dict(autorange="reversed"), "showlegend": False, "margin": dict(l=10, r=10, t=30, b=10), "hovermode": "y unified"}
    if config['show_fs']: layout['xaxis2'] = dict(overlaying="x", side="top", showgrid=False)
    fig.update_layout(**layout)
    return fig

# ==============================================================================
# 5. PAGE RENDERERS
# ==============================================================================
inject_custom_css()
st.title("🇧🇪 CPT Explorer")

raw_df = load_and_process_data(FILE_VITO, FILE_DOV, FILE_SMOOTHED)
if raw_df.empty: st.error("No data loaded. Check parquet files."); st.stop()

# --- STATE 1: MAP VIEW ---
if st.session_state['view_mode'] == 'map':

    if 'selected_cpt' not in st.session_state: st.session_state['selected_cpt'] = None
    if 'dd_extra_cpts' not in st.session_state: st.session_state['dd_extra_cpts'] = []

    st.sidebar.header("1. Filters")
    show_vito = st.sidebar.checkbox("Show VITO Data (Blue)", True)
    show_pred = st.sidebar.checkbox("Show Predicted Data (Purple)", True)
    show_dov = st.sidebar.checkbox("Show DOV Data (Red)", True)

    st.sidebar.divider()
    st.sidebar.header("2. Search CPT")
    all_cpt_ids = sorted(raw_df['sondering_id'].unique())

    current_idx = 0
    if st.session_state['selected_cpt'] in all_cpt_ids:
        current_idx = all_cpt_ids.index(st.session_state['selected_cpt']) + 1

    def on_search_change():
        val = st.session_state['cpt_search_box']
        st.session_state['selected_cpt'] = val if val != "Select a CPT..." else None

    selected_from_list = st.sidebar.selectbox("Find by ID:", options=["Select a CPT..."]+all_cpt_ids, index=current_idx, key="cpt_search_box", on_change=on_search_change)

    st.sidebar.divider()
    st.sidebar.header("3. Layers")
    show_surfaces = st.sidebar.checkbox("Show Surfaces", True)
    formations_in_data = set(raw_df['formation'].unique().astype(str))
    valid_layers = [f for f in PRIORITY_LAYERS if f in formations_in_data]
    sel_forms = st.sidebar.multiselect("Interpolate Layers", valid_layers, default=valid_layers[:3])

    st.sidebar.header("4. Map Style")
    map_style_name = st.sidebar.selectbox("Style", ["Road", "Satellite", "Light", "Dark"], index=2)
    style_urls = {
        "Satellite": "mapbox://styles/mapbox/satellite-streets-v12",
        "Road": "mapbox://styles/mapbox/streets-v12",
        "Light": "mapbox://styles/mapbox/light-v11",
        "Dark": "mapbox://styles/mapbox/dark-v11"
    }

    df = raw_df.copy()
    srcs = []
    if show_vito: srcs.append("VITO Data")
    if show_pred: srcs.append("Predicted Data")
    if show_dov: srcs.append("DOV Public")
    df = df[df['source'].isin(srcs)]

    deck_event = None
    if not df.empty:
        layers = []
        if show_surfaces and sel_forms:
            poly_df = generate_triangulated_surfaces(df, target_layers=sel_forms)
            if not poly_df.empty:
                layers.append(pdk.Layer("PolygonLayer", poly_df, get_polygon="polygon", get_fill_color="color", wireframe=True, filled=True, opacity=0.4, pickable=False, id="geo_surfaces"))

        vis_cols = ['sondering_id', 'lon', 'lat', 'depth_mtaw', 'color', 'source_color', 'source', 'formation']
        df_map = df[vis_cols].iloc[::20].copy()
        df_map['scaled_elevation'] = df_map['depth_mtaw'] * Z_SCALE
        df_tops = df_map.loc[df_map.groupby(['sondering_id'])['scaled_elevation'].idxmax()]

        target_id = st.session_state['selected_cpt']
        if target_id:
            highlight_color = [255, 255, 0, 255]
            df_tops['color'] = df_tops.apply(lambda row: highlight_color if row['sondering_id'] == target_id else row['color'], axis=1)

        layers.append(pdk.Layer("PointCloudLayer", df_map, get_position=["lon", "lat", "scaled_elevation"], get_color="color", point_size=8, pickable=True, auto_highlight=True, id="cpt_pillars"))
        layers.append(pdk.Layer("ScatterplotLayer", df_tops, get_position=["lon", "lat", "scaled_elevation"], get_line_color="source_color", get_radius=30, stroked=True, pickable=True, id="cpt_rings"))

        initial_view = pdk.ViewState(latitude=df_map['lat'].mean(), longitude=df_map['lon'].mean(), zoom=11, pitch=60)
        if target_id:
            target_loc = df_tops[df_tops['sondering_id'] == target_id]
            if not target_loc.empty:
                initial_view = pdk.ViewState(latitude=target_loc.iloc[0]['lat'], longitude=target_loc.iloc[0]['lon'], zoom=14, pitch=60)

        col1, col2 = st.columns([3, 2])
        with col1:
            deck_event = st.pydeck_chart(
                pdk.Deck(
                    layers=layers,
                    initial_view_state=initial_view,
                    tooltip={"html": "<b>ID:</b> {sondering_id}<br/><b>Source:</b> {source}<br/><b>Fmt:</b> {formation}"},
                    map_style=style_urls[map_style_name]
                ),
                on_select="rerun",
                height=700
            )

    with col2:
        st.subheader("Interactive Analysis")
        if deck_event and deck_event.selection and "objects" in deck_event.selection:
            for ln in ["cpt_pillars", "cpt_rings"]:
                if ln in deck_event.selection["objects"] and deck_event.selection["objects"][ln]:
                    clicked_id = deck_event.selection["objects"][ln][0].get("sondering_id")
                    if clicked_id and clicked_id != st.session_state['selected_cpt']:
                        st.session_state['selected_cpt'] = clicked_id
                        st.rerun()

        selected_id = st.session_state['selected_cpt']
        if selected_id:
            cpt_main = raw_df[raw_df['sondering_id'] == selected_id]
            available_sources = cpt_main['source'].unique()
            source_type = "Unknown"
            if len(available_sources) > 0:
                if "Predicted Data" in available_sources and show_pred: source_type = "Predicted Data"
                elif "VITO Data" in available_sources and show_vito: source_type = "VITO Data"
                elif "DOV Public" in available_sources and show_dov: source_type = "DOV Public"
                else: source_type = available_sources[0]
                cpt_main = cpt_main[cpt_main['source'] == source_type]

            if not cpt_main.empty:
                st.markdown(f"### Selected: `{selected_id}`")
                st.caption(f"Source: {source_type}")

                if st.button("🔍 Go to Detailed Analysis", type="primary", key="btn_dd_map"):
                    st.session_state['deep_dive_target'] = selected_id
                    st.session_state['dd_extra_cpts'] = [] # Reset on new entry
                    st.session_state['view_mode'] = 'deep_dive'
                    st.rerun()

                fig = plot_cpt_interactive(cpt_main, f"{selected_id}")
                st.plotly_chart(fig, use_container_width=True)

                if source_type in ["Predicted Data", "VITO Data"]:
                    with st.expander("Quick Edit (Tables)"):
                        intervals = extract_intervals_from_cpt(cpt_main)
                        edited = st.data_editor(intervals, num_rows="dynamic", key=f"quick_edit_{selected_id}", column_config={"Formation": st.column_config.SelectboxColumn(options=PRIORITY_LAYERS)})
                        if st.button("💾 Save Quick Edit"):
                            updated_df = apply_intervals_to_dataframe(cpt_main, selected_id, edited)
                            save_amendment(updated_df, selected_id)
                            st.toast("Saved to Amendments!")
                            time.sleep(1)
                            st.rerun()
            else: st.warning(f"Data for {selected_id} is hidden by filters.")
        else: st.info("Select a CPT from the sidebar list or click a point on the map.")

# --- STATE 2: DETAILED VIEW ---
elif st.session_state['view_mode'] == 'deep_dive':
    try:
        main_target_id = st.session_state['deep_dive_target']
        if 'dd_extra_cpts' not in st.session_state: st.session_state['dd_extra_cpts'] = []

        # Calculate Nearest Neighbors (VITO & Predicted) for Reference
        # This is needed for map context and the "Add Neighbor" button
        vito_neighbors = find_n_nearest_by_source(main_target_id, raw_df, "VITO Data", n=5)
        pred_neighbors = find_n_nearest_by_source(main_target_id, raw_df, "Predicted Data", n=10) # Get more to allow sequential adding

        # Header
        h1, h2 = st.columns([1, 6])
        with h1:
            if st.button("← Back"):
                st.session_state['view_mode'] = 'map'
                st.rerun()
        with h2: st.subheader(f"Detailed Analysis: {main_target_id}")

        # --- SETTINGS PANEL (MAP + CONTROLS) ---
        with st.expander("📊 Settings & Context Map", expanded=True):
            col_settings, col_map = st.columns([1, 1])

            with col_settings:
                c1, c2 = st.columns(2)
                with c1:
                    selected_model = st.selectbox("Model", list(MODEL_OPTIONS.keys()))
                    model_prefix = MODEL_OPTIONS[selected_model]
                with c2:
                    smooth_mode = st.radio("Mode", ["Unsmoothed", "Smoothed", "Viterbi"], index=1, horizontal=True)
                    mode_key = smooth_mode.lower()

                chart_height = st.slider("Chart Height (px)", 300, 1200, 500, step=50)

                # --- ADD NEIGHBOR BUTTON ---
                st.divider()
                st.markdown("**Add Predicted Neighbors for Amendment**")

                # Determine next candidate
                current_shown = [main_target_id] + st.session_state['dd_extra_cpts']
                next_candidate = None
                for pid, dist in pred_neighbors:
                    if pid not in current_shown and pid is not None:
                        next_candidate = (pid, dist)
                        break

                if next_candidate:
                    cand_id, cand_dist = next_candidate
                    if st.button(f"➕ Add Nearest: {cand_id} ({cand_dist:.0f}m)"):
                        st.session_state['dd_extra_cpts'].append(cand_id)
                        st.rerun()
                else:
                    st.info("No more nearby neighbors found.")

                if st.session_state['dd_extra_cpts']:
                    if st.button("🗑️ Clear Added CPTs"):
                        st.session_state['dd_extra_cpts'] = []
                        st.rerun()

            with col_map:
                # --- MINI CONTEXT MAP ---
                # 1. Main Target (Red)
                # 2. Added/Amended (Purple)
                # 3. VITO Reference (Blue)

                map_data = []

                # Helper to get coords
                def get_coords(cid):
                    r = raw_df[raw_df['sondering_id'] == cid]
                    if not r.empty: return r.iloc[0]['lat'], r.iloc[0]['lon']
                    return None, None

                # Main
                lat, lon = get_coords(main_target_id)
                if lat: map_data.append({"id": main_target_id, "lat": lat, "lon": lon, "color": [255, 0, 0, 200], "role": "Main"})

                # Added
                for cid in st.session_state['dd_extra_cpts']:
                    lat, lon = get_coords(cid)
                    if lat: map_data.append({"id": cid, "lat": lat, "lon": lon, "color": [128, 0, 128, 200], "role": "Added"})

                # VITO (First 3)
                for cid, _ in vito_neighbors[:3]:
                    if cid:
                        lat, lon = get_coords(cid)
                        if lat: map_data.append({"id": cid, "lat": lat, "lon": lon, "color": [0, 0, 255, 150], "role": "Ref"})

                df_mini = pd.DataFrame(map_data)
                if not df_mini.empty:
                    view_state = pdk.ViewState(latitude=df_mini['lat'].mean(), longitude=df_mini['lon'].mean(), zoom=13)
                    layer = pdk.Layer(
                        "ScatterplotLayer", df_mini,
                        get_position=["lon", "lat"], get_fill_color="color",
                        get_radius=100, pickable=True, stroked=True, get_line_color=[0,0,0], line_width_min_pixels=1
                    )
                    st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip={"text": "{role}: {id}"}, map_style="mapbox://styles/mapbox/light-v10"), height=300)

        # --- MAIN CPT LOOP (STACKED) ---
        cpts_to_show = [main_target_id] + st.session_state['dd_extra_cpts']

        for i, target_id in enumerate(cpts_to_show):
            st.markdown(f"### {i+1}. CPT: `{target_id}`")

            df_target = load_amended_data(target_id)
            is_amended = False

            if df_target is not None:
                is_amended = True
                df_target['color'] = df_target['formation'].apply(lambda x: COLOR_MAP.get(x, DEFAULT_COLOR))
                if 'hover_text' not in df_target.columns: df_target['hover_text'] = df_target['formation']
            else:
                if mode_key == 'smoothed':
                    df_subset = raw_df[(raw_df['sondering_id'] == target_id) & (raw_df['source'] == 'Predicted Data')]
                    if not df_subset.empty: df_target = prepare_prediction_data(df_subset, model_prefix, mode='smoothed')
                else:
                    df_subset = load_deep_dive_data(target_id, mode=mode_key)
                    if not df_subset.empty:
                        df_target = prepare_prediction_data(df_subset, model_prefix, mode=mode_key)
                        if 'qc' not in df_target.columns:
                            base = raw_df[(raw_df['sondering_id'] == target_id) & (raw_df['source'] == 'Predicted Data')]
                            if not base.empty: df_target = pd.merge_asof(df_target.sort_values('depth'), base[['depth','qc','fs']].sort_values('depth'), on='depth')

            if df_target is None or df_target.empty:
                st.error(f"No data found for {target_id}"); continue

            col_plot, col_tools = st.columns([3, 1])
            with col_plot:
                title_txt = f"{target_id} ({'Amended' if is_amended else smooth_mode})"
                # PASS DYNAMIC HEIGHT HERE
                fig_main = plot_cpt_interactive(df_target, title_txt, height=chart_height)
                sel_pts = st.plotly_chart(fig_main, use_container_width=True, on_select="rerun", selection_mode="points", key=f"plot_{target_id}")

                clicked_depth = None
                if sel_pts and sel_pts.selection["points"]:
                    clicked_depth = sel_pts.selection["points"][0]["y"]
                    st.info(f"📍 Selected Depth: **{clicked_depth:.2f} m**")

            with col_tools:
                st.markdown("#### Tools")
                if is_amended: st.caption("⚠️ Editing Amendment")

                if clicked_depth:
                    if st.button("✂️ Split", key=f"split_{target_id}"):
                        df_target = split_layer_at_depth(df_target, target_id, clicked_depth)
                        save_amendment(df_target, target_id)
                        st.rerun()
                    if st.button("↔️ Move", key=f"move_{target_id}"):
                        df_target = move_nearest_boundary(df_target, target_id, clicked_depth)
                        save_amendment(df_target, target_id)
                        st.rerun()

                with st.expander("Table", expanded=False):
                    intervals = extract_intervals_from_cpt(df_target)
                    edited = st.data_editor(intervals, num_rows="dynamic", column_config={"Formation": st.column_config.SelectboxColumn(options=PRIORITY_LAYERS)}, key=f"table_{target_id}")
                    if st.button("💾 Save Table", key=f"save_table_{target_id}"):
                        df_target = apply_intervals_to_dataframe(df_target, target_id, edited)
                        save_amendment(df_target, target_id)
                        st.rerun()

                if is_amended:
                    if st.button("❌ Revert", type="secondary", key=f"revert_{target_id}"):
                        try:
                            df_am = pd.read_parquet(FILE_AMENDED)
                            df_am = df_am[df_am['sondering_id'] != str(target_id)]
                            df_am.to_parquet(FILE_AMENDED, index=False)
                            st.rerun()
                        except: pass

            st.divider()

        # --- NEIGHBORS (Context for Main Target Only) ---
        st.subheader(f"Reference Neighbors (VITO)")
        # Show Top 3 VITO
        cols = st.columns(3)
        viz = {'show_qc':True, 'show_fs':True, 'show_geo':True, 'show_rf': False}
        for i, (nid, dist) in enumerate(vito_neighbors[:3]):
            if nid:
                d = raw_df[(raw_df['sondering_id']==nid) & (raw_df['source']=='VITO Data')]
                cols[i].plotly_chart(plot_mini_cpt(d, f"{nid} ({dist:.0f}m)", viz), use_container_width=True)

    except Exception as e:
        st.error(f"⚠️ An error occurred in Detailed Analysis: {e}")