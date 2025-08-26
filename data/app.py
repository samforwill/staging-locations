import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import json
import plotly.express as px
import numpy as np

st.set_page_config(page_title="Turf Map", layout="wide")
st.title("Turf Mapping")

# -----------------------------
# Data loaders
# -----------------------------
@st.cache_data
def load_metrics():
    df = pd.read_csv("output/precincts_metrics.csv")
    df["van_precinct_id"] = df["van_precinct_id"].astype(str)
    return df

@st.cache_data
def load_geojson():
    with open("output/precincts_simplified.geojson", "r") as f:
        data = json.load(f)
    for feature in data["features"]:
        feature["properties"]["van_precinct_id"] = str(feature["properties"]["van_precinct_id"])
    return data

@st.cache_data
def load_ooc_points():
    return pd.read_csv("output/ooc_sls_points.csv")

@st.cache_data  
def load_regular_points():
    return pd.read_csv("output/regular_sls_points.csv")

# -----------------------------
# Load data
# -----------------------------
df_metrics = load_metrics()
geojson_data = load_geojson()
ooc_points = load_ooc_points()
regular_points = load_regular_points()

# -----------------------------
# Sidebar: filters only
# -----------------------------
with st.sidebar:
    st.header("Filters")

    # Region filter
    regions = sorted(df_metrics["Current Region"].dropna().unique())
    regions_with_all = ["VA All Regions"] + regions

    selected_regions_raw = st.multiselect("Select Region(s):", regions_with_all, default=["VA All Regions"])
    if "VA All Regions" in selected_regions_raw:
        selected_regions = regions
    else:
        selected_regions = [r for r in selected_regions_raw if r != "VA All Regions"]

    # Turf filter (based on selected regions)
    if selected_regions:
        region_filter = df_metrics["Current Region"].isin(selected_regions)
        available_turfs = sorted(
            df_metrics[region_filter]["Current Turf"].dropna().unique()
        )
        selected_turfs = st.multiselect("Select Turf(s):", available_turfs, default=[])
    else:
        selected_turfs = []

# -----------------------------
# Filter for view
# -----------------------------
if selected_regions:
    filtered_metrics = df_metrics[
        df_metrics["Current Region"].isin(selected_regions)
    ].copy()
    if selected_turfs:
        filtered_metrics = filtered_metrics[filtered_metrics["Current Turf"].isin(selected_turfs)]
else:
    filtered_metrics = pd.DataFrame(columns=df_metrics.columns).copy()

# Filter points based on same criteria
def filter_points_by_selection(points_df, selected_regions, selected_turfs):
    if len(points_df) == 0:
        return points_df
    
    filtered_points = points_df.copy()
    
    # Filter by region if available
    if 'Region' in filtered_points.columns and selected_regions:
        filtered_points = filtered_points[filtered_points['Region'].isin(selected_regions)]
    
    # Filter by turf if available and selected
    if 'fo_name' in filtered_points.columns and selected_turfs:
        filtered_points = filtered_points[filtered_points['fo_name'].isin(selected_turfs)]
    
    return filtered_points

filtered_ooc = filter_points_by_selection(ooc_points, selected_regions, selected_turfs)
filtered_regular = filter_points_by_selection(regular_points, selected_regions, selected_turfs)

# -----------------------------
# KPIs
# -----------------------------
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Total Precincts in Filter", f"{len(filtered_metrics):,}")
with c2:
    st.metric(
        "Total Voters in Filter",
        f"{int(filtered_metrics['voters'].sum()):,}" if "voters" in filtered_metrics.columns else "0",
    )
with c3:
    st.metric(
        "Total Targets in Filter",
        f"{int(filtered_metrics['supporters'].sum()):,}" if "supporters" in filtered_metrics.columns else "0",
    )
with c4:
    st.metric(
        "SLS Points in Filter", 
        f"OOC: {len(filtered_ooc)}, Regular: {len(filtered_regular)}"
    )

# -----------------------------
# Create choropleth color mapping
# -----------------------------
def create_choropleth_colors(df, value_col='supporters', bins=6):
    """Create color mapping for choropleth based on target values"""
    if len(df) == 0 or value_col not in df.columns:
        return {}, []
    
    values = df[value_col].fillna(0)
    if values.max() == 0:
        return {}, []
    
    # Create quantile-based bins for more even distribution
    quantiles = np.linspace(0, 1, bins + 1)
    bin_edges = values.quantile(quantiles).unique()
    
    # If we have fewer unique values than bins, adjust
    if len(bin_edges) < bins + 1:
        bin_edges = np.linspace(values.min(), values.max(), len(bin_edges))
    
    # Blue palette from light to dark with low opacity for lowest bin
    blues = [
        "#deebf7",  # Very light blue
        "#c6dbef",  # Light blue  
        "#9ecae1",  # Medium light blue
        "#6baed6",  # Medium blue
        "#3182bd",  # Dark blue
        "#08519c"   # Very dark blue
    ]
    
    # Opacity values - lowest bin gets 20%, others get 80%
    opacities = [0.2] + [0.8] * (bins - 1)
    
    # Create color mapping
    color_map = {}
    bin_labels = []
    
    for idx, row in df.iterrows():
        val = row[value_col] if pd.notna(row[value_col]) else 0
        
        # Find which bin this value belongs to
        bin_idx = 0
        for i in range(len(bin_edges) - 1):
            if val >= bin_edges[i]:
                bin_idx = i
        
        # Make sure we don't exceed our color array
        bin_idx = min(bin_idx, len(blues) - 1)
        
        color_map[row['van_precinct_id']] = {
            'color': blues[bin_idx],
            'opacity': opacities[bin_idx],
            'value': val,
            'bin': bin_idx
        }
    
    # Create bin labels for legend
    for i in range(len(bin_edges) - 1):
        if i == 0:
            label = f"{int(bin_edges[i])}-{int(bin_edges[i+1])}"
        else:
            label = f"{int(bin_edges[i])+1}-{int(bin_edges[i+1])}"
        bin_labels.append(label)
    
    return color_map, bin_labels

# Create choropleth colors
choropleth_colors, bin_labels = create_choropleth_colors(filtered_metrics, 'supporters', 6)

# -----------------------------
# Map (Folium with Choropleth)
# -----------------------------
m = folium.Map(location=[37.5407, -77.4360], zoom_start=7, prefer_canvas=True)

filtered_features = []
if len(filtered_metrics) > 0:
    selected_ids = set(filtered_metrics["van_precinct_id"])
    filtered_features = [f for f in geojson_data["features"] if f["properties"]["van_precinct_id"] in selected_ids]

    if filtered_features and {"min_lat", "min_lon", "max_lat", "max_lon"}.issubset(filtered_metrics.columns):
        bounds = [
            [filtered_metrics["min_lat"].min(), filtered_metrics["min_lon"].min()],
            [filtered_metrics["max_lat"].max(), filtered_metrics["max_lon"].max()],
        ]
        m.fit_bounds(bounds)

    if filtered_features:
        filtered_geojson = {"type": "FeatureCollection", "features": filtered_features}
        
        def style_function(feature):
            precinct_id = feature["properties"]["van_precinct_id"]
            color_info = choropleth_colors.get(precinct_id, {'color': '#3388ff', 'opacity': 0.8})
            
            return {
                "fillColor": color_info['color'],
                "color": "#000",
                "weight": 0.5,
                "fillOpacity": color_info['opacity'],
            }
        
        folium.GeoJson(
            filtered_geojson,
            style_function=style_function,
            tooltip=folium.GeoJsonTooltip(
                fields=[
                    "van_precinct_name",
                    "van_precinct_id",
                    "county_name",
                    "Current Region",
                    "Current Turf",
                    *(["voters"] if "voters" in filtered_metrics.columns else []),
                    *(["supporters"] if "supporters" in filtered_metrics.columns else []),
                ],
                aliases=[
                    "Precinct Name:",
                    "Precinct ID:",
                    "County:",
                    "Region:",
                    "Turf:",
                    *(["Voters:"] if "voters" in filtered_metrics.columns else []),
                    *(["Targets:"] if "supporters" in filtered_metrics.columns else []),
                ],
                localize=True,
            ) if len(filtered_features) < 3000 else None,
        ).add_to(m)

# Add OOC points (orange map pins)
for _, point in filtered_ooc.iterrows():
    folium.Marker(
        location=[point['latitude'], point['longitude']],
        popup=f"<b>OOC SL</b><br>{point.get('location', 'Unknown')}<br>Region: {point.get('Region', 'N/A')}<br>Precinct: {point.get('van_precinct_name', 'N/A')}",
        icon=folium.Icon(color='orange')
    ).add_to(m)

# Add Regular SLS points (purple map pins)  
for _, point in filtered_regular.iterrows():
    folium.Marker(
        location=[point['latitude'], point['longitude']],
        popup=f"<b>Staging Lo</b><br>{point.get('location', 'Unknown')}<br>Region: {point.get('Region', 'N/A')}<br>Precinct: {point.get('van_precinct_name', 'N/A')}",
        icon=folium.Icon(color='purple')
    ).add_to(m)

# Add legend if we have data
if bin_labels and len(filtered_metrics) > 0:
    legend_html = '''
    <div style="position: fixed; 
                top: 10px; right: 10px; width: 220px; height: auto; 
                background-color: white; border:2px solid grey; z-index:9999; 
                font-size:14px; padding: 10px">
    <p><b>Targets by Precinct</b></p>
    '''
    
    blues = ["#deebf7", "#c6dbef", "#9ecae1", "#6baed6", "#3182bd", "#08519c"]
    opacities = [0.2] + [0.8] * 5
    
    for i, (label, color, opacity) in enumerate(zip(bin_labels, blues[:len(bin_labels)], opacities[:len(bin_labels)])):
        legend_html += f'''
        <p><i class="fa fa-square" style="color:{color}; opacity:{opacity}"></i> {label}</p>
        '''
    
    # Add point legend
    legend_html += '''
    <hr style="margin: 8px 0;">
    <p><b>SLS Points</b></p>
    <p><i class="fa fa-map-marker" style="color:purple; font-size:14px;"></i> Staging Locations</p>
    <p><i class="fa fa-map-marker" style="color:orange; font-size:14px;"></i> OOC Locations</p>
    '''
    
    legend_html += '</div>'
    m.get_root().html.add_child(folium.Element(legend_html))

st_folium(m, key="map", width=None, height=600, returned_objects=[])

# =========================================================
# Section 1: Precincts in selection (table + checkbox + DL)
# =========================================================
if len(filtered_metrics) > 0:
    st.subheader("Precincts in selection")
    table_cols = [
        "Current Turf",
        "van_precinct_name",
        "van_precinct_id",
        *(["voters"] if "voters" in filtered_metrics.columns else []),
        *(["supporters"] if "supporters" in filtered_metrics.columns else []),
    ]
    
    # Rename supporters to targets in display
    precinct_table = (
        filtered_metrics[table_cols]
        .sort_values(["Current Turf", "van_precinct_name"], ascending=[True, True])
        .reset_index(drop=True)
    )
    
    # Rename column for display
    if "supporters" in precinct_table.columns:
        precinct_table = precinct_table.rename(columns={"supporters": "targets"})
    
    st.dataframe(precinct_table, use_container_width=True, hide_index=True)

    # Small download button for this table
    st.download_button(
        "Download table as CSV",
        data=precinct_table.to_csv(index=False),
        file_name="precincts_in_selection.csv",
        mime="text/csv",
        use_container_width=False,
    )

    # Checkbox + chart (voter distribution by precinct)
    show_precinct_hist = st.checkbox("Show voter distribution by precinct", value=False, key="show_precincts_chart")
    if show_precinct_hist and "voters" in filtered_metrics.columns:
        hist_df = filtered_metrics.sort_values(["Current Turf", "van_precinct_name"]).copy()
        hist_df["display_name"] = hist_df["van_precinct_name"]
        
        # Use blues for the chart to match choropleth
        fig_precincts = px.bar(
            hist_df,
            x="display_name",
            y="voters",
            color="supporters",  # Color by targets
            color_continuous_scale="Blues",
            labels={"display_name": "Precinct", "voters": "Voters", "supporters": "Targets"},
            title="Voter Distribution by Precinct (Colored by Targets)",
            height=420,
        )
        fig_precincts.update_layout(
            xaxis_tickangle=-45, xaxis_title="Precinct", yaxis_title="Number of Voters"
        )
        st.plotly_chart(fig_precincts, use_container_width=True)

# =====================================================
# Section 2: Breakdown by Turf (table + checkbox + DL)
# =====================================================
if len(filtered_metrics) > 0:
    st.subheader("Breakdown by Turf")

    turf_summary = (
        filtered_metrics.groupby("Current Turf")
        .agg(
            voters=("voters", "sum") if "voters" in filtered_metrics.columns else ("van_precinct_id", "size"),
            targets=("supporters", "sum") if "supporters" in filtered_metrics.columns else ("van_precinct_id", "size"),
            precinct_count=("van_precinct_id", "count"),
        )
        .sort_index()
    )
    st.dataframe(turf_summary, use_container_width=True)

    # Small download button for this breakdown table
    st.download_button(
        "Download breakdown as CSV",
        data=turf_summary.reset_index().to_csv(index=False),
        file_name="breakdown_by_turf.csv",
        mime="text/csv",
        use_container_width=False,
    )

    # Checkbox + chart (compare turfs)
    compare_turfs = st.checkbox("Compare turfs in filter", value=False, key="compare_turfs_chart")
    if compare_turfs and "voters" in filtered_metrics.columns:
        agg_turf = (
            filtered_metrics.groupby("Current Turf", as_index=False)
            .agg({
                "voters": "sum",
                "supporters": "sum"
            })
            .sort_values("Current Turf", ascending=True)
        )
        
        fig_turf = px.bar(
            agg_turf,
            x="Current Turf",
            y="voters",
            color="supporters",  # Color by targets
            color_continuous_scale="Blues",
            title="Voters by Turf (Colored by Targets)",
            height=380,
            labels={"Current Turf": "Turf", "voters": "Voters", "supporters": "Targets"},
        )
        fig_turf.update_layout(xaxis_title="Turf", yaxis_title="Voters")
        st.plotly_chart(fig_turf, use_container_width=True)