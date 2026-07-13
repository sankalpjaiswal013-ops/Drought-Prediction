import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import skill_metrics as sm

print("Generating remaining Research Figures (1-5, 11)...")
os.makedirs("outputs/figures", exist_ok=True)

# Load data
df = pd.read_csv("data/processed_features.csv", parse_dates=['time'])
metrics_df = pd.read_csv("outputs/results/metrics.csv")
metrics_df_2 = pd.read_csv("outputs/results/metrics_2018_2023.csv")

# ---------------------------------------------------------
# Figure 1: Study Area Map (3-Panel Circular Layout - Zero Dependency)
# ---------------------------------------------------------
try:
    import struct
    import json
    import urllib.request
    from matplotlib.patches import Circle, Polygon as MPolygon
    from matplotlib.collections import PatchCollection

    # Helper function to parse shapefile polygons in pure Python
    def parse_shp_polygons(file_path):
        polygons = []
        with open(file_path, 'rb') as f:
            header = f.read(100)
            file_code = struct.unpack('>i', header[0:4])[0]
            if file_code != 9994:
                return polygons
            while True:
                rec_header = f.read(8)
                if len(rec_header) < 8:
                    break
                rec_num, content_word_len = struct.unpack('>ii', rec_header)
                content_byte_len = content_word_len * 2
                content = f.read(content_byte_len)
                if len(content) < content_byte_len:
                    break
                shape_type = struct.unpack('<i', content[0:4])[0]
                if shape_type in [5, 15]: # Polygon or PolygonZ
                    num_parts = struct.unpack('<i', content[36:40])[0]
                    num_points = struct.unpack('<i', content[40:44])[0]
                    parts_format = f'<{num_parts}i'
                    parts_size = num_parts * 4
                    parts = list(struct.unpack(parts_format, content[44:44 + parts_size]))
                    points_offset = 44 + parts_size
                    points = []
                    for i in range(num_points):
                        offset = points_offset + i * 16
                        if offset + 16 > len(content):
                            break
                        x, y = struct.unpack('<dd', content[offset:offset+16])
                        points.append((x, y))
                    parts.append(len(points))
                    for i in range(num_parts):
                        part_points = points[parts[i]:parts[i+1]]
                        if len(part_points) > 0:
                            polygons.append(part_points)
        return polygons

    # Helper to download and load geojson
    def load_geojson(url):
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=15) as response:
            return json.loads(response.read().decode('utf-8'))

    # Helper to plot GeoJSON geometry
    def plot_geojson_geometry(ax, geometry, facecolor, edgecolor, linewidth=0.5, zorder=2, clip_path=None):
        if not geometry:
            return
        geom_type = geometry.get('type')
        coords = geometry.get('coordinates', [])
        patches = []
        if geom_type == 'Polygon':
            for ring in coords:
                patches.append(MPolygon(np.array(ring), closed=True))
        elif geom_type == 'MultiPolygon':
            for poly in coords:
                for ring in poly:
                    patches.append(MPolygon(np.array(ring), closed=True))
        if patches:
            p_col = PatchCollection(patches, facecolor=facecolor, edgecolor=edgecolor, linewidth=linewidth, zorder=zorder)
            if clip_path:
                p_col.set_clip_path(clip_path)
            ax.add_collection(p_col)

    print("Generating Fig 1: Study Area Map (Zero-Dependency)...")
    
    # 1. Load boundaries
    world_data = load_geojson("https://raw.githubusercontent.com/johan/world.geo.json/master/countries.geo.json")
    india_geom = None
    if world_data:
        for feature in world_data.get('features', []):
            if feature.get('properties', {}).get('name') == 'India':
                india_geom = feature.get('geometry')
                break

    states_data = load_geojson("https://raw.githubusercontent.com/geohacker/india/master/state/india_state.geojson")
    up_geom = None
    if states_data:
        for feature in states_data.get('features', []):
            props = feature.get('properties', {})
            for col in ['NAME_1', 'ST_NM', 'NAME', 'state_name', 'State']:
                if col in props:
                    if str(props[col]).strip().lower() == 'uttar pradesh':
                        up_geom = feature.get('geometry')
                        break
            if up_geom:
                break

    # 3. Parse shapefile
    east_up_polys = parse_shp_polygons("raw_data/east up/eastern_up.shp")

    # Generate 3-panel figure
    fig = plt.figure(figsize=(14, 8), facecolor='white')
    ax1 = fig.add_axes([0.05, 0.52, 0.38, 0.38]) # India inset
    ax2 = fig.add_axes([0.05, 0.08, 0.38, 0.38]) # UP inset
    ax3 = fig.add_axes([0.48, 0.10, 0.48, 0.80]) # Detailed Eastern UP map
    
    # --- PANEL 1: INDIA INSET ---
    cx1, cy1 = 80.0, 22.0
    r1 = 15.0
    ax1.set_xlim([cx1 - r1, cx1 + r1])
    ax1.set_ylim([cy1 - r1, cy1 + r1])
    ax1.set_aspect('equal')
    ax1.set_axis_off()
    
    circle1 = Circle((cx1, cy1), r1, facecolor='#0f172a', edgecolor='#d97706', linewidth=2, zorder=1)
    ax1.add_patch(circle1)
    
    if india_geom:
        plot_geojson_geometry(ax1, india_geom, facecolor='#334155', edgecolor='#64748b', linewidth=0.5, zorder=2, clip_path=circle1)
    if up_geom:
        plot_geojson_geometry(ax1, up_geom, facecolor='#f97316', edgecolor='#fdba74', linewidth=0.5, zorder=3, clip_path=circle1)
        
    ax1.text(cx1, cy1 - 0.7 * r1, "INDIA", color='white', fontsize=11, fontweight='bold', ha='center', va='center', zorder=10)
    ax1.text(cx1 - 0.6 * r1, cy1 - 0.3 * r1, "Arabian\nSea", color='#94a3b8', fontsize=7, style='italic', ha='center', va='center', zorder=10)
    ax1.text(cx1 + 0.6 * r1, cy1 - 0.3 * r1, "Bay of\nBengal", color='#94a3b8', fontsize=7, style='italic', ha='center', va='center', zorder=10)
    
    # --- PANEL 2: UTTAR PRADESH INSET ---
    cx2, cy2 = 81.0, 27.5
    r2 = 4.5
    ax2.set_xlim([cx2 - r2, cx2 + r2])
    ax2.set_ylim([cy2 - r2, cy2 + r2])
    ax2.set_aspect('equal')
    ax2.set_axis_off()
    
    circle2 = Circle((cx2, cy2), r2, facecolor='#0f172a', edgecolor='#d97706', linewidth=2, zorder=1)
    ax2.add_patch(circle2)
    
    if up_geom:
        plot_geojson_geometry(ax2, up_geom, facecolor='#334155', edgecolor='#64748b', linewidth=0.7, zorder=2, clip_path=circle2)
    if east_up_polys:
        patches = [MPolygon(np.array(poly), closed=True) for poly in east_up_polys]
        p_col = PatchCollection(patches, facecolor='#10b981', edgecolor='#a7f3d0', linewidth=0.5, zorder=3)
        p_col.set_clip_path(circle2)
        ax2.add_collection(p_col)
        
    ax2.text(cx2, cy2 - 0.7 * r2, "UTTAR PRADESH", color='white', fontsize=11, fontweight='bold', ha='center', va='center', zorder=10)
    
    # --- PANEL 3: DETAILED STUDY AREA ---
    cx3, cy3 = 82.75, 26.0
    r3 = 2.5
    ax3.set_xlim([cx3 - r3, cx3 + r3])
    ax3.set_ylim([cy3 - r3, cy3 + r3])
    ax3.set_aspect('equal')
    ax3.set_axis_off()
    
    circle3 = Circle((cx3, cy3), r3, facecolor='#f8fafc', edgecolor='#0f172a', linewidth=2.5, zorder=1)
    ax3.add_patch(circle3)
    
    if east_up_polys:
        patches = [MPolygon(np.array(poly), closed=True) for poly in east_up_polys]
        p_col = PatchCollection(patches, facecolor='#e2e8f0', edgecolor='#64748b', linewidth=0.8, zorder=2)
        p_col.set_clip_path(circle3)
        ax3.add_collection(p_col)
        
    for lon in np.arange(80, 86, 1):
        line = ax3.axvline(lon, color='#cbd5e1', linestyle=':', linewidth=0.8, zorder=3)
        line.set_clip_path(circle3)
    for lat in np.arange(23, 29, 1):
        line = ax3.axhline(lat, color='#cbd5e1', linestyle=':', linewidth=0.8, zorder=3)
        line.set_clip_path(circle3)
        
    for lon in [81.0, 82.0, 83.0, 84.0]:
        if abs(lon - cx3) < r3:
            y_top = cy3 + np.sqrt(r3**2 - (lon - cx3)**2)
            ax3.text(lon, y_top + 0.08, f"{lon}°E", fontsize=8, color='#475569', ha='center', va='bottom', fontweight='semibold')
            y_bottom = cy3 - np.sqrt(r3**2 - (lon - cx3)**2)
            ax3.text(lon, y_bottom - 0.08, f"{lon}°E", fontsize=8, color='#475569', ha='center', va='top', fontweight='semibold')
            
    for lat in [24.0, 25.0, 26.0, 27.0, 28.0]:
        if abs(lat - cy3) < r3:
            x_right = cx3 + np.sqrt(r3**2 - (lat - cy3)**2)
            ax3.text(x_right + 0.08, lat, f"{lat}°N", fontsize=8, color='#475569', ha='left', va='center', fontweight='semibold')
            x_left = cx3 - np.sqrt(r3**2 - (lat - cy3)**2)
            ax3.text(x_left - 0.08, lat, f"{lat}°N", fontsize=8, color='#475569', ha='right', va='center', fontweight='semibold')
            
    major_districts = {
        'Varanasi': (82.97, 25.32),
        'Gorakhpur': (83.37, 26.76),
        'Prayagraj': (81.85, 25.45),
        'Mirzapur': (82.56, 25.15),
        'Jaunpur': (82.68, 25.75),
        'Ghazipur': (83.58, 25.58),
        'Azamgarh': (83.19, 26.07),
        'Basti': (82.72, 26.79)
    }
    for name, (lon, lat) in major_districts.items():
        dist_from_center = np.sqrt((lon - cx3)**2 + (lat - cy3)**2)
        if dist_from_center < (r3 - 0.2):
            ax3.text(lon, lat, name, fontsize=8, color='#1e293b', 
                     fontweight='bold', ha='center', va='center', zorder=5,
                     bbox=dict(boxstyle='round,pad=0.15', facecolor='#f8fafc', edgecolor='none', alpha=0.7))
                        
    ax3.annotate('N', xy=(cx3 - 1.9, cy3 + 1.8), xytext=(cx3 - 1.9, cy3 + 1.4),
                 arrowprops=dict(facecolor='#0f172a', width=2.5, headwidth=7, shrink=0.05),
                 ha='center', va='bottom', fontsize=9, fontweight='bold', color='#0f172a')
                 
    leg_x = cx3 - 2.0
    leg_y = cy3 - 2.1
    rect = plt.Rectangle((leg_x, leg_y), 1.9, 0.75, facecolor='white', edgecolor='#cbd5e1', alpha=0.9, zorder=6)
    ax3.add_patch(rect)
    
    item1 = plt.Rectangle((leg_x + 0.1, leg_y + 0.45), 0.22, 0.15, facecolor='#e2e8f0', edgecolor='#64748b', zorder=7)
    ax3.add_patch(item1)
    ax3.text(leg_x + 0.4, leg_y + 0.50, 'Eastern UP Districts', fontsize=7.5, color='#334155', fontweight='semibold', zorder=7)
    
    item2 = plt.Rectangle((leg_x + 0.1, leg_y + 0.15), 0.22, 0.15, facecolor='#10b981', edgecolor='#a7f3d0', zorder=7)
    ax3.add_patch(item2)
    ax3.text(leg_x + 0.4, leg_y + 0.20, 'Eastern UP Bounds', fontsize=7.5, color='#334155', fontweight='semibold', zorder=7)
    
    plt.suptitle("Figure 1: Study Area Map (Eastern Uttar Pradesh)", fontsize=14, fontweight='bold', y=0.95)
    plt.savefig("outputs/figures/Fig1_Study_Area.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated Fig 1: Study Area Map (3-Panel Circular Layout - Zero Dependency)")
except Exception as e:
    import traceback
    print(f"Skipping Fig 1 due to issue: {e}")
    traceback.print_exc()

# ---------------------------------------------------------
# Figure 3: Box Plots of Variables
# ---------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(12, 5))
df['Rainfall'].plot.box(ax=axes[0])
axes[0].set_ylabel("mm / week")

df['Max_Temp'].plot.box(ax=axes[1])
axes[1].set_ylabel("°C")

df['Soil_Moisture'].plot.box(ax=axes[2])
axes[2].set_ylabel("kg/m²")

plt.suptitle("Figure 3: Distribution of Climate Variables (JJAS 1981-2023)", fontsize=14)
plt.tight_layout()
plt.savefig("outputs/figures/Fig3_Boxplots.png", dpi=300)
plt.close()
print("Generated Fig 3: Box Plots")

# ---------------------------------------------------------
# Figure 11: Taylor Diagrams
# ---------------------------------------------------------
def generate_taylor_diagram(metrics_data, title, filename):
    try:
        ref_std = df['SPI'].std()
        
        sdev = [ref_std]
        crmse = [0]
        ccoef = [1.0]
        labels = ['Reference']
        
        for index, row in metrics_data.iterrows():
            r2 = row['R2']
            if r2 < 0: 
                r2 = 0 
                
            std_est = np.sqrt(row['RMSE']**2 / (1.0001 - r2)) 
            
            sdev.append(std_est)
            crmse.append(row['RMSE'])
            ccoef.append(row['Correlation'])
            labels.append(row['Model'])

        fig = plt.figure(figsize=(10, 8)) # Increased width slightly for legend
        
        # Do not pass markerLabel to avoid overlapping text on the plot (we have a legend instead)
        sm.taylor_diagram(np.array(sdev), np.array(crmse), np.array(ccoef))
        
        # Overlay custom markers to distinguish models by shape and color
        shapes = ['*', 'o', 'v', '^', 's', 'D', 'P', 'X']
        colors = ['k', 'blue', 'green', 'magenta', 'cyan', 'orange', 'purple', 'brown']
        
        for i in range(len(sdev)):
            # Taylor diagram coordinate transformation: r = sdev, theta = arccos(ccoef)
            x = sdev[i] * ccoef[i]
            y = sdev[i] * np.sqrt(max(0, 1 - ccoef[i]**2))
            
            # Use smaller size, slight transparency, and dark edges to make overlapping shapes distinct
            plt.scatter(x, y, marker=shapes[i % len(shapes)], color=colors[i % len(colors)], 
                        s=80, alpha=0.8, edgecolors='black', linewidths=1.0, zorder=10, label=labels[i])

        # Add a custom legend to make it compact
        plt.legend(loc='upper right', title="Models", scatterpoints=1)
        
        plt.title(title, y=1.05)
        plt.savefig(f"outputs/figures/{filename}", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Generated {title} (with distinguished markers)")
    except Exception as e:
        print(f"Skipping {title} due to skill_metrics issue: {e}")

generate_taylor_diagram(metrics_df, "Figure 11a: Taylor Diagram (2013-2017)", "Fig11a_Taylor_Diagram_2013_2017.png")
generate_taylor_diagram(metrics_df_2, "Figure 11b: Taylor Diagram (2018-2023)", "Fig11b_Taylor_Diagram_2018_2023.png")

print("\nFinished generating auxiliary figures!")
print("Check outputs/figures/ for the new images.")
