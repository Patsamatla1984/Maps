import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
import googlemaps
import os

# --- Streamlit Web App for Location Clustering ---
#test comment to trigger deployment

# Configure Streamlit for Azure deployment
st.set_page_config(
    page_title="Location Clustering with Google Maps API", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add error handling for WebSocket issues
if 'initialized' not in st.session_state:
    st.session_state.initialized = True

st.title("Location Clustering with Google Maps API and DBSCAN")

st.markdown("""
This app geocodes addresses, clusters them by driving time using DBSCAN, and allows you to download the results.\
**Note:** You need a Google Maps API key with Geocoding and Distance Matrix APIs enabled.
""")


# --- Sidebar Inputs ---
st.sidebar.header("Configuration")
# Try to get API key from Azure Web App configuration (environment variable)
api_key = "AIzaSyDOJnxLg4A7joiItXeUFpkK5v_c-wi8xfw"  #os.environ.get("GOOGLE_MAPS_API_KEY")
if not api_key:
    api_key = st.sidebar.text_input("Google Maps API Key", type="password")
else:
    st.sidebar.success("Google Maps API Key loaded from Azure configuration.")
eps_minutes = st.sidebar.number_input("Max Driving Time for Cluster (minutes)", min_value=1, max_value=240, value=45)
min_samples = st.sidebar.number_input("Min Points to Form Cluster", min_value=1, max_value=20, value=2)

uploaded_file = st.file_uploader("Upload your CSV file (Address, City, State / Prov, Country required)", type=["csv"]) 

if uploaded_file and api_key:
    # Log window
    log_placeholder = st.empty()
    logs = []
    def log(msg):
        logs.append(str(msg))
        log_placeholder.code('\n'.join(logs), language='text')
    # Try reading with utf-8, then latin1, then ISO-8859-1, and handle empty file
    import pandas.errors
    import csv
    from io import StringIO, BytesIO
    def try_read_csv_auto(file):
        # Try different encodings
        encodings = ['utf-8', 'latin1', 'ISO-8859-1']
        for enc in encodings:
            file.seek(0)
            try:
                # Read a small sample to sniff delimiter
                sample = file.read(4096)
                if isinstance(sample, bytes):
                    sample = sample.decode(enc, errors='replace')
                sniffer = csv.Sniffer()
                try:
                    dialect = sniffer.sniff(sample)
                    delimiter = dialect.delimiter
                except Exception:
                    delimiter = ','
                file.seek(0)
                # Try with header
                try:
                    df = pd.read_csv(file, encoding=enc, delimiter=delimiter)
                    if len(df.columns) < 2:
                        raise ValueError('Too few columns')
                    return df
                except Exception:
                    file.seek(0)
                    # Try without header
                    df = pd.read_csv(file, encoding=enc, delimiter=delimiter, header=None)
                    if len(df.columns) < 2:
                        continue
                    return df
            except Exception:
                continue
        return None

    df = try_read_csv_auto(uploaded_file)
    if df is None or df.empty:
        st.error("The uploaded file is empty, not a valid CSV, or the delimiter/header could not be detected. Please check your file and try again.")
        st.stop()
    address_cols = ['Address', 'City', 'State / Prov', 'Country']
    if not all(col in df.columns for col in address_cols):
        st.error(f"CSV must contain columns: {', '.join(address_cols)}")
        st.stop()
    df['full_address'] = df[address_cols].fillna('').agg(', '.join, axis=1)
    gmaps = googlemaps.Client(key=api_key)
    latitudes, longitudes = [], []
    log("Starting geocoding process with Google Maps API...")
    with st.spinner("Geocoding addresses..."):
        for idx, address in enumerate(df['full_address']):
            try:
                geocode_result = gmaps.geocode(address)
                if geocode_result:
                    lat = geocode_result[0]['geometry']['location']['lat']
                    lng = geocode_result[0]['geometry']['location']['lng']
                    log(f"{idx+1}/{len(df)} Geocoded: {address} -> ({lat:.4f}, {lng:.4f})")
                else:
                    lat, lng = None, None
                    log(f"{idx+1}/{len(df)} Could not geocode: {address}")
            except Exception as e:
                lat, lng = None, None
                log(f"{idx+1}/{len(df)} Error geocoding '{address}': {e}")
            latitudes.append(lat)
            longitudes.append(lng)
    df['latitude'] = latitudes
    df['longitude'] = longitudes
    df.dropna(subset=['latitude', 'longitude'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    log(f"Geocoding complete. {len(df)} locations geocoded.")
    st.success(f"Geocoded {len(df)} locations.")
    st.dataframe(df.head())
    # --- Duration Matrix ---
    locations = list(zip(df['latitude'], df['longitude']))
    n = len(locations)
    duration_matrix = np.zeros((n, n))
    with st.spinner("Calculating driving duration matrix (may take time)..."):
        if n > 25:
            batch_size = 10  # 10x10 = 100 elements per request
            for i in range(0, n, batch_size):
                origin_batch = locations[i:i+batch_size]
                for j in range(0, n, batch_size):
                    dest_batch = locations[j:j+batch_size]
                    try:
                        response = gmaps.distance_matrix(origin_batch, dest_batch, mode="driving")
                        for origin_idx, row in enumerate(response['rows']):
                            for dest_idx, element in enumerate(row['elements']):
                                global_origin_idx = i + origin_idx
                                global_dest_idx = j + dest_idx
                                if element['status'] == 'OK':
                                    duration_min = element['duration']['value'] / 60.0
                                    duration_matrix[global_origin_idx, global_dest_idx] = duration_min
                                    log(f"Duration: {global_origin_idx+1},{global_dest_idx+1} = {duration_min:.2f} min")
                                else:
                                    duration_matrix[global_origin_idx, global_dest_idx] = np.inf
                                    log(f"Duration: {global_origin_idx+1},{global_dest_idx+1} = inf (not found)")
                    except Exception as e:
                        log(f"Error in batch ({i}:{i+batch_size}, {j}:{j+batch_size}): {e}")
                        duration_matrix[i:i+len(origin_batch), j:j+len(dest_batch)] = np.inf
            log("Batched duration matrix calculation complete.")
        else:
            # Use largest possible batch that does not exceed 100 elements
            batch_size = min(25, 100 // n if n else 1)
            if batch_size == 0:
                batch_size = 1
            for i in range(0, n, batch_size):
                origin_batch = locations[i:i + batch_size]
                try:
                    response = gmaps.distance_matrix(origin_batch, locations, mode="driving")
                    for origin_idx, row in enumerate(response['rows']):
                        for dest_idx, element in enumerate(row['elements']):
                            global_origin_idx = i + origin_idx
                            global_dest_idx = dest_idx
                            if element['status'] == 'OK':
                                duration_min = element['duration']['value'] / 60.0
                                duration_matrix[global_origin_idx, global_dest_idx] = duration_min
                                log(f"Duration: {global_origin_idx+1},{global_dest_idx+1} = {duration_min:.2f} min")
                            else:
                                duration_matrix[global_origin_idx, global_dest_idx] = np.inf
                                log(f"Duration: {global_origin_idx+1},{global_dest_idx+1} = inf (not found)")
                except Exception as e:
                    log(f"Error with Google Maps API request for origins {i}-{i+len(origin_batch)-1}, destinations 0-{n-1}: {e}")
                    duration_matrix[i:i+len(origin_batch), :] = np.inf
            log("Duration matrix calculation complete.")
    st.success("Duration matrix calculation complete.")
    # --- Clustering ---
    safe_matrix = np.where(np.isinf(duration_matrix), 999999, duration_matrix)
    dbscan = DBSCAN(eps=eps_minutes, min_samples=min_samples, metric='precomputed', n_jobs=-1)
    labels = dbscan.fit_predict(safe_matrix)
    df['Cluster'] = labels
    log(f"Clustering complete. Found {len(set(labels)) - (1 if -1 in labels else 0)} clusters and {list(labels).count(-1)} noise points.")
    st.write(f"Clustering complete. Found {len(set(labels)) - (1 if -1 in labels else 0)} clusters and {list(labels).count(-1)} noise points.")
    # --- Cluster Centers ---
    df['Center_Latitude'] = np.nan
    df['Center_Longitude'] = np.nan
    df['Distance_to_Center'] = np.nan
    for cluster_id in df['Cluster'].unique():
        if cluster_id == -1:
            continue
        cluster_indices = df[df['Cluster'] == cluster_id].index
        intra_cluster_distances = duration_matrix[np.ix_(cluster_indices, cluster_indices)]
        avg_dist_to_others = intra_cluster_distances.mean(axis=1)
        center_point_local_idx = np.argmin(avg_dist_to_others)
        center_point_global_idx = cluster_indices[center_point_local_idx]
        center_coords = (
            df.loc[center_point_global_idx, 'latitude'],
            df.loc[center_point_global_idx, 'longitude']
        )
        log(f"Cluster {cluster_id}: center at index {center_point_global_idx} ({center_coords[0]:.4f}, {center_coords[1]:.4f})")
        for idx in cluster_indices:
            from_idx = idx
            to_idx = center_point_global_idx
            driving_time = duration_matrix[from_idx, to_idx]
            df.at[idx, 'Center_Latitude'] = center_coords[0]
            df.at[idx, 'Center_Longitude'] = center_coords[1]
            df.at[idx, 'Distance_to_Center'] = driving_time
            log(f"Cluster {cluster_id}: {idx} -> center {center_point_global_idx}, driving time = {driving_time}")
    df['Distance_to_Center'] = df['Distance_to_Center'].fillna(-1).round().astype(int)
    def hub_spoke_label(row):
        if row['Cluster'] == -1:
            return "Not Applicable"
        elif row['Distance_to_Center'] == 0:
            return f"Hub-{int(row['Cluster'])}"
        else:
            return f"Spoke-{int(row['Cluster'])}"
    df['Hub / Spoke'] = df.apply(hub_spoke_label, axis=1)
    columns_to_keep = [
        'Address', 'City', 'State / Prov', 'Country',
        'latitude', 'longitude', 'Cluster',
        'Center_Latitude', 'Center_Longitude', 'Distance_to_Center',
        'Hub / Spoke'
    ]
    output_df = df[[col for col in columns_to_keep if col in df.columns]]
    st.dataframe(output_df)
    log("Processing complete. Ready for download.")
    csv = output_df.to_csv(index=False, float_format='%.6f').encode('utf-8')
    st.download_button("Download Results as CSV", data=csv, file_name="clustered_locations_gmaps.csv", mime="text/csv")
else:
    st.info("Please provide your Google Maps API key and upload a CSV file to begin.")
