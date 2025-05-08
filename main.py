import streamlit as st
import os
import tempfile
from datetime import datetime
import json
import sys
import logging
from utils.logger import setup_logger
from utils.config import load_default_config
from utils.visualization import draw_timeline, create_heatmap, plot_metrics

# Add modules directory to path
modules_dir = os.path.join(os.path.dirname(__file__), 'modules')
if modules_dir not in sys.path:
    sys.path.append(modules_dir)

# Setup logger
logger = setup_logger(name="main", log_dir="logs")

# Load default configuration
config = load_default_config()

def save_uploaded_file(uploaded_file):
    """Save uploaded file to a temporary directory"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            return tmp_file.name
    except Exception as e:
        logger.error(f"Error saving uploaded file: {str(e)}")
        raise

def process_video(video_path: str, selected_models: list) -> dict:
    """Process video with selected models"""
    try:
        from modules.run_all import run_all_modules
        return run_all_modules(video_path)
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        raise

def display_results(results: dict):
    """Display results in the Streamlit UI"""
    try:
        # Create results directory
        results_dir = os.path.join('results', datetime.now().strftime('%Y%m%d_%H%M%S'))
        os.makedirs(results_dir, exist_ok=True)
        
        # Save results to JSON
        with open(os.path.join(results_dir, 'results.json'), 'w') as f:
            json.dump(results, f, indent=4)
        
        # Display risk score and level
        st.subheader("Risk Assessment")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Risk Score", f"{results.get('risk_score', 0):.1f}/100")
        with col2:
            risk_level = results.get('risk_level', 'LOW')
            st.metric("Risk Level", risk_level)
        
        # Display detection counts
        st.subheader("Detection Summary")
        detections = []
        if 'screen_switch' in results:
            detections.append(f"Screen Switch Events: {results['screen_switch']['scene_switch_events']}")
        if 'gaze' in results:
            detections.append(f"Gaze Deviations: {results['gaze']['frames_looked_away']}")
        if 'audio' in results:
            detections.append(f"Audio Anomalies: {results['audio']['anomaly_count']}")
        if 'multi_person' in results:
            detections.append(f"Multiple Persons Detected: {results['multi_person']['detection_count']}")
        
        for detection in detections:
            st.write(detection)
        
        # Create and display visualizations
        st.subheader("Visualizations")
        
        # Convert results to events format for timeline
        events = []
        if 'screen_switch' in results:
            events.append({
                'type': 'screen_switch',
                'timestamp': 0,  # Placeholder - you might want to add actual timestamps
                'confidence': results['screen_switch']['suspicion_score'] / 100
            })
        if 'gaze' in results:
            events.append({
                'type': 'gaze',
                'timestamp': 0,
                'confidence': results['gaze']['score'] / 100
            })
        if 'audio' in results:
            events.append({
                'type': 'audio',
                'timestamp': 0,
                'confidence': results['audio'].get('suspicion_score', 0) / 100
            })
        if 'multi_person' in results:
            events.append({
                'type': 'multi_person',
                'timestamp': 0,
                'confidence': results['multi_person'].get('suspicion_score', 0) / 100
            })
        
        # Timeline
        timeline_path = os.path.join(results_dir, 'timeline.png')
        draw_timeline(events, duration=60, save_path=timeline_path)  # Using 60 seconds as default duration
        st.image(timeline_path, caption="Detection Timeline")
        
        # Heatmap
        heatmap_path = os.path.join(results_dir, 'heatmap.png')
        create_heatmap(events, image_shape=(480, 640), save_path=heatmap_path)  # Using standard video dimensions
        st.image(heatmap_path, caption="Detection Heatmap")
        
        # Metrics plot
        metrics_path = os.path.join(results_dir, 'metrics.png')
        metrics = {
            'Risk Score': [results.get('risk_score', 0)],
            'Screen Switch': [results.get('screen_switch', {}).get('suspicion_score', 0)],
            'Gaze': [results.get('gaze', {}).get('score', 0)],
            'Audio': [results.get('audio', {}).get('suspicion_score', 0)],
            'Multi-Person': [results.get('multi_person', {}).get('suspicion_score', 0)]
        }
        plot_metrics(metrics, save_path=metrics_path)
        st.image(metrics_path, caption="Metrics Over Time")
        
    except Exception as e:
        logger.error(f"Error displaying results: {str(e)}")
        st.error(f"Error displaying results: {str(e)}")

def main():
    st.set_page_config(
        page_title="Anti-Cheating Detection System",
        page_icon="🎥",
        layout="wide"
    )
    
    st.title("Anti-Cheating Detection System")
    
    # File uploader
    uploaded_file = st.file_uploader("Upload a video file", type=['mp4', 'avi', 'mov'])
    
    # Model selection
    st.subheader("Select Detection Models")
    col1, col2 = st.columns(2)
    with col1:
        screen_switch = st.checkbox("Screen Switch Detection", value=True)
        gaze_tracking = st.checkbox("Gaze Tracking", value=True)
    with col2:
        audio_analysis = st.checkbox("Audio Analysis", value=True)
        multi_person = st.checkbox("Multiple Person Detection", value=True)
    
    selected_models = []
    if screen_switch:
        selected_models.append('screen_switch')
    if gaze_tracking:
        selected_models.append('gaze_tracking')
    if audio_analysis:
        selected_models.append('audio_analysis')
    if multi_person:
        selected_models.append('multi_person')
    
    if uploaded_file is not None:
        # Display video preview
        st.video(uploaded_file)
        
        # Process button
        if st.button("Analyze Video"):
            if not selected_models:
                st.warning("Please select at least one detection model.")
                return
                
            with st.spinner("Processing video..."):
                try:
                    # Save uploaded file
                    video_path = save_uploaded_file(uploaded_file)
                    
                    # Process video
                    results = process_video(video_path, selected_models)
                    
                    # Display results
                    display_results(results)
                    
                    # Cleanup
                    os.unlink(video_path)
                    
                except Exception as e:
                    logger.error(f"Error in main: {str(e)}")
                    st.error(f"Error processing video: {str(e)}")

if __name__ == "__main__":
    main() 