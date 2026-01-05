"""
EarthPulse AI - Real-Time Dashboard with Hardware Integration
Live elephant detection using actual ESP32-S3 + Geophone hardware
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dash import Dash, html, dcc, Input, Output, State
import plotly.graph_objs as go
import numpy as np
from datetime import datetime
import time

# Import hardware bridge
from hardware.hardware_dashboard_bridge import get_hardware_bridge

# Import detection system
from edge_firmware_simulated.detection_system import ElephantDetectionSystem


class RealTimeHardwareDashboard:
    """Real-time dashboard with actual hardware sensor"""
    
    def __init__(self, com_port: str = None):
        """Initialize dashboard with hardware"""
        print("="*70)
        print("🐘 EarthPulse AI - Real-Time Hardware Dashboard")
        print("="*70)
        
        # Initialize hardware bridge
        self.hardware = get_hardware_bridge(com_port)
        
        # Initialize detection system
        print("\nInitializing AI detection system...")
        self.system = ElephantDetectionSystem(model_path="./models/lstm_model.h5")
        
        # Detection history
        self.detection_history = []
        
        # Create Dash app
        self.app = Dash(__name__)
        self.setup_layout()
        self.setup_callbacks()
        
        print("\n" + "="*70)
        print("✓ Dashboard initialized with hardware!")
        print("="*70)
    
    def setup_layout(self):
        """Create dashboard layout"""
        self.app.layout = html.Div([
            html.H1("🐘 EarthPulse AI - Live Hardware Detection", 
                   style={'textAlign': 'center', 'color': '#2c3e50'}),
            
            html.Div([
                html.Div([
                    html.H3("📊 Real-Time Signal", style={'color': '#34495e'}),
                    dcc.Graph(id='live-signal', style={'height': '400px'})
                ], style={'width': '70%', 'display': 'inline-block', 'padding': '10px'}),
                
                html.Div([
                    html.H3("🎯 Detection Status", style={'color': '#34495e'}),
                    html.Div(id='detection-status', style={
                        'padding': '20px',
                        'backgroundColor': '#ecf0f1',
                        'borderRadius': '10px',
                        'fontSize': '16px'
                    })
                ], style={'width': '30%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '10px'})
            ]),
            
            html.Div([
                html.Div([
                    html.H3("📈 Frequency Spectrum", style={'color': '#34495e'}),
                    dcc.Graph(id='spectrum', style={'height': '300px'})
                ], style={'width': '50%', 'display': 'inline-block', 'padding': '10px'}),
                
                html.Div([
                    html.H3("� Detection Log", style={'color': '#34495e'}),
                    html.Div(id='detection-log', style={
                        'height': '300px',
                        'overflowY': 'scroll',
                        'padding': '10px',
                        'backgroundColor': '#f8f9fa',
                        'borderRadius': '5px',
                        'fontSize': '14px'
                    })
                ], style={'width': '50%', 'display': 'inline-block', 'padding': '10px'})
            ]),
            
            html.Div([
                html.H4("� System Statistics", style={'color': '#34495e'}),
                html.Div(id='system-stats', style={
                    'padding': '15px',
                    'backgroundColor': '#e8f5e9',
                    'borderRadius': '5px',
                    'fontSize': '14px'
                })
            ], style={'padding': '10px'}),
            
            # Update interval
            dcc.Interval(
                id='interval-component',
                interval=500,  # Update every 500ms
                n_intervals=0
            )
        ], style={'padding': '20px', 'fontFamily': 'Arial'})
    
    def setup_callbacks(self):
        """Setup dashboard callbacks"""
        
        @self.app.callback(
            [Output('live-signal', 'figure'),
             Output('spectrum', 'figure'),
             Output('detection-status', 'children'),
             Output('detection-log', 'children'),
             Output('system-stats', 'children')],
            [Input('interval-component', 'n_intervals')]
        )
        def update_dashboard(n):
            """Update all dashboard components"""
            
            # Get signal from hardware
            signal = self.hardware.get_signal(1000)
            
            # Time axis
            time_axis = np.arange(len(signal)) / 1000  # Convert to seconds
            
            # Create signal plot
            signal_fig = go.Figure()
            signal_fig.add_trace(go.Scatter(
                x=time_axis,
                y=signal,
                mode='lines',
                line=dict(color='#3498db', width=1),
                name='Voltage'
            ))
            signal_fig.update_layout(
                xaxis_title='Time (s)',
                yaxis_title='Voltage (V)',
                margin=dict(l=40, r=20, t=20, b=40),
                plot_bgcolor='white',
                xaxis=dict(showgrid=True, gridcolor='#ecf0f1'),
                yaxis=dict(showgrid=True, gridcolor='#ecf0f1')
            )
            
            # Compute FFT for spectrum
            fft = np.fft.fft(signal)
            freqs = np.fft.fftfreq(len(signal), 1/1000)
            magnitude = np.abs(fft)[:len(fft)//2]
            freqs = freqs[:len(freqs)//2]
            
            # Create spectrum plot
            spectrum_fig = go.Figure()
            spectrum_fig.add_trace(go.Scatter(
                x=freqs,
                y=magnitude,
                mode='lines',
                line=dict(color='#e74c3c', width=1),
                fill='tozeroy',
                name='Magnitude'
            ))
            spectrum_fig.update_layout(
                xaxis_title='Frequency (Hz)',
                yaxis_title='Magnitude',
                xaxis=dict(range=[0, 100]),  # Focus on 0-100 Hz
                margin=dict(l=40, r=20, t=20, b=40),
                plot_bgcolor='white',
                xaxis_showgrid=True,
                yaxis_showgrid=True
            )
            
            # Run detection
            result = self.system.process_signal(signal)
            
            # Create detection status display
            if result.get('detected', False) and result.get('class_name') == 'elephant_footfall':
                if result.get('confirmation') == 'multi_frame_confirmed':
                    # Confirmed detection!
                    self.detection_history.append({
                        'time': datetime.now(),
                        'result': result
                    })
                    
                    status = html.Div([
                        html.H2("🐘 ELEPHANT DETECTED!", style={'color': '#27ae60', 'marginBottom': '10px'}),
                        html.P(f"Confidence: {result['confidence']:.1%}", style={'fontSize': '18px'}),
                        html.Hr(),
                        html.P("📍 Direction:", style={'fontWeight': 'bold', 'marginTop': '10px'}),
                        html.P(f"  {result.get('direction', {}).get('cardinal', 'Unknown')}"),
                        html.P(f"  {'Approaching' if result.get('direction', {}).get('approaching') else 'Moving away'}"),
                        html.P(f"  Distance: {result.get('direction', {}).get('distance', 0):.1f} m"),
                        html.P(f"  Velocity: {result.get('direction', {}).get('velocity', 0):.2f} m/s"),
                        html.Hr(),
                        html.P("🐘 Behavior:", style={'fontWeight': 'bold', 'marginTop': '10px'}),
                        html.P(f"  {result.get('behavior', {}).get('type', 'Unknown')}"),
                        html.P(f"  {result.get('behavior', {}).get('activity_level', 'Unknown')} activity"),
                        html.P(f"  Weight: {result.get('behavior', {}).get('estimated_weight', 0):.0f} kg")
                    ])
                else:
                    status = html.Div([
                        html.H3("⏳ Awaiting Confirmation", style={'color': '#f39c12'}),
                        html.P(f"Confidence: {result['confidence']:.1%}"),
                        html.P(f"Frames: {result.get('frames_confirmed', 0)}/2")
                    ])
            else:
                # Calculate signal strength
                rms = np.sqrt(np.mean(signal**2))
                peak_to_peak = signal.max() - signal.min()
                
                status = html.Div([
                    html.H3("👂 Listening...", style={'color': '#95a5a6'}),
                    html.P(f"Signal RMS: {rms:.6f} V"),
                    html.P(f"Peak-to-Peak: {peak_to_peak:.6f} V"),
                    html.P(f"Detected: {result.get('class_name', 'background_noise')}"),
                    html.P(f"Confidence: {result.get('confidence', 0):.1%}")
                ])
            
            # Create detection log
            log_entries = []
            for entry in reversed(self.detection_history[-10:]):  # Last 10
                time_str = entry['time'].strftime('%H:%M:%S')
                conf = entry['result']['confidence']
                log_entries.append(
                    html.Div([
                        html.Span(f"🐘 {time_str}", style={'fontWeight': 'bold', 'color': '#27ae60'}),
                        html.Span(f" - Confidence: {conf:.1%}"),
                        html.Br()
                    ])
                )
            
            if not log_entries:
                log_entries = [html.P("No detections yet...", style={'color': '#95a5a6'})]
            
            # System statistics
            hw_stats = self.hardware.get_stats()
            det_stats = self.system.stats
            
            stats_display = html.Div([
                html.Div([
                    html.Span("📡 Hardware: ", style={'fontWeight': 'bold'}),
                    html.Span(f"{hw_stats['samples_received']:,} samples | "),
                    html.Span(f"Buffer: {hw_stats['buffer_size']}/1000 | "),
                    html.Span(f"{'🟢 Streaming' if hw_stats['is_streaming'] else '🔴 Stopped'}")
                ], style={'marginBottom': '5px'}),
                html.Div([
                    html.Span("🤖 AI Detection: ", style={'fontWeight': 'bold'}),
                    html.Span(f"{det_stats['total_predictions']} predictions | "),
                    html.Span(f"{det_stats['confirmed_elephants']} elephants | "),
                    html.Span(f"{det_stats['false_positives_suppressed']} false positives suppressed")
                ])
            ])
            
            return signal_fig, spectrum_fig, status, log_entries, stats_display
    
    def run(self, debug=False):
        """Run the dashboard"""
        print("\n" + "="*70)
        print("Starting Real-Time Hardware Dashboard...")
        print("="*70)
        print("\nDashboard URL: http://localhost:8050")
        print("\n⚡ Hardware streaming active - Tap table to test!")
        print("🐘 AI detection running - Waiting for elephant footfalls...")
        print("\nPress Ctrl+C to stop\n")
        
        try:
            self.app.run(debug=debug, host='0.0.0.0', port=8050)
        except KeyboardInterrupt:
            print("\n\nStopping dashboard...")
            self.hardware.stop_streaming()
            print("✓ Dashboard stopped")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='EarthPulse AI Real-Time Hardware Dashboard')
    parser.add_argument('--port', type=str, default=None,
                       help='Serial port (e.g., COM5)')
    parser.add_argument('--debug', action='store_true',
                       help='Run in debug mode')
    
    args = parser.parse_args()
    
    try:
        dashboard = RealTimeHardwareDashboard(com_port=args.port)
        dashboard.run(debug=args.debug)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
