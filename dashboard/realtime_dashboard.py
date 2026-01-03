"""
EarthPulse AI - Real-Time Dashboard
Interactive visualization of detection system
Supports both simulated data and real hardware (ESP32-S3 + Geophone)
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
from collections import deque
import time
from scipy.fft import fft, fftfreq
import argparse
import threading

# Import our components
from synthetic_generator.seismic_signal_generator import SeismicSignalGenerator
from synthetic_generator.jungle_environment_generator import JungleEnvironmentGenerator
from edge_firmware_simulated.detection_system import ElephantDetectionSystem
from hardware_interface.esp32_serial_reader import ESP32SerialReader


class RealTimeDashboard:
    """Interactive real-time dashboard"""
    
    def __init__(self, buffer_size: int = 1000, hardware_mode: bool = False, port: str = None):
        """
        Initialize dashboard
        
        Args:
            buffer_size: Size of data buffers
            hardware_mode: If True, read from ESP32 hardware instead of simulation
            port: Serial port for ESP32 (e.g., 'COM5' or '/dev/ttyUSB0')
        """
        self.buffer_size = buffer_size
        self.hardware_mode = hardware_mode
        self.port = port
        
        # Data buffers
        self.time_buffer = deque(maxlen=buffer_size)
        self.signal_buffer = deque(maxlen=buffer_size)
        self.processed_buffer = deque(maxlen=buffer_size)
        
        # Detection history
        self.detection_history = []
        
        # Initialize detection system
        self.system = ElephantDetectionSystem(model_path="./models/lstm_model.h5")
        
        # Hardware or simulation components
        if self.hardware_mode:
            print(f"🔧 Initializing hardware mode on port {self.port}")
            self.hardware_reader = ESP32SerialReader()
            if not self.hardware_reader.connect(self.port):
                raise RuntimeError(f"Failed to connect to ESP32 on {self.port}")
            self.latest_hardware_signal = None
            self.hardware_reader.start_reading(callback=self._hardware_callback)
            print("✓ Hardware connected and reading")
        else:
            print("🎮 Initializing simulation mode")
            self.generator = SeismicSignalGenerator(sampling_rate=1000)
            self.jungle_generator = JungleEnvironmentGenerator(fs=1000)
        
        # Current state
        self.soil_moisture = 20.0
        self.current_scenario = "jungle_elephant_day"
        self.time_of_day = "day"
        self.activity_level = "medium"
        self.is_running = False
        
        # Create Dash app
        self.app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
        self._create_layout()
        self._setup_callbacks()
    
    def _hardware_callback(self, signal: np.ndarray):
        """Callback for hardware data"""
        self.latest_hardware_signal = signal
        
    def _create_layout(self):
        """Create dashboard layout"""
        
        # Title based on mode
        if self.hardware_mode:
            title = f"🐘 EarthPulse AI - Real-Time Detection Dashboard (Hardware Mode - {self.port})"
            mode_badge = dbc.Badge("🔧 HARDWARE MODE", color="success", className="ms-2")
        else:
            title = "🐘 EarthPulse AI - Real-Time Detection Dashboard"
            mode_badge = dbc.Badge("🎮 SIMULATION MODE", color="info", className="ms-2")
        
        self.app.layout = dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.H1(title, className="text-center mb-2 mt-4", style={'display': 'inline-block'}),
                        mode_badge
                    ], className="text-center")
                ])
            ]),
            
            # Control Panel
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Control Panel", className="card-title"),
                            
                            # Show scenario dropdown only in simulation mode
                            html.Div([
                                html.Label("Jungle Scenario (Real-World):"),
                                dcc.Dropdown(
                                    id='scenario-dropdown',
                                    options=[
                                        {'label': '🌳🐘 Jungle - Elephant Present (Day)', 'value': 'jungle_elephant_day'},
                                        {'label': '🌙🐘 Jungle - Elephant Present (Night)', 'value': 'jungle_elephant_night'},
                                        {'label': '☀️🐘 Jungle - Distant Elephant', 'value': 'jungle_elephant_far'},
                                        {'label': '🌳 Active Jungle - No Elephant', 'value': 'jungle_no_elephant_active'},
                                        {'label': '🌙 Quiet Night - No Elephant', 'value': 'jungle_no_elephant_quiet'},
                                        {'label': '🐄 Cattle in Jungle (Challenging)', 'value': 'jungle_cattle'},
                                        {'label': '🌧️ Rainy Jungle - No Elephant', 'value': 'jungle_rain'},
                                        {'label': '--- Basic Signals (Testing) ---', 'value': 'separator', 'disabled': True},
                                        {'label': '🐘 Pure Elephant (No Noise)', 'value': 'elephant_footfall'},
                                        {'label': '👤 Human Footsteps', 'value': 'human_footsteps'},
                                        {'label': '🐄 Cattle Movement', 'value': 'cattle_movement'},
                                        {'label': '💨 Wind Vibration', 'value': 'wind_vibration'}
                                    ],
                                    value='jungle_elephant_day',
                                    className="mb-3"
                                ),
                            ], style={'display': 'none' if self.hardware_mode else 'block'}),
                            
                            # Hardware info
                            html.Div([
                                html.Label("Hardware Status:"),
                                html.P(f"✓ Connected to {self.port}", className="text-success"),
                                html.P("Reading from ESP32-S3 + ADS1115 + Geophone", className="text-muted small")
                            ], style={'display': 'block' if self.hardware_mode else 'none'}),
                            
                            html.Label("Soil Moisture (%):"),
                            dcc.Slider(
                                id='moisture-slider',
                                min=5, max=40, step=5,
                                value=20,
                                marks={i: f'{i}%' for i in range(5, 45, 5)},
                                className="mb-3"
                            ),
                            
                            dbc.Button("Start Detection", id="start-button", 
                                      color="success", className="me-2"),
                            dbc.Button("Stop", id="stop-button",
                                      color="danger", className="me-2"),
                            dbc.Button("Reset", id="reset-button",
                                      color="warning")
                        ])
                    ])
                ], width=12, lg=3),
                
                # Status Panel
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Detection Status", className="card-title"),
                            html.Div(id='status-display', children=[
                                html.H2("Standby", className="text-info"),
                                html.P("Waiting to start...")
                            ])
                        ])
                    ])
                ], width=12, lg=3),
                
                # Statistics Panel
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Statistics", className="card-title"),
                            html.Div(id='stats-display')
                        ])
                    ])
                ], width=12, lg=3),
                
                # Direction & Behavior Panel  
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Direction & Behavior", className="card-title"),
                            html.Div(id='behavior-display', children=[
                                html.P("No elephant detected yet")
                            ])
                        ])
                    ])
                ], width=12, lg=3)
            ], className="mb-4"),
            
            # Vibration Plot
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id='vibration-plot', style={'height': '350px'})
                ])
            ], className="mb-4"),
            
            # FFT and STFT Plots
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id='fft-plot', style={'height': '300px'})
                ], width=12, lg=6),
                dbc.Col([
                    dcc.Graph(id='stft-plot', style={'height': '300px'})
                ], width=12, lg=6)
            ], className="mb-4"),
            
            # Prediction History
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Detection History", className="card-title"),
                            html.Div(id='history-display', 
                                    style={'maxHeight': '200px', 'overflowY': 'scroll'})
                        ])
                    ])
                ])
            ]),
            
            # Update interval
            dcc.Interval(id='interval-component', interval=1000, n_intervals=0)
            
        ], fluid=True, className="dbc")
        
    def _setup_callbacks(self):
        """Setup dashboard callbacks"""
        
        @self.app.callback(
            [Output('status-display', 'children'),
             Output('stats-display', 'children'),
             Output('behavior-display', 'children'),
             Output('vibration-plot', 'figure'),
             Output('fft-plot', 'figure'),
             Output('stft-plot', 'figure'),
             Output('history-display', 'children')],
            [Input('interval-component', 'n_intervals'),
             Input('start-button', 'n_clicks'),
             Input('stop-button', 'n_clicks'),
             Input('reset-button', 'n_clicks')],
            [State('scenario-dropdown', 'value'),
             State('moisture-slider', 'value')]
        )
        def update_dashboard(n, start_clicks, stop_clicks, reset_clicks,
                           scenario, moisture):
            # Determine which button was clicked
            ctx = dash.callback_context
            if ctx.triggered:
                button_id = ctx.triggered[0]['prop_id'].split('.')[0]
                
                if button_id == 'start-button':
                    self.is_running = True
                    self.current_scenario = scenario
                    self.soil_moisture = moisture
                elif button_id == 'stop-button':
                    self.is_running = False
                elif button_id == 'reset-button':
                    self.is_running = False
                    self.time_buffer.clear()
                    self.signal_buffer.clear()
                    self.processed_buffer.clear()
                    self.detection_history = []
            
            # Generate and process signal if running
            if self.is_running:
                # Get signal from hardware or simulation
                if self.hardware_mode:
                    # Use latest hardware signal
                    if self.latest_hardware_signal is not None:
                        signal_data = self.latest_hardware_signal
                        self.latest_hardware_signal = None  # Clear it
                    else:
                        # No new data yet, skip this update
                        return dash.no_update
                else:
                    # Generate simulated signal
                    signal_data = self._generate_signal(self.current_scenario)
                
                # Process through detection system
                result = self.system.process_signal(signal_data, self.soil_moisture)
                
                # Update buffers
                t = np.linspace(0, len(signal_data)/1000, len(signal_data))
                self.time_buffer.extend(t)
                self.signal_buffer.extend(signal_data)
                
                # Add ALL predictions to history (for transparency)
                self.detection_history.append({
                    'time': time.time(),
                    'class': result.get('class_name', 'unknown'),
                    'confidence': result.get('confidence', 0.0),
                    'detected': result.get('detected', False),
                    'reason': result.get('reason', 'N/A'),
                    'direction': result.get('direction'),
                    'behavior': result.get('behavior')
                })
            
            # Create plots
            status = self._create_status(self.is_running)
            stats = self._create_stats()
            behavior_info = self._create_behavior(self.is_running)
            vib_fig = self._create_vibration_plot()
            fft_fig = self._create_fft_plot()
            stft_fig = self._create_stft_plot()
            history = self._create_history()
            
            return status, stats, behavior_info, vib_fig, fft_fig, stft_fig, history
    
    def _generate_signal(self, scenario: str) -> np.ndarray:
        """Generate signal based on scenario - using REALISTIC JUNGLE DATA"""
        self.generator.set_soil_conditions(moisture=self.soil_moisture)
        
        # REALISTIC JUNGLE SCENARIOS
        if scenario == 'jungle_elephant_day':
            # Daytime jungle with elephant at medium distance
            signal, _ = self.jungle_generator.generate_realistic_jungle_with_elephant(
                duration=1.0,
                elephant_distance=np.random.uniform(35, 50),
                elephant_weight=np.random.uniform(3500, 4500),
                soil_moisture=self.soil_moisture,
                time_of_day='day'
            )
        elif scenario == 'jungle_elephant_night':
            # Nighttime jungle with elephant (quieter ambient)
            signal, _ = self.jungle_generator.generate_realistic_jungle_with_elephant(
                duration=1.0,
                elephant_distance=np.random.uniform(30, 45),
                elephant_weight=np.random.uniform(3800, 5000),
                soil_moisture=self.soil_moisture,
                time_of_day='night'
            )
        elif scenario == 'jungle_elephant_far':
            # Distant elephant in jungle
            signal, _ = self.jungle_generator.generate_realistic_jungle_with_elephant(
                duration=1.0,
                elephant_distance=np.random.uniform(60, 80),
                elephant_weight=np.random.uniform(3000, 4000),
                soil_moisture=self.soil_moisture,
                time_of_day='day'
            )
        elif scenario == 'jungle_no_elephant_active':
            # Active jungle with various animals but NO elephant
            signal, _ = self.jungle_generator.generate_realistic_jungle_without_elephant(
                duration=1.0,
                soil_moisture=self.soil_moisture,
                time_of_day='day',
                activity_level='high'
            )
        elif scenario == 'jungle_no_elephant_quiet':
            # Quiet night jungle
            signal, _ = self.jungle_generator.generate_realistic_jungle_without_elephant(
                duration=1.0,
                soil_moisture=self.soil_moisture,
                time_of_day='night',
                activity_level='low'
            )
        elif scenario == 'jungle_cattle':
            # Cattle in jungle (challenging false positive test)
            signal, _ = self.jungle_generator.generate_realistic_cattle_near_elephant_path(
                duration=1.0,
                num_cattle=np.random.randint(2, 5),
                soil_moisture=self.soil_moisture
            )
        elif scenario == 'jungle_rain':
            # Rainy jungle without elephant
            signal, _ = self.jungle_generator.generate_realistic_jungle_without_elephant(
                duration=1.0,
                soil_moisture=min(40, self.soil_moisture + 10),
                time_of_day='day',
                activity_level='low'
            )
        
        # BASIC TESTING SIGNALS (Clean, no jungle noise)
        elif scenario == 'elephant_footfall':
            signal, _ = self.generator.generate_elephant_footfall(duration=1.0)
        elif scenario == 'human_footsteps':
            signal, _ = self.generator.generate_human_footsteps(duration=1.0)
        elif scenario == 'cattle_movement':
            signal, _ = self.generator.generate_cattle_movement(duration=1.0)
        elif scenario == 'wind_vibration':
            signal, _ = self.generator.generate_wind_vibration(duration=1.0)
        else:
            # Default to active jungle without elephant
            signal, _ = self.jungle_generator.generate_realistic_jungle_without_elephant(
                duration=1.0,
                soil_moisture=self.soil_moisture,
                time_of_day='day',
                activity_level='medium'
            )
        
        return signal
    
    def _create_status(self, is_running: bool):
        """Create status display"""
        if not is_running:
            return html.Div([
                html.H2("Standby", className="text-warning"),
                html.P("System ready - Select scenario and click Start")
            ])
        
        # Get last prediction
        if self.detection_history:
            last = self.detection_history[-1]
            class_name = last['class'].replace('_', ' ').title()
            confidence = last['confidence']
            detected = last['detected']
            reason = last.get('reason', '')
            
            if last['class'] == 'elephant_footfall' and detected:
                return html.Div([
                    html.H2("🐘 ELEPHANT DETECTED!", className="text-danger"),
                    html.P(f"Confidence: {confidence:.2%}", className="lead"),
                    html.P(f"Soil Moisture: {self.soil_moisture}%")
                ])
            elif last['class'] == 'elephant_footfall' and not detected:
                return html.Div([
                    html.H2("⏳ Elephant Signal Detected", className="text-warning"),
                    html.P(f"Confidence: {confidence:.2%}", className="lead"),
                    html.P(f"Status: {reason.replace('_', ' ').title()}"),
                    html.P(f"Awaiting multi-frame confirmation...")
                ])
            elif detected:
                return html.Div([
                    html.H2("📊 Monitoring Active", className="text-success"),
                    html.P(f"Current: {class_name} ({confidence:.2%})")
                ])
            else:
                return html.Div([
                    html.H2("📊 Monitoring Active", className="text-info"),
                    html.P(f"Predicted: {class_name} ({confidence:.2%})"),
                    html.P(f"Status: {reason.replace('_', ' ')}", className="text-muted")
                ])
        else:
            return html.Div([
                html.H2("Monitoring...", className="text-success"),
                html.P("Waiting for first signal...")
            ])
    
    def _create_stats(self):
        """Create statistics display"""
        stats = self.system.get_statistics()
        
        return html.Div([
            html.P(f"Total Predictions: {stats['total_predictions']}"),
            html.P(f"Elephant Detections: {stats['confirmed_elephants']}"),
            html.P(f"False Positives Filtered: {stats['false_positives_suppressed']}"),
            html.P(f"Confirmation Rate: {stats.get('confirmation_rate', 0):.1%}")
        ])
    
    def _create_behavior(self, is_running: bool):
        """Create direction and behavior display"""
        if not is_running or not self.detection_history:
            return html.Div([
                html.P("No data available", className="text-muted")
            ])
        
        # Get last elephant detection with behavior info
        for detection in reversed(self.detection_history):
            if detection.get('behavior') and detection.get('direction'):
                direction = detection['direction']
                behavior = detection['behavior']
                
                # Direction icon
                direction_icon = "⬆️" if direction['approaching'] else "⬇️"
                
                # Behavior emoji
                behavior_emojis = {
                    'walking': '🚶',
                    'running': '🏃',
                    'feeding': '🌿',
                    'standing': '🧍',
                    'bathing': '💦'
                }
                behavior_emoji = behavior_emojis.get(behavior['type'], '❓')
                
                return html.Div([
                    html.H5([direction_icon, " Direction"], className="text-primary"),
                    html.P([
                        html.Strong("Status: "),
                        f"{'Approaching' if direction['approaching'] else 'Moving Away'}"
                    ], style={'marginBottom': '5px'}),
                    html.P([
                        html.Strong("Distance: "),
                        f"{direction['distance']:.1f}m"
                    ], style={'marginBottom': '5px'}),
                    html.P([
                        html.Strong("Velocity: "),
                        f"{direction['velocity']:.2f} m/s"
                    ], style={'marginBottom': '15px'}),
                    
                    html.H5([behavior_emoji, " Behavior"], className="text-success"),
                    html.P([
                        html.Strong("Activity: "),
                        f"{behavior['type'].title()}"
                    ], style={'marginBottom': '5px'}),
                    html.P([
                        html.Strong("Speed: "),
                        f"{behavior['gait_speed']:.2f} m/s"
                    ], style={'marginBottom': '5px'}),
                    html.P([
                        html.Strong("Level: "),
                        f"{behavior['activity_level'].title()}"
                    ], style={'marginBottom': '5px'}),
                    html.P([
                        html.Strong("Est. Weight: "),
                        f"{behavior['estimated_weight']:.0f} kg"
                    ], style={'marginBottom': '0px'})
                ])
        
        return html.Div([
            html.P("Waiting for elephant detection...", className="text-muted")
        ])
    
    def _create_vibration_plot(self):
        """Create vibration time series plot"""
        fig = go.Figure()
        
        if len(self.signal_buffer) > 0:
            # Keep only recent data for display
            display_size = min(1000, len(self.signal_buffer))
            time_data = list(self.time_buffer)[-display_size:]
            signal_data = list(self.signal_buffer)[-display_size:]
            
            fig.add_trace(go.Scatter(
                x=time_data,
                y=signal_data,
                mode='lines',
                name='Raw Signal',
                line=dict(color='cyan', width=1)
            ))
        
        fig.update_layout(
            title="Seismic Vibration Stream",
            xaxis_title="Time (s)",
            yaxis_title="Amplitude",
            template="plotly_dark",
            showlegend=True,
            margin=dict(l=50, r=50, t=50, b=50)
        )
        
        return fig
    
    def _create_fft_plot(self):
        """Create FFT plot"""
        fig = go.Figure()
        
        if len(self.signal_buffer) > 100:
            signal_data = np.array(list(self.signal_buffer)[-1000:])
            
            # Compute FFT
            N = len(signal_data)
            fft_vals = fft(signal_data)
            fft_freqs = fftfreq(N, 1/1000)  # 1000 Hz sampling rate
            
            # Only positive frequencies
            pos_mask = fft_freqs > 0
            freqs = fft_freqs[pos_mask]
            magnitude = np.abs(fft_vals[pos_mask])
            
            fig.add_trace(go.Scatter(
                x=freqs[:500],  # Up to 500 Hz
                y=magnitude[:500],
                mode='lines',
                name='FFT',
                line=dict(color='yellow', width=2)
            ))
        
        fig.update_layout(
            title="Frequency Spectrum (FFT)",
            xaxis_title="Frequency (Hz)",
            yaxis_title="Magnitude",
            template="plotly_dark",
            margin=dict(l=50, r=50, t=50, b=50)
        )
        
        return fig
    
    def _create_stft_plot(self):
        """Create STFT spectrogram"""
        fig = go.Figure()
        
        if len(self.signal_buffer) > 256:
            from scipy.signal import stft
            
            signal_data = np.array(list(self.signal_buffer)[-2000:])
            
            # Compute STFT
            f, t, Zxx = stft(signal_data, fs=1000, nperseg=256)
            
            fig.add_trace(go.Heatmap(
                z=np.abs(Zxx),
                x=t,
                y=f,
                colorscale='Viridis'
            ))
        
        fig.update_layout(
            title="Spectrogram (STFT)",
            xaxis_title="Time (s)",
            yaxis_title="Frequency (Hz)",
            template="plotly_dark",
            margin=dict(l=50, r=50, t=50, b=50)
        )
        
        return fig
    
    def _create_history(self):
        """Create detection history display"""
        if not self.detection_history:
            return html.P("No detections yet", className="text-muted")
        
        # Show last 10 detections
        recent = self.detection_history[-10:]
        
        items = []
        for det in reversed(recent):
            class_name = det['class'].replace('_', ' ').title()
            confidence = det['confidence']
            
            color = "danger" if det['class'] == 'elephant_footfall' else "info"
            
            items.append(
                dbc.Alert([
                    html.Strong(class_name),
                    f" - Confidence: {confidence:.2%}"
                ], color=color, className="mb-2 p-2")
            )
        
        return html.Div(items)
    
    def run(self, debug: bool = False, port: int = 8050):
        """Run the dashboard"""
        print("=" * 70)
        mode = "Hardware" if self.hardware_mode else "Simulation"
        print(f"EarthPulse AI - Real-Time Dashboard ({mode} Mode)")
        print("=" * 70)
        if self.hardware_mode:
            print(f"✓ Connected to ESP32 on {self.port}")
            print("✓ Reading real geophone data")
        else:
            print("🎮 Using simulated jungle environment data")
        print(f"\nStarting dashboard on http://localhost:{port}")
        print("Press Ctrl+C to stop\n")
        
        try:
            self.app.run(debug=debug, port=port, host='127.0.0.1')
        finally:
            if self.hardware_mode:
                print("\nShutting down hardware connection...")
                self.hardware_reader.stop_reading()
                self.hardware_reader.disconnect()
                print("✓ Hardware disconnected")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='EarthPulse AI Real-Time Dashboard')
    parser.add_argument('--hardware', action='store_true',
                       help='Use hardware mode (ESP32-S3 + Geophone) instead of simulation')
    parser.add_argument('--port', type=str, default='COM5',
                       help='Serial port for ESP32 (e.g., COM5, /dev/ttyUSB0)')
    parser.add_argument('--web-port', type=int, default=8050,
                       help='Web server port (default: 8050)')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode')
    
    args = parser.parse_args()
    
    # Create dashboard
    if args.hardware:
        print("🔧 Initializing hardware mode...")
        print(f"📡 Connecting to ESP32 on {args.port}...")
        dashboard = RealTimeDashboard(hardware_mode=True, port=args.port)
    else:
        print("🎮 Initializing simulation mode...")
        dashboard = RealTimeDashboard(hardware_mode=False)
    
    # Run dashboard
    dashboard.run(debug=args.debug, port=args.web_port)
