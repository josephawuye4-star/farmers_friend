#!/usr/bin/env python3
"""
Farm Robot Dashboard - Raspberry Pi Control Interface
UPDATED: Soil moisture sensor support added
"""

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import serial
import json
import time
import threading
import logging
from datetime import datetime

# ============================================================================
# FLASK APP CONFIGURATION
# ============================================================================

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# SERIAL COMMUNICATION
# ============================================================================

class ESP32Controller:
    def __init__(self, port='/dev/ttyUSB0', baudrate=115200, timeout=2):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.serial_connection = None
        self.connected = False
        self.last_status = {}
        self.last_water_levels = {}
        self.last_soil_moisture = {}
        
        # Threading lock
        self.serial_lock = threading.Lock()
        
    def connect(self):
        """Establish serial connection to ESP32"""
        try:
            logger.info(f"Attempting to connect to {self.port} at {self.baudrate} baud...")
            
            self.serial_connection = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=self.timeout,
                # Explicitly disable flow control
                rtscts=False,
                dsrdtr=False,
                xonxoff=False
            )
            
            logger.info("Serial port opened, waiting for ESP32 to boot...")
            time.sleep(2)
            
            # DIAGNOSTIC: Read any boot messages from ESP32
            logger.info("\n" + "="*70)
            logger.info("READING ESP32 BOOT MESSAGES (next 3 seconds)...")
            logger.info("="*70)
            
            boot_end_time = time.time() + 3
            boot_messages = []
            
            while time.time() < boot_end_time:
                if self.serial_connection.in_waiting > 0:
                    try:
                        line = self.serial_connection.readline()
                        boot_messages.append(line)
                        logger.info(f"BOOT MSG: {repr(line)}")
                    except Exception as e:
                        logger.warning(f"Error reading boot message: {e}")
                time.sleep(0.05)
            
            logger.info(f"Boot phase complete. Received {len(boot_messages)} messages\n")
            
            # Now clear buffers after boot phase is complete
            logger.info("Clearing serial buffers...")
            self.serial_connection.reset_input_buffer()
            self.serial_connection.reset_output_buffer()
            time.sleep(0.1)
            
            self.connected = True
            logger.info(f"✓ Successfully connected to ESP32 on {self.port}")
            logger.info(f"  Baudrate: {self.baudrate}")
            logger.info(f"  Timeout: {self.timeout}s")
            logger.info(f"  Flow Control: DISABLED (rtscts=False, dsrdtr=False, xonxoff=False)")
            return True
            
        except serial.SerialException as e:
            logger.error(f"✗ Failed to connect to ESP32: {e}")
            logger.info(f"\nTroubleshooting:")
            logger.info(f"  1. Check ESP32 is plugged in: lsusb")
            logger.info(f"  2. Check port exists: ls -la /dev/ttyUSB*")
            logger.info(f"  3. Check permissions: sudo usermod -a -G dialout $USER")
            logger.info(f"  4. Try 'minicom -D /dev/ttyUSB0 -b 115200' to test manually")
            self.connected = False
            return False
    
    def disconnect(self):
        """Close serial connection"""
        if self.serial_connection and self.serial_connection.is_open:
            self.serial_connection.close()
            self.connected = False
            logger.info("Disconnected from ESP32")
    
    def send_command(self, command_dict):
        """Send JSON command to ESP32"""
        if not self.connected:
            logger.error("Not connected to ESP32")
            return {"status": "error", "error": "Not connected to ESP32"}
        
        try:
            with self.serial_lock:
                json_str = json.dumps(command_dict)
                
                # Send command
                logger.debug(f"Sending: {json_str}")
                self.serial_connection.write((json_str + '\n').encode('utf-8'))
                self.serial_connection.flush()
                
                logger.info(f"↑ SENT: {json_str}")
                
                # Wait for ESP32 to process
                time.sleep(0.5)
                
                # Read response
                response = self.read_response(timeout=4)
                return response
        
        except Exception as e:
            logger.error(f"Error sending command: {e}")
            return {"status": "error", "error": str(e)}
    
    def read_response(self, timeout=4):
        """
        Read JSON response from ESP32 using readline()
        DIAGNOSTIC VERSION: Extensive logging
        """
        if not self.connected:
            return None
        
        try:
            logger.debug(f"Waiting for response (timeout={timeout}s)...")
            start_time = time.time()
            
            while time.time() - start_time < timeout:
                # Check how many bytes are waiting
                bytes_waiting = self.serial_connection.in_waiting
                
                if bytes_waiting > 0:
                    logger.debug(f"  Bytes waiting: {bytes_waiting}")
                    
                    # Use readline() which blocks until newline or timeout
                    raw_bytes = self.serial_connection.readline()
                    logger.info(f"  RAW BYTES: {repr(raw_bytes)}")
                    
                    if not raw_bytes:
                        logger.warning("  readline() returned empty bytes")
                        continue
                    
                    # Decode
                    try:
                        text_response = raw_bytes.decode('utf-8').strip()
                        logger.info(f"  DECODED TEXT: {text_response}")
                        
                        # Parse JSON
                        if text_response and len(text_response) > 5:
                            try:
                                response = json.loads(text_response)
                                logger.info(f"↓ RECEIVED: {text_response}")
                                return response
                            except json.JSONDecodeError as je:
                                logger.error(f"  JSON PARSE ERROR: {je}")
                                logger.error(f"  Problematic text: {repr(text_response)}")
                                continue
                    except UnicodeDecodeError as ue:
                        logger.error(f"  DECODE ERROR: {ue}")
                        logger.error(f"  Raw bytes: {repr(raw_bytes)}")
                        continue
                else:
                    # No bytes waiting, wait a bit
                    time.sleep(0.05)
            
            # Timeout
            elapsed = time.time() - start_time
            logger.warning(f"NO RESPONSE (timeout after {elapsed:.2f}s)")
            return None
        
        except Exception as e:
            logger.error(f"Error reading response: {e}")
            return None

# ============================================================================
# GLOBAL ESP32 CONTROLLER
# ============================================================================

esp32 = ESP32Controller()

# ============================================================================
# INITIALIZATION
# ============================================================================

def initialize():
    """Initialize connection on startup"""
    if esp32.connect():
        logger.info("\n" + "="*70)
        logger.info("INITIALIZATION PHASE: Getting system state...")
        logger.info("="*70 + "\n")
        
        # Get initial status
        logger.info("Requesting system status...")
        status_resp = esp32.send_command({"command": "get_status"})
        if status_resp:
            esp32.last_status = status_resp
            logger.info(f"✓ Status: {status_resp}\n")
        else:
            logger.error("✗ Failed to get status\n")
        
        # Get water levels
        logger.info("Requesting water levels...")
        water_resp = esp32.send_command({"command": "get_water_levels"})
        if water_resp:
            esp32.last_water_levels = water_resp
            logger.info(f"✓ Water levels: {water_resp}\n")
        else:
            logger.error("✗ Failed to get water levels\n")
        
        # Get soil moisture
        logger.info("Requesting soil moisture...")
        soil_resp = esp32.send_command({"command": "get_soil_moisture"})
        if soil_resp:
            esp32.last_soil_moisture = soil_resp
            logger.info(f"✓ Soil moisture: {soil_resp}\n")
        else:
            logger.error("✗ Failed to get soil moisture\n")
    else:
        logger.warning("Starting in offline mode - ESP32 not detected")

# ============================================================================
# FLASK ROUTES - WEB INTERFACE
# ============================================================================

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('dashboard.html')

# ============================================================================
# FLASK ROUTES - STATUS & DIAGNOSTICS
# ============================================================================

@app.route('/api/status')
def api_status():
    """Get current system status"""
    if not esp32.connected:
        return jsonify({"status": "error", "error": "Not connected to ESP32"}), 503
    
    response = esp32.send_command({"command": "get_status"})
    if response:
        esp32.last_status = response
        return jsonify(response)
    else:
        return jsonify({"status": "error", "error": "No response from ESP32"}), 504

@app.route('/api/water_levels')
def api_water_levels():
    """Get water level status"""
    if not esp32.connected:
        return jsonify({"status": "error", "error": "Not connected to ESP32"}), 503
    
    response = esp32.send_command({"command": "get_water_levels"})
    if response:
        esp32.last_water_levels = response
        return jsonify(response)
    else:
        return jsonify({"status": "error", "error": "No response from ESP32"}), 504

@app.route('/api/soil_moisture')
def api_soil_moisture():
    """Get soil moisture status"""
    if not esp32.connected:
        return jsonify({"status": "error", "error": "Not connected to ESP32"}), 503
    
    response = esp32.send_command({"command": "get_soil_moisture"})
    if response:
        esp32.last_soil_moisture = response
        return jsonify(response)
    else:
        return jsonify({"status": "error", "error": "No response from ESP32"}), 504

@app.route('/api/connection_status')
def api_connection_status():
    """Check ESP32 connection status"""
    return jsonify({
        "connected": esp32.connected,
        "port": esp32.port,
        "baudrate": esp32.baudrate,
        "timeout": esp32.timeout,
        "last_status": esp32.last_status,
        "last_water_levels": esp32.last_water_levels,
        "last_soil_moisture": esp32.last_soil_moisture
    })

# ============================================================================
# FLASK ROUTES - POWER CONTROL
# ============================================================================

@app.route('/api/power_on', methods=['POST'])
def api_power_on():
    """Turn system power ON"""
    if not esp32.connected:
        return jsonify({"status": "error", "error": "Not connected to ESP32"}), 503
    response = esp32.send_command({"command": "power_on"})
    return jsonify(response) if response else jsonify({"status": "error"}), 504

@app.route('/api/power_off', methods=['POST'])
def api_power_off():
    """Turn system power OFF"""
    if not esp32.connected:
        return jsonify({"status": "error", "error": "Not connected to ESP32"}), 503
    response = esp32.send_command({"command": "power_off"})
    return jsonify(response) if response else jsonify({"status": "error"}), 504

# ============================================================================
# FLASK ROUTES - MOTOR CONTROL
# ============================================================================

@app.route('/api/motor_control', methods=['POST'])
def api_motor_control():
    """Control motor direction and duration"""
    data = request.get_json()
    
    if not all(k in data for k in ['motor', 'direction']):
        return jsonify({"status": "error", "error": "Missing motor or direction"}), 400
    
    if not esp32.connected:
        return jsonify({"status": "error", "error": "Not connected to ESP32"}), 503
    
    motor = data.get('motor')
    direction = data.get('direction')
    duration = data.get('duration', 0)
    
    valid_motors = ['slasher', 'irrigation_a', 'irrigation_b']
    valid_directions = ['forward', 'reverse']
    
    if motor not in valid_motors:
        return jsonify({"status": "error", "error": f"Invalid motor: {motor}"}), 400
    if direction not in valid_directions:
        return jsonify({"status": "error", "error": f"Invalid direction: {direction}"}), 400
    
    command = {
        "command": "motor_control",
        "motor": motor,
        "direction": direction,
        "duration": int(duration)
    }
    
    response = esp32.send_command(command)
    return jsonify(response) if response else jsonify({"status": "error"}), 504

@app.route('/api/motor_stop', methods=['POST'])
def api_motor_stop():
    """Stop specific motor"""
    data = request.get_json()
    if 'motor' not in data:
        return jsonify({"status": "error", "error": "Missing motor parameter"}), 400
    if not esp32.connected:
        return jsonify({"status": "error", "error": "Not connected to ESP32"}), 503
    
    response = esp32.send_command({
        "command": "motor_stop",
        "motor": data.get('motor')
    })
    return jsonify(response) if response else jsonify({"status": "error"}), 504

@app.route('/api/stop_all', methods=['POST'])
def api_stop_all():
    """Stop all motors"""
    if not esp32.connected:
        return jsonify({"status": "error", "error": "Not connected to ESP32"}), 503
    response = esp32.send_command({"command": "stop_all"})
    return jsonify(response) if response else jsonify({"status": "error"}), 504

# ============================================================================
# FLASK ROUTES - SIMPLE RELAY CONTROL
# ============================================================================

@app.route('/api/slasher_on', methods=['POST'])
def api_slasher_on():
    if not esp32.connected:
        return jsonify({"status": "error", "error": "Not connected to ESP32"}), 503
    return jsonify(esp32.send_command({"command": "slasher_on"})) if esp32.send_command({"command": "slasher_on"}) else jsonify({"status": "error"}), 504

@app.route('/api/slasher_off', methods=['POST'])
def api_slasher_off():
    if not esp32.connected:
        return jsonify({"status": "error", "error": "Not connected to ESP32"}), 503
    return jsonify(esp32.send_command({"command": "slasher_off"})) if esp32.send_command({"command": "slasher_off"}) else jsonify({"status": "error"}), 504

@app.route('/api/irrigation_on', methods=['POST'])
def api_irrigation_on():
    if not esp32.connected:
        return jsonify({"status": "error", "error": "Not connected to ESP32"}), 503
    return jsonify(esp32.send_command({"command": "irrigation_on"})) if esp32.send_command({"command": "irrigation_on"}) else jsonify({"status": "error"}), 504

@app.route('/api/irrigation_off', methods=['POST'])
def api_irrigation_off():
    if not esp32.connected:
        return jsonify({"status": "error", "error": "Not connected to ESP32"}), 503
    return jsonify(esp32.send_command({"command": "irrigation_off"})) if esp32.send_command({"command": "irrigation_off"}) else jsonify({"status": "error"}), 504

@app.route('/api/sprayer_on', methods=['POST'])
def api_sprayer_on():
    if not esp32.connected:
        return jsonify({"status": "error", "error": "Not connected to ESP32"}), 503
    return jsonify(esp32.send_command({"command": "sprayer_on"})) if esp32.send_command({"command": "sprayer_on"}) else jsonify({"status": "error"}), 504

@app.route('/api/sprayer_off', methods=['POST'])
def api_sprayer_off():
    if not esp32.connected:
        return jsonify({"status": "error", "error": "Not connected to ESP32"}), 503
    return jsonify(esp32.send_command({"command": "sprayer_off"})) if esp32.send_command({"command": "sprayer_off"}) else jsonify({"status": "error"}), 504

@app.route('/api/transfer_pump_on', methods=['POST'])
def api_transfer_pump_on():
    if not esp32.connected:
        return jsonify({"status": "error", "error": "Not connected to ESP32"}), 503
    return jsonify(esp32.send_command({"command": "transfer_pump_on"})) if esp32.send_command({"command": "transfer_pump_on"}) else jsonify({"status": "error"}), 504

@app.route('/api/transfer_pump_off', methods=['POST'])
def api_transfer_pump_off():
    if not esp32.connected:
        return jsonify({"status": "error", "error": "Not connected to ESP32"}), 503
    return jsonify(esp32.send_command({"command": "transfer_pump_off"})) if esp32.send_command({"command": "transfer_pump_off"}) else jsonify({"status": "error"}), 504

# ============================================================================
# FLASK ROUTES - MIXER CONTROL
# ============================================================================

@app.route('/api/mixer_on', methods=['POST'])
def api_mixer_on():
    if not esp32.connected:
        return jsonify({"status": "error", "error": "Not connected to ESP32"}), 503
    return jsonify(esp32.send_command({"command": "mixer_on"})) if esp32.send_command({"command": "mixer_on"}) else jsonify({"status": "error"}), 504

@app.route('/api/mixer_off', methods=['POST'])
def api_mixer_off():
    if not esp32.connected:
        return jsonify({"status": "error", "error": "Not connected to ESP32"}), 503
    return jsonify(esp32.send_command({"command": "mixer_off"})) if esp32.send_command({"command": "mixer_off"}) else jsonify({"status": "error"}), 504

# ============================================================================
# FLASK ROUTES - PUMP CONTROL
# ============================================================================

@app.route('/api/pump_on', methods=['POST'])
def api_pump_on():
    if not esp32.connected:
        return jsonify({"status": "error", "error": "Not connected to ESP32"}), 503
    return jsonify(esp32.send_command({"command": "pump_on"})) if esp32.send_command({"command": "pump_on"}) else jsonify({"status": "error"}), 504

@app.route('/api/pump_off', methods=['POST'])
def api_pump_off():
    if not esp32.connected:
        return jsonify({"status": "error", "error": "Not connected to ESP32"}), 503
    return jsonify(esp32.send_command({"command": "pump_off"})) if esp32.send_command({"command": "pump_off"}) else jsonify({"status": "error"}), 504

# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    try:
        logger.info("\n" + "="*70)
        logger.info("Farm Robot Dashboard - Raspberry Pi")
        logger.info("Updated: Soil moisture sensor support added")
        logger.info("="*70 + "\n")
        
        initialize()
        
        logger.info("\n" + "="*70)
        logger.info("Starting Flask server...")
        logger.info("Dashboard: http://localhost:5000")
        logger.info("Status: http://localhost:5000/api/connection_status")
        logger.info("="*70)
        logger.info("Press Ctrl+C to stop\n")
        
        app.run(
            host='0.0.0.0',
            port=5000,
            debug=False,
            use_reloader=False,
            threaded=True
        )
    
    except KeyboardInterrupt:
        logger.info("\n\nShutting down...")
        esp32.disconnect()
        logger.info("Goodbye!")
    
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        esp32.disconnect()
