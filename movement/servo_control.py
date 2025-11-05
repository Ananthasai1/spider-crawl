#!/usr/bin/env python3
"""
Servo Controller for CyberCrawl Spider Robot
Controls 12 servos (4 legs × 3 joints) via PCA9685
Based on Arduino quadruped inverse kinematics
"""

import time
import math
from adafruit_pca9685 import PCA9685
import board
import busio
import config

class SpiderServoController:
    def __init__(self):
        """Initialize PCA9685 and servo parameters"""
        print("  🦾 Initializing servo controller...")
        
        # Initialize I2C bus and PCA9685
        i2c = busio.I2C(board.SCL, board.SDA)
        self.pca = PCA9685(i2c, address=config.PCA9685_ADDRESS)
        self.pca.frequency = config.PCA9685_FREQUENCY
        
        # Servo channel mapping
        self.servo_channels = config.SERVO_CHANNELS
        
        # Robot dimensions
        self.length_a = config.LENGTH_A
        self.length_b = config.LENGTH_B
        self.length_c = config.LENGTH_C
        self.length_side = config.LENGTH_SIDE
        
        # Movement parameters
        self.z_default = config.Z_DEFAULT
        self.z_up = config.Z_UP
        self.z_boot = config.Z_BOOT
        self.x_default = config.X_DEFAULT
        self.x_offset = config.X_OFFSET
        self.y_start = config.Y_START
        self.y_step = config.Y_STEP
        
        # Current and expected positions [leg][axis: x, y, z]
        self.site_now = [[0.0, 0.0, 0.0] for _ in range(4)]
        self.site_expect = [[0.0, 0.0, 0.0] for _ in range(4)]
        
        # Movement speeds
        self.leg_move_speed = config.LEG_MOVE_SPEED
        self.body_move_speed = config.BODY_MOVE_SPEED
        self.spot_turn_speed = config.SPOT_TURN_SPEED
        self.stand_seat_speed = config.STAND_SEAT_SPEED
        self.move_speed = self.leg_move_speed
        self.speed_multiple = config.SPEED_MULTIPLE
        
        # Calculate turn constants
        self._calculate_turn_constants()
        
        # Initialize leg positions
        self._initialize_positions()
        
        print("  ✅ Servo controller ready")
        
    def _calculate_turn_constants(self):
        """Calculate turn position constants"""
        temp_a = math.sqrt((2 * self.x_default + self.length_side)**2 + self.y_step**2)
        temp_b = 2 * (self.y_start + self.y_step) + self.length_side
        temp_c = math.sqrt((2 * self.x_default + self.length_side)**2 + 
                          (2 * self.y_start + self.y_step + self.length_side)**2)
        temp_alpha = math.acos((temp_a**2 + temp_b**2 - temp_c**2) / (2 * temp_a * temp_b))
        
        self.turn_x1 = (temp_a - self.length_side) / 2
        self.turn_y1 = self.y_start + self.y_step / 2
        self.turn_x0 = self.turn_x1 - temp_b * math.cos(temp_alpha)
        self.turn_y0 = temp_b * math.sin(temp_alpha) - self.turn_y1 - self.length_side
        
    def _initialize_positions(self):
        """Initialize default leg positions"""
        # Front legs (0, 1)
        for leg in [0, 1]:
            self.site_now[leg] = [
                self.x_default - self.x_offset,
                self.y_start + self.y_step,
                self.z_boot
            ]
            self.site_expect[leg] = self.site_now[leg].copy()
        
        # Rear legs (2, 3)
        for leg in [2, 3]:
            self.site_now[leg] = [
                self.x_default + self.x_offset,
                self.y_start,
                self.z_boot
            ]
            self.site_expect[leg] = self.site_now[leg].copy()
    
    def set_servo_pulse(self, channel, pulse):
        """Set servo pulse width (150-600 for 0-180°)"""
        pulse = max(config.SERVO_PULSE_RANGE[0], min(config.SERVO_PULSE_RANGE[1], pulse))
        self.pca.channels[channel].duty_cycle = int(pulse)
    
    def set_servo_angle(self, leg, joint, angle):
        """Set servo angle (0-180 degrees)"""
        # Constrain angle
        angle = max(0, min(180, angle))
        
        # Convert angle to pulse width
        pulse_range = config.SERVO_PULSE_RANGE[1] - config.SERVO_PULSE_RANGE[0]
        pulse = int((angle / 180.0) * pulse_range + config.SERVO_PULSE_RANGE[0])
        
        # Get channel and set pulse
        channel = self.servo_channels[leg][joint]
        self.set_servo_pulse(channel, pulse)
    
    def cartesian_to_polar(self, x, y, z):
        """Convert cartesian coordinates to servo angles (inverse kinematics)"""
        try:
            # Calculate w-z plane
            w = (1 if x >= 0 else -1) * math.sqrt(x**2 + y**2)
            v = w - self.length_c
            
            # Calculate alpha (coxa-femur angle)
            sqrt_term = math.sqrt(v**2 + z**2)
            acos_term = (self.length_a**2 - self.length_b**2 + v**2 + z**2) / (2 * self.length_a * sqrt_term)
            acos_term = max(-1, min(1, acos_term))  # Constrain to valid range
            
            alpha = math.atan2(z, v) + math.acos(acos_term)
            
            # Calculate beta (femur-tibia angle)
            beta_term = (self.length_a**2 + self.length_b**2 - v**2 - z**2) / (2 * self.length_a * self.length_b)
            beta_term = max(-1, min(1, beta_term))
            beta = math.acos(beta_term)
            
            # Calculate gamma (coxa rotation)
            gamma = math.atan2(y, x) if w >= 0 else math.atan2(-y, -x)
            
            # Convert to degrees
            return (math.degrees(alpha), math.degrees(beta), math.degrees(gamma))
            
        except Exception as e:
            print(f"IK calculation error: {e}")
            return (90, 90, 90)  # Safe default
    
    def polar_to_servo(self, leg, alpha, beta, gamma):
        """Map polar coordinates to actual servo angles"""
        if leg == 0:  # Front-right
            alpha = 90 - alpha
            gamma += 90
        elif leg == 1:  # Front-left
            alpha += 90
            beta = 180 - beta
            gamma = 90 - gamma
        elif leg == 2:  # Rear-left
            alpha += 90
            beta = 180 - beta
            gamma = 90 - gamma
        elif leg == 3:  # Rear-right
            alpha = 90 - alpha
            gamma += 90
        
        # Set servos
        self.set_servo_angle(leg, 0, alpha)
        self.set_servo_angle(leg, 1, beta)
        self.set_servo_angle(leg, 2, gamma)
    
    def set_site(self, leg, x, y, z):
        """Set leg position"""
        # Update expected position
        self.site_expect[leg] = [x, y, z]
        
        # Calculate angles
        alpha, beta, gamma = self.cartesian_to_polar(x, y, z)
        
        # Set servo positions
        self.polar_to_servo(leg, alpha, beta, gamma)
        
        # Update current position
        self.site_now[leg] = [x, y, z]
    
    def wait_all_reach(self, delay=0.3):
        """Wait for all legs to reach their positions"""
        time.sleep(delay)
    
    # ========== Basic Postures ==========
    
    def stand(self):
        """Stand up position"""
        self.move_speed = self.stand_seat_speed
        for leg in range(4):
            if leg in [0, 1]:  # Front legs
                self.set_site(leg, self.x_default - self.x_offset, 
                            self.y_start + self.y_step, self.z_default)
            else:  # Rear legs
                self.set_site(leg, self.x_default + self.x_offset, 
                            self.y_start, self.z_default)
        self.wait_all_reach(0.5)
    
    def sit(self):
        """Sit down position"""
        self.move_speed = self.stand_seat_speed
        for leg in range(4):
            if leg in [0, 1]:
                self.set_site(leg, self.x_default - self.x_offset, 
                            self.y_start + self.y_step, self.z_boot)
            else:
                self.set_site(leg, self.x_default + self.x_offset, 
                            self.y_start, self.z_boot)
        self.wait_all_reach(0.5)
    
    # ========== Movement Functions ==========
    
    def step_forward(self, steps=1):
        """Move forward"""
        self.move_speed = self.leg_move_speed
        for _ in range(steps):
            if self.site_now[2][1] == self.y_start:
                # Leg 2 & 1 move
                self.set_site(2, self.x_default + self.x_offset, self.y_start, self.z_up)
                self.wait_all_reach(0.15)
                self.set_site(2, self.x_default + self.x_offset, 
                             self.y_start + 2 * self.y_step, self.z_up)
                self.wait_all_reach(0.15)
                self.set_site(2, self.x_default + self.x_offset, 
                             self.y_start + 2 * self.y_step, self.z_default)
                self.wait_all_reach(0.15)
                
                self.move_speed = self.body_move_speed
                self.set_site(0, self.x_default + self.x_offset, self.y_start, self.z_default)
                self.set_site(1, self.x_default + self.x_offset, 
                             self.y_start + 2 * self.y_step, self.z_default)
                self.set_site(2, self.x_default - self.x_offset, 
                             self.y_start + self.y_step, self.z_default)
                self.set_site(3, self.x_default - self.x_offset, 
                             self.y_start + self.y_step, self.z_default)
                self.wait_all_reach(0.2)
                
                self.move_speed = self.leg_move_speed
                self.set_site(1, self.x_default + self.x_offset, 
                             self.y_start + 2 * self.y_step, self.z_up)
                self.wait_all_reach(0.15)
                self.set_site(1, self.x_default + self.x_offset, self.y_start, self.z_up)
                self.wait_all_reach(0.15)
                self.set_site(1, self.x_default + self.x_offset, self.y_start, self.z_default)
                self.wait_all_reach(0.15)
            else:
                # Leg 0 & 3 move
                self.set_site(0, self.x_default + self.x_offset, self.y_start, self.z_up)
                self.wait_all_reach(0.15)
                self.set_site(0, self.x_default + self.x_offset, 
                             self.y_start + 2 * self.y_step, self.z_up)
                self.wait_all_reach(0.15)
                self.set_site(0, self.x_default + self.x_offset, 
                             self.y_start + 2 * self.y_step, self.z_default)
                self.wait_all_reach(0.15)
                
                self.move_speed = self.body_move_speed
                self.set_site(0, self.x_default - self.x_offset, 
                             self.y_start + self.y_step, self.z_default)
                self.set_site(1, self.x_default - self.x_offset, 
                             self.y_start + self.y_step, self.z_default)
                self.set_site(2, self.x_default + self.x_offset, self.y_start, self.z_default)
                self.set_site(3, self.x_default + self.x_offset, 
                             self.y_start + 2 * self.y_step, self.z_default)
                self.wait_all_reach(0.2)
                
                self.move_speed = self.leg_move_speed
                self.set_site(3, self.x_default + self.x_offset, 
                             self.y_start + 2 * self.y_step, self.z_up)
                self.wait_all_reach(0.15)
                self.set_site(3, self.x_default + self.x_offset, self.y_start, self.z_up)
                self.wait_all_reach(0.15)
                self.set_site(3, self.x_default + self.x_offset, self.y_start, self.z_default)
                self.wait_all_reach(0.15)
    
    def step_back(self, steps=1):
        """Move backward"""
        self.move_speed = self.leg_move_speed
        for _ in range(steps):
            if self.site_now[3][1] == self.y_start:
                # Leg 3 & 0 move
                self.set_site(3, self.x_default + self.x_offset, self.y_start, self.z_up)
                self.wait_all_reach(0.15)
                self.set_site(3, self.x_default + self.x_offset, 
                             self.y_start + 2 * self.y_step, self.z_up)
                self.wait_all_reach(0.15)
                self.set_site(3, self.x_default + self.x_offset, 
                             self.y_start + 2 * self.y_step, self.z_default)
                self.wait_all_reach(0.15)
                
                self.move_speed = self.body_move_speed
                self.set_site(0, self.x_default + self.x_offset, 
                             self.y_start + 2 * self.y_step, self.z_default)
                self.set_site(1, self.x_default + self.x_offset, self.y_start, self.z_default)
                self.set_site(2, self.x_default - self.x_offset, 
                             self.y_start + self.y_step, self.z_default)
                self.set_site(3, self.x_default - self.x_offset, 
                             self.y_start + self.y_step, self.z_default)
                self.wait_all_reach(0.2)
                
                self.move_speed = self.leg_move_speed
                self.set_site(0, self.x_default + self.x_offset, 
                             self.y_start + 2 * self.y_step, self.z_up)
                self.wait_all_reach(0.15)
                self.set_site(0, self.x_default + self.x_offset, self.y_start, self.z_up)
                self.wait_all_reach(0.15)
                self.set_site(0, self.x_default + self.x_offset, self.y_start, self.z_default)
                self.wait_all_reach(0.15)
            else:
                # Leg 1 & 2 move
                self.set_site(1, self.x_default + self.x_offset, self.y_start, self.z_up)
                self.wait_all_reach(0.15)
                self.set_site(1, self.x_default + self.x_offset, 
                             self.y_start + 2 * self.y_step, self.z_up)
                self.wait_all_reach(0.15)
                self.set_site(1, self.x_default + self.x_offset, 
                             self.y_start + 2 * self.y_step, self.z_default)
                self.wait_all_reach(0.15)
                
                self.move_speed = self.body_move_speed
                self.set_site(0, self.x_default - self.x_offset, 
                             self.y_start + self.y_step, self.z_default)
                self.set_site(1, self.x_default - self.x_offset, 
                             self.y_start + self.y_step, self.z_default)
                self.set_site(2, self.x_default + self.x_offset, 
                             self.y_start + 2 * self.y_step, self.z_default)
                self.set_site(3, self.x_default + self.x_offset, self.y_start, self.z_default)
                self.wait_all_reach(0.2)
                
                self.move_speed = self.leg_move_speed
                self.set_site(2, self.x_default + self.x_offset, 
                             self.y_start + 2 * self.y_step, self.z_up)
                self.wait_all_reach(0.15)
                self.set_site(2, self.x_default + self.x_offset, self.y_start, self.z_up)
                self.wait_all_reach(0.15)
                self.set_site(2, self.x_default + self.x_offset, self.y_start, self.z_default)
                self.wait_all_reach(0.15)
    
    def turn_left(self, steps=1):
        """Turn left"""
        self.move_speed = self.spot_turn_speed
        for _ in range(steps):
            if self.site_now[3][1] == self.y_start:
                # Leg 3 & 1 sequence
                self.set_site(3, self.x_default + self.x_offset, self.y_start, self.z_up)
                self.wait_all_reach(0.2)
                
                self.set_site(0, self.turn_x1 - self.x_offset, self.turn_y1, self.z_default)
                self.set_site(1, self.turn_x0 - self.x_offset, self.turn_y0, self.z_default)
                self.set_site(2, self.turn_x1 + self.x_offset, self.turn_y1, self.z_default)
                self.set_site(3, self.turn_x0 + self.x_offset, self.turn_y0, self.z_up)
                self.wait_all_reach(0.2)
                
                self.set_site(3, self.turn_x0 + self.x_offset, self.turn_y0, self.z_default)
                self.wait_all_reach(0.2)
                
                self.set_site(0, self.turn_x1 + self.x_offset, self.turn_y1, self.z_default)
                self.set_site(1, self.turn_x0 + self.x_offset, self.turn_y0, self.z_default)
                self.set_site(2, self.turn_x1 - self.x_offset, self.turn_y1, self.z_default)
                self.set_site(3, self.turn_x0 - self.x_offset, self.turn_y0, self.z_default)
                self.wait_all_reach(0.2)
                
                self.set_site(1, self.turn_x0 + self.x_offset, self.turn_y0, self.z_up)
                self.wait_all_reach(0.2)
                
                self.set_site(0, self.x_default + self.x_offset, self.y_start, self.z_default)
                self.set_site(1, self.x_default + self.x_offset, self.y_start, self.z_up)
                self.set_site(2, self.x_default - self.x_offset, 
                             self.y_start + self.y_step, self.z_default)
                self.set_site(3, self.x_default - self.x_offset, 
                             self.y_start + self.y_step, self.z_default)
                self.wait_all_reach(0.2)
                
                self.set_site(1, self.x_default + self.x_offset, self.y_start, self.z_default)
                self.wait_all_reach(0.2)
            else:
                # Leg 0 & 2 sequence
                self.set_site(0, self.x_default + self.x_offset, self.y_start, self.z_up)
                self.wait_all_reach(0.2)
                
                self.set_site(0, self.turn_x0 + self.x_offset, self.turn_y0, self.z_up)
                self.set_site(1, self.turn_x1 + self.x_offset, self.turn_y1, self.z_default)
                self.set_site(2, self.turn_x0 - self.x_offset, self.turn_y0, self.z_default)
                self.set_site(3, self.turn_x1 - self.x_offset, self.turn_y1, self.z_default)
                self.wait_all_reach(0.2)
                
                self.set_site(0, self.turn_x0 + self.x_offset, self.turn_y0, self.z_default)
                self.wait_all_reach(0.2)
                
                self.set_site(0, self.turn_x0 - self.x_offset, self.turn_y0, self.z_default)
                self.set_site(1, self.turn_x1 - self.x_offset, self.turn_y1, self.z_default)
                self.set_site(2, self.turn_x0 + self.x_offset, self.turn_y0, self.z_default)
                self.set_site(3, self.turn_x1 + self.x_offset, self.turn_y1, self.z_default)
                self.wait_all_reach(0.2)
                
                self.set_site(2, self.turn_x0 + self.x_offset, self.turn_y0, self.z_up)
                self.wait_all_reach(0.2)
                
                self.set_site(0, self.x_default - self.x_offset, 
                             self.y_start + self.y_step, self.z_default)
                self.set_site(1, self.x_default - self.x_offset, 
                             self.y_start + self.y_step, self.z_default)
                self.set_site(2, self.x_default + self.x_offset, self.y_start, self.z_up)
                self.set_site(3, self.x_default + self.x_offset, self.y_start, self.z_default)
                self.wait_all_reach(0.2)
                
                self.set_site(2, self.x_default + self.x_offset, self.y_start, self.z_default)
                self.wait_all_reach(0.2)
    
    def turn_right(self, steps=1):
        """Turn right"""
        self.move_speed = self.spot_turn_speed
        for _ in range(steps):
            if self.site_now[2][1] == self.y_start:
                # Leg 2 & 0 sequence
                self.set_site(2, self.x_default + self.x_offset, self.y_start, self.z_up)
                self.wait_all_reach(0.2)
                
                self.set_site(0, self.turn_x0 - self.x_offset, self.turn_y0, self.z_default)
                self.set_site(1, self.turn_x1 - self.x_offset, self.turn_y1, self.z_default)
                self.set_site(2, self.turn_x0 + self.x_offset, self.turn_y0, self.z_up)
                self.set_site(3, self.turn_x1 + self.x_offset, self.turn_y1, self.z_default)
                self.wait_all_reach(0.2)
                
                self.set_site(2, self.turn_x0 + self.x_offset, self.turn_y0, self.z_default)
                self.wait_all_reach(0.2)
                
                self.set_site(0, self.turn_x0 + self.x_offset, self.turn_y0, self.z_default)
                self.set_site(1, self.turn_x1 + self.x_offset, self.turn_y1, self.z_default)
                self.set_site(2, self.turn_x0 - self.x_offset, self.turn_y0, self.z_default)
                self.set_site(3, self.turn_x1 - self.x_offset, self.turn_y1, self.z_default)
                self.wait_all_reach(0.2)
                
                self.set_site(0, self.turn_x0 + self.x_offset, self.turn_y0, self.z_up)
                self.wait_all_reach(0.2)
                
                self.set_site(0, self.x_default + self.x_offset, self.y_start, self.z_up)
                self.set_site(1, self.x_default + self.x_offset, self.y_start, self.z_default)
                self.set_site(2, self.x_default - self.x_offset, 
                             self.y_start + self.y_step, self.z_default)
                self.set_site(3, self.x_default - self.x_offset, 
                             self.y_start + self.y_step, self.z_default)
                self.wait_all_reach(0.2)
                
                self.set_site(0, self.x_default + self.x_offset, self.y_start, self.z_default)
                self.wait_all_reach(0.2)
            else:
                # Leg 1 & 3 sequence
                self.set_site(1, self.x_default + self.x_offset, self.y_start, self.z_up)
                self.wait_all_reach(0.2)
                
                self.set_site(0, self.turn_x1 + self.x_offset, self.turn_y1, self.z_default)
                self.set_site(1, self.turn_x0 + self.x_offset, self.turn_y0, self.z_up)
                self.set_site(2, self.turn_x1 - self.x_offset, self.turn_y1, self.z_default)
                self.set_site(3, self.turn_x0 - self.x_offset, self.turn_y0, self.z_default)
                self.wait_all_reach(0.2)
                
                self.set_site(1, self.turn_x0 + self.x_offset, self.turn_y0, self.z_default)
                self.wait_all_reach(0.2)
                
                self.set_site(0, self.turn_x1 - self.x_offset, self.turn_y1, self.z_default)
                self.set_site(1, self.turn_x0 - self.x_offset, self.turn_y0, self.z_default)
                self.set_site(2, self.turn_x1 + self.x_offset, self.turn_y1, self.z_default)
                self.set_site(3, self.turn_x0 + self.x_offset, self.turn_y0, self.z_default)
                self.wait_all_reach(0.2)
                
                self.set_site(3, self.turn_x0 + self.x_offset, self.turn_y0, self.z_up)
                self.wait_all_reach(0.2)
                
                self.set_site(0, self.x_default - self.x_offset, 
                             self.y_start + self.y_step, self.z_default)
                self.set_site(1, self.x_default - self.x_offset, 
                             self.y_start + self.y_step, self.z_default)
                self.set_site(2, self.x_default + self.x_offset, self.y_start, self.z_default)
                self.set_site(3, self.x_default + self.x_offset, self.y_start, self.z_up)
                self.wait_all_reach(0.2)
                
                self.set_site(3, self.x_default + self.x_offset, self.y_start, self.z_default)
                self.wait_all_reach(0.2)
    
    # ========== Gesture Functions ==========
    
    def body_left(self, offset=15):
        """Lean body to the left"""
        self.set_site(0, self.site_now[0][0] + offset, 
                     self.site_now[0][1], self.site_now[0][2])
        self.set_site(1, self.site_now[1][0] + offset, 
                     self.site_now[1][1], self.site_now[1][2])
        self.set_site(2, self.site_now[2][0] - offset, 
                     self.site_now[2][1], self.site_now[2][2])
        self.set_site(3, self.site_now[3][0] - offset, 
                     self.site_now[3][1], self.site_now[3][2])
        self.wait_all_reach(0.3)
    
    def body_right(self, offset=15):
        """Lean body to the right"""
        self.set_site(0, self.site_now[0][0] - offset, 
                     self.site_now[0][1], self.site_now[0][2])
        self.set_site(1, self.site_now[1][0] - offset, 
                     self.site_now[1][1], self.site_now[1][2])
        self.set_site(2, self.site_now[2][0] + offset, 
                     self.site_now[2][1], self.site_now[2][2])
        self.set_site(3, self.site_now[3][0] + offset, 
                     self.site_now[3][1], self.site_now[3][2])
        self.wait_all_reach(0.3)
    
    def hand_wave(self, waves=3):
        """Wave hand gesture"""
        self.move_speed = 1
        if self.site_now[3][1] == self.y_start:
            self.body_right(15)
            x_tmp, y_tmp, z_tmp = self.site_now[2]
            self.move_speed = self.body_move_speed
            for _ in range(waves):
                self.set_site(2, self.turn_x1, self.turn_y1, 50)
                self.wait_all_reach(0.2)
                self.set_site(2, self.turn_x0, self.turn_y0, 50)
                self.wait_all_reach(0.2)
            self.set_site(2, x_tmp, y_tmp, z_tmp)
            self.wait_all_reach(0.3)
            self.move_speed = 1
            self.body_left(15)
        else:
            self.body_left(15)
            x_tmp, y_tmp, z_tmp = self.site_now[0]
            self.move_speed = self.body_move_speed
            for _ in range(waves):
                self.set_site(0, self.turn_x1, self.turn_y1, 50)
                self.wait_all_reach(0.2)
                self.set_site(0, self.turn_x0, self.turn_y0, 50)
                self.wait_all_reach(0.2)
            self.set_site(0, x_tmp, y_tmp, z_tmp)
            self.wait_all_reach(0.3)
            self.move_speed = 1
            self.body_right(15)
    
    def hand_shake(self, shakes=3):
        """Hand shake gesture"""
        self.move_speed = 1
        if self.site_now[3][1] == self.y_start:
            self.body_right(15)
            x_tmp, y_tmp, z_tmp = self.site_now[2]
            self.move_speed = self.body_move_speed
            for _ in range(shakes):
                self.set_site(2, self.x_default - 30, 
                             self.y_start + 2 * self.y_step, 55)
                self.wait_all_reach(0.2)
                self.set_site(2, self.x_default - 30, 
                             self.y_start + 2 * self.y_step, 10)
                self.wait_all_reach(0.2)
            self.set_site(2, x_tmp, y_tmp, z_tmp)
            self.wait_all_reach(0.3)
            self.move_speed = 1
            self.body_left(15)
        else:
            self.body_left(15)
            x_tmp, y_tmp, z_tmp = self.site_now[0]
            self.move_speed = self.body_move_speed
            for _ in range(shakes):
                self.set_site(0, self.x_default - 30, 
                             self.y_start + 2 * self.y_step, 55)
                self.wait_all_reach(0.2)
                self.set_site(0, self.x_default - 30, 
                             self.y_start + 2 * self.y_step, 10)
                self.wait_all_reach(0.2)
            self.set_site(0, x_tmp, y_tmp, z_tmp)
            self.wait_all_reach(0.3)
            self.move_speed = 1
            self.body_right(15)
    
    def dance(self):
        """Fun dance routine"""
        for _ in range(2):
            self.body_left(20)
            self.body_right(20)
        self.body_left(10)
        for _ in range(2):
            self.hand_wave(1)
        self.stand()
    
    def cleanup(self):
        """Cleanup PCA9685"""
        self.pca.deinit()