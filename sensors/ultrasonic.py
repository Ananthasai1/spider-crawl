#!/usr/bin/env python3
"""
Ultrasonic Sensor Module for CyberCrawl Spider Robot
HC-SR04 Distance Sensor
"""

import time
import RPi.GPIO as GPIO

class UltrasonicSensor:
    def __init__(self, trigger_pin, echo_pin, max_distance=200):
        """
        Initialize ultrasonic sensor
        
        Args:
            trigger_pin: GPIO pin for trigger
            echo_pin: GPIO pin for echo
            max_distance: Maximum detection distance in cm
        """
        print("  📡 Initializing ultrasonic sensor...")
        
        self.trigger_pin = trigger_pin
        self.echo_pin = echo_pin
        self.max_distance = max_distance
        
        # Setup GPIO
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        GPIO.setup(self.trigger_pin, GPIO.OUT)
        GPIO.setup(self.echo_pin, GPIO.IN)
        
        # Initialize trigger to LOW
        GPIO.output(self.trigger_pin, GPIO.LOW)
        time.sleep(0.1)
        
        print("  ✅ Ultrasonic sensor ready")
    
    def get_distance(self, timeout=0.5):
        """
        Measure distance in centimeters
        
        Args:
            timeout: Maximum time to wait for echo (seconds)
            
        Returns:
            Distance in cm, or -1 if measurement failed
        """
        try:
            # Send 10us pulse to trigger
            GPIO.output(self.trigger_pin, GPIO.HIGH)
            time.sleep(0.00001)  # 10 microseconds
            GPIO.output(self.trigger_pin, GPIO.LOW)
            
            # Wait for echo to go HIGH
            pulse_start = time.time()
            timeout_start = time.time()
            while GPIO.input(self.echo_pin) == GPIO.LOW:
                pulse_start = time.time()
                if (pulse_start - timeout_start) > timeout:
                    return -1
            
            # Wait for echo to go LOW
            pulse_end = time.time()
            timeout_start = time.time()
            while GPIO.input(self.echo_pin) == GPIO.HIGH:
                pulse_end = time.time()
                if (pulse_end - timeout_start) > timeout:
                    return -1
            
            # Calculate distance
            pulse_duration = pulse_end - pulse_start
            distance = (pulse_duration * 34300) / 2  # Speed of sound: 343 m/s
            
            # Validate distance
            if distance > 0 and distance <= self.max_distance:
                return round(distance, 1)
            else:
                return -1
                
        except Exception as e:
            print(f"Ultrasonic sensor error: {e}")
            return -1
    
    def get_average_distance(self, samples=3):
        """
        Get average distance over multiple samples
        
        Args:
            samples: Number of measurements to average
            
        Returns:
            Average distance in cm
        """
        distances = []
        for _ in range(samples):
            dist = self.get_distance()
            if dist > 0:
                distances.append(dist)
            time.sleep(0.02)  # Small delay between measurements
        
        if distances:
            return sum(distances) / len(distances)
        else:
            return -1
    
    def cleanup(self):
        """Cleanup GPIO pins"""
        print("  Cleaning up ultrasonic sensor...")
        GPIO.cleanup([self.trigger_pin, self.echo_pin])