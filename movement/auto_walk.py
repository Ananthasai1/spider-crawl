#!/usr/bin/env python3
"""
Autonomous Walking Controller for CyberCrawl Spider Robot
Handles auto mode with obstacle avoidance
"""

import time
import random
import config

class AutoWalkController:
    def __init__(self, servo_controller, ultrasonic_sensor, camera):
        """
        Initialize auto walk controller
        
        Args:
            servo_controller: SpiderServoController instance
            ultrasonic_sensor: UltrasonicSensor instance
            camera: CameraYOLO instance
        """
        self.servo = servo_controller
        self.ultrasonic = ultrasonic_sensor
        self.camera = camera
        
        self.is_running = False
        self.obstacle_threshold = config.OBSTACLE_THRESHOLD
        
        print("  🤖 Auto walk controller initialized")
    
    def start(self):
        """Start autonomous walking mode"""
        print("🚀 Starting Auto Mode...")
        self.is_running = True
        
        # Start camera detection if not already running
        if not self.camera.is_running:
            self.camera.start_detection()
        
        # Initial greeting
        print("  👋 Greeting sequence...")
        self.servo.stand()
        time.sleep(0.5)
        self.servo.hand_wave(3)
        time.sleep(0.5)
        
        # Start walking loop
        self._walk_loop()
    
    def _walk_loop(self):
        """Main autonomous walking loop"""
        print("  🚶 Starting autonomous navigation...")
        
        consecutive_obstacles = 0
        last_turn_direction = None
        
        while self.is_running:
            try:
                # Get distance reading
                distance = self.ultrasonic.get_average_distance(samples=2)
                
                # Check for obstacle
                if 0 < distance < self.obstacle_threshold:
                    consecutive_obstacles += 1
                    print(f"  ⚠️  Obstacle detected at {distance:.1f} cm!")
                    
                    # Stop and back up
                    print("  ⬅️  Backing up...")
                    self.servo.step_back(3)
                    time.sleep(0.3)
                    
                    # Decide turn direction
                    if consecutive_obstacles > 3:
                        # If stuck, try alternating directions
                        if last_turn_direction == 'left':
                            turn_direction = 'right'
                        elif last_turn_direction == 'right':
                            turn_direction = 'left'
                        else:
                            turn_direction = random.choice(['left', 'right'])
                    else:
                        # Random turn
                        turn_direction = random.choice(['left', 'right'])
                    
                    # Execute turn
                    turn_steps = random.randint(3, 6)
                    if turn_direction == 'left':
                        print(f"  ↩️  Turning left ({turn_steps} steps)...")
                        self.servo.turn_left(turn_steps)
                    else:
                        print(f"  ↪️  Turning right ({turn_steps} steps)...")
                        self.servo.turn_right(turn_steps)
                    
                    last_turn_direction = turn_direction
                    time.sleep(0.3)
                    
                    # Check if path is clear after turn
                    new_distance = self.ultrasonic.get_distance()
                    if 0 < new_distance < self.obstacle_threshold:
                        # Still blocked, try opposite direction
                        print("  ⚠️  Still blocked, trying opposite direction...")
                        opposite = 'right' if turn_direction == 'left' else 'left'
                        if opposite == 'left':
                            self.servo.turn_left(turn_steps * 2)
                        else:
                            self.servo.turn_right(turn_steps * 2)
                        time.sleep(0.3)
                    
                else:
                    # Path is clear
                    consecutive_obstacles = 0
                    
                    # Show detected objects occasionally
                    detections = self.camera.get_detections()
                    if detections and random.random() < 0.1:  # 10% chance
                        detected_objects = [d['class'] for d in detections]
                        print(f"  👁️  Detected: {', '.join(detected_objects)}")
                    
                    # Move forward
                    if distance > 0:
                        print(f"  ➡️  Path clear ({distance:.1f} cm) - moving forward")
                    
                    self.servo.step_forward(1)
                    
                    # Occasional gesture (5% chance)
                    if random.random() < 0.05:
                        gesture = random.choice(['wave', 'shake', None, None])
                        if gesture == 'wave':
                            print("  👋 Waving...")
                            self.servo.hand_wave(2)
                        elif gesture == 'shake':
                            print("  🤝 Shaking hand...")
                            self.servo.hand_shake(2)
                
                # Small delay between iterations
                time.sleep(config.AUTO_MODE_LOOP_DELAY)
                
            except KeyboardInterrupt:
                print("\n  ⏸️  Keyboard interrupt - stopping auto mode")
                self.stop()
                break
            except Exception as e:
                print(f"  ❌ Error in walk loop: {e}")
                time.sleep(0.5)
    
    def stop(self):
        """Stop autonomous mode"""
        print("  🛑 Stopping Auto Mode...")
        self.is_running = False
        time.sleep(0.2)
        
        # Sit down
        print("  💺 Sitting down...")
        self.servo.sit()
        
        print("  ✅ Auto Mode stopped")