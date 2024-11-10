import pygame
import random
import mediapipe as mp
import cv2
import math
from enum import Enum
import pandas as pd
from datetime import datetime
import os

# Initialize Pygame and mixer for sound effects
pygame.init()
pygame.mixer.init()

# Define Game States
class GameState(Enum):
    MENU = 1
    PLAYING = 2
    ROUND_END = 3
    GAME_OVER = 4

# Define Constants
WIDTH, HEIGHT = 800, 600
FPS = 30
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)
BLUE = (0, 100, 255)
PURPLE = (147, 0, 211)

# Initialize MediaPipe Hands with improved settings
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.6
)

# Initialize camera with preferred resolution
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

class RockPaperScissors:
    def __init__(self):
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Rock Paper Scissors Ultimate")
        self.clock = pygame.time.Clock()
        
        # Game state and scores
        self.state = GameState.MENU
        self.score_player = 5
        self.score_computer = 5
        self.round_number = 1
        self.player_streak = 0
        
        # Initialize fonts
        self.title_font = pygame.font.Font(None, 72)
        self.large_font = pygame.font.Font(None, 48)
        self.medium_font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 24)
        
        # Animation variables
        self.fade_alpha = 0
        self.animation_surface = pygame.Surface((WIDTH, HEIGHT))
        self.animation_surface.set_alpha(self.fade_alpha)
        
        # Gesture detection cooldown
        self.last_gesture_time = 0
        self.gesture_cooldown = 1.0  # seconds
        
        # Round history
        self.round_history = []
        
        # Data logging setup
        self.data_path = r"C:\Users\rahulrajpvr7d\OneDrive\Desktop\GAMES\DATA\game_history.csv"
        self.ensure_data_directory()
        self.game_data = []
        
        # Load existing data if available
        if os.path.exists(self.data_path):
            self.existing_data = pd.read_csv(self.data_path)
        else:
            self.existing_data = pd.DataFrame(columns=[
                'timestamp', 'player_choice', 'computer_choice', 
                'result', 'player_score', 'computer_score', 
                'round_number', 'streak'
            ])
        
        # Load and scale images
        self.load_images()

    def ensure_data_directory(self):
        """Ensure the data directory exists"""
        os.makedirs(os.path.dirname(self.data_path), exist_ok=True)

    def log_round_data(self, player_choice, computer_choice, result):
        """Log the round data to be saved"""
        round_data = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'player_choice': player_choice,
            'computer_choice': computer_choice,
            'result': result,
            'player_score': self.score_player,
            'computer_score': self.score_computer,
            'round_number': self.round_number,
            'streak': self.player_streak
        }
        self.game_data.append(round_data)
        
        # Convert to DataFrame and append to CSV
        new_data = pd.DataFrame([round_data])
        if os.path.exists(self.data_path):
            new_data.to_csv(self.data_path, mode='a', header=False, index=False)
        else:
            new_data.to_csv(self.data_path, index=False)

    def load_images(self):
        # Create simple geometric shapes for gestures
        self.gesture_images = {
            'rock': self.create_circle_surface(60, RED),
            'paper': self.create_square_surface(120, BLUE),
            'scissors': self.create_triangle_surface(80, GREEN)
        }

    def create_circle_surface(self, radius, color):
        surface = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(surface, color, (radius, radius), radius)
        return surface

    def create_square_surface(self, size, color):
        surface = pygame.Surface((size, size), pygame.SRCALPHA)
        pygame.draw.rect(surface, color, (0, 0, size, size))
        return surface

    def create_triangle_surface(self, size, color):
        surface = pygame.Surface((size, size), pygame.SRCALPHA)
        points = [(size//2, 0), (0, size), (size, size)]
        pygame.draw.polygon(surface, color, points)
        return surface

    def get_player_gesture(self, landmarks):
        """Enhanced gesture detection with additional checks for accuracy"""
        if landmarks is None:
            return None
        
        # Extract key landmarks
        thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
        index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        middle_tip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        ring_tip = landmarks[mp_hands.HandLandmark.RING_FINGER_TIP]
        pinky_tip = landmarks[mp_hands.HandLandmark.PINKY_TIP]
        
        # Calculate distances and angles for better gesture recognition
        fingers_extended = [
            middle_tip.y < landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y,
            index_tip.y < landmarks[mp_hands.HandLandmark.INDEX_FINGER_PIP].y,
            ring_tip.y < landmarks[mp_hands.HandLandmark.RING_FINGER_PIP].y,
            pinky_tip.y < landmarks[mp_hands.HandLandmark.PINKY_TIP].y
        ]
        
        # Enhanced gesture detection logic
        extended_count = sum(fingers_extended)
        
        if extended_count <= 1:  # Closed fist
            return 'rock'
        elif extended_count >= 3:  # Open hand
            return 'paper'
        elif extended_count == 2 and fingers_extended[0] and fingers_extended[1]:  # Victory sign
            return 'scissors'
        
        return None

    def draw_menu(self):
        self.screen.fill(BLACK)
        
        # Draw title
        title = self.title_font.render("Rock Paper Scissors Ultimate", True, WHITE)
        self.screen.blit(title, (WIDTH//2 - title.get_width()//2, 100))
        
        # Draw instructions
        instructions = [
            "Show your hand to the camera to play",
            "Rock: Closed fist",
            "Paper: Open hand",
            "Scissors: Victory sign",
            "Press SPACE to start",
            "Press Q to quit"
        ]
        
        for i, text in enumerate(instructions):
            instruction = self.medium_font.render(text, True, WHITE)
            self.screen.blit(instruction, (WIDTH//2 - instruction.get_width()//2, 250 + i * 50))

    def draw_game_ui(self):
        self.screen.fill(BLACK)
        
        # Draw scores with health bar style
        self.draw_health_bar(50, 30, self.score_player, 5, "Player")
        self.draw_health_bar(WIDTH - 250, 30, self.score_computer, 5, "Computer")
        
        # Draw round number
        round_text = self.medium_font.render(f"Round {self.round_number}", True, WHITE)
        self.screen.blit(round_text, (WIDTH//2 - round_text.get_width()//2, 20))
        
        # Draw streak if any
        if self.player_streak > 1:
            streak_text = self.medium_font.render(f"Streak: {self.player_streak}!", True, YELLOW)
            self.screen.blit(streak_text, (WIDTH//2 - streak_text.get_width()//2, 60))

    def draw_health_bar(self, x, y, current, maximum, label):
        bar_width = 200
        bar_height = 30
        fill_width = int((current / maximum) * bar_width)
        
        # Draw label
        label_text = self.small_font.render(label, True, WHITE)
        self.screen.blit(label_text, (x + bar_width//2 - label_text.get_width()//2, y - 25))
        
        # Draw border
        pygame.draw.rect(self.screen, WHITE, (x, y, bar_width, bar_height), 2)
        
        # Draw fill
        color = GREEN if current > maximum//2 else YELLOW if current > maximum//4 else RED
        pygame.draw.rect(self.screen, color, (x, y, fill_width, bar_height))
        
        # Draw pip markers
        for i in range(1, maximum):
            pip_x = x + (i * bar_width // maximum)
            pygame.draw.line(self.screen, WHITE, (pip_x, y), (pip_x, y + bar_height), 2)

    def display_round_result(self, player_choice, computer_choice, result):
        self.screen.fill(BLACK)
        
        # Draw choices with images
        player_img = self.gesture_images[player_choice]
        computer_img = self.gesture_images[computer_choice]
        
        # Position images
        self.screen.blit(player_img, (WIDTH//4 - player_img.get_width()//2, HEIGHT//2 - player_img.get_height()//2))
        self.screen.blit(computer_img, (3*WIDTH//4 - computer_img.get_width()//2, HEIGHT//2 - computer_img.get_height()//2))
        
        # Draw labels
        player_text = self.medium_font.render(f"You chose {player_choice}", True, WHITE)
        computer_text = self.medium_font.render(f"Computer chose {computer_choice}", True, WHITE)
        self.screen.blit(player_text, (WIDTH//4 - player_text.get_width()//2, HEIGHT//2 + 100))
        self.screen.blit(computer_text, (3*WIDTH//4 - computer_text.get_width()//2, HEIGHT//2 + 100))
        
        # Draw result
        result_color = GREEN if result == "Win" else RED if result == "Lose" else YELLOW
        result_text = self.large_font.render(result, True, result_color)
        self.screen.blit(result_text, (WIDTH//2 - result_text.get_width()//2, 100))

    def display_game_over(self, winner):
        self.screen.fill(BLACK)
        
        # Draw game over message
        game_over_text = self.title_font.render("Game Over!", True, YELLOW)
        winner_text = self.large_font.render(f"{winner} Wins!", True, GREEN)
        
        self.screen.blit(game_over_text, (WIDTH//2 - game_over_text.get_width()//2, HEIGHT//3))
        self.screen.blit(winner_text, (WIDTH//2 - winner_text.get_width()//2, HEIGHT//2))
        
        # Draw final scores
        score_text = self.medium_font.render(f"Final Score - Player: {self.score_player} Computer: {self.score_computer}", True, WHITE)
        self.screen.blit(score_text, (WIDTH//2 - score_text.get_width()//2, HEIGHT*2//3))
        
        # Draw restart instruction
        restart_text = self.small_font.render("Press SPACE to play again or Q to quit", True, WHITE)
        self.screen.blit(restart_text, (WIDTH//2 - restart_text.get_width()//2, HEIGHT*5//6))

    def run(self):
        running = True
        while running:
            self.clock.tick(FPS)
            
            # Event handling
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        if self.state in [GameState.MENU, GameState.GAME_OVER]:
                            self.state = GameState.PLAYING
                            self.score_player = 5
                            self.score_computer = 5
                            self.round_number = 1
                            self.player_streak = 0
                            self.round_history = []

            # Camera capture and hand detection
            ret, frame = cap.read()
            if not ret:
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            # State machine
            if self.state == GameState.MENU:
                self.draw_menu()
            
            elif self.state == GameState.PLAYING:
                self.draw_game_ui()
                
                # Process hand gesture
                if results.multi_hand_landmarks:
                    current_time = pygame.time.get_ticks() / 1000
                    if current_time - self.last_gesture_time >= self.gesture_cooldown:
                        landmarks = results.multi_hand_landmarks[0].landmark
                        player_gesture = self.get_player_gesture(landmarks)
                        
                        if player_gesture:
                            self.last_gesture_time = current_time
                            computer_gesture = random.choice(['rock', 'paper', 'scissors'])
                            
                            # Determine winner
                            if player_gesture == computer_gesture:
                                result = "Tie"
                                self.player_streak = 0
                            elif ((player_gesture == 'rock' and computer_gesture == 'scissors') or
                                  (player_gesture == 'scissors' and computer_gesture == 'paper') or
                                  (player_gesture == 'paper' and computer_gesture == 'rock')):
                                result = "Win"
                                self.score_computer -= 1
                                self.player_streak += 1
                            else:
                                result = "Lose"
                                self.score_player -= 1
                                self.player_streak = 0
                            
                            # Log the round data
                            self.log_round_data(player_gesture, computer_gesture, result)
                            
                            self.round_history.append((player_gesture, computer_gesture, result))
                            self.state = GameState.ROUND_END
                            self.round_number += 1
            
            elif self.state == GameState.ROUND_END:
                last_round = self.round_history[-1]
                self.display_round_result(last_round[0], last_round[1], last_round[2])
                
                # Check for game over
                if self.score_player <= 0:
                    self.state = GameState.GAME_OVER
                    winner = "Computer"
                elif self.score_computer <= 0:
                    self.state = GameState.GAME_OVER
                    winner = "Player"
                else:
                    # Auto-continue after delay
                    current_time = pygame.time.get_ticks() / 1000
                    if current_time - self.last_gesture_time >= 3:
                        self.state = GameState.PLAYING
            
            elif self.state == GameState.GAME_OVER:
                self.display_game_over("Player" if self.score_computer <= 0 else "Computer")

            pygame.display.flip()

        # Cleanup
        cap.release()
        pygame.quit()

# Start the game
if __name__ == "__main__":
    game = RockPaperScissors()
    game.run()