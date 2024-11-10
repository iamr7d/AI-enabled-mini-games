import pygame
import random
import cv2
import mediapipe as mp
from enum import Enum
import time
import threading
from groq import Groq
import pyttsx3
from queue import Queue

# Initialize Pygame and mixer
pygame.init()
pygame.mixer.init()

class GameState(Enum):
    MENU = 1
    BATTING = 2
    BOWLING = 3
    INNINGS_BREAK = 4
    GAME_OVER = 5

# Enhanced Constants
WIDTH, HEIGHT = 800, 600
FPS = 30
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 100, 255)
YELLOW = (255, 255, 0)
PURPLE = (147, 0, 211)
GOLD = (255, 215, 0)
class CommentaryEngine:
    def __init__(self, api_key):
        self.client = Groq(api_key=api_key)
        # Initialize text-to-speech with proper settings
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)    # Speed of speech
        self.engine.setProperty('volume', 1.0)  # Volume level
        
        # Initialize thread-safe queue and event flags
        self.commentary_queue = Queue()
        self.is_speaking = threading.Event()
        self.should_stop = threading.Event()
        
        # Start the commentary processing thread
        self.commentary_thread = threading.Thread(target=self._process_commentary, daemon=True)
        self.commentary_thread.start()
    
    def _process_commentary(self):
        """Process commentary from queue and convert to speech"""
        while not self.should_stop.is_set():
            try:
                # Get commentary with timeout to allow checking should_stop
                commentary = self.commentary_queue.get(timeout=0.5)
                
                if commentary:
                    self.is_speaking.set()
                    # Configure speech properties for this commentary
                    self.engine.setProperty('rate', 150)  # Adjust rate for clarity
                    
                    def onEnd(name, completed):
                        self.is_speaking.clear()
                    
                    # Set callback for speech completion
                    self.engine.connect('finished-utterance', onEnd)
                    
                    # Speak the commentary
                    self.engine.say(commentary)
                    self.engine.runAndWait()
                    
            except Queue.Empty:
                continue
            except Exception as e:
                print(f"Error in commentary processing: {e}")
                self.is_speaking.clear()
    
    def generate_commentary(self, game_state, player_score, balls_bowled, runs_scored, target=None):
        """Generate contextual commentary based on game state"""
        try:
            # Build context for commentary
            context = f"Score: {player_score} runs from {balls_bowled} balls."
            if target:
                context += f" Target: {target} runs."
            
            # Create appropriate prompt based on game situation
            if runs_scored is not None:
                if runs_scored == 6:
                    prompt = f"Generate a short, excited cricket commentary for a SIX! {context}"
                elif runs_scored == 4:
                    prompt = f"Generate a short, excited cricket commentary for a FOUR! {context}"
                elif runs_scored == 0:
                    prompt = f"Generate a short cricket commentary for a dot ball. {context}"
                else:
                    prompt = f"Generate a short cricket commentary for {runs_scored} runs. {context}"
            else:
                prompt = f"Generate a short status update for: {context}"
            
            # Get commentary from Groq
            response = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama3-8b-8192",
                max_tokens=50,
                temperature=0.7  # Add some variety to commentary
            )
            
            commentary = response.choices[0].message.content
            
            # Clean up commentary text
            commentary = commentary.strip('"').strip()
            if not self.is_speaking.is_set():
                self.commentary_queue.put(commentary)
            
        except Exception as e:
            print(f"Commentary generation error: {e}")
    
    def cleanup(self):
        """Clean up resources when shutting down"""
        self.should_stop.set()
        if self.commentary_thread.is_alive():
            self.commentary_thread.join(timeout=1.0)
        self.engine.stop()
        
class HandCricket:
    def __init__(self, groq_api_key):
        # Existing initialization code remains the same
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Hand Cricket Pro")
        self.clock = pygame.time.Clock()
        
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.8,
            min_tracking_confidence=0.6
        )
        
        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # Game state and scores
        self.state = GameState.MENU
        self.player_score = 0
        self.computer_score = 0
        self.current_over = 0
        self.max_overs = 2
        self.balls_bowled = 0
        self.target = None
        self.is_first_innings = True
        self.last_gesture_time = 0
        self.gesture_cooldown = 1.0
        
        # Initialize Commentary Engine
        self.commentary = CommentaryEngine(groq_api_key)
        
        # Rest of the initialization remains the same...
        self.title_font = pygame.font.Font(None, 72)
        self.large_font = pygame.font.Font(None, 48)
        self.medium_font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 24)
        
        self.stats = {
            'boundaries': 0,
            'sixes': 0,
            'dots': 0,
            'strike_rate': 0.0,
            'last_6_balls': []
        }
        
        self.create_cricket_field()

    def create_cricket_field(self):
        """Create a simple cricket field background"""
        self.field = pygame.Surface((WIDTH, HEIGHT))
        self.field.fill((34, 139, 34))  # Green field
        
        # Draw cricket pitch
        pygame.draw.rect(self.field, (193, 154, 107), (WIDTH//2 - 30, 100, 60, 400))  # Brown pitch
        
        # Draw crease lines
        pygame.draw.rect(self.field, WHITE, (WIDTH//2 - 40, 150, 80, 5))  # Upper crease
        pygame.draw.rect(self.field, WHITE, (WIDTH//2 - 40, 450, 80, 5))  # Lower crease
        
        # Draw boundary circle
        pygame.draw.circle(self.field, WHITE, (WIDTH//2, HEIGHT//2), 250, 2)

    def get_player_gesture(self, landmarks):
        """Enhanced gesture detection for cricket shots"""
        if landmarks is None:
            return None
        
        # Extract key landmarks
        thumb_tip = landmarks[self.mp_hands.HandLandmark.THUMB_TIP]
        index_tip = landmarks[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        middle_tip = landmarks[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        ring_tip = landmarks[self.mp_hands.HandLandmark.RING_FINGER_TIP]
        pinky_tip = landmarks[self.mp_hands.HandLandmark.PINKY_TIP]
        
        # Count extended fingers
        fingers_extended = [
            thumb_tip.y < landmarks[self.mp_hands.HandLandmark.THUMB_IP].y,
            index_tip.y < landmarks[self.mp_hands.HandLandmark.INDEX_FINGER_PIP].y,
            middle_tip.y < landmarks[self.mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y,
            ring_tip.y < landmarks[self.mp_hands.HandLandmark.RING_FINGER_PIP].y,
            pinky_tip.y < landmarks[self.mp_hands.HandLandmark.PINKY_TIP].y
        ]
        
        extended_count = sum(fingers_extended)
        
        # Map finger count to cricket shots
        if extended_count == 1:
            return 1  # Single
        elif extended_count == 2:
            return 2  # Double
        elif extended_count == 3:
            return 3  # Triple
        elif extended_count == 4:
            return 4  # Boundary
        elif extended_count == 5:
            return 6  # Six
        elif extended_count == 0:
            return 0  # Dot ball
        
        return None

    def draw_scoreboard(self):
        """Draw an enhanced cricket scoreboard"""
        # Draw scoreboard background
        pygame.draw.rect(self.screen, (50, 50, 50), (WIDTH - 250, 0, 250, 200))
        pygame.draw.rect(self.screen, WHITE, (WIDTH - 250, 0, 250, 200), 2)
        
        # Current score
        score_text = self.large_font.render(f"{self.player_score}/{self.current_over}.{self.balls_bowled}", True, WHITE)
        self.screen.blit(score_text, (WIDTH - 230, 20))
        
        # Target if applicable
        if self.target:
            target_text = self.medium_font.render(f"Target: {self.target}", True, GOLD)
            self.screen.blit(target_text, (WIDTH - 230, 70))
        
        # Required run rate
        if self.target and self.balls_bowled < self.max_overs * 6:
            balls_remaining = (self.max_overs * 6) - self.balls_bowled
            runs_needed = self.target - self.player_score
            if balls_remaining > 0:
                req_rate = (runs_needed * 6) / balls_remaining
                rr_text = self.small_font.render(f"RRR: {req_rate:.2f}", True, YELLOW)
                self.screen.blit(rr_text, (WIDTH - 230, 110))
        
        # Statistics
        stats_y = 140
        stats_texts = [
            f"4s: {self.stats['boundaries']}",
            f"6s: {self.stats['sixes']}",
            f"SR: {self.stats['strike_rate']:.1f}"
        ]
        
        for text in stats_texts:
            stat_text = self.small_font.render(text, True, WHITE)
            self.screen.blit(stat_text, (WIDTH - 230, stats_y))
            stats_y += 20

    def draw_shot_animation(self, runs):
        """Animate the cricket shot"""
        animation_frames = 30
        ball_pos = [WIDTH//2, HEIGHT - 100]
        
        for _ in range(animation_frames):
            self.screen.blit(self.field, (0, 0))
            
            if runs == 4:
                # Ground shot animation
                ball_pos[0] += random.randint(-5, 5)
                ball_pos[1] -= 3
            elif runs == 6:
                # Aerial shot animation
                ball_pos[0] += random.randint(-3, 3)
                ball_pos[1] -= 5
            
            pygame.draw.circle(self.screen, WHITE, (int(ball_pos[0]), int(ball_pos[1])), 5)
            pygame.display.flip()
            self.clock.tick(60)

    def draw_menu(self):
        """Draw the main menu"""
        self.screen.fill(BLACK)
        
        # Title
        title = self.title_font.render("Hand Cricket Pro", True, GOLD)
        self.screen.blit(title, (WIDTH//2 - title.get_width()//2, 100))
        
        # Instructions
        instructions = [
            "Show fingers to play shots:",
            "1 finger = Single",
            "2 fingers = Double",
            "3 fingers = Triple",
            "4 fingers = Four",
            "5 fingers = Six",
            "Fist = Dot ball",
            "",
            "Press SPACE to start",
            "Press Q to quit"
        ]
        
        for i, text in enumerate(instructions):
            instruction = self.medium_font.render(text, True, WHITE)
            self.screen.blit(instruction, (WIDTH//2 - instruction.get_width()//2, 250 + i * 40))

    def draw_innings_break(self):
        """Draw innings break screen"""
        self.screen.fill(BLACK)
        
        texts = [
            f"First Innings Score: {self.player_score}",
            f"Target: {self.target}",
            "",
            "Press SPACE to start second innings"
        ]
        
        for i, text in enumerate(texts):
            text_surface = self.large_font.render(text, True, WHITE)
            self.screen.blit(text_surface, (WIDTH//2 - text_surface.get_width()//2, HEIGHT//3 + i * 50))

    def draw_game_over(self):
        """Draw game over screen"""
        self.screen.fill(BLACK)
        
        if self.is_first_innings:
            result_text = f"Final Score: {self.player_score}"
        else:
            if self.player_score >= self.target:
                result_text = "You Won!"
            else:
                result_text = f"You Lost by {self.target - self.player_score} runs"
        
        text_surface = self.title_font.render(result_text, True, GOLD)
        self.screen.blit(text_surface, (WIDTH//2 - text_surface.get_width()//2, HEIGHT//3))
        
        # Display final statistics
        stats_text = [
            f"Boundaries: {self.stats['boundaries']}",
            f"Sixes: {self.stats['sixes']}",
            f"Strike Rate: {self.stats['strike_rate']:.1f}",
            "",
            "Press SPACE to play again",
            "Press Q to quit"
        ]
        
        for i, text in enumerate(stats_text):
            stat_surface = self.medium_font.render(text, True, WHITE)
            self.screen.blit(stat_surface, (WIDTH//2 - stat_surface.get_width()//2, HEIGHT//2 + i * 40))

    def update_statistics(self, runs):
            """Update game statistics and generate commentary"""
            if runs == 4:
                self.stats['boundaries'] += 1
            elif runs == 6:
                self.stats['sixes'] += 1
            elif runs == 0:
                self.stats['dots'] += 1
            
            self.stats['last_6_balls'].append(runs)
            if len(self.stats['last_6_balls']) > 6:
                self.stats['last_6_balls'].pop(0)
            
            # Update strike rate
            if self.balls_bowled > 0:
                self.stats['strike_rate'] = (self.player_score * 100) / self.balls_bowled
            
            # Generate commentary for the runs scored
            self.commentary.generate_commentary(
                self.state,
                self.player_score,
                self.balls_bowled,
                runs,
                self.target
            )
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
                        if self.state == GameState.MENU:
                            self.state = GameState.BATTING
                            self.commentary.generate_commentary(
                                self.state,
                                self.player_score,
                                self.balls_bowled,
                                None
                            )
                        elif self.state == GameState.INNINGS_BREAK:
                            self.state = GameState.BATTING
                            self.is_first_innings = False
                            self.balls_bowled = 0
                            self.player_score = 0
                            self.stats = {
                                'boundaries': 0,
                                'sixes': 0,
                                'dots': 0,
                                'strike_rate': 0.0,
                                'last_6_balls': []
                            }
                            self.commentary.generate_commentary(
                                self.state,
                                self.player_score,
                                self.balls_bowled,
                                None,
                                self.target
                            )
                        elif self.state == GameState.GAME_OVER:
                            self.__init__()
            
            # Camera capture and processing
            ret, frame = self.cap.read()
            if not ret:
                continue  # Now properly inside the while loop
                
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(frame_rgb)
            
            if self.state == GameState.MENU:
                self.draw_menu()
            
            elif self.state == GameState.BATTING:
                self.screen.blit(self.field, (0, 0))
                self.draw_scoreboard()
                
                if results.multi_hand_landmarks:
                    current_time = time.time()
                    if current_time - self.last_gesture_time >= self.gesture_cooldown:
                        self.last_gesture_time = current_time
                        
                        gesture = self.get_player_gesture(results.multi_hand_landmarks[0].landmark)
                        if gesture is not None:
                            self.player_score += gesture
                            self.balls_bowled += 1
                            self.update_statistics(gesture)
                            
                            if gesture >= 4:
                                self.draw_shot_animation(gesture)
                            
                            if self.balls_bowled >= self.max_overs * 6:
                                if self.is_first_innings:
                                    self.target = self.player_score + 1
                                    self.state = GameState.INNINGS_BREAK
                                else:
                                    self.state = GameState.GAME_OVER
                            elif not self.is_first_innings and self.player_score >= self.target:
                                self.state = GameState.GAME_OVER
            
            elif self.state == GameState.INNINGS_BREAK:
                self.draw_innings_break()
            
            elif self.state == GameState.GAME_OVER:
                self.draw_game_over()
            
            pygame.display.flip()
        
        # Cleanup
        self.cap.release()
        pygame.quit()  # Now properly inside the while loop
# Start the game
if __name__ == "__main__":
    GROQ_API_KEY = "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"  # Replace with your actual API key
    game = HandCricket(GROQ_API_KEY)
    game.run()