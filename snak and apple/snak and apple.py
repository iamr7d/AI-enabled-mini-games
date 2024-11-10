import pygame
import random
import cv2
import mediapipe as mp
from enum import Enum
import numpy as np
import time
import sys

class Direction(Enum):
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4
    NEUTRAL = 5

class PowerUpType(Enum):
    SPEED = 1
    SLOW = 2
    INVINCIBILITY = 3
    DOUBLE_POINTS = 4

class GameState:
    def __init__(self):
        # Initialize pygame
        # Initialize Pygame
        pygame.init()

        # Set up display
        #screen = pygame.display.set_mode((800, 600))
        
        pygame.display.set_caption("Snake & Apple AI Game")
        
        # Get the screen dimensions
        self.screen_info = pygame.display.Info()
        self.width = self.screen_info.current_w
        self.height = self.screen_info.current_h
        
        # Display setup
        self.display = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption('Enhanced Snake Game with Hand Gestures')
        
        # Colors
        self.COLORS = {
            'WHITE': (255, 255, 255),
            'BLACK': (0, 0, 0),
            'RED': (213, 50, 80),
            'GREEN': (0, 255, 0),
            'BLUE': (50, 153, 213),
            'YELLOW': (255, 255, 0),
            'PURPLE': (128, 0, 128)
        }
        
        # Game settings
        self.snake_block = max(25, self.width // 80)
        self.food_size = max(25, self.width // 100)
        self.base_speed = 20
        self.snake_speed = self.base_speed
        
        # Fonts
        self.font_style = pygame.font.SysFont("bahnschrift", 25)
        self.score_font = pygame.font.SysFont("bahnschrift", 35)
        
        # MediaPipe setup
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Game state
        self.clock = pygame.time.Clock()
        self.is_fullscreen = False
        self.power_ups = []
        self.current_power_ups = set()
        self.power_up_timers = {}
        self.gesture_smoothing_buffer = []
        self.buffer_size = 5
        self.obstacles = []  # Initialize the obstacles list
        # Initialize game variables
        self.game_over = False
        self.game_close = False
        self.paused = False
        self.score = 0
        self.snake_list = []
        self.length_of_snake = 1
        self.x1 = self.width / 2
        self.y1 = self.height / 2
        self.x1_change = 0
        self.y1_change = 0

    def init_game_variables(self):
        self.game_over = False
        self.game_close = False
        self.paused = False
        self.score = 0
        self.snake_list = []
        self.length_of_snake = 2
        self.x1 = self.width / 2
        self.y1 = self.height / 2
        self.x1_change = 0
        self.y1_change = 0
        self.obstacles = []  # Initialize obstacles here
        self.spawn_initial_objects()

    def spawn_initial_objects(self):
        self.foodx, self.foody = self.spawn_food()
        self.obstacles = self.generate_obstacles(10)
        self.spawn_power_up()

    def spawn_food(self):
        while True:
            x = round(random.randrange(0, self.width - self.food_size) / 10.0) * 10.0
            y = round(random.randrange(0, self.height - self.food_size) / 10.0) * 10.0
            if not self.check_position_conflicts(x, y):
                return x, y

    def display_start_screen(self):
        # Initialize game background and variables
        self.init_game_variables()
        self.display.fill((0, 0, 0))  # Black background

        # Title and subtitle animation variables
        title_y = -200
        subtitle_y = -150
        start_y = self.height / 2 + 100
        title_alpha = 0
        subtitle_alpha = 0
        start_alpha = 0
        blink_visible = True
        blink_timer = 0

        # Load snake and apple images
        snake_image = pygame.font.SysFont("Arial", 50).render("🐍", True, (0, 255, 0))
        apple_image = pygame.font.SysFont("Arial", 50).render("🍎", True, (255, 0, 0))

        # Multiple flying snake and apple smileys
        num_snakes = 5
        num_apples = 5
        snake_x = [random.randint(0, self.width) for _ in range(num_snakes)]
        snake_y = [random.randint(0, self.height) for _ in range(num_snakes)]
        snake_speed_x = [random.choice([-5, 5]) for _ in range(num_snakes)]
        snake_speed_y = [random.choice([-5, 5]) for _ in range(num_snakes)]
        apple_x = [random.randint(0, self.width) for _ in range(num_apples)]
        apple_y = [random.randint(0, self.height) for _ in range(num_apples)]
        apple_speed_x = [random.choice([-5, 5]) for _ in range(num_apples)]
        apple_speed_y = [random.choice([-5, 5]) for _ in range(num_apples)]

        # Animation loop
        clock = pygame.time.Clock()
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        running = False

            # Clear screen
            self.display.fill((0, 0, 0))  # Black background

            # Draw and update multiple flying snake and apple smileys
            for i in range(num_snakes):
                self.display.blit(snake_image, (snake_x[i], snake_y[i]))
                snake_x[i] += snake_speed_x[i]
                snake_y[i] += snake_speed_y[i]

                # Bounce snake off edges
                if snake_x[i] < 0 or snake_x[i] > self.width - 50:
                    snake_speed_x[i] *= -1
                if snake_y[i] < 0 or snake_y[i] > self.height - 50:
                    snake_speed_y[i] *= -1

            for i in range(num_apples):
                self.display.blit(apple_image, (apple_x[i], apple_y[i]))
                apple_x[i] += apple_speed_x[i]
                apple_y[i] += apple_speed_y[i]

                # Bounce apple off edges
                if apple_x[i] < 0 or apple_x[i] > self.width - 50:
                    apple_speed_x[i] *= -1
                if apple_y[i] < 0 or apple_y[i] > self.height - 50:
                    apple_speed_y[i] *= -1

            # Game Title
            font = pygame.font.SysFont("bahnschrift", 80, bold=True)
            game_title_text = font.render("Snake & Apple", True, (255, 255, 255))
            game_title_text.set_alpha(title_alpha)
            game_title_rect = game_title_text.get_rect(center=(self.width / 2, title_y))
            self.display.blit(game_title_text, game_title_rect)

            # Subtitle
            font = pygame.font.SysFont("bahnschrift", 30, bold=True)
            sub_text = font.render("(AI Enabled)", True, (255, 255, 0))  # Yellow color
            sub_text.set_alpha(subtitle_alpha)
            sub_text_rect = sub_text.get_rect(center=(self.width / 2, subtitle_y))
            self.display.blit(sub_text, sub_text_rect)

            # AI sparkle effect
            sparkle_font = pygame.font.SysFont("bahnschrift", 20, bold=True)
            sparkle_text = sparkle_font.render("*", True, (255, 255, 0))  # Yellow sparkle
            sparkle_rect = sparkle_text.get_rect(center=(self.width / 2 + 130, subtitle_y - 10))
            self.display.blit(sparkle_text, sparkle_rect)
            sparkle_text = sparkle_font.render("*", True, (255, 255, 0))
            sparkle_rect = sparkle_text.get_rect(center=(self.width / 2 + 115, subtitle_y))
            self.display.blit(sparkle_text, sparkle_rect)

            # Start text animation
            start_font = pygame.font.SysFont("bahnschrift", 40)
            start_text = start_font.render("Press SPACE to Start", True, (255, 255, 255))
            start_text.set_alpha(start_alpha if blink_visible else 0)  # Blink effect
            start_text_rect = start_text.get_rect(center=(self.width / 2, start_y))
            self.display.blit(start_text, start_text_rect)

            # Update animation variables
            if title_y < self.height / 2 - 100:
                title_y += 5
                title_alpha = min(title_alpha + 5, 255)
            if subtitle_y < self.height / 2 - 40:
                subtitle_y += 5
                subtitle_alpha = min(subtitle_alpha + 5, 255)
            if start_alpha < 255:
                start_alpha += 5

            # Blink timer
            blink_timer += 1
            if blink_timer >= 30:
                blink_visible = not blink_visible
                blink_timer = 0

            # Update display and set frame rate
            pygame.display.update()
            clock.tick(30)

    def spawn_power_up(self):
        if len(self.power_ups) < 3:  # Limit number of power-ups on screen
            power_up = {
                'type': random.choice(list(PowerUpType)),
                'position': self.get_valid_position(),
                'spawn_time': time.time()
            }
            self.power_ups.append(power_up)

    def get_valid_position(self):
        while True:
            x = round(random.randrange(0, self.width - self.food_size) / 10.0) * 10.0
            y = round(random.randrange(0, self.height - self.food_size) / 10.0) * 10.0
            if not self.check_position_conflicts(x, y):
                return (x, y)

    def check_position_conflicts(self, x, y):
        # Check obstacles
        for obstacle in self.obstacles:
            if abs(x - obstacle[0]) < self.snake_block and abs(y - obstacle[1]) < self.snake_block:
                return True
        
        # Check existing power-ups
        for power_up in self.power_ups:
            pos = power_up['position']
            if abs(x - pos[0]) < self.food_size and abs(y - pos[1]) < self.food_size:
                return True

        # Check snake body
        for segment in self.snake_list:
            if abs(x - segment[0]) < self.snake_block and abs(y - segment[1]) < self.snake_block:
                return True

        return False

    def generate_obstacles(self, num_obstacles):
        obstacles = []
        for _ in range(num_obstacles):
            while True:
                x = round(random.randrange(0, self.width - self.snake_block) / 10.0) * 10.0
                y = round(random.randrange(0, self.height - self.snake_block) / 10.0) * 10.0
                if abs(x - self.width/2) > self.snake_block * 3 and abs(y - self.height/2) > self.snake_block * 3:
                    obstacles.append((x, y))
                    break
        return obstacles

    def get_smoothed_direction(self, new_direction):
        self.gesture_smoothing_buffer.append(new_direction)
        if len(self.gesture_smoothing_buffer) > self.buffer_size:
            self.gesture_smoothing_buffer.pop(0)
        
        direction_counts = {}
        for direction in self.gesture_smoothing_buffer:
            direction_counts[direction] = direction_counts.get(direction, 0) + 1
            
        return max(direction_counts.items(), key=lambda x: x[1])[0]

    def get_hand_direction(self, hand_landmarks):
        index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        index_mcp = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_MCP]
        
        y_diff = index_tip.y - index_mcp.y
        x_diff = index_tip.x - index_mcp.x
        
        threshold = 0.1
        if abs(y_diff) > abs(x_diff):
            if y_diff < -threshold:
                return Direction.UP
            elif y_diff > threshold:
                return Direction.DOWN
        else:
            if x_diff < -threshold:
                return Direction.LEFT
            elif x_diff > threshold:
                return Direction.RIGHT
        
        return Direction.NEUTRAL

    def apply_power_up(self, power_up_type):
        if power_up_type == PowerUpType.SPEED:
            self.snake_speed = min(self.base_speed * 1.5, 30)
            self.power_up_timers[PowerUpType.SPEED] = time.time() + 5
        elif power_up_type == PowerUpType.SLOW:
            self.snake_speed = max(self.base_speed * 0.5, 5)
            self.power_up_timers[PowerUpType.SLOW] = time.time() + 5
        elif power_up_type == PowerUpType.INVINCIBILITY:
            self.current_power_ups.add(PowerUpType.INVINCIBILITY)
            self.power_up_timers[PowerUpType.INVINCIBILITY] = time.time() + 3
        elif power_up_type == PowerUpType.DOUBLE_POINTS:
            self.current_power_ups.add(PowerUpType.DOUBLE_POINTS)
            self.power_up_timers[PowerUpType.DOUBLE_POINTS] = time.time() + 10

    def play_background_music(self):
        try:
            pygame.mixer.init()
            pygame.mixer.music.load(r"C:\Users\rahulrajpvr7d\Music\8-bit-retro-game-music-233964.mp3")
            pygame.mixer.music.set_volume(0.5)  # Adjust volume (0.0 to 1.0)
            pygame.mixer.music.play(-1)  # Loop music indefinitely
        except pygame.error as e:
            print(f"Error loading music: {e}")  # Handle any errors that occur

    def update_power_ups(self):
        current_time = time.time()
        expired_power_ups = []
        
        for power_up_type, end_time in self.power_up_timers.items():
            if current_time > end_time:
                expired_power_ups.append(power_up_type)
        
        for power_up_type in expired_power_ups:
            self.power_up_timers.pop(power_up_type)
            self.current_power_ups.discard(power_up_type)
            if power_up_type in (PowerUpType.SPEED, PowerUpType.SLOW):
                self.snake_speed = self.base_speed
        
        self.power_ups = [p for p in self.power_ups if current_time - p['spawn_time'] < 10]
        
        if random.random() < 0.01:
            self.spawn_power_up()

    def draw_game_objects(self):
        self.display.fill(self.COLORS['BLUE'])
        
        pygame.draw.rect(self.display, self.COLORS['GREEN'],
                        [self.foodx, self.foody, self.food_size, self.food_size])
        
        for obstacle in self.obstacles:
            pygame.draw.rect(self.display, self.COLORS['BLACK'],
                           [obstacle[0], obstacle[1], self.snake_block, self.snake_block])
        
        for power_up in self.power_ups:
            color = self.COLORS['YELLOW']
            if power_up['type'] == PowerUpType.SPEED:
                color = self.COLORS['RED']
            elif power_up['type'] == PowerUpType.INVINCIBILITY:
                color = self.COLORS['PURPLE']
            pygame.draw.rect(self.display, color,
                           [power_up['position'][0], power_up['position'][1],
                            self.food_size, self.food_size])
        
        snake_color = self.COLORS['GREEN']
        if PowerUpType.INVINCIBILITY in self.current_power_ups:
            snake_color = self.COLORS['YELLOW']
        for x in self.snake_list:
            pygame.draw.rect(self.display, snake_color,
                           [x[0], x[1], self.snake_block, self.snake_block])

    def display_game_info(self):
        score_text = self.score_font.render(f"Score: {self.score}", True, self.COLORS['WHITE'])
        self.display.blit(score_text, [10, 10])
        
        y_offset = 50
        for power_up_type in self.current_power_ups:
            time_left = round(self.power_up_timers[power_up_type] - time.time(), 1)
            if time_left > 0:
                power_up_text = self.font_style.render(
                    f"{power_up_type.name}: {time_left}s", True, self.COLORS['WHITE'])
                self.display.blit(power_up_text, [10, y_offset])
                y_offset += 30

    def run_game(self):
        cap = cv2.VideoCapture(0)
        self.play_background_music()
        self.display_start_screen()
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        waiting = False
            self.clock.tick(10)  # Limit frame rate to prevent high CPU usage
        with self.mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:
            self.init_game_variables()
            
            while not self.game_over:
                if self.game_close:
                    self.handle_game_over()
                    continue
                
                if self.handle_events():
                    break
                
                if self.paused:
                    continue
                
                ret, frame = cap.read()
                if not ret:
                    continue
                
                frame = cv2.flip(frame, 1)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(frame_rgb)
                
                if results.multi_hand_landmarks:
                    for landmarks in results.multi_hand_landmarks:
                        self.mp_drawing.draw_landmarks(frame, landmarks,
                                                     self.mp_hands.HAND_CONNECTIONS)
                        direction = self.get_hand_direction(landmarks)
                        smoothed_direction = self.get_smoothed_direction(direction)
                        self.update_snake_direction(smoothed_direction)
                
                self.update_snake_position()
                self.check_collisions()
                self.update_power_ups()
                
                self.draw_game_objects()
                self.display_game_info()
                pygame.display.update()
                
                self.clock.tick(self.snake_speed)
            
            cap.release()
            pygame.quit()

    def update_snake_direction(self, direction):
        if direction == Direction.UP and self.y1_change <= 0:
            self.y1_change = -self.snake_block
            self.x1_change = 0
        elif direction == Direction.DOWN and self.y1_change >= 0:
            self.y1_change = self.snake_block
            self.x1_change = 0
        elif direction == Direction.LEFT and self.x1_change <= 0:
            self.x1_change = -self.snake_block
            self.y1_change = 0
        elif direction == Direction.RIGHT and self.x1_change >= 0:
            self.x1_change = self.snake_block
            self.y1_change = 0

    def update_snake_position(self):
        self.x1 += self.x1_change
        self.y1 += self.y1_change
        
        self.x1 = self.x1 % self.width
        self.y1 = self.y1 % self.height
        
        snake_head = [self.x1, self.y1]
        self.snake_list.append(snake_head)
        if len(self.snake_list) > self.length_of_snake:
            del self.snake_list[0]
    def check_collisions(self):
        # Check for collision with food
        if abs(self.x1 - self.foodx) < self.food_size and abs(self.y1 - self.foody) < self.food_size:
            self.foodx, self.foody = self.spawn_food()
            self.length_of_snake += 1
            self.score += 10
            self.spawn_power_up()  # Chance to spawn a power-up after eating food

        # Check for collision with obstacles
        for obstacle in self.obstacles:
            if abs(self.x1 - obstacle[0]) < self.snake_block and abs(self.y1 - obstacle[1]) < self.snake_block:
                self.game_close = True

        # Check for collision with power-ups
        for power_up in self.power_ups:
            if abs(self.x1 - power_up['position'][0]) < self.food_size and abs(self.y1 - power_up['position'][1]) < self.food_size:
                self.apply_power_up(power_up['type'])
                self.power_ups.remove(power_up)
    def handle_game_over(self):
        """Handles game over screen."""
        
        # Fill screen with black
        self.display.fill(self.COLORS['RED'])
        
        # Render game over text
        font = pygame.font.SysFont("bahnschrift", 120)
        game_over_text = font.render("Game Over!", True, self.COLORS['WHITE'])
        text_rect = game_over_text.get_rect(center=(self.width / 2, self.height / 2 - 100))
        self.display.blit(game_over_text, text_rect)
        
        # Render final score
        score_font = pygame.font.SysFont("bahnschrift", 60)
        score_text = score_font.render(f"Final Score: {self.score}", True, self.COLORS['WHITE'])
        score_rect = score_text.get_rect(center=(self.width / 2, self.height / 2))
        self.display.blit(score_text, score_rect)
        
        # Render restart and quit options
        restart_font = pygame.font.SysFont("bahnschrift", 40)
        restart_text = restart_font.render("Press Space to Restart", True, self.COLORS['WHITE'])
        restart_rect = restart_text.get_rect(center=(self.width / 2, self.height / 2 + 150))
        self.display.blit(restart_text, restart_rect)
        
        quit_font = pygame.font.SysFont("bahnschrift", 40)
        quit_text = quit_font.render("Press Q to Quit", True, self.COLORS['WHITE'])
        quit_rect = quit_text.get_rect(center=(self.width / 2, self.height / 2 + 200))
        self.display.blit(quit_text, quit_rect)
        
        # Update display
        pygame.display.update()
        
        # Wait for user input
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        self.init_game_variables()
                        waiting = False
                    elif event.key == pygame.K_q:
                        pygame.quit()
                        sys.exit()

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.game_over = True
                return True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_p:  # Pause game
                    self.paused = not self.paused
                if event.key == pygame.K_q:  # Quit game
                    self.game_over = True
                    return True
        return False

# Game loop
if __name__ == "__main__":
    game = GameState()
    game.run_game()