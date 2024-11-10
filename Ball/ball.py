import pygame
import cv2
import mediapipe as mp
import random
import sys

# Initialize Pygame and MediaPipe
pygame.init()
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

# Screen dimensions
SCREEN_WIDTH, SCREEN_HEIGHT = 800, 600
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Hand-Tracking Game")

# Colors and game settings
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
PLAYER_COLOR = (0, 255, 0)  # Green color for the player
OBSTACLE_SIZE = 30
OBSTACLE_SPEED = 5
PLAYER_WIDTH = 60
PLAYER_HEIGHT = 10
FPS = 30

# Load and play background music
pygame.mixer.music.load(r"C:\Users\rahulrajpvr7d\Music\8-bit-retro-game-music-233964.mp3")  # Replace with your music file
pygame.mixer.music.set_volume(0.5)

# Function to generate a random color
def random_color():
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

# Function to generate a gradient for the obstacle box
def generate_gradient_box(surface, width, height):
    # Generate two random colors for the gradient
    color1 = random_color()
    color2 = random_color()
    
    for i in range(height):
        # Calculate the blend ratio
        ratio = i / height
        # Blend the two colors based on the ratio
        r = int(color1[0] * (1 - ratio) + color2[0] * ratio)
        g = int(color1[1] * (1 - ratio) + color2[1] * ratio)
        b = int(color1[2] * (1 - ratio) + color2[2] * ratio)
        # Draw a line of the gradient at the i-th row
        pygame.draw.line(surface, (r, g, b), (0, i), (width, i))

# Player class
class Player:
    def __init__(self):
        self.x = SCREEN_WIDTH // 2
        self.y = SCREEN_HEIGHT - 40
        self.width = PLAYER_WIDTH
        self.height = PLAYER_HEIGHT

    def draw(self):
        pygame.draw.rect(screen, PLAYER_COLOR, (self.x, self.y, self.width, self.height))

# Obstacle class
class Obstacle:
    def __init__(self, x):
        self.x = x
        self.y = 0
        self.size = OBSTACLE_SIZE
        self.surface = pygame.Surface((self.size, self.size))
        generate_gradient_box(self.surface, self.size, self.size)  # Generate gradient for the obstacle

    def move(self):
        self.y += OBSTACLE_SPEED

    def draw(self):
        screen.blit(self.surface, (self.x, self.y))  # Draw the obstacle with the gradient

    def check_collision(self, player):
        return (self.y + self.size >= player.y and self.x < player.x + player.width and self.x + self.size > player.x)

# Hand detection function
def detect_hand_position(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    hand_position = None
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            x = int(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x * SCREEN_WIDTH)
            hand_position = x
            break
    return hand_position

# Display title screen
def show_title_screen():
    screen.fill(BLACK)
    font = pygame.font.Font(None, 74)
    title_text = font.render("Hand-Tracking Game", True, WHITE)
    screen.blit(title_text, (SCREEN_WIDTH // 2 - title_text.get_width() // 2, SCREEN_HEIGHT // 2 - 50))
    
    font_small = pygame.font.Font(None, 36)
    instruction_text = font_small.render("Use your hand to control the player. Press SPACE to start.", True, WHITE)
    screen.blit(instruction_text, (SCREEN_WIDTH // 2 - instruction_text.get_width() // 2, SCREEN_HEIGHT // 2 + 50))
    
    pygame.display.flip()
    wait_for_key_press()

# Wait for a specific key press
def wait_for_key_press():
    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                waiting = False

# Display game over screen
def show_game_over_screen(score):
    screen.fill(BLACK)
    font = pygame.font.Font(None, 74)
    game_over_text = font.render("Game Over", True, WHITE)
    screen.blit(game_over_text, (SCREEN_WIDTH // 2 - game_over_text.get_width() // 2, SCREEN_HEIGHT // 2 - 50))
    
    font_small = pygame.font.Font(None, 36)
    score_text = font_small.render(f"Your Score: {score}", True, WHITE)
    screen.blit(score_text, (SCREEN_WIDTH // 2 - score_text.get_width() // 2, SCREEN_HEIGHT // 2 + 20))
    
    restart_text = font_small.render("Press SPACE to restart", True, WHITE)
    screen.blit(restart_text, (SCREEN_WIDTH // 2 - restart_text.get_width() // 2, SCREEN_HEIGHT // 2 + 60))
    
    pygame.display.flip()
    wait_for_key_press()

# Main game loop
def game_loop():
    pygame.mixer.music.play(-1)  # Loop background music
    
    clock = pygame.time.Clock()
    cap = cv2.VideoCapture(0)
    player = Player()
    obstacles = []
    score = 0
    running = True
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        ret, frame = cap.read()
        if not ret:
            break
        hand_position = detect_hand_position(frame)

        if hand_position is not None:
            player.x = max(0, min(hand_position - player.width // 2, SCREEN_WIDTH - player.width))

        if random.randint(0, 20) == 0:
            obstacles.append(Obstacle(random.randint(0, SCREEN_WIDTH - OBSTACLE_SIZE)))

        screen.fill(BLACK)
        for obstacle in obstacles[:]:
            obstacle.move()
            obstacle.draw()
            if obstacle.check_collision(player):
                pygame.mixer.music.stop()
                show_game_over_screen(score)
                return
            if obstacle.y > SCREEN_HEIGHT:
                obstacles.remove(obstacle)
                score += 1

        player.draw()

        font = pygame.font.Font(None, 36)
        score_text = font.render(f"Score: {score}", True, WHITE)
        screen.blit(score_text, (20, 20))

        pygame.display.flip()
        clock.tick(FPS)

    cap.release()
    pygame.quit()
    sys.exit()

# Run the game
show_title_screen()
while True:
    game_loop()
# Only run when executed directly
if __name__ == "__main__":
    show_title_screen()  # Optional: show title screen before starting game loop
    while True:
        game_loop()