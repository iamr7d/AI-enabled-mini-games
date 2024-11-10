import pygame
import sys
from Ball import game_loop as BallGame
from RockPaperScissors import RockPaperScissors as RPSGame
from snak_and_apple import GameState as SnakeGame
from AI_Cricket import CricketGame  # Ensure CricketGame matches the main function or class name in cricket.py

# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((800, 600))
pygame.display.set_caption("AI Games Menu")
font = pygame.font.Font(None, 74)

# Button setup
button_texts = ["Ball Game", "Rock Paper Scissors", "Snake & Apple", "AI Cricket"]
button_actions = [BallGame, RPSGame, SnakeGame, CricketGame]
buttons = []

for i, text in enumerate(button_texts):
    btn_rect = pygame.Rect(200, 150 + i * 100, 400, 75)
    buttons.append((btn_rect, text, button_actions[i]))

def draw_menu():
    screen.fill((0, 0, 0))
    for btn_rect, text, _ in buttons:
        pygame.draw.rect(screen, (0, 255, 0), btn_rect)
        txt_surface = font.render(text, True, (255, 255, 255))
        screen.blit(txt_surface, (btn_rect.x + 50, btn_rect.y + 20))
    pygame.display.flip()

def main():
    while True:
        draw_menu()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                for btn_rect, _, action in buttons:
                    if btn_rect.collidepoint(event.pos):
                        if callable(action):
                            action().run()  # Run games with class structure
                        else:
                            action()  # Run function-based game directly
                            pygame.display.set_mode((800, 600))  # Reset display after exit
        pygame.display.update()

main()
