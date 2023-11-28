import random
import curses
import time
import pygame.mixer
import os
import sys

def resource_path(relative_path): # usage: LANDING = pygame.mixer.Sound(resource_path('landing.mp3'))
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

pygame.mixer.init()

LANDING = pygame.mixer.Sound('landing.mp3')
LINECLEAR = pygame.mixer.Sound('lineclear.mp3')
GAMEOVER = pygame.mixer.Sound('gameover.mp3')
ROTATE = pygame.mixer.Sound('rotate.mp3')
MOVE = pygame.mixer.Sound('move.mp3')
pygame.mixer.music.load(resource_path('background_music.mp3')) # comment these lines for compilation

'''
# for compilation using pyinstaller: pyinstaller  --add-data "landing.mp3;." --add-data "lineclear.mp3;." --add-data "gameover.mp3;." --add-data "rotate.mp3;." --add-data "move.mp3;." --add-data "background_music.mp3;." Tetris.py --onefile
# for linux replace the ';' with ':'


LANDING = pygame.mixer.Sound(resource_path('landing.mp3'))
LINECLEAR = pygame.mixer.Sound(resource_path('lineclear.mp3'))
GAMEOVER = pygame.mixer.Sound(resource_path('gameover.mp3'))
ROTATE = pygame.mixer.Sound(resource_path('rotate.mp3'))
MOVE = pygame.mixer.Sound(resource_path('move.mp3'))
pygame.mixer.music.load(resource_path('background_music.mp3')) # Uncomment these lines for compilation
'''

pygame.mixer.music.play(-1)
paused = False
stdscr = curses.initscr()
curses.start_color()
curses.curs_set(0)
sh, sw = stdscr.getmaxyx()
w = stdscr.subwin(sh, sw, 0, 0)
w.keypad(1)
w.timeout(100)

WIDTH = 10
HEIGHT = 20
SHAPES = [
    (1, [[1, 1, 1, 1]]), 
    (2, [[1, 1], [1, 1]]), 
    (3, [[1, 1, 1], [0, 1, 0]]), 
    (4, [[1, 1, 1], [1, 0, 0]]), 
    (5, [[1, 1, 1], [0, 0, 1]]), 
    (6, [[1, 1, 0], [0, 1, 1]]), 
    (7, [[0, 1, 1], [1, 1, 0]])
]
# Rotation points (row, col) relative to the shape's top-left corner
ROTATION_POINTS = {
    1: (1, 2),  # I-piece
    2: (0, 0),  # O-piece (doesn't rotate)
    3: (1, 1),  # T-piece
    4: (1, 1),  # J-piece
    5: (1, 1),  # L-piece
    6: (0, 1),  # S-piece
    7: (0, 1)   # Z-piece
}

curses.init_pair(1, curses.COLOR_CYAN, curses.COLOR_BLACK)
curses.init_pair(2, curses.COLOR_YELLOW, curses.COLOR_BLACK)
curses.init_pair(3, curses.COLOR_MAGENTA, curses.COLOR_BLACK)
curses.init_pair(4, 202, curses.COLOR_BLACK)
curses.init_pair(5, curses.COLOR_BLUE, curses.COLOR_BLACK)
curses.init_pair(6, curses.COLOR_GREEN, curses.COLOR_BLACK)
curses.init_pair(7, curses.COLOR_RED, curses.COLOR_BLACK)

BLOCK = "[ ]"
EMPTY = " . "
grid = [[0] * WIDTH for _ in range(HEIGHT)]
current_piece = random.choice(SHAPES)
current_x = (WIDTH - len(current_piece[1][0])) // 2  # Center the piece
current_y = 0
score = 0
fall_speed = 70  
last_fall_speed = fall_speed
lines_cleared = 0
fall_counter = 0
next_piece = random.choice(SHAPES)  
can_swap = True

landing_timestamp = None
lines_cleared_at_once = 0

def can_place(piece, x, y):
    _, shape = piece
    for row in range(len(shape)):
        for col in range(len(shape[row])):
            if shape[row][col] == 1:
                if (
                    x + col < 0
                    or x + col >= WIDTH
                    or y + row >= HEIGHT
                    or grid[y + row][x + col] != 0
                ):
                    return False
    return True


def place_piece(piece, x, y):
    piece_id, shape = piece
    LANDING.play()
    for row in range(len(shape)):
        for col in range(len(shape[row])):
            if shape[row][col] == 1:
                grid[y + row][x + col] = piece_id

def flash_game_over():
    pygame.mixer.music.stop()
    GAMEOVER.play()
    for y in range(0, HEIGHT, 2):  # Iterate over lines two at a time
        # Flash the lines
        for line in range(y, min(y + 2, HEIGHT)):
            w.move(line, 0)
            for x in range(WIDTH):
                w.addstr(BLOCK, curses.color_pair(0))  # Set to default color
        w.refresh()
        time.sleep(0.1)  # Short pause for the flash effect

        # Clear the lines
        for line in range(y, min(y + 2, HEIGHT)):
            w.move(line, 0)
            for x in range(WIDTH):
                w.addstr("   ", curses.color_pair(0))  # Clear the blocks
        w.refresh()
        time.sleep(0.01)  # Another short pause



def flash_lines(lines_to_flash):
    LINECLEAR.play()
    for _ in range(3):  # Flash 3 times
        for y in lines_to_flash:
            w.move(y, 0)
            for x in range(WIDTH):
                w.addstr(BLOCK, curses.color_pair(0))  # Set to default color (usually white on black)
                
        w.refresh()
        time.sleep(0.1)  # Pause for 100ms
       
        for y in lines_to_flash:
            w.move(y, 0)
            for x in range(WIDTH):
                w.addstr("   ", curses.color_pair(0))  # Clear the blocks
        w.refresh()
        time.sleep(0.1)  # Pause for 100ms


def clear_lines():
    global score, lines_cleared, fall_speed, lines_cleared_at_once

    lines_to_clear = [i for i, row in enumerate(grid) if all(cell != 0 for cell in row)]
    lines_cleared_at_once = len(lines_to_clear)

    if lines_cleared_at_once > 0:
        lines_cleared += lines_cleared_at_once
        flash_lines(lines_to_clear)

        # Basic difficulty adjustment based on lines cleared
        fall_speed_adjustment = 5 + 5 * lines_cleared_at_once
        fall_speed -= fall_speed_adjustment

        # Additional adjustment based on grid density
        sponge = calculate_density()
        if sponge > 0.7:  # Adjust threshold as needed
            score += lines_cleared_at_once * 100  # Adjust scoring as needed
            fall_speed += 7 * lines_cleared_at_once # Harder if the grid is less spongy
        elif sponge < 0.1:  # Adjust threshold as needed
            fall_speed += -4 * lines_cleared_at_once * 0.1 # Easier if the grid is spongier
            score += lines_cleared_at_once * 4000  # Adjust scoring as needed

        else:
            fall_speed += 2 * lines_cleared_at_once
        score += lines_cleared_at_once * 2000  # Adjust scoring as needed


        for line in lines_to_clear:
            del grid[line]
            grid.insert(0, [0] * WIDTH)
    else:
        lines_cleared_at_once = 0

def calculate_density():
    spongy_spaces = 0
    total_spongy_checkable_spaces = 0

    # Iterate through each cell, except the bottom row, as nothing is beneath it
    for y in range(HEIGHT - 1):
        for x in range(WIDTH):
            if grid[y][x] != 0:  # If the current block is filled
                total_spongy_checkable_spaces += 1
                if grid[y + 1][x] == 0:  # Check if the space directly beneath is empty
                    spongy_spaces += 1

    # Avoid division by zero
    if total_spongy_checkable_spaces == 0:
        return 0

    return spongy_spaces / total_spongy_checkable_spaces



def is_game_over():
    middle = WIDTH // 2
    return grid[0][middle-1] or grid[0][middle] or grid[0][middle+1]

def generate_piece():
    global next_piece
    current_piece = next_piece
    next_piece = random.choice(SHAPES)
    return current_piece

def swap_piece():
    global current_piece, next_piece, can_swap, current_x, current_y
    if can_swap:
        current_piece, next_piece = next_piece, current_piece
        current_x = (WIDTH - len(current_piece[1][0])) // 2  # Center the piece
        current_y = 0
        can_swap = False

def draw_next_piece(piece, start_x, start_y):
    piece_id, shape = piece

    # Get the dimensions of the shape
    shape_height = len(shape)
    shape_width = len(shape[0])

    # Calculate the centered starting positions
    center_y = (7 - shape_height) // 2
    center_x = (7 - shape_width) // 2

    for y in range(7):
        for x in range(7):
            if center_y <= y < center_y + shape_height and center_x <= x < center_x + shape_width and shape[y - center_y][x - center_x]:
                w.attron(curses.color_pair(piece_id))
                w.addstr(start_y + y, start_x + x * len(BLOCK), BLOCK, curses.color_pair(piece_id))
                w.attroff(curses.color_pair(piece_id))
            else:
                w.addstr(start_y + y, start_x + x * len(BLOCK), EMPTY)
    w.refresh()


def rotate_piece(piece):
    global current_x, current_y  # Access the global variables directly

    _, shape = piece

    if piece is None:
        return piece

    rotated_shape = [list(row) for row in zip(*reversed(shape))]

    # Kick data for Tetris, where each tuple represents (dx, dy)
    # This will try current, left, right, up (for floor kick)
    kicks = [(0, 0), (-1, 0), (1, 0), (0, -1)]

    for dx, dy in kicks:
        if can_place((_, rotated_shape), current_x + dx, current_y + dy):
            # Adjust the piece's x and y position directly
            current_x += dx  
            current_y += dy  
            return (_, rotated_shape)  # return the updated piece

    return piece  # If no adjustment works, return the original piece


def draw_piece(piece, x, y):
    piece_id, shape = piece
    for row in range(len(shape)):
        for col in range(len(shape[row])):
            if shape[row][col] == 1:
                w.attron(curses.color_pair(piece_id))
                w.addstr(y + row, (x + col) * len(BLOCK), BLOCK)
                w.attroff(curses.color_pair(piece_id))

speed_up_counter = 0
LOCK_DELAY_THRESHOLD = 1
lock_delay_counter = 0
last_move_time = time.time()  # Initialize the last move time to the current time
last_fall_time = time.time()

while True:
    current_time = time.time()  
    key = w.getch()

    # Update the last move time whenever a key is pressed
    if key in [curses.KEY_RIGHT, curses.KEY_LEFT, curses.KEY_DOWN, curses.KEY_UP]:
        last_move_time = time.time()

    if key == ord('p'):
        paused = not paused

    if paused:
        # Display a pause message
        pause_msg = "PAUSED"
        w.addstr(sh // 2, (sw - len(pause_msg)) // 2, pause_msg)
        w.refresh()
        continue  # Skip the rest of the loop
        
    if key == ord('q'):
        pygame.mixer.music.stop()
        break
    elif key == curses.KEY_RIGHT:
        if can_place(current_piece, current_x + 1, current_y):
            MOVE.play()
            current_x += 1

    elif key == ord('s'):
        swap_piece()
    elif key == curses.KEY_LEFT:
        if can_place(current_piece, current_x - 1, current_y):
            MOVE.play()
            current_x -= 1

    elif key == curses.KEY_DOWN:
        if can_place(current_piece, current_x, current_y + 1):
            MOVE.play()
            current_y += 1
    elif key == ord(' '):
        while can_place(current_piece, current_x, current_y + 1):
            current_y += 1
    elif key == curses.KEY_UP:
        rotated_piece = rotate_piece(current_piece)
        if can_place(rotated_piece, current_x, current_y):
            ROTATE.play()
            current_piece = rotated_piece

    if current_time - last_fall_time >= fall_speed / 1000.0:  # Assuming fall_speed is in milliseconds
        fall_counter += 1
        last_fall_time = current_time

    if fall_counter >= fall_speed / 10:
        if can_place(current_piece, current_x, current_y + 1):
            current_y += 1
            lock_delay_counter = 0
        else:
            lock_delay_counter += 1
            if lock_delay_counter >= LOCK_DELAY_THRESHOLD or (time.time() - last_move_time > 0.5):  # Check if 1 second has passed since the last move
                place_piece(current_piece, current_x, current_y)
                clear_lines()
                current_piece = None
                lock_delay_counter = 0
                if is_game_over():
                    flash_game_over()
                    curses.endwin()
                    # pygame.mixer.music.stop()
                    
                    print("Game Over. Your Score:", score)
                    break
                
                
                # Generate a new piece
                current_piece = generate_piece()
                current_x = (WIDTH - len(current_piece[1][0])) // 2  # Center the piece
                current_y = 0
                can_swap = True
        fall_counter = 0

    if key in [curses.KEY_RIGHT, curses.KEY_LEFT, curses.KEY_UP]:
        lock_delay_counter = 0

    w.clear()

    for y, row in enumerate(grid):
        for x, cell in enumerate(row):
            if cell != 0:
                color_code = cell
                w.attron(curses.color_pair(color_code))
                w.addstr(y, x * len(BLOCK), BLOCK)
                w.attroff(curses.color_pair(color_code))
            else:
                w.addstr(y, x * len(BLOCK), EMPTY)

    if current_piece is not None:
        draw_piece(current_piece, current_x, current_y)

    next_piece_label_x = (WIDTH + 2) * len(BLOCK)
    next_piece_label_y = 2
    w.addstr(next_piece_label_y, next_piece_label_x, "Next Piece:")
    next_piece_display_x = next_piece_label_x
    next_piece_display_y = next_piece_label_y + 2
    draw_next_piece(next_piece, next_piece_display_x, next_piece_display_y)
    w.addstr(sh - 1, 0, "Score: " + str(score))
    # w.addstr(sh - 2, 0, "Sponginess: " + str(calculate_density())) # TODO: Remove debug prints
    w.refresh()
