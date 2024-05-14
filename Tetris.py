import torch
import torch.nn as nn
import random
from collections import deque
import curses
import time
import numpy as np
# import pygame.mixer

# pygame.mixer.init()
# pygame.mixer.music.load('background_music.mp3')
# pygame.mixer.music.play(-1)
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
curses.init_pair(1, curses.COLOR_CYAN, curses.COLOR_BLACK)
curses.init_pair(2, curses.COLOR_YELLOW, curses.COLOR_BLACK)
curses.init_pair(3, curses.COLOR_MAGENTA, curses.COLOR_BLACK)
curses.init_pair(4, curses.COLOR_YELLOW, curses.COLOR_BLACK)
curses.init_pair(5, curses.COLOR_CYAN, curses.COLOR_BLACK)
curses.init_pair(6, curses.COLOR_GREEN, curses.COLOR_BLACK)
curses.init_pair(7, curses.COLOR_RED, curses.COLOR_BLACK)

BLOCK = "[ ]"
CUDA = True
EMPTY = " . "
grid = [[0] * WIDTH for _ in range(HEIGHT)]
current_piece = random.choice(SHAPES)
current_x = WIDTH // 2
current_y = 0
score = 0
fall_speed = 40
last_fall_speed = fall_speed
lines_cleared = 0
fall_counter = 0
next_piece = random.choice(SHAPES)  
can_swap = True
AI_PLAY = True
landing_timestamp = None
lines_cleared_at_once = 0


class TetrisModel(nn.Module):
    def __init__(self, input_shape):
        super(TetrisModel, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 512, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        
        # Calculate the size of the output from the last convolutional layer
        self.feature_dim = self._get_conv_out(input_shape)
        
        self.fc = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 7)  # Setting the output dimension to 6
        )
        
    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(o.size()[1:].numel())

    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0], -1)
        return self.fc(conv_out)


class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)


def train_model():
    if len(replay_buffer) < BATCH_SIZE:
        return
    state, action, reward, next_state, done = replay_buffer.sample(BATCH_SIZE)
    if CUDA:
        state = torch.stack(state).to('cuda:0')
        next_state = torch.stack(next_state).to('cuda:0')
        action = torch.tensor(action).to('cuda:0')
        reward = torch.tensor(reward).to('cuda:0')
        done = torch.tensor(done).to('cuda:0')
    else:
        state = torch.stack(state)
        next_state = torch.stack(next_state)
        action = torch.tensor(action)
        reward = torch.tensor(reward)
        done = torch.tensor(done)
        
    current_q = model(state).gather(1, action.unsqueeze(1))
    max_next_q = model(next_state).max(1)[0].detach()
    expected_q = reward + (1 - done) * GAMMA * max_next_q

    loss = criterion(current_q, expected_q.unsqueeze(1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss

def board_to_state(board):
    state = [[1 if cell != 0 else 0 for cell in row] for row in board]
    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0) # [32, 1, HEIGHT, WIDTH]
    return state_tensor

def choose_action(model, state, epsilon):
    actions = [0, 1, 2, 3, 4, 5, 6]

    if random.random() < epsilon:
        # Exploration: choose a random action
        return random.choice(actions)
    else:
        # Exploitation: choose the best action according to the model
        with torch.no_grad():
            if CUDA:
                q_values = model(state.to('cuda:0'))
            else:
                q_values = model(state)
            action_index = torch.argmax(q_values).item()
            return actions[action_index]

def reward_function(board, lines_cleared):
    # Simple reward constants
    background_reward = -1
    line_clear_reward = 10  # Reward per line cleared
    game_over_penalty = -100  # Large penalty for ending the game
    
    # Calculate rewards and penalties
    reward = lines_cleared * line_clear_reward
    if reward == 0:
        reward += -1
    
    # Check for game over state to apply a significant penalty
    if is_game_over():
        reward += game_over_penalty
    
    return reward



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
    for row in range(len(shape)):
        for col in range(len(shape[row])):
            if shape[row][col] == 1:
                grid[y + row][x + col] = piece_id

def flash_game_over():
    pygame.mixer.music.stop()
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

    if lines_to_clear:
        lines_cleared_at_once = len(lines_to_clear)
        lines_cleared += len(lines_to_clear)
        score += len(lines_to_clear)
        flash_lines(lines_to_clear)
        fall_speed -= 10
    else:
        lines_cleared_at_once = 0
    for line in lines_to_clear:
        del grid[line]
        grid.insert(0, [0] * WIDTH)
        
        lines_cleared_at_once = 0 if lines_cleared_at_once == None else lines_cleared_at_once
        
    return lines_cleared_at_once

def is_game_over():
    middle = WIDTH // 2
    return grid[0][middle-1] or grid[0][middle] or grid[0][middle+1]

def generate_piece():
    global next_piece
    current_piece = next_piece
    next_piece = random.choice(SHAPES)
    return current_piece

def swap_piece():
    global current_piece, next_piece, current_x, current_y, can_swap
    if can_swap:
        current_piece, next_piece = next_piece, current_piece
        current_x = (WIDTH - len(current_piece[1][0])) // 2  # Reset to the top-middle position
        current_y = 0  # Reset to the top of the grid
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
                draw_y = y + row
                draw_x = (x + col) * len(BLOCK)
                if 0 <= draw_y < sh and 0 <= draw_x < sw:  # Check if within window
                    w.attron(curses.color_pair(piece_id))
                    w.addstr(draw_y, draw_x, BLOCK)
                    w.attroff(curses.color_pair(piece_id))


if AI_PLAY == False:
    speed_up_counter = 0
    LOCK_DELAY_THRESHOLD = 3
    lock_delay_counter = 0
    last_move_time = time.time()  # Initialize the last move time to the current time
    fall_counter = 0

    while True:
        key = w.getch()

        # Update the last move time whenever a key is pressed
        if key in [curses.KEY_RIGHT, curses.KEY_LEFT, curses.KEY_DOWN, curses.KEY_UP, ord('s'), ord(' ')]:
            last_move_time = time.time()
    
        if key == ord('q'):
            pygame.mixer.music.stop()
            break
        elif key == curses.KEY_RIGHT:
            if can_place(current_piece, current_x + 1, current_y):
                current_x += 1
        elif key == curses.KEY_LEFT:
            if can_place(current_piece, current_x - 1, current_y):
                current_x -= 1
        elif key == curses.KEY_UP:
            rotated_piece = rotate_piece(current_piece)
            if can_place(rotated_piece, current_x, current_y):
                current_piece = rotated_piece
        elif key == curses.KEY_DOWN:
            fall_speed += -90
            speed_up_counter = 0
            fall_speed += 90
        elif key == ord('s'):
            swap_piece()
        elif key == ord(' '):
            while can_place(current_piece, current_x, current_y + 1):
                current_y += 1
            fall_speed = last_fall_speed


        if fall_counter >= fall_speed / 10:
            if can_place(current_piece, current_x, current_y + 1):
                current_y += 1
                lock_delay_counter = 0
            else:
                lock_delay_counter += 1
                if lock_delay_counter >= LOCK_DELAY_THRESHOLD or (time.time() - last_move_time > 0.5):  # Check if 1 second has passed since the last move
                    place_piece(current_piece, current_x, current_y)
                    lines_cleared = clear_lines()
                    reward_function(grid, lines_cleared)
                    current_piece = None
                    lock_delay_counter = 0
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
        w.refresh()
        fall_counter += 1

        if key != curses.KEY_DOWN:
            fall_speed = last_fall_speed

else:
    # Hyperparameters
    EPISODES = 10000
    BATCH_SIZE = 64
    LEARNING_RATE = 0.2
    GAMMA = 0.95
    EPSILON_START = 1.0
    EPSILON_END = 0.01
    EPSILON_DECAY = 0.995
    model = TetrisModel((1, HEIGHT, WIDTH))
    try:
        model.load_state_dict(torch.load('model_state_dict.pth'))
    except Exception as e:
        print(f"Tried to load model, failed: {e}")
    if CUDA:
        model = model.to('cuda:0')
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    replay_buffer = ReplayBuffer()
    epsilon = EPSILON_START
    for i in range(EPISODES):



        # Reset the game state
        grid = [[0] * WIDTH for _ in range(HEIGHT)]
        current_piece = random.choice(SHAPES)
        current_x = WIDTH // 2
        current_y = 0
        score = 0
        reward = 0
        fall_speed = 100
        last_fall_speed = fall_speed
        lines_cleared = 0
        fall_counter = 0
        next_piece = random.choice(SHAPES)
        can_swap = True
        last_lines_cleared = 0
        speed_up_counter = 0
        LOCK_DELAY_THRESHOLD = 3
        lock_delay_counter = 0
        last_move_time = time.time()  # Initialize the last move time to the current time

        while True:
            state = board_to_state(grid)
            key = choose_action(model, state, epsilon)


            # Update the last move time whenever a key is pressed
            if key in [5, 3, 2, 1]:
                last_move_time = time.time()
    

            if key == 0: # Go right
                if can_place(current_piece, current_x + 1, current_y):
                    current_x += 1
            elif key == 1: # Go left
                if can_place(current_piece, current_x - 1, current_y):
                    current_x -= 1
            elif key == 2: # Rotate piece
                rotated_piece = rotate_piece(current_piece)
                if can_place(rotated_piece, current_x, current_y):
                    current_piece = rotated_piece
            elif key == 3: # Soft-drop
                fall_speed += -90
                speed_up_counter = 0
                fall_speed += 90
            elif key == 4: # Switch piece
                swap_piece()
            elif key == 5: # Hard-drop
                while can_place(current_piece, current_x, current_y + 1):
                    current_y += 1
                fall_speed = last_fall_speed

            # if fall_counter >= fall_speed / 10:
            if 1 == 1:
                if can_place(current_piece, current_x, current_y + 1):
                    current_y += 1
                    lock_delay_counter = 0
                else:
                    lock_delay_counter += 1
                    if lock_delay_counter >= LOCK_DELAY_THRESHOLD or (time.time() - last_move_time > 0.5):  # Check if 1 second has passed since the last move
                        place_piece(current_piece, current_x, current_y)
                        lines_cleared = clear_lines()
                        reward = reward_function(grid, lines_cleared)
                        current_piece = None
                        lock_delay_counter = 0
                        # Generate a new piece
                        current_piece = generate_piece()
                        current_x = (WIDTH - len(current_piece[1][0])) // 2  # Center the piece
                        current_y = 0
                        can_swap = True
                    fall_counter = 0

                if key in [6, 5, 3]:
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
                # w.addstr(sh - 1, 0, "Score: " + str(score))
                w.addstr(sh - 1, 0, "Reward: " + str(reward))
                w.refresh()
                fall_counter += 1

                if key != curses.KEY_DOWN:
                    fall_speed = last_fall_speed

                reward = reward_function(grid, lines_cleared)

                last_lines_cleared = lines_cleared

                # Store this experience in the replay buffer
                next_state = board_to_state(grid)
                replay_buffer.push(state, key, reward, next_state, is_game_over())

                # Train the model
                loss = train_model()
                w.addstr(sh - 2, 0, "Loss: " + str(loss))
                w.refresh()

                if is_game_over():
                    epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
                    train_model()

                    break
    torch.save(model.state_dict(), 'model_state_dict.pth')
    print("Saved model in 'model_state_dict.pth'")
