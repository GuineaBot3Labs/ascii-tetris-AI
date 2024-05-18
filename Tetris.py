import torch
import torch.nn as nn
import random
from collections import deque
import curses
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

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
fall_speed = 100
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
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
        )
        
        self.feature_dim = self._get_conv_out((1, HEIGHT + 4, WIDTH))
        
        self.fc = nn.Sequential(
            nn.Linear(self.feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 20),
            nn.ReLU(),
            nn.Linear(20, 7),
            nn.ReLU(),
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

def update_model(state, action, reward, next_state, done):
    state = state.unsqueeze(0)
    next_state = next_state.unsqueeze(0)
    action = torch.tensor([action])
    reward = torch.tensor([reward])
    done = torch.tensor([done], dtype=torch.float32)

    if CUDA:
        state = state.to('cuda:0')
        next_state = next_state.to('cuda:0')
        action = action.to('cuda:0')
        reward = reward.to('cuda:0')
        done = done.to('cuda:0')
        
    current_q = model(state).gather(1, action.unsqueeze(1))
    max_next_q = model(next_state).max(1)[0].detach()
    expected_q = reward + (1 - done) * GAMMA * max_next_q

    loss = criterion(current_q, expected_q.unsqueeze(1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss

def train_model():
    if len(replay_buffer) < BATCH_SIZE:
        return
    state, action, reward, next_state, done = replay_buffer.sample(BATCH_SIZE)
    
    state = torch.stack(state)
    next_state = torch.stack(next_state)
    action = torch.tensor(action)
    reward = torch.tensor(reward)
    done = torch.tensor(done, dtype=torch.float32)

    if CUDA:
        state = state.to('cuda:0')
        next_state = next_state.to('cuda:0')
        action = action.to('cuda:0')
        reward = reward.to('cuda:0')
        done = done.to('cuda:0')
        
    current_q = model(state).gather(1, action.unsqueeze(1))
    max_next_q = model(next_state).max(1)[0].detach()
    expected_q = reward + (1 - done) * GAMMA * max_next_q

    loss = criterion(current_q, expected_q.unsqueeze(1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss

def board_to_state(board, next_piece):
    state_flattened = [1 if cell != 0 else 0 for row in board for cell in row]
    
    _, next_shape = next_piece
    next_piece_grid = [[0] * 4 for _ in range(4)]
    for i in range(len(next_shape)):
        for j in range(len(next_shape[i])):
            if next_shape[i][j] == 1:
                next_piece_grid[i][j] = 1
    next_piece_flattened = [cell for row in next_piece_grid for cell in row]
    
    combined_state = state_flattened + next_piece_flattened
    
    target_size = (HEIGHT + 4) * WIDTH
    padding = [0] * (target_size - len(combined_state))
    padded_combined_state = combined_state + padding
    
    assert len(padded_combined_state) == target_size, f"Expected padded_combined_state to have {target_size} elements, but got {len(padded_combined_state)}"

    state_tensor = torch.tensor(padded_combined_state, dtype=torch.float32).view(1, HEIGHT + 4, WIDTH)
    
    return state_tensor

def choose_action(model, state, epsilon):
    actions = [0, 1, 2, 3, 4, 5, 6]

    if random.random() < epsilon:
        return random.choice(actions)
    else:
        with torch.no_grad():
            if CUDA:
                q_values = model(state.to('cuda:0'))
            else:
                q_values = model(state)
            action_index = torch.argmax(q_values).item()
            return actions[action_index]

def reward_function(grid, lines_cleared):
    line_clear_reward = 10.0  # Reward per line cleared
    max_height_penalty_factor = 0.1  # Penalty factor for height
    compactness_bonus_factor = 0.3  # Max bonus factor for compactness, could be less based on actual compactness

    # Initialize metrics
    filled_cells = 0
    min_x, max_x = WIDTH, -1
    min_y, max_y = HEIGHT, -1
    max_height = 0

    # Calculate filled cells, bounding rectangle, and max height
    for y in range(HEIGHT):
        for x in range(WIDTH):
            if grid[y][x] != 0:
                filled_cells += 1
                min_x = min(min_x, x)
                max_x = max(max_x, x)
                min_y = min(min_y, y)
                max_y = max(max_y, y)
                current_height = HEIGHT - y
                if current_height > max_height:
                    max_height = current_height

    # Compactness and area calculations
    compactness = 0
    if filled_cells > 0:
        bounding_area = (max_x - min_x + 1) * (max_y - min_y + 1)
        compactness = filled_cells / bounding_area
        compactness = min(compactness, 1.0)  # Ensure compactness does not exceed 1

    # Calculate reward components
    line_clear_component = lines_cleared * line_clear_reward
    height_penalty_component = (max_height / HEIGHT) * max_height_penalty_factor
    compactness_component = compactness * compactness_bonus_factor

    # Combine components to get the final reward
    reward = line_clear_component + compactness_component - height_penalty_component

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
    for y in range(0, HEIGHT, 2):
        for line in range(y, min(y + 2, HEIGHT)):
            w.move(line, 0)
            for x in range(WIDTH):
                w.addstr(BLOCK, curses.color_pair(0))
        w.refresh()
        time.sleep(0.1)

        for line in range(y, min(y + 2, HEIGHT)):
            w.move(line, 0)
            for x in range(WIDTH):
                w.addstr("   ", curses.color_pair(0))
        w.refresh()
        time.sleep(0.01)

def flash_lines(lines_to_flash):
    for _ in range(3):
        for y in lines_to_flash:
            w.move(y, 0)
            for x in range(WIDTH):
                w.addstr(BLOCK, curses.color_pair(0))
        w.refresh()
        time.sleep(0.1)
       
        for y in lines_to_flash:
            w.move(y, 0)
            for x in range(WIDTH):
                w.addstr("   ", curses.color_pair(0))
        w.refresh()
        time.sleep(0.1)

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
        current_x = (WIDTH - len(current_piece[1][0])) // 2
        current_y = 0
        can_swap = False

def draw_next_piece(piece, start_x, start_y):
    piece_id, shape = piece

    shape_height = len(shape)
    shape_width = len(shape[0])

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
    global current_x, current_y

    _, shape = piece

    if piece is None:
        return piece

    rotated_shape = [list(row) for row in zip(*reversed(shape))]

    kicks = [(0, 0), (-1, 0), (1, 0), (0, -1)]

    for dx, dy in kicks:
        if can_place((_, rotated_shape), current_x + dx, current_y + dy):
            current_x += dx  
            current_y += dy  
            return (_, rotated_shape)

    return piece

def draw_piece(piece, x, y):
    piece_id, shape = piece
    for row in range(len(shape)):
        for col in range(len(shape[row])):
            if shape[row][col] == 1:
                draw_y = y + row
                draw_x = (x + col) * len(BLOCK)
                if 0 <= draw_y < sh and 0 <= draw_x < sw:
                    w.attron(curses.color_pair(piece_id))
                    w.addstr(draw_y, draw_x, BLOCK)
                    w.attroff(curses.color_pair(piece_id))

if AI_PLAY == False:
    speed_up_counter = 0
    LOCK_DELAY_THRESHOLD = 3
    lock_delay_counter = 0
    last_move_time = time.time()
    fall_counter = 0

    while True:
        key = w.getch()

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
                if lock_delay_counter >= LOCK_DELAY_THRESHOLD or (time.time() - last_move_time > 0.5):
                    place_piece(current_piece, current_x, current_y)
                    lines_cleared = clear_lines()
                    reward_function(grid, lines_cleared)
                    current_piece = None
                    lock_delay_counter = 0
                    current_piece = generate_piece()
                    current_x = (WIDTH - len(current_piece[1][0])) // 2
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
    EPISODES = 10000
    BATCH_SIZE = 270
    LEARNING_RATE = 0.01
    GAMMA = 0.96
    EPSILON_START = 1.0
    EPSILON_END = 0.01
    EPSILON_DECAY = 0.997
    model = TetrisModel((1, HEIGHT + 4, WIDTH))
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

    losses = []
    fig, ax = plt.subplots()
    ax.set_title('Real-time Loss Plot')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Loss')
    line, = ax.plot([], [], 'r-')

    def update_plot(frame):
        line.set_data(range(len(losses)), losses)
        ax.relim()
        ax.autoscale_view()
        return line,

    ani = FuncAnimation(fig, update_plot, blit=True)
    plt.show(block=False)

    for i in range(EPISODES):
        if i % 100 == 0:
            torch.save(model.state_dict(), f'model_checkpoint_{i}.pth')
            print(f"Model saved at episode {i}")
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
        last_move_time = time.time()

        while True:
            state = board_to_state(grid, next_piece)
            key = choose_action(model, state, epsilon)

            if key in [5, 3, 2, 1]:
                last_move_time = time.time()
    
            if key == 0:
                if can_place(current_piece, current_x + 1, current_y):
                    current_x += 1
            elif key == 1:
                if can_place(current_piece, current_x - 1, current_y):
                    current_x -= 1
            elif key == 2:
                rotated_piece = rotate_piece(current_piece)
                if can_place(rotated_piece, current_x, current_y):
                    current_piece = rotated_piece
            elif key == 3:
                fall_speed += -90
                speed_up_counter = 0
                fall_speed += 90
            elif key == 4:
                swap_piece()
            elif key == 5:
                while can_place(current_piece, current_x, current_y + 1):
                    current_y += 1
                fall_speed = last_fall_speed

            if 1 == 1:
                if can_place(current_piece, current_x, current_y + 1):
                    current_y += 1
                    lock_delay_counter = 0
                else:
                    lock_delay_counter += 1
                    if lock_delay_counter >= LOCK_DELAY_THRESHOLD or (time.time() - last_move_time > 0.5):
                        place_piece(current_piece, current_x, current_y)
                        lines_cleared = clear_lines()
                        reward = reward_function(grid, lines_cleared)
                        current_piece = None
                        lock_delay_counter = 0
                        current_piece = generate_piece()
                        current_x = (WIDTH - len(current_piece[1][0])) // 2
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
                w.addstr(sh - 1, 0, "Reward: " + str(reward))
                w.refresh()
                fall_counter += 1

                if key != 3:
                    fall_speed = last_fall_speed

                reward = reward_function(grid, lines_cleared)

                last_lines_cleared = lines_cleared

                next_state = board_to_state(grid, next_piece)
                replay_buffer.push(state, key, reward, next_state, is_game_over())

                loss = update_model(state, key, reward, next_state, is_game_over())
                losses.append(loss.item())
                line.set_data(range(len(losses)), losses)
                plt.draw()
                plt.pause(0.01)

                if is_game_over():
                    epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
                    train_model()
                    break

    torch.save(model.state_dict(), 'model_state_dict.pth')
    print("Saved model in 'model_state_dict.pth'")
