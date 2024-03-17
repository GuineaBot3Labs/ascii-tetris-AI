# ascii-tetris-AI

This is an implementation of AI inside tetris, it involves PPO learning (Post-Proximal-Optimization) to play tetris, currently, rewarding is very simple (i.e., nothing happens = -1, lines cleared = +1 * num_lines_cleared)

To use CUDA, set the CUDA variable from:

    CUDA = False

to:

    CUDA = True
