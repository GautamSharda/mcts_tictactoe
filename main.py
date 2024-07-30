from dataclasses import dataclass
import random
from enum import Enum
from math import sqrt, log
from typing import Optional, Tuple

class Turn(Enum):
    x = 1
    o = 2

@dataclass
class State:
    board: list[int]
    turn: Turn

@dataclass
class Node:
    parent: Optional['Node']
    state: State
    simulations: int
    wins: int
    children: Optional[list['Node']] = None

@dataclass
class SearchTree:
    root: Node

def get_valid_actions(state):
    return [
        [x, y]
        for x in range(3)
        for y in range(3)
        if state.board[x][y] == 0
    ]

def apply_action(action, state):
    x, y = action
    new_board = [row[:] for row in state.board]
    new_board[x][y] = state.turn.value
    new_turn = Turn.o if state.turn == Turn.x else Turn.x
    return State(board=new_board, turn=new_turn)

def expand(node: Node):
    valid_actions = get_valid_actions(node.state)
    node.children = [Node(
            parent = node,
            state = apply_action(action, node.state),
            simulations = 0,
            wins = 0
        ) for action in valid_actions]
    return node

def check_win_conditions(state):
    lines = (
        state.board + # rows
        [list(col) for col in zip(*state.board)] +
        [[state.board[i][i] for i in range(3)],
        [state.board[i][2-i] for i in range(3)]] 
    )

    return state.turn if [state.turn.value] * 3 in lines else None


def select_child(node: Node):
    exploration_constant = sqrt(100)
    return max(
        node.children,
        key=lambda child: (child.wins / child.simulations if child.simulations else float('inf')) + 
                          exploration_constant * sqrt(log(node.simulations) / child.simulations) if child.simulations else float('inf')
    )

def mcts(search_tree: SearchTree, max_steps: int):
    root = search_tree.root
    for _ in range(max_steps):
        if not root.children:
            expand(root)
        
        curr_node = root
        while curr_node.children:
            if all(child.simulations > 0 for child in curr_node.children):
                curr_node = select_child(curr_node)
            else:
                curr_node = next(child for child in curr_node.children if child.simulations == 0) # This biases exploration to first child in order
                break
        
        if curr_node.simulations > 0:
            expand(curr_node)
            curr_node = random.choice(curr_node.children) if curr_node.children else curr_node
        
        random_rollout(curr_node)

def random_rollout(node: Node):
    state = node.state
    initial_turn = state.turn
    while True:
        winner = check_win_conditions(state)
        if winner or not get_valid_actions(state):
            break
        action = random.choice(get_valid_actions(state))
        state = apply_action(action, state)
    
    result = 1 if winner == initial_turn else 0 if winner else 0.5
    
    while node:
        node.simulations += 1
        node.wins += result
        node = node.parent

def best_action(node):
    return max(node.children, key=lambda child: child.simulations)

def play_game(tree: SearchTree):
    current_state = tree.root.state
    print("You are 'O', the AI is 'X'")
    
    def print_board(board):
        symbols = {0: ' ', 1: 'X', 2: 'O'}
        for row in board:
            print('|' + '|'.join(symbols[cell] for cell in row) + '|')
            print('-' * 7)

    def get_human_action(state):
        while True:
            try:
                x, y = map(int, input("Enter your move (row column): ").split())
                if 0 <= x < 3 and 0 <= y < 3 and state.board[x][y] == 0:
                    return [x, y]
                else:
                    print("Invalid move. Try again.")
            except ValueError:
                print("Invalid input. Please enter two numbers separated by a space.")

    while True:
        print_board(current_state.board)
        
        if current_state.turn == Turn.x: 
            if not tree.root.children:
                expand(tree.root)
            best_child = best_action(tree.root)
            for x in range(3):
                for y in range(3):
                    if tree.root.state.board[x][y] != best_child.state.board[x][y]:
                        action = [x, y]
                        break
                else:
                    continue
                break
            print(f"AI plays: {action[0]}, {action[1]}")
        else:
            action = get_human_action(current_state)
        
        current_state = apply_action(action, current_state)
        
        if tree.root.children:
            tree.root = next((child for child in tree.root.children 
                              if child.state.board == current_state.board), None)
            if tree.root is None:
                tree.root = Node(parent=None, state=current_state, simulations=0, wins=0)
            else:
                tree.root.parent = None
        else:
            tree.root = Node(parent=None, state=current_state, simulations=0, wins=0)
        
        winner = check_win_conditions(current_state)
        if winner:
            print_board(current_state.board)
            print(f"{'You win!' if winner == Turn.o else 'AI wins!'}")
            break
        elif not get_valid_actions(current_state):
            print_board(current_state.board)
            print("It's a draw!")
            break

if __name__ == "__main__":
    initial_state = State(
        board=[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
        turn=Turn.x
    )
    tree = SearchTree(
        root=Node(
            parent=None,
            state=initial_state,
            simulations=0,
            wins=0
        )
    )
    print("Training AI...")
    mcts(tree, 10000000)
    print("AI trained. Let's play!")
    play_game(tree)

