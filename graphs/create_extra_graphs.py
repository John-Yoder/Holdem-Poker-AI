import json
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

# --- CONFIGURATION ---
DATA_FILE = 'poker_data.jsonl'

# 1. LOAD THE DATA
data = []
try:
    with open(DATA_FILE, 'r') as f:
        for line_num, line in enumerate(f):
            if line.strip():
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    print(f"Skipping invalid JSON on line {line_num + 1}")
    print(f"Successfully loaded {len(data)} hands from {DATA_FILE}.")
except FileNotFoundError:
    print(f"Error: Could not find '{DATA_FILE}'.")
    exit()

# --- GRAPH 3: WIN/LOSS DISTRIBUTION (HISTOGRAM) ---
def plot_win_loss_distribution():
    hero_nets = []
    for entry in data:
        hand_data = entry.get('hand', entry)
        net = hand_data.get('hero_net', 0)
        hero_nets.append(net)
    
    plt.figure(figsize=(10, 6))
    
    # Use more bins and set a reasonable range
    plt.hist(hero_nets, bins=40, color='#1f77b4', edgecolor='black', alpha=0.7, range=(-10000, 10000))
    plt.axvline(0, color='red', linestyle='--', linewidth=2, label='Break Even')
    
    # Set y-axis to show detail better
    plt.ylim(0, max(plt.ylim()[1] * 0.8, 10))  # Cap the y-axis to show more detail
    
    plt.title('Win/Loss Distribution Per Hand', fontsize=14, fontweight='bold')
    plt.xlabel('Chips Won/Lost (BB)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('graph_win_loss_distribution.png', dpi=300)
    print("Saved 'graph_win_loss_distribution.png'")
    plt.show()

# --- GRAPH 5: AGGRESSION ANALYSIS ---
def plot_aggression_analysis():
    aggressive_actions = 0  # BET + RAISE
    passive_actions = 0     # CHECK + CALL
    folds = 0
    
    for entry in data:
        hand_data = entry.get('hand', entry)
        actions = hand_data.get('actions', [])
        
        for action in actions:
            if action.get('actor') == 1:  # Hero's actions
                name = action.get('act_name', '')
                if "BET" in name or "RAISE" in name:
                    aggressive_actions += 1
                elif "CHECK" in name or "CALL" in name:
                    passive_actions += 1
                elif "FOLD" in name:
                    folds += 1
    
    total = aggressive_actions + passive_actions + folds
    
    # Create pie chart
    plt.figure(figsize=(10, 7))
    sizes = [aggressive_actions, passive_actions, folds]
    labels = [f'Aggressive\n({aggressive_actions})', 
              f'Passive\n({passive_actions})', 
              f'Fold\n({folds})']
    colors = ['#2ca02c', '#1f77b4', '#d62728']
    explode = (0.05, 0, 0)  # Emphasize aggressive actions
    
    plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
            shadow=True, startangle=90, textprops={'fontsize': 12, 'fontweight': 'bold'})
    
    plt.title('Aggression Profile: Action Distribution', fontsize=14, fontweight='bold', pad=20)
    
    # Add aggression factor
    agg_factor = aggressive_actions / passive_actions if passive_actions > 0 else float('inf')
    plt.text(0, -1.4, f'Aggression Factor: {agg_factor:.2f}', 
            ha='center', fontsize=12, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('graph_aggression_analysis.png', dpi=300)
    print("Saved 'graph_aggression_analysis.png'")
    plt.show()


# --- GRAPH 8: MATCH PERFORMANCE BREAKDOWN ---
def plot_match_breakdown():
    match_results = defaultdict(int)
    
    for entry in data:
        match_id = entry.get('matchId', 'Unknown')
        hand_data = entry.get('hand', entry)
        net = hand_data.get('hero_net', 0)
        match_results[match_id] += net
    
    # Sort by performance
    sorted_matches = sorted(match_results.items(), key=lambda x: x[1], reverse=True)
    
    plt.figure(figsize=(12, 6))
    match_labels = [f'Match {i+1}' for i in range(len(sorted_matches))]
    values = [v[1] for v in sorted_matches]
    colors = ['#2ca02c' if v > 0 else '#d62728' for v in values]
    
    bars = plt.bar(match_labels, values, color=colors, edgecolor='black', linewidth=1.5)
    plt.axhline(0, color='black', linewidth=1)
    plt.title('Performance by Match', fontsize=14, fontweight='bold')
    plt.xlabel('Match', fontsize=12)
    plt.ylabel('Net Chips (BB)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', 
                va='bottom' if height > 0 else 'top', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('graph_match_breakdown.png', dpi=300)
    print("Saved 'graph_match_breakdown.png'")
    plt.show()

# --- RUN ALL GRAPHS ---
if __name__ == "__main__":
    print("\nGenerating advanced analytics graphs...\n")
    plot_win_loss_distribution()
    plot_position_analysis()
    plot_aggression_analysis()
    plot_hand_outcomes()
    plot_rolling_average()
    plot_match_breakdown()
    print("\n✓ All graphs generated successfully!")
