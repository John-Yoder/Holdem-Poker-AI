import json
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
# We use .jsonl extension now to match Option 2
DATA_FILE = 'poker_data.jsonl' 

# 1. LOAD THE DATA
data = []
try:
    with open(DATA_FILE, 'r') as f:
        for line_num, line in enumerate(f):
            # Skip empty lines
            if line.strip(): 
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    print(f"Skipping invalid JSON on line {line_num + 1}")
    print(f"Successfully loaded {len(data)} hands from {DATA_FILE}.")
except FileNotFoundError:
    print(f"Error: Could not find '{DATA_FILE}'. Make sure it is in the same folder.")
    exit()

# 2. PROCESS DATA FOR GRAPH 1 (CUMULATIVE PERFORMANCE)
cumulative_net = 0
hero_nets = []
hand_labels = []

for i, entry in enumerate(data):
    # Handle nested 'hand' key if present, otherwise assume flat structure
    hand_data = entry.get('hand', entry) 
    
    # "hero_net" is the key for chips won/lost in that hand
    net = hand_data.get('hero_net', 0)
    cumulative_net += net
    hero_nets.append(cumulative_net)
    hand_labels.append(i + 1)

# 3. PLOT GRAPH 1: CUMULATIVE BANKROLL
plt.figure(figsize=(10, 6))
plt.plot(hand_labels, hero_nets, marker='o', linestyle='-', color='#2ca02c', linewidth=2, markersize=4)
plt.axhline(15000, color='red', linestyle='--', linewidth=1, label='Break Even')
plt.title('Agent Performance: Cumulative Chip Count', fontsize=14)
plt.xlabel('Hand Number', fontsize=12)
plt.ylabel('Net Chips (BB)', fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('graph_performance.png') # Saves the image to your folder
print("Saved 'graph_performance.png'")
plt.show()

# 4. PROCESS DATA FOR GRAPH 2 (ACTION DISTRIBUTION)
action_counts = {"FOLD": 0, "CHECK": 0, "CALL": 0, "BET": 0, "RAISE": 0}

for entry in data:
    hand_data = entry.get('hand', entry)
    actions = hand_data.get('actions', [])
    
    for action in actions:
        # In your logs, actor 1 is the bot (Hero)
        if action.get('actor') == 1: 
            name = action.get('act_name', '')
            
            # Categorize the action string
            if "BET" in name: action_counts["BET"] += 1
            elif "RAISE" in name: action_counts["RAISE"] += 1
            elif "CHECK" in name: action_counts["CHECK"] += 1
            elif "CALL" in name: action_counts["CALL"] += 1
            elif "FOLD" in name: action_counts["FOLD"] += 1

# 5. PLOT GRAPH 2: POLICY DISTRIBUTION
plt.figure(figsize=(8, 5))
# Colors: Fold=Red, Check=Gray, Call=Orange, Bet=Blue, Raise=Green
colors = ['#d62728', '#7f7f7f', '#ff7f0e', '#1f77b4', '#2ca02c']
bars = plt.bar(action_counts.keys(), action_counts.values(), color=colors)
plt.title('Agent Policy Distribution (Action Frequency)', fontsize=14)
plt.xlabel('Action Type', fontsize=12)
plt.ylabel('Frequency (Count)', fontsize=12)

# Add numbers on top of bars
for bar in bars:
    yval = bar.get_height()
    if yval > 0: # Only label if bar exists
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.1, int(yval), ha='center', va='bottom')

plt.tight_layout()
plt.savefig('graph_policy.png') # Saves the image
print("Saved 'graph_policy.png'")
plt.show()