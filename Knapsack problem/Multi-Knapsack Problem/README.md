# Multi-Knapsack Problem with Low Balance Utilization

## Problem Description

The **Multi-Knapsack Problem** is a variant of the classic 0/1 Knapsack Problem. In this problem, multiple knapsacks are available, and a set of items is given, each with a weight but no value. The objective is to distribute all items across the knapsacks such that each item is assigned to exactly one knapsack, and the "low balance" of the weight utilization across knapsacks is minimized.

### Constraints:

- Multiple knapsacks, each with a specific capacity.
- Each item has a weight but no associated value.
- Each item must be placed in exactly one knapsack.
- The total weight of items in each knapsack should not exceed its capacity.
- The objective is to minimize the low balance in weight utilization among the knapsacks.

### Fitness Evaluation:

- The fitness value is determined based on the "low balance" of the knapsacks’ weight utilization, aiming to achieve a balanced distribution of weights across all knapsacks.

## Data

The instance data file for this problem contains the following:

- **Number of knapsacks**: The total number of available knapsacks, each with a specific capacity.
- **Number of items**: The total number of items to be distributed among the knapsacks.
- **Knapsack capacities**: A list of capacities, where each capacity represents the maximum weight a knapsack can carry.
- **Item weights**: A list of weights, where each weight represents the weight of an item.