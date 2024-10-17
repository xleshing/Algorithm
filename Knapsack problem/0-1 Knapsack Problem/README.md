# 0/1 Knapsack Problem

## Problem Description

The **0/1 Knapsack Problem** is a classic combinatorial optimization problem. In this problem, a set of items is given, each with a value and a weight. The objective is to select a subset of items such that the total value is maximized, while the total weight of the selected items does not exceed the knapsack's capacity. Each item can only be selected once, either taken (1) or not taken (0), hence the name "0/1 Knapsack Problem."

### Constraints:

- Each item has a corresponding value and weight.
- The knapsack has a limited capacity, which cannot be exceeded by the total weight of the selected items.
- Each item can only be selected once (i.e., 0 or 1), and items cannot be partially selected.
- The objective is to maximize the total value of the selected items.

## Data

The instance data file for this problem contains the following:

- **Number of items**: The total number of items to be considered.
- **Knapsack capacity**: The maximum weight the knapsack can carry.
- **Item values**: A list of values, where each value represents the value of an item.
- **Item weights**: A list of weights, where each weight represents the weight of an item.
