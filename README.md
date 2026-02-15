# Euler Path Solver

A Python application that detects graphs from images using OpenCV and solves Euler Path/Circuit problems.

## What Does It Do?

1. **Image Processing**: Detects circles (nodes) and lines (edges) from JPG/PNG images
2. **Graph Construction**: Represents the detected structure as a mathematical graph
3. **Euler Analysis**: Checks whether an Euler path/circuit exists
4. **Solution**: Finds the path using Hierholzer's algorithm
5. **Visualization**: Shows the step-by-step solution

## Requirements

```bash
pip install opencv-python numpy pyefd scikit-image networkx matplotlib scipy
```

## Usage

### Command Line

```bash
# Create test graphs
python solve_euler.py --create-test

# Solve a graph
python solve_euler.py test_euler_path.png

# Save the result
python solve_euler.py graph.jpg -o result.png
```

### Python API

```python
from euler_path_solver import EulerPathSolver

# Create solver
solver = EulerPathSolver(
    min_node_area=100,          # Min node area (pixels²)
    max_node_area=15000,        # Max node area (pixels²)
    circularity_threshold=0.4,  # Circle threshold (0-1)
    connection_distance=50      # Connection distance (pixels)
)

# Solve
graph_data, result = solver.solve("graph.jpg")

# Check results
print(f"Has Euler Path? {result.has_euler_path}")
print(f"Has Euler Circuit? {result.has_euler_circuit}")
print(f"Path: {result.path}")

# Visualize
solver.visualize(graph_data, result, save_path="result.png")
```

## Graph Rules

For proper detection of the graph in the image:

- **Nodes**: Filled circles (black or dark colored)
- **Edges**: Straight lines (connections between nodes)
- **Background**: Light colored (white is ideal)
- **Resolution**: At least 400x400 pixels recommended


## Algorithm Details

### 1. Node Detection (HoughCircles)
```
Gray image → GaussianBlur → HoughCircles → Circle centers
```

### 2. Edge Detection (Skeleton)
```
Binary image → Remove node mask → Skeletonize → Find endpoints → Trace
```

### 3. Euler Conditions
- **Circuit**: All nodes have even degree
- **Path**: Exactly 2 nodes have odd degree
- **None**: More than 2 odd degree nodes

### 4. Hierholzer's Algorithm
Finds the Euler path using stack-based DFS (O(E) complexity).

## File Structure

```
euler_solver/
├── euler_path_solver.py   # Main module
├── solve_euler.py         # CLI interface
├── README.md              # This file
├── test_*.png             # Test graphs
└── result_*.png           # Result images
```

## Parameter Tuning

If you're having issues with hand-drawn graphs:

```python
solver = EulerPathSolver(
    min_node_area=50,           # Lower for small nodes
    max_node_area=20000,        # Increase for large nodes
    circularity_threshold=0.3,  # Lower for imperfect circles
    connection_distance=70      # Increase for distant connections
)
```

## Output Example

The visualization shows:
- **Green node**: START
- **Red node**: END (not shown for circuits)
- **Numbered paths**: Follow in order (1, 2, 3...)
- **Rainbow colors**: Path progression

## Troubleshooting

**"No nodes found"**: 
- Make sure circles are filled
- Lower the `circularity_threshold` value

**"No edges found"**:
- Make sure lines touch the nodes
- Increase the `connection_distance` value

**Wrong connections**:
- Intersection of lines "may" confuse the program
- Draw larger nodes
