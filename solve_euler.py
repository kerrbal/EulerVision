#!/usr/bin/env python3
"""
Euler Path Solver - CLI

Usage:
    python solve_euler.py image.jpg                    # Fundamental Use
    python solve_euler.py image.jpg -o result.png      # Specify output file
    python solve_euler.py image.jpg --no-display      # Show the image
    
Example:
    python solve_euler.py test_euler_path.png -o solution.png
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from euler_path_solver import EulerPathSolver


def main():
    parser = argparse.ArgumentParser(
        description='Solve Euler path/circuit from image',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s graf.jpg                    Solve the graph and show the result
  %(prog)s graf.jpg -o result.png      Save the result
  
Supported Formats: JPG, PNG, BMP, TIFF
        """
    )
    
    parser.add_argument('image', nargs='?', help='Graph image path')
    parser.add_argument('-o', '--output', help='Output file (PNG)')
    parser.add_argument('--no-display', action='store_true', help='Dont show the solution on the screen')
    
    # settings
    parser.add_argument('--min-area', type=int, default=100, help='Min Node Area (piksel²)')
    parser.add_argument('--max-area', type=int, default=15000, help='Max Node are (piksel²)')
    parser.add_argument('--circularity', type=float, default=0.4, help='Circularity threshold (0-1)')
    parser.add_argument('--connection-dist', type=int, default=50, help='Connection distance (piksel)')
    
    args = parser.parse_args()
    
    
    #file path of image is necessary
    if not args.image:
        parser.print_help()
        print("\nError: No image file specified!")
        print("For test: python solve_euler.py --create-test")
        sys.exit(1)
    
    if not os.path.exists(args.image):
        print(f"Error: file couldnt be found: {args.image}")
        sys.exit(1)
    
    # Create solver
    solver = EulerPathSolver(
        min_node_area=args.min_area,
        max_node_area=args.max_area,
        circularity_threshold=args.circularity,
        connection_distance=args.connection_dist
    )
    
    print(f"Image is loading: {args.image}")
    print("-" * 50)
    
    try:
        # solve the problem.
        graph_data, result = solver.solve(args.image)
        
        # see the results
        print(f"Node number: {len(graph_data.nodes)}")
        print(f"Edge number: {len(graph_data.edges)}")
        print(f"Node degrees: {result.node_degrees}")
        print()
        print(f"Result: {result.message}")
        
        if result.path:
            print(f"Path: {' → '.join(map(str, result.path))}")
        
        #visualize the picture
        output_path = args.output
        if not output_path and not args.no_display:
            output_path = None  #Show only
        
        solver.visualize(
            graph_data, 
            result,
            save_path=output_path,
            show=not args.no_display
        )
        
        if output_path:
            print(f"\n Result Saved: {output_path}")
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
