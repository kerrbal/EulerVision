
import cv2
import numpy as np
import pyefd
from skimage.morphology import skeletonize
from skimage import img_as_ubyte
import networkx as nx
from collections import defaultdict
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import os


@dataclass
class Node:
    id: int
    center: Tuple[int, int]
    radius: float
    contour: np.ndarray


@dataclass
class Edge:
    node1_id: int
    node2_id: int
    pixels: List[Tuple[int, int]]  #all pixel coordinates of the edge.
    def __repr__(self):
        return f"node 1 id: {self.node1_id}, node 2 id: {self.node2_id}"


@dataclass
class GraphData:
    nodes: List[Node]
    edges: List[Edge]
    adjacency: Dict[int, List[int]]
    original_image: np.ndarray
    processed_image: np.ndarray


@dataclass
class EulerResult:
    has_euler_path: bool
    has_euler_circuit: bool
    path: List[int]
    odd_degree_nodes: List[int]
    node_degrees: Dict[int, int]
    message: str


class EulerPathSolver:
    """
    Class that detects nodes, edges, and solves the path.
    """
    
    def __init__(self, 
                 min_node_area: int = 100,
                 max_node_area: int = 10000,
                 circularity_threshold: float = 0.5,
                 connection_distance: int = 30):

        self.min_node_area = min_node_area
        self.max_node_area = max_node_area
        self.circularity_threshold = circularity_threshold
        self.connection_distance = connection_distance
    
    # Visuion Processing 
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess the image, apply blur -> make gray -> reduce noise"""
        # make gray
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # reduce noise
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)

        # adaptive thresholding (it is better for hand drawn vision.)
        binary = cv2.adaptiveThreshold(
            blurred, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 
            blockSize=11, 
            C=2
        )
        
        #Morphological cleaning
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        
        return binary
    
    #to increase robustness, calculate circularity with 3 different ways.
    def calculate_circularity(self, contour: np.ndarray) -> float:
        
        if len(contour) < 5:
            return 0.0
        
        # Approach 1: classic circularity (4π × Area / Perimeter²)
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        if perimeter == 0:
            return 0.0
        
        classic_circularity = 4 * np.pi * area / (perimeter ** 2)
        
        # Approach 2: Minimum enclosing circle
        (_, _), radius = cv2.minEnclosingCircle(contour)
        circle_area = np.pi * radius ** 2
        
        if circle_area == 0:
            return 0.0
        
        area_ratio = area / circle_area
        
        # Approach 3: EFD (Use fourier to capture general trend.)
        efd_score = 1.0
        if len(contour) >= 10:
            try:
                pts = contour.reshape(-1, 2).astype(np.float64)
                coeffs = pyefd.elliptic_fourier_descriptors(pts, order=10, normalize=False)
                a1, b1, c1, d1 = coeffs[0]
                matrix = np.array([[a1, b1], [c1, d1]])
                _, s, _ = np.linalg.svd(matrix)
                efd_score = s[1] / (s[0] + 1e-6)
            except Exception:
                efd_score = 1.0
        
        # combine score.
        combined = (classic_circularity * 0.4 + area_ratio * 0.4 + efd_score * 0.2)
        return min(combined, 1.0)
    
    
    #detect circles in the vision using houghcircles
    def detect_nodes(self, binary: np.ndarray, original_gray: np.ndarray = None) -> List[Node]:
        nodes = []
        

        if original_gray is None:
            original_gray = binary
        
        # apply blur
        blurred = cv2.GaussianBlur(original_gray, (11, 11), 2)
        
        # HoughCircles circle detection
        circles = cv2.HoughCircles(
            blurred, 
            cv2.HOUGH_GRADIENT, 
            dp=1,
            minDist=40,
            param1=50,
            param2=25,
            minRadius=15,
            maxRadius=60
        )
        
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i, (x, y, r) in enumerate(circles[0]):
                # For plotting, create dummy contours.
                theta = np.linspace(0, 2*np.pi, 36)
                contour = np.array([
                    [[int(x + r*np.cos(t)), int(y + r*np.sin(t))]] 
                    for t in theta
                ], dtype=np.int32)
                
                nodes.append(Node(
                    id=i,
                    center=(int(x), int(y)),
                    radius=float(r),
                    contour=contour
                ))
        
        # spare approach if houghcircles does not find anything.
        if len(nodes) == 0:
            nodes = self._detect_nodes_contour(binary)
        
        return nodes
    
    def _detect_nodes_contour(self, binary: np.ndarray) -> List[Node]:
        nodes = []
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        dilated = cv2.dilate(binary, kernel, iterations=3)
        eroded = cv2.erode(dilated, kernel, iterations=3)
        
        contours, _ = cv2.findContours(eroded, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        node_id = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if area < self.min_node_area or area > self.max_node_area:
                continue
            
            circularity = self.calculate_circularity(contour)
            
            if circularity >= self.circularity_threshold:
                (cx, cy), radius = cv2.minEnclosingCircle(contour)
                
                nodes.append(Node(
                    id=node_id,
                    center=(int(cx), int(cy)),
                    radius=radius,
                    contour=contour
                ))
                node_id += 1
        
        return nodes
    #mask nodes to create skeleton
    def create_node_mask(self, binary: np.ndarray, nodes: List[Node]) -> np.ndarray:
        
        mask = np.zeros_like(binary)
        
        for node in nodes:
            cv2.drawContours(mask, [node.contour], -1, 255, -1)

            cv2.circle(mask, node.center, int(node.radius * 1.3), 255, -1)
        
        return mask
    
    #EDGE DETECTION
    def extract_edges_skeleton(self, binary: np.ndarray, node_mask: np.ndarray) -> np.ndarray:
        # remove nodes
        edges_only = cv2.bitwise_and(binary, cv2.bitwise_not(node_mask))

        #form skeleton
        skeleton = skeletonize(edges_only > 0)
        skeleton =  (skeleton)
        
        return skeleton
    
    def find_skeleton_endpoints(self, skeleton: np.ndarray) -> List[Tuple[int, int]]:
        """find edge endpoints"""
        from scipy import ndimage
        
        # neighboring kernel.
        kernel = np.array([[1, 1, 1],
                          [1, 0, 1],
                          [1, 1, 1]], dtype=np.int32)
        
        # calculate neighbor number for each pixel
        neighbors = ndimage.convolve(
            (skeleton > 0).astype(np.int32), 
            kernel, 
            mode='constant', 
            cval=0
        )
        
        # two endpoints of a line is if it has only one neighbor (suppose line is continuous)
        endpoint_mask = (skeleton > 0) & (neighbors == 1)
        
        ys, xs = np.where(endpoint_mask)
        endpoints = list(zip(xs.astype(int), ys.astype(int)))
        
        return endpoints
    
    def trace_edge_from_point(self, skeleton: np.ndarray, start: Tuple[int, int], 
                              visited: set) -> List[Tuple[int, int]]:
        """Follow the edge from start"""
        path = [start]
        visited.add(start)
        
        current = start
        
        while True:
            x, y = current
            found_next = False
            
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    
                    nx, ny = x + dx, y + dy
                    
                    if (0 <= nx < skeleton.shape[1] and 
                        0 <= ny < skeleton.shape[0] and
                        skeleton[ny, nx] > 0 and 
                        (nx, ny) not in visited):
                        
                        visited.add((nx, ny))
                        path.append((nx, ny))
                        current = (nx, ny)
                        found_next = True
                        break
                
                if found_next:
                    break
            
            if not found_next:
                break
        
        return path
    
    def find_nearest_node(self, point: Tuple[int, int], nodes: List[Node]) -> Optional[int]:

        min_dist = float('inf')
        nearest_id = None
        
        for node in nodes:
            dist = np.sqrt((point[0] - node.center[0])**2 + 
                          (point[1] - node.center[1])**2)
            
            # if the edge end point is inside the radius plus the connection_distance
            if dist < node.radius + self.connection_distance and dist < min_dist:
                min_dist = dist
                nearest_id = node.id
        
        return nearest_id
    
    def detect_edges(self, skeleton: np.ndarray, nodes: List[Node]) -> List[Edge]:
        #extract edges from skeleton and connect it to each other.
        edges = []
        visited = set()
        #start from endpoints
        endpoints = self.find_skeleton_endpoints(skeleton)

        for endpoint in endpoints:
            if endpoint in visited:
                continue
            
            # from endpoint save the path to other end.
            path = self.trace_edge_from_point(skeleton, endpoint, visited)
            
            #Skip too short lines that cannot be real
            if len(path) < min(nodes, key = lambda x: x.radius).radius * 2:
                continue
            
        
            start_node = self.find_nearest_node(path[0], nodes)
            end_node = self.find_nearest_node(path[-1], nodes)
            
            if start_node is not None and end_node is not None:
                edges.append(Edge(
                    node1_id=start_node,
                    node2_id=end_node,
                    pixels=path
                ))
        
        return edges
    
    #create the graph    
    def build_adjacency(self, nodes: List[Node], edges: List[Edge]) -> Dict[int, List[int]]:
        #adjaceny list
        adjacency = defaultdict(list)
        
        # create all nodes if there is no connections to it.
        for node in nodes:
            if node.id not in adjacency:
                adjacency[node.id] = []
        
        # add edges
        for edge in edges:
            adjacency[edge.node1_id].append(edge.node2_id)
            adjacency[edge.node2_id].append(edge.node1_id)
        
        return dict(adjacency)
    
    def detect_graph(self, image: np.ndarray) -> GraphData:
        """extract full graph structure from the image array"""
        # Gray
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        binary = self.preprocess_image(image)
        
        nodes = self.detect_nodes(binary, gray)
        
        if len(nodes) == 0:
            raise ValueError("Node cannot be found!")
        
        #Node mask
        node_mask = self.create_node_mask(binary, nodes)
        
        #skeleton
        skeleton = self.extract_edges_skeleton(binary, node_mask)

        #find edges
        edges = self.detect_edges(skeleton, nodes)
        
        #form adjaceny
        adjacency = self.build_adjacency(nodes, edges)
        
        return GraphData(
            nodes=nodes,
            edges=edges,
            adjacency=adjacency,
            original_image=image,
            processed_image=binary
        )
    
    #euler algorithm    
    def check_euler_conditions(self, adjacency: Dict[int, List[int]]) -> EulerResult:
        degrees = {node: len(neighbors) for node, neighbors in adjacency.items()}
        
        # odd degree nodes
        odd_nodes = [node for node, deg in degrees.items() if deg % 2 == 1]
        
        # Connectiviness control
        G = nx.Graph()
        for node, neighbors in adjacency.items():
            for neighbor in neighbors:
                G.add_edge(node, neighbor)
        
        # empty graph control
        if G.number_of_edges() == 0:
            return EulerResult(
                has_euler_path=False,
                has_euler_circuit=False,
                path=[],
                odd_degree_nodes=odd_nodes,
                node_degrees=degrees,
                message="No edge is detected!"
            )
        
        # connectiviness cnotrol
        if not nx.is_connected(G):
            return EulerResult(
                has_euler_path=False,
                has_euler_circuit=False,
                path=[],
                odd_degree_nodes=odd_nodes,
                node_degrees=degrees,
                message="Not connected!"
            )
        
        # Euler condition
        if len(odd_nodes) == 0:
            return EulerResult(
                has_euler_path=True,
                has_euler_circuit=True,
                path=[],
                odd_degree_nodes=odd_nodes,
                node_degrees=degrees,
                message="There is Euler Circuit"
            )
        elif len(odd_nodes) == 2:
            return EulerResult(
                has_euler_path=True,
                has_euler_circuit=False,
                path=[],
                odd_degree_nodes=odd_nodes,
                node_degrees=degrees,
                message=f"There is Euler Path"
            )
        else:
            return EulerResult(
                has_euler_path=False,
                has_euler_circuit=False,
                path=[],
                odd_degree_nodes=odd_nodes,
                node_degrees=degrees,
                message=f"No Euler Path"
            )
    
    def hierholzer_algorithm(self, adjacency: Dict[int, List[int]], 
                             start_node: Optional[int] = None) -> List[int]:
        adj = defaultdict(list)
        for node, neighbors in adjacency.items():
            adj[node] = neighbors.copy()
        
        if start_node is None:
            odd_nodes = [n for n, neigh in adj.items() if len(neigh) % 2 == 1]
            if odd_nodes:
                start_node = odd_nodes[0]
            else:
                start_node = next(iter(adj))
        
        stack = [start_node]
        path = []
        
        while stack:
            current = stack[-1]
            
            if adj[current]:
                #go to the neighbor
                next_node = adj[current].pop()
                #remove the reverse way.
                adj[next_node].remove(current)
                stack.append(next_node)
            else:
                # backwards.
                path.append(stack.pop())
        
        return path[::-1]
    
    def solve_euler(self, graph_data: GraphData) -> EulerResult:
        """find the euler path for graph"""
        result = self.check_euler_conditions(graph_data.adjacency)
        
        if result.has_euler_path:
            start = result.odd_degree_nodes[0] if result.odd_degree_nodes else None
            path = self.hierholzer_algorithm(graph_data.adjacency, start)
            result.path = path
        
        return result
    
    # main solve function
    
    def solve(self, image_path: str) -> Tuple[GraphData, EulerResult]:
        """
        main solve function
        
        Args:
            image_path: vision file image
            
        Returns:
            (GraphData, EulerResult) tuple
        """

        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"cannot read: {image_path}")
        
        
        graph_data = self.detect_graph(image)
        
        
        euler_result = self.solve_euler(graph_data)
        
        return graph_data, euler_result
    
    def solve_from_image(self, image: np.ndarray) -> Tuple[GraphData, EulerResult]:
        graph_data = self.detect_graph(image)
        euler_result = self.solve_euler(graph_data)
        return graph_data, euler_result
    

    #visualization    
    def visualize(self, graph_data: GraphData, euler_result: EulerResult,
                  save_path: Optional[str] = None, show: bool = True) -> np.ndarray:
        
        #upscale the image for plausible experience
        img = graph_data.original_image.copy()

        h, w = img.shape[:2]
        scale = max(1, 750 // min(h, w))
        img = np.ones((h * scale, w * scale, 3), dtype=np.uint8) * 255

        if scale > 1:
            img = cv2.resize(img, (w * scale, h * scale), interpolation=cv2.INTER_LINEAR)
            scale_factor = scale
        else:
            scale_factor = 1
        
        if euler_result.has_euler_path and euler_result.path:
            path = euler_result.path

            #rainbow color
            #colors = plt.cm.rainbow(np.linspace(0, 1, len(path) - 1))
            
            #soft green color
            colors = np.ones(shape = (len(path), 3), dtype = np.float64)
            colors *= np.array([110 / 255, 193 / 255, 117 / 255], dtype = np.float64)
            # draw paths
            for i in range(len(path) - 1):
                node1 = next(n for n in graph_data.nodes if n.id == path[i])
                node2 = next(n for n in graph_data.nodes if n.id == path[i + 1])
                
                pt1 = (node1.center[0] * scale_factor, node1.center[1] * scale_factor)
                pt2 = (node2.center[0] * scale_factor, node2.center[1] * scale_factor)
                
                # normalize color range
                color = tuple(int(c * 255) for c in colors[i][:3])
                cv2.line(img, pt1, pt2, color, 6)
            
            # Draw path order
            for i in range(len(path) - 1):
                node1 = next(n for n in graph_data.nodes if n.id == path[i])
                node2 = next(n for n in graph_data.nodes if n.id == path[i + 1])
                
                # draw it in the middle
                mid_x = (node1.center[0] + node2.center[0]) // 2 * scale_factor
                mid_y = (node1.center[1] + node2.center[1]) // 2 * scale_factor
                
                # draw it on a white circle.
                cv2.circle(img, (mid_x, mid_y), 18, (255, 255, 255), -1)
                cv2.circle(img, (mid_x, mid_y), 18, (0, 0, 0), 2)
                
                # num
                text = str(i + 1)
                font_scale = 0.7
                thickness = 2

                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                cv2.putText(img, text, 
                           (mid_x - tw // 2, mid_y + th // 2),
                           cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)
            
            # draw nodes
            for node in graph_data.nodes:
                center = (node.center[0] * scale_factor, node.center[1] * scale_factor)
                radius = int(node.radius * scale_factor)
                
                # start node is green, finish node is red
                if node.id == path[0]:
                    color = (0, 200, 0)  # green start
                elif node.id == path[-1]:
                    color = (0, 0, 200)  # red finish
                else:
                    color = (200, 100, 0)  # otherwise blue
                
                cv2.circle(img, center, radius, color, -1)
                cv2.circle(img, center, radius, (0, 0, 0), 2)
            
            # start end labels
            #next -> take first one
            start_node = next(n for n in graph_data.nodes if n.id == path[0])
            end_node = next(n for n in graph_data.nodes if n.id == path[-1])
            
            start_pt = (start_node.center[0] * scale_factor, 
                       start_node.center[1] * scale_factor - int(start_node.radius * scale_factor) - 10)
            cv2.putText(img, "START", start_pt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 150, 0), 2)
            
            if path[0] != path[-1]:  # Circuit değilse
                end_pt = (end_node.center[0] * scale_factor, 
                         end_node.center[1] * scale_factor + int(end_node.radius * scale_factor) + 25)
                cv2.putText(img, "FINISH", end_pt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 150), 2)
            
            title = f"Euler {'Circuit' if euler_result.has_euler_circuit else 'Path'}"
        else:
            #If there is no euler path or circuit, just draw the nodes and edges.
            for node in graph_data.nodes:
                center = (node.center[0] * scale_factor, node.center[1] * scale_factor)
                radius = int(node.radius * scale_factor)
                cv2.circle(img, center, radius, (0, 0, 200), -1)
                cv2.circle(img, center, radius, (0, 0, 0), 2)
            
            title = "No Euler Path!"
        #show image

        
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        if show:
            plt.show()
        
        plt.close()
        
        return img

