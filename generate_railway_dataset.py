"""
Railway Rolling Stock Cycle Selection - Synthetic Data Generator
Generates realistic test datasets in CSV format for the MWIS optimization problem.
Automatically converts to QCentroid Platform JSON contract.

Use Case: Deutsche Bahn Railway Rolling Stock Cycle Selection via MWIS
Compliance: QCentroid Platform v1.0
"""

import csv
import json
import os
from datetime import datetime, timedelta
from typing import List, Dict, Set, Tuple
import random


class RailwayDatasetGenerator:
    """
    Generates synthetic railway rolling stock data for MWIS optimization testing.
    Creates realistic cycles, trips, and conflict relationships.
    """
    
    def __init__(self, num_cycles: int = 20, num_trips: int = 50, seed: int = 42):
        """
        Initialize the dataset generator.
        
        Args:
            num_cycles: Number of rolling stock cycles to generate
            num_trips: Number of train services (trips) to generate
            seed: Random seed for reproducibility
        """
        self.num_cycles = num_cycles
        self.num_trips = num_trips
        random.seed(seed)
        
        self.cycles_data: List[Dict] = []
        self.trips_data: List[Dict] = []
        self.conflict_edges: List[Tuple[str, str]] = []
        
        print(f"✓ Initialized RailwayDatasetGenerator (cycles={num_cycles}, trips={num_trips})")
    
    def generate_trips(self) -> List[Dict]:
        """
        Generate synthetic train service data (trips).
        
        Returns:
            List of trip dictionaries with id, origin, destination, times
        """
        print("\n[1/4] Generating synthetic train services (trips)...")
        
        # Define major German cities for realistic routes
        cities = [
            "Berlin", "München", "Köln", "Hamburg", "Frankfurt",
            "Stuttgart", "Düsseldorf", "Dortmund", "Essen", "Leipzig",
            "Dresden", "Hannover", "Nürnberg", "Duisburg", "Bochum"
        ]
        
        trips = []
        base_time = datetime(2026, 9, 11, 6, 0)  # Start at 6 AM
        
        for i in range(1, self.num_trips + 1):
            origin = random.choice(cities)
            destination = random.choice([c for c in cities if c != origin])
            
            # Generate realistic departure/arrival times
            departure = base_time + timedelta(hours=random.randint(0, 18))
            duration_hours = random.randint(1, 8)
            arrival = departure + timedelta(hours=duration_hours)
            
            trip = {
                "trip_id": f"TRIP_{i:03d}",
                "origin": origin,
                "destination": destination,
                "departure_time": departure.strftime("%H:%M"),
                "arrival_time": arrival.strftime("%H:%M"),
                "train_type": random.choice(["ICE", "IC", "EC", "RB", "S-Bahn"]),
                "distance_km": random.randint(50, 800)
            }
            trips.append(trip)
        
        self.trips_data = trips
        print(f"✓ Generated {len(trips)} train services")
        return trips
    
    def generate_cycles(self) -> List[Dict]:
        """
        Generate synthetic rolling stock cycles.
        Each cycle covers multiple trips with associated weight (optimization metric).
        
        Weight represents: (paying_km - empty_km) / total_km ratio
        Higher weight = better utilization
        
        Returns:
            List of cycle dictionaries with id, weight, trips_covered
        """
        print("\n[2/4] Generating rolling stock cycles...")
        
        cycles = []
        
        for i in range(1, self.num_cycles + 1):
            # Each cycle covers 2-6 trips
            num_trips_in_cycle = random.randint(2, 6)
            covered_trips = random.sample(
                [t["trip_id"] for t in self.trips_data],
                k=min(num_trips_in_cycle, len(self.trips_data))
            )
            
            # Calculate weight: based on distance and utilization
            total_distance = sum(
                t["distance_km"] for t in self.trips_data
                if t["trip_id"] in covered_trips
            )
            
            # Simulate realistic weight calculation
            # Weight = (utilized_km - empty_km) / total_km
            # Range: 0.3 to 1.0 (higher is better utilization)
            base_weight = random.uniform(0.3, 1.0)
            weight = total_distance * base_weight * random.uniform(0.8, 1.2)
            weight = max(50.0, min(weight, 800.0))  # Clamp to realistic range
            
            cycle = {
                "cycle_id": f"CYCLE_{i:02d}",
                "weight": round(weight, 2),
                "trips_covered": covered_trips,
                "total_distance_km": total_distance,
                "efficiency_ratio": round(base_weight, 3)
            }
            cycles.append(cycle)
        
        self.cycles_data = cycles
        print(f"✓ Generated {len(cycles)} rolling stock cycles")
        return cycles
    
    def calculate_conflicts(self) -> List[Tuple[str, str]]:
        """
        Calculate conflict relationships between cycles.
        Two cycles conflict if they share at least one trip (cannot be selected together).
        
        Returns:
            List of conflict edges [(cycle_id_1, cycle_id_2), ...]
        """
        print("\n[3/4] Calculating conflict relationships...")
        
        conflicts = []
        
        # Compare all pairs of cycles
        for i in range(len(self.cycles_data)):
            for j in range(i + 1, len(self.cycles_data)):
                cycle_a = self.cycles_data[i]
                cycle_b = self.cycles_data[j]
                
                # Get sets of trips covered by each cycle
                trips_a = set(cycle_a["trips_covered"])
                trips_b = set(cycle_b["trips_covered"])
                
                # Check for intersection (shared trips = conflict)
                if trips_a & trips_b:
                    conflicts.append((cycle_a["cycle_id"], cycle_b["cycle_id"]))
        
        self.conflict_edges = conflicts
        
        # Calculate conflict statistics
        num_possible_edges = self.num_cycles * (self.num_cycles - 1) / 2
        conflict_density = len(conflicts) / num_possible_edges if num_possible_edges > 0 else 0
        
        print(f"✓ Calculated {len(conflicts)} conflict relationships")
        print(f"  Conflict density: {conflict_density:.2%}")
        print(f"  (Possible edges: {int(num_possible_edges)}, Actual conflicts: {len(conflicts)})")
        
        return conflicts
    
    def save_cycles_csv(self, filename: str = "cycles.csv") -> str:
        """
        Save cycles data to CSV file.
        
        Args:
            filename: Output CSV filename
            
        Returns:
            Path to saved CSV file
        """
        print(f"\n[4a/4] Saving cycles to {filename}...")
        
        filepath = os.path.abspath(filename)
        
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(
                f,
                fieldnames=['cycle_id', 'weight', 'trips_covered', 'total_distance_km', 'efficiency_ratio']
            )
            writer.writeheader()
            
            for cycle in self.cycles_data:
                writer.writerow({
                    'cycle_id': cycle['cycle_id'],
                    'weight': cycle['weight'],
                    'trips_covered': '|'.join(cycle['trips_covered']),  # Pipe-separated
                    'total_distance_km': cycle['total_distance_km'],
                    'efficiency_ratio': cycle['efficiency_ratio']
                })
        
        print(f"✓ Saved {len(self.cycles_data)} cycles to: {filepath}")
        return filepath
    
    def save_trips_csv(self, filename: str = "trips.csv") -> str:
        """
        Save trips (train services) data to CSV file.
        
        Args:
            filename: Output CSV filename
            
        Returns:
            Path to saved CSV file
        """
        print(f"\n[4b/4] Saving trips to {filename}...")
        
        filepath = os.path.abspath(filename)
        
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(
                f,
                fieldnames=['trip_id', 'origin', 'destination', 'departure_time', 'arrival_time', 'train_type', 'distance_km']
            )
            writer.writeheader()
            
            for trip in self.trips_data:
                writer.writerow(trip)
        
        print(f"✓ Saved {len(self.trips_data)} train services to: {filepath}")
        return filepath


class CSVToQCentroidConverter:
    """
    Converts CSV data to QCentroid Platform JSON contract format.
    """
    
    def __init__(self, cycles_csv: str = "cycles.csv"):
        """
        Initialize converter with CSV file path.
        
        Args:
            cycles_csv: Path to cycles CSV file
        """
        self.cycles_csv = cycles_csv
        self.nodes = []
        self.edges = []
        
        print(f"\n✓ Initialized CSVToQCentroidConverter")
    
    def load_cycles_from_csv(self) -> List[Dict]:
        """
        Load cycle data from CSV file.
        
        Returns:
            List of cycle dictionaries
        """
        print(f"\n[Converter] Reading cycles from: {self.cycles_csv}")
        
        cycles = []
        
        if not os.path.exists(self.cycles_csv):
            raise FileNotFoundError(f"CSV file not found: {self.cycles_csv}")
        
        with open(self.cycles_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                cycle = {
                    "id": row["cycle_id"],
                    "weight": float(row["weight"]),
                    "trips": row["trips_covered"].split("|")  # Split pipe-separated trips
                }
                cycles.append(cycle)
        
        print(f"✓ Loaded {len(cycles)} cycles from CSV")
        return cycles
    
    def calculate_edges(self, nodes: List[Dict]) -> List[List[str]]:
        """
        Calculate conflict edges from nodes.
        Two nodes conflict if their trip sets intersect.
        
        Args:
            nodes: List of node dictionaries with id and trips
            
        Returns:
            List of edge pairs [[node_id_1, node_id_2], ...]
        """
        print(f"\n[Converter] Calculating conflict edges...")
        
        edges = []
        
        # Compare all pairs
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                node_a = nodes[i]
                node_b = nodes[j]
                
                # Get trip sets
                trips_a = set(node_a.get("trips", []))
                trips_b = set(node_b.get("trips", []))
                
                # Check intersection
                if trips_a & trips_b:
                    edges.append([node_a["id"], node_b["id"]])
        
        print(f"✓ Calculated {len(edges)} conflict edges")
        return edges
    
    def convert_to_qcentroid_json(self, output_file: str = "input_dataset.json") -> Dict:
        """
        Convert CSV data to QCentroid Platform JSON contract.
        
        Args:
            output_file: Output JSON filename
            
        Returns:
            Dictionary with QCentroid contract structure
        """
        print(f"\n[Converter] Converting to QCentroid JSON contract...")
        
        # Load nodes from CSV
        nodes = self.load_cycles_from_csv()
        
        # Calculate edges (conflicts)
        edges = self.calculate_edges(nodes)
        
        # Build QCentroid contract
        qcentroid_data = {
            "nodes": nodes,
            "edges": edges,
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "num_nodes": len(nodes),
                "num_edges": len(edges),
                "problem_type": "MWIS",
                "use_case": "Railway Rolling Stock Cycle Selection",
                "platform": "QCentroid v1.0"
            }
        }
        
        # Save to JSON file
        filepath = os.path.abspath(output_file)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(qcentroid_data, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Converted to QCentroid JSON contract")
        print(f"✓ Saved to: {filepath}")
        
        return qcentroid_data


def generate_complete_dataset(num_cycles: int = 20, num_trips: int = 50, output_dir: str = ".") -> Dict:
    """
    Complete workflow: Generate synthetic data and convert to QCentroid JSON.
    
    Args:
        num_cycles: Number of rolling stock cycles
        num_trips: Number of train services
        output_dir: Directory for output files
        
    Returns:
        Dictionary with QCentroid contract
    """
    print("=" * 80)
    print("Railway Rolling Stock MWIS - Synthetic Dataset Generator")
    print("=" * 80)
    
    # Create output directory if needed
    if output_dir != ".":
        os.makedirs(output_dir, exist_ok=True)
    
    # Change to output directory
    original_dir = os.getcwd()
    if output_dir != ".":
        os.chdir(output_dir)
    
    try:
        # Step 1: Generate synthetic data
        generator = RailwayDatasetGenerator(num_cycles=num_cycles, num_trips=num_trips)
        generator.generate_trips()
        generator.generate_cycles()
        generator.calculate_conflicts()
        
        # Step 2: Save to CSV
        cycles_csv = generator.save_cycles_csv("cycles.csv")
        trips_csv = generator.save_trips_csv("trips.csv")
        
        # Step 3: Convert to QCentroid JSON
        converter = CSVToQCentroidConverter(cycles_csv="cycles.csv")
        qcentroid_data = converter.convert_to_qcentroid_json("input_dataset.json")
        
        # Print summary
        print("\n" + "=" * 80)
        print("DATASET GENERATION SUMMARY")
        print("=" * 80)
        print(f"✓ Rolling Stock Cycles: {len(qcentroid_data['nodes'])}")
        print(f"✓ Train Services (Trips): {num_trips}")
        print(f"✓ Conflict Relationships: {len(qcentroid_data['edges'])}")
        print(f"✓ Conflict Density: {len(qcentroid_data['edges']) / (len(qcentroid_data['nodes']) * (len(qcentroid_data['nodes']) - 1) / 2) * 100:.2f}%")
        print(f"\nGenerated Files:")
        print(f"  • cycles.csv (rolling stock data)")
        print(f"  • trips.csv (train services)")
        print(f"  • input_dataset.json (QCentroid contract)")
        print("=" * 80)
        
        return qcentroid_data
    
    finally:
        # Restore original directory
        os.chdir(original_dir)


def print_qcentroid_contract_preview(data: Dict, max_nodes: int = 5, max_edges: int = 10):
    """
    Print a preview of the QCentroid contract.
    
    Args:
        data: QCentroid contract dictionary
        max_nodes: Max nodes to display
        max_edges: Max edges to display
    """
    print("\n" + "=" * 80)
    print("QCentroid JSON CONTRACT PREVIEW")
    print("=" * 80)
    
    print("\n[NODES] (First 5 of {})".format(len(data['nodes'])))
    print("-" * 80)
    for node in data['nodes'][:max_nodes]:
        print(f"  {node['id']}: weight={node['weight']}, trips={len(node['trips'])}")
        print(f"    Trips: {node['trips'][:3]}{'...' if len(node['trips']) > 3 else ''}")
    
    print("\n[EDGES] (First 10 of {})".format(len(data['edges'])))
    print("-" * 80)
    for i, edge in enumerate(data['edges'][:max_edges]):
        print(f"  [{i+1}] {edge[0]} <--conflict--> {edge[1]}")
    
    print("\n[METADATA]")
    print("-" * 80)
    for key, value in data['metadata'].items():
        print(f"  {key}: {value}")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    """
    Main execution: Generate complete dataset with default parameters.
    """
    
    # Generate dataset
    qcentroid_contract = generate_complete_dataset(
        num_cycles=20,      # 20 rolling stock cycles
        num_trips=50,       # 50 train services
        output_dir="."      # Save in current directory
    )
    
    # Print preview
    print_qcentroid_contract_preview(qcentroid_contract)
    
    # Display JSON snippet
    print("\n[JSON SNIPPET] Full contract saved to input_dataset.json")
    print("=" * 80)
    print(json.dumps(qcentroid_contract, indent=2, ensure_ascii=False)[:500] + "\n... (truncated)")
