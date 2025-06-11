import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
from pathlib import Path

def load_problem5_data(data_dir="problem5"):
    """Load all problem5 CSV files and organize by distribution type"""
    
    # Define the mapping for distribution types
    distribution_map = {
        'even': 'a. All groups 25% each',
        'more_short': 'b. Length 1: 1%, rest equally',
        'more_long': 'c. Length 20: 1%, rest equally'
    }
    
    # Find all CSV files
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    
    data = {}
    
    for file in csv_files:
        filename = os.path.basename(file)
        
        # Extract parameters from filename
        parts = filename.replace('.csv', '').split('_')
        
        # Find clause count
        clauses = None
        atoms = None
        distribution = None
        
        for i, part in enumerate(parts):
            if part.startswith('c') and part[1:].isdigit():
                clauses = int(part[1:])
            elif part.startswith('a') and part[1:].isdigit():
                atoms = int(part[1:])
            elif part in ['even', 'more']:
                if part == 'more' and i+1 < len(parts):
                    if parts[i+1] == 'short':
                        distribution = 'more_short'
                    elif parts[i+1] == 'long':
                        distribution = 'more_long'
                else:
                    distribution = part
        
        if clauses and distribution:
            # Read the CSV file
            df = pd.read_csv(file, sep=';')
            
            # Filter to get only the average row
            avg_row = df[df['Run Number'] == 'Average'].iloc[0]
            
            # Store in our data structure
            if distribution not in data:
                data[distribution] = []
            
            data[distribution].append({
                'clauses': clauses,
                'atoms': atoms,
                'filename': filename,
                'data': avg_row
            })
    
    # Sort by number of clauses
    for dist in data:
        data[dist].sort(key=lambda x: x['clauses'])
    
    return data, distribution_map

def extract_solver_metrics(data_row, solver_name):
    """Extract memory and time metrics for a specific solver"""
    memory_col = f"{solver_name} Memory (MB)"
    time_col = f"{solver_name} Time (s)"
    
    memory = data_row[memory_col] if memory_col in data_row else 0
    time = data_row[time_col] if time_col in data_row else 0
    
    # Handle string values with commas (European decimal notation)
    if isinstance(memory, str):
        memory = float(memory.replace(',', '.')) if memory != '0,0' else 0
    if isinstance(time, str):
        time = float(time.replace(',', '.')) if time != '0,0' else 0
        
    return memory, time

def create_plots(data, distribution_map):
    """Create comprehensive plots for problem5 analysis"""
    
    # Define solvers to analyze
    solvers = ['vampire', 'snake', 'z3', 'cvc5', 'e', 'inkresat']
    solver_colors = {
        'vampire': '#1f77b4',
        'snake': '#ff7f0e', 
        'z3': '#2ca02c',
        'cvc5': '#d62728',
        'e': '#9467bd',
        'inkresat': '#8c564b'
    }
    
    # Create figure with subplots
    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    fig.suptitle('Problem 5: Impact of Formula Size on Time/Memory\n(Constant Clause Length Groups)', fontsize=16, fontweight='bold')
    
    # Plot for each distribution type
    distributions = ['even', 'more_short', 'more_long']
    
    for dist_idx, dist in enumerate(distributions):
        if dist not in data:
            continue
            
        dist_data = data[dist]
        clause_counts = [item['clauses'] for item in dist_data]
        
        # Memory plot
        ax_mem = axes[dist_idx, 0]
        # Time plot  
        ax_time = axes[dist_idx, 1]
        
        for solver in solvers:
            memories = []
            times = []
            
            for item in dist_data:
                memory, time = extract_solver_metrics(item['data'], solver)
                memories.append(memory)
                times.append(time)
            
            # Plot only if there's meaningful data
            if any(m > 0 for m in memories):
                ax_mem.plot(clause_counts, memories, 'o-', label=solver, 
                           color=solver_colors[solver], linewidth=2, markersize=6)
            
            if any(t > 0 for t in times):
                ax_time.plot(clause_counts, times, 'o-', label=solver,
                            color=solver_colors[solver], linewidth=2, markersize=6)
        
        # Customize memory plot
        ax_mem.set_title(f'Memory Usage - {distribution_map[dist]}', fontweight='bold')
        ax_mem.set_xlabel('Number of Clauses')
        ax_mem.set_ylabel('Memory (MB)')
        ax_mem.set_xscale('log')
        ax_mem.set_yscale('log')
        ax_mem.grid(True, alpha=0.3)
        ax_mem.legend()
        
        # Customize time plot
        ax_time.set_title(f'Execution Time - {distribution_map[dist]}', fontweight='bold')
        ax_time.set_xlabel('Number of Clauses')
        ax_time.set_ylabel('Time (s)')
        ax_time.set_xscale('log')
        ax_time.set_yscale('log')
        ax_time.grid(True, alpha=0.3)
        ax_time.legend()
    
    plt.tight_layout()
    plt.savefig('problem5_plots/p05_analysis_overview.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_comparison_plots(data, distribution_map):
    """Create comparison plots showing different distributions side by side"""
    
    solvers = ['vampire', 'snake', 'z3', 'cvc5', 'e', 'inkresat']
    solver_colors = {
        'vampire': '#1f77b4',
        'snake': '#ff7f0e', 
        'z3': '#2ca02c',
        'cvc5': '#d62728',
        'e': '#9467bd',
        'inkresat': '#8c564b'
    }
    
    # Create comparison plots for each solver
    for solver in solvers:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle(f'Problem 5: {solver.upper()} Performance Across Distributions', fontsize=14, fontweight='bold')
        
        for dist in ['even', 'more_short', 'more_long']:
            if dist not in data:
                continue
                
            dist_data = data[dist]
            clause_counts = [item['clauses'] for item in dist_data]
            
            memories = []
            times = []
            
            for item in dist_data:
                memory, time = extract_solver_metrics(item['data'], solver)
                memories.append(memory)
                times.append(time)
            
            label = distribution_map[dist]
            
            # Plot memory
            if any(m > 0 for m in memories):
                ax1.plot(clause_counts, memories, 'o-', label=label, linewidth=2, markersize=6)
            
            # Plot time
            if any(t > 0 for t in times):
                ax2.plot(clause_counts, times, 'o-', label=label, linewidth=2, markersize=6)
        
        # Customize plots
        ax1.set_title('Memory Usage')
        ax1.set_xlabel('Number of Clauses')
        ax1.set_ylabel('Memory (MB)')
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        ax2.set_title('Execution Time')
        ax2.set_xlabel('Number of Clauses')
        ax2.set_ylabel('Time (s)')
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(f'problem5_plots/p05_{solver}_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()

def print_data_summary(data, distribution_map):
    """Print a summary of the loaded data"""
    print("=== Problem 5 Data Summary ===")
    print("Formula structure: 50% liveness + 50% safety clauses")
    print("Clause lengths: 1, 5, 10, 20 (constant within each formula)")
    print()
    
    for dist, dist_name in distribution_map.items():
        if dist in data:
            print(f"{dist_name}:")
            clause_counts = [item['clauses'] for item in data[dist]]
            print(f"  Clause counts: {clause_counts}")
            print(f"  Number of test cases: {len(data[dist])}")
            print()

def main():
    """Main function to run the analysis"""
    # Create output directory
    os.makedirs('problem5_plots', exist_ok=True)
    
    # Load data
    print("Loading Problem 5 data...")
    data, distribution_map = load_problem5_data()
    
    # Print summary
    print_data_summary(data, distribution_map)
    
    # Create plots
    print("Creating overview plots...")
    create_plots(data, distribution_map)
    
    print("Creating solver comparison plots...")
    create_comparison_plots(data, distribution_map)
    
    print("Analysis complete! Plots saved in problem5_plots/ directory.")

if __name__ == "__main__":
    main() 