import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
from pathlib import Path

def load_problem6_data(data_dir="problem6/results/benchmark-test-20250606-072605/results"):
    """Load all problem6 CSV files and organize by liveness/safety ratio"""
    
    # Find all CSV files
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    
    data = {}
    
    for file in csv_files:
        filename = os.path.basename(file)
        
        # Extract parameters from filename
        parts = filename.replace('.csv', '').split('_')
        
        # Find clause count and safety percentage
        clauses = None
        atoms = None
        safety_prec = None
        
        for i, part in enumerate(parts):
            if part.startswith('c') and part[1:].isdigit():
                clauses = int(part[1:])
            elif part.startswith('a') and part[1:].isdigit():
                atoms = int(part[1:])
            elif part.startswith('prec') and part[4:].isdigit():
                safety_prec = int(part[4:])
        
        if clauses and safety_prec is not None:
            # Read the CSV file
            df = pd.read_csv(file, sep=';')
            
            # Filter to get only the average row
            avg_row = df[df['Run Number'] == 'Average'].iloc[0]
            
            # Calculate liveness percentage
            liveness_prec = 100 - safety_prec
            
            # Store in our data structure
            if clauses not in data:
                data[clauses] = []
            
            data[clauses].append({
                'clauses': clauses,
                'atoms': atoms,
                'safety_prec': safety_prec,
                'liveness_prec': liveness_prec,
                'ratio_label': f"{liveness_prec}:{safety_prec}",
                'filename': filename,
                'data': avg_row
            })
    
    # Sort by safety percentage for each clause count
    for clause_count in data:
        data[clause_count].sort(key=lambda x: x['safety_prec'])
    
    return data

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

def create_overview_plots(data):
    """Create comprehensive overview plots for problem6 analysis"""
    
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
    
    # Get clause counts (sorted)
    clause_counts = sorted(data.keys())
    n_clause_counts = len(clause_counts)
    
    # Create figure with subplots
    fig, axes = plt.subplots(n_clause_counts, 2, figsize=(16, 6*n_clause_counts))
    if n_clause_counts == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle('Problem 6: Impact of Liveness/Safety Ratio on Solver Performance', fontsize=16, fontweight='bold')
    
    for idx, clause_count in enumerate(clause_counts):
        clause_data = data[clause_count]
        
        # Extract ratios for x-axis
        safety_percentages = [item['safety_prec'] for item in clause_data]
        ratio_labels = [item['ratio_label'] for item in clause_data]
        
        # Memory plot
        ax_mem = axes[idx, 0]
        # Time plot  
        ax_time = axes[idx, 1]
        
        for solver in solvers:
            memories = []
            times = []
            
            for item in clause_data:
                memory, time = extract_solver_metrics(item['data'], solver)
                memories.append(memory)
                times.append(time)
            
            # Plot only if there's meaningful data
            if any(m > 0 for m in memories):
                ax_mem.plot(safety_percentages, memories, 'o-', label=solver, 
                           color=solver_colors[solver], linewidth=2, markersize=6)
            
            if any(t > 0 for t in times):
                ax_time.plot(safety_percentages, times, 'o-', label=solver,
                            color=solver_colors[solver], linewidth=2, markersize=6)
        
        # Customize memory plot
        ax_mem.set_title(f'Memory Usage - {clause_count} Clauses', fontweight='bold')
        ax_mem.set_xlabel('Safety Percentage (%)')
        ax_mem.set_ylabel('Memory (MB)')
        ax_mem.set_yscale('log')
        ax_mem.grid(True, alpha=0.3)
        ax_mem.legend()
        ax_mem.set_xticks(safety_percentages)
        
        # Add secondary x-axis labels for liveness:safety ratios
        ax_mem_twin = ax_mem.twiny()
        ax_mem_twin.set_xlim(ax_mem.get_xlim())
        ax_mem_twin.set_xticks(safety_percentages)
        ax_mem_twin.set_xticklabels(ratio_labels, rotation=45, ha='left')
        ax_mem_twin.set_xlabel('Liveness:Safety Ratio')
        
        # Customize time plot
        ax_time.set_title(f'Execution Time - {clause_count} Clauses', fontweight='bold')
        ax_time.set_xlabel('Safety Percentage (%)')
        ax_time.set_ylabel('Time (s)')
        ax_time.set_yscale('log')
        ax_time.grid(True, alpha=0.3)
        ax_time.legend()
        ax_time.set_xticks(safety_percentages)
        
        # Add secondary x-axis labels for liveness:safety ratios
        ax_time_twin = ax_time.twiny()
        ax_time_twin.set_xlim(ax_time.get_xlim())
        ax_time_twin.set_xticks(safety_percentages)
        ax_time_twin.set_xticklabels(ratio_labels, rotation=45, ha='left')
        ax_time_twin.set_xlabel('Liveness:Safety Ratio')
    
    plt.tight_layout()
    plt.savefig('problem6_plots/p06_analysis_overview.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_solver_comparison_plots(data):
    """Create individual solver comparison plots across different clause counts"""
    
    solvers = ['vampire', 'snake', 'z3', 'cvc5', 'e', 'inkresat']
    clause_counts = sorted(data.keys())
    
    # Create color map for clause counts
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    for solver in solvers:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle(f'Problem 6: {solver.upper()} Performance Across Liveness/Safety Ratios', fontsize=14, fontweight='bold')
        
        for idx, clause_count in enumerate(clause_counts):
            clause_data = data[clause_count]
            
            safety_percentages = [item['safety_prec'] for item in clause_data]
            ratio_labels = [item['ratio_label'] for item in clause_data]
            
            memories = []
            times = []
            
            for item in clause_data:
                memory, time = extract_solver_metrics(item['data'], solver)
                memories.append(memory)
                times.append(time)
            
            label = f'{clause_count} clauses'
            color = colors[idx % len(colors)]
            
            # Plot memory
            if any(m > 0 for m in memories):
                ax1.plot(safety_percentages, memories, 'o-', label=label, color=color, linewidth=2, markersize=6)
            
            # Plot time
            if any(t > 0 for t in times):
                ax2.plot(safety_percentages, times, 'o-', label=label, color=color, linewidth=2, markersize=6)
        
        # Customize plots
        ax1.set_title('Memory Usage')
        ax1.set_xlabel('Safety Percentage (%)')
        ax1.set_ylabel('Memory (MB)')
        ax1.set_yscale('log')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        if clause_data:  # Use last clause_data for x-ticks
            ax1.set_xticks([item['safety_prec'] for item in clause_data])
        
        ax2.set_title('Execution Time')
        ax2.set_xlabel('Safety Percentage (%)')
        ax2.set_ylabel('Time (s)')
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        if clause_data:
            ax2.set_xticks([item['safety_prec'] for item in clause_data])
        
        plt.tight_layout()
        plt.savefig(f'problem6_plots/p06_{solver}_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()

def create_ratio_impact_heatmap(data):
    """Create heatmap showing performance across different ratios and clause counts"""
    
    solvers = ['vampire', 'snake', 'z3', 'cvc5', 'e']  # Exclude inkresat for cleaner visualization
    clause_counts = sorted(data.keys())
    
    # Get all safety percentages (should be consistent across clause counts)
    if clause_counts:
        safety_percentages = sorted(set(item['safety_prec'] for item in data[clause_counts[0]]))
    else:
        return
    
    for metric in ['memory', 'time']:
        fig, axes = plt.subplots(1, len(solvers), figsize=(20, 6))
        fig.suptitle(f'Problem 6: {metric.title()} Performance Heatmap (Liveness/Safety Impact)', fontsize=16, fontweight='bold')
        
        for solver_idx, solver in enumerate(solvers):
            # Create matrix for heatmap
            matrix = np.zeros((len(clause_counts), len(safety_percentages)))
            
            for clause_idx, clause_count in enumerate(clause_counts):
                clause_data = data[clause_count]
                for item in clause_data:
                    safety_idx = safety_percentages.index(item['safety_prec'])
                    memory, time = extract_solver_metrics(item['data'], solver)
                    value = memory if metric == 'memory' else time
                    matrix[clause_idx, safety_idx] = value if value > 0 else np.nan
            
            # Create heatmap
            ax = axes[solver_idx]
            im = ax.imshow(matrix, cmap='viridis', aspect='auto', interpolation='nearest')
            
            # Set labels
            ax.set_title(f'{solver.upper()}')
            ax.set_xlabel('Safety Percentage (%)')
            ax.set_ylabel('Number of Clauses')
            
            # Set ticks
            ax.set_xticks(range(len(safety_percentages)))
            ax.set_xticklabels(safety_percentages)
            ax.set_yticks(range(len(clause_counts)))
            ax.set_yticklabels(clause_counts)
            
            # Add colorbar
            plt.colorbar(im, ax=ax, label=f'{metric.title()} ({"MB" if metric == "memory" else "s"})')
        
        plt.tight_layout()
        plt.savefig(f'problem6_plots/p06_{metric}_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()

def print_data_summary(data):
    """Print a summary of the loaded data"""
    print("=== Problem 6 Data Summary ===")
    print("Study: Impact of Liveness/Safety ratio on solver performance")
    print("Clause lengths: 2, 3, 4, 6, 8, 10")
    print()
    
    clause_counts = sorted(data.keys())
    print(f"Formula sizes tested: {clause_counts} clauses")
    
    if clause_counts:
        ratios = [(item['liveness_prec'], item['safety_prec']) for item in data[clause_counts[0]]]
        ratios.sort(key=lambda x: x[1])  # Sort by safety percentage
        print("Liveness:Safety ratios tested:")
        for liveness, safety in ratios:
            print(f"  {liveness}:{safety}")
        print(f"Total test cases per clause count: {len(ratios)}")
    print()

def main():
    """Main function to run the analysis"""
    # Create output directory
    os.makedirs('problem6_plots', exist_ok=True)
    
    # Load data
    print("Loading Problem 6 data...")
    data = load_problem6_data()
    
    if not data:
        print("No data found! Please check the data directory path.")
        return
    
    # Print summary
    print_data_summary(data)
    
    # Create plots
    print("Creating overview plots...")
    create_overview_plots(data)
    
    print("Creating solver comparison plots...")
    create_solver_comparison_plots(data)
    
    print("Creating heatmap visualizations...")
    create_ratio_impact_heatmap(data)
    
    print("Analysis complete! Plots saved in problem6_plots/ directory.")

if __name__ == "__main__":
    main() 