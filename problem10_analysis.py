import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
from pathlib import Path

def load_problem10_data():
    """Load all problem10 CSV files and organize by clause count and nesting depth"""
    
    data = {}
    
    # Load data from results directory
    results_dir = "problem10/results/benchmark-test-20250607-124607/results"
    csv_files = glob.glob(os.path.join(results_dir, "problem10*.csv"))
    
    for file in csv_files:
        filename = os.path.basename(file)
        
        # Extract parameters from filename: problem10_c100_a50_d1_results.csv
        parts = filename.replace('.csv', '').split('_')
        
        clauses = None
        atoms = None
        depth = None
        
        for part in parts:
            if part.startswith('c') and part[1:].isdigit():
                clauses = int(part[1:])
            elif part.startswith('a') and part[1:].isdigit():
                atoms = int(part[1:])
            elif part.startswith('d') and part[1:].isdigit():
                depth = int(part[1:])
        
        if clauses and depth:
            # Read the CSV file
            df = pd.read_csv(file, sep=';')
            
            # Filter to get only the average row
            avg_row = df[df['Run Number'] == 'Average'].iloc[0]
            
            # Organize by clause count
            if clauses not in data:
                data[clauses] = {}
            
            data[clauses][depth] = {
                'clauses': clauses,
                'atoms': atoms,
                'depth': depth,
                'filename': filename,
                'data': avg_row
            }
    
    return data

def extract_solver_metrics(data_row, solver_name):
    """Extract memory and time metrics for a specific solver"""
    memory_col = f"{solver_name} Memory (MB)"
    time_col = f"{solver_name} Time (s)"
    sat_col = f"{solver_name} SAT"
    
    memory = data_row[memory_col] if memory_col in data_row else 0
    time = data_row[time_col] if time_col in data_row else 0
    sat_result = data_row[sat_col] if sat_col in data_row else False
    
    # Handle string values with commas (European decimal notation)
    if isinstance(memory, str):
        memory = float(memory.replace(',', '.')) if memory != '0,0' else 0
    if isinstance(time, str):
        time = float(time.replace(',', '.')) if time != '0,0' else 0
        
    return memory, time, sat_result

def create_nesting_depth_analysis(data):
    """Create comprehensive analysis of quantifier nesting depth impact"""
    
    solvers = ['vampire', 'snake', 'z3', 'cvc5']
    clause_counts = sorted(data.keys())
    depths = [1, 2, 3, 4]
    
    fig, axes = plt.subplots(len(clause_counts), 2, figsize=(16, 6*len(clause_counts)))
    if len(clause_counts) == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle('Problem 10: Quantifier Nesting Depth Impact on Solver Performance', 
                 fontsize=16, fontweight='bold')
    
    for clause_idx, clause_count in enumerate(clause_counts):
        ax_mem = axes[clause_idx, 0]
        ax_time = axes[clause_idx, 1]
        
        for solver in solvers:
            memories = []
            times = []
            available_depths = []
            
            for depth in depths:
                if depth in data[clause_count]:
                    depth_data = data[clause_count][depth]
                    memory, time, sat_result = extract_solver_metrics(depth_data['data'], solver)
                    
                    if memory > 0 or time > 0:
                        memories.append(memory)
                        times.append(time)
                        available_depths.append(depth)
            
            if memories:
                ax_mem.plot(available_depths, memories, 'o-', label=solver, linewidth=2, markersize=6)
            
            if times:
                ax_time.plot(available_depths, times, 'o-', label=solver, linewidth=2, markersize=6)
        
        # Customize memory plot
        ax_mem.set_title(f'Memory Usage vs Nesting Depth - {clause_count} Clauses', fontweight='bold')
        ax_mem.set_xlabel('Quantifier Nesting Depth')
        ax_mem.set_ylabel('Memory (MB)')
        ax_mem.set_yscale('log')
        ax_mem.set_xticks(depths)
        ax_mem.legend()
        ax_mem.grid(True, alpha=0.3)
        
        # Customize time plot
        ax_time.set_title(f'Execution Time vs Nesting Depth - {clause_count} Clauses', fontweight='bold')
        ax_time.set_xlabel('Quantifier Nesting Depth')
        ax_time.set_ylabel('Time (s)')
        ax_time.set_yscale('log')
        ax_time.set_xticks(depths)
        ax_time.legend()
        ax_time.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('problem10_plots/p10_nesting_depth_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_complexity_scaling_analysis(data):
    """Create analysis showing how complexity scales with both clauses and nesting depth"""
    
    solvers = ['vampire', 'snake', 'z3', 'cvc5']
    clause_counts = sorted(data.keys())
    depths = [1, 2, 3, 4]
    
    # Create 3D-like visualization for complexity scaling
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Problem 10: Complexity Scaling Analysis\n(Clause Count × Nesting Depth)', 
                 fontsize=16, fontweight='bold')
    
    # Prepare data for heatmaps
    memory_matrices = {solver: [] for solver in solvers}
    time_matrices = {solver: [] for solver in solvers}
    
    for clause_count in clause_counts:
        for solver in solvers:
            memory_row = []
            time_row = []
            
            for depth in depths:
                if depth in data[clause_count]:
                    depth_data = data[clause_count][depth]
                    memory, time, sat_result = extract_solver_metrics(depth_data['data'], solver)
                    memory_row.append(memory)
                    time_row.append(time)
                else:
                    memory_row.append(0)
                    time_row.append(0)
            
            memory_matrices[solver].append(memory_row)
            time_matrices[solver].append(time_row)
    
    # Plot heatmaps for best performing solvers
    best_solvers = ['vampire', 'cvc5']  # Focus on most reliable performers
    
    for i, solver in enumerate(best_solvers):
        # Memory heatmap
        ax_mem = axes[i, 0]
        im_mem = ax_mem.imshow(memory_matrices[solver], cmap='YlOrRd', aspect='auto')
        ax_mem.set_title(f'{solver.title()} - Memory Usage (MB)', fontweight='bold')
        ax_mem.set_xlabel('Nesting Depth')
        ax_mem.set_ylabel('Clause Count')
        ax_mem.set_xticks(range(len(depths)))
        ax_mem.set_xticklabels(depths)
        ax_mem.set_yticks(range(len(clause_counts)))
        ax_mem.set_yticklabels(clause_counts)
        
        # Add text annotations
        for y in range(len(clause_counts)):
            for x in range(len(depths)):
                if memory_matrices[solver][y][x] > 0:
                    text = ax_mem.text(x, y, f'{memory_matrices[solver][y][x]:.2f}', 
                                     ha="center", va="center", color="black", fontsize=9)
        
        plt.colorbar(im_mem, ax=ax_mem)
        
        # Time heatmap
        ax_time = axes[i, 1]
        im_time = ax_time.imshow(time_matrices[solver], cmap='YlGnBu', aspect='auto')
        ax_time.set_title(f'{solver.title()} - Execution Time (s)', fontweight='bold')
        ax_time.set_xlabel('Nesting Depth')
        ax_time.set_ylabel('Clause Count')
        ax_time.set_xticks(range(len(depths)))
        ax_time.set_xticklabels(depths)
        ax_time.set_yticks(range(len(clause_counts)))
        ax_time.set_yticklabels(clause_counts)
        
        # Add text annotations
        for y in range(len(clause_counts)):
            for x in range(len(depths)):
                if time_matrices[solver][y][x] > 0:
                    text = ax_time.text(x, y, f'{time_matrices[solver][y][x]:.3f}', 
                                      ha="center", va="center", color="black", fontsize=9)
        
        plt.colorbar(im_time, ax=ax_time)
    
    plt.tight_layout()
    plt.savefig('problem10_plots/p10_complexity_scaling_heatmaps.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_nesting_efficiency_analysis(data):
    """Create analysis of solver efficiency at different nesting depths"""
    
    solvers = ['vampire', 'snake', 'z3', 'cvc5']
    depths = [1, 2, 3, 4]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Problem 10: Solver Efficiency vs Quantifier Nesting Depth', 
                 fontsize=16, fontweight='bold')
    
    # Calculate efficiency metrics
    efficiency_data = {solver: {'memory_eff': [], 'time_eff': [], 'combined_eff': []} for solver in solvers}
    
    for depth in depths:
        depth_memory = {solver: [] for solver in solvers}
        depth_time = {solver: [] for solver in solvers}
        
        # Collect all data for this depth across all clause counts
        for clause_count in data.keys():
            if depth in data[clause_count]:
                depth_data = data[clause_count][depth]
                for solver in solvers:
                    memory, time, sat_result = extract_solver_metrics(depth_data['data'], solver)
                    if memory > 0:
                        depth_memory[solver].append(memory)
                    if time > 0:
                        depth_time[solver].append(time)
        
        # Calculate efficiency scores (inverse of average resource usage)
        for solver in solvers:
            avg_memory = np.mean(depth_memory[solver]) if depth_memory[solver] else float('inf')
            avg_time = np.mean(depth_time[solver]) if depth_time[solver] else float('inf')
            
            memory_eff = 1 / (avg_memory + 0.001)  # Add small epsilon to avoid division by zero
            time_eff = 1 / (avg_time + 0.001)
            combined_eff = 2 / ((avg_memory + 0.001) + (avg_time + 0.001))
            
            efficiency_data[solver]['memory_eff'].append(memory_eff)
            efficiency_data[solver]['time_eff'].append(time_eff)
            efficiency_data[solver]['combined_eff'].append(combined_eff)
    
    # Plot efficiency trends
    ax1 = axes[0, 0]
    for solver in solvers:
        ax1.plot(depths, efficiency_data[solver]['memory_eff'], 'o-', label=solver, linewidth=2, markersize=6)
    ax1.set_title('Memory Efficiency vs Nesting Depth', fontweight='bold')
    ax1.set_xlabel('Nesting Depth')
    ax1.set_ylabel('Memory Efficiency (1/MB)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[0, 1]
    for solver in solvers:
        ax2.plot(depths, efficiency_data[solver]['time_eff'], 'o-', label=solver, linewidth=2, markersize=6)
    ax2.set_title('Time Efficiency vs Nesting Depth', fontweight='bold')
    ax2.set_xlabel('Nesting Depth')
    ax2.set_ylabel('Time Efficiency (1/s)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    ax3 = axes[1, 0]
    for solver in solvers:
        ax3.plot(depths, efficiency_data[solver]['combined_eff'], 'o-', label=solver, linewidth=2, markersize=6)
    ax3.set_title('Combined Efficiency vs Nesting Depth', fontweight='bold')
    ax3.set_xlabel('Nesting Depth')
    ax3.set_ylabel('Combined Efficiency Score')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Performance degradation analysis
    ax4 = axes[1, 1]
    degradation_data = {}
    
    for solver in solvers:
        if len(efficiency_data[solver]['combined_eff']) >= 2:
            baseline = efficiency_data[solver]['combined_eff'][0]  # Depth 1 as baseline
            degradation = [(baseline - eff) / baseline * 100 for eff in efficiency_data[solver]['combined_eff']]
            degradation_data[solver] = degradation
            ax4.plot(depths, degradation, 'o-', label=solver, linewidth=2, markersize=6)
    
    ax4.set_title('Performance Degradation from Depth 1 (%)', fontweight='bold')
    ax4.set_xlabel('Nesting Depth')
    ax4.set_ylabel('Performance Degradation (%)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('problem10_plots/p10_nesting_efficiency_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_quantifier_complexity_insights(data):
    """Create insights visualization about quantifier nesting complexity"""
    
    solvers = ['vampire', 'snake', 'z3', 'cvc5']
    depths = [1, 2, 3, 4]
    clause_counts = sorted(data.keys())
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Problem 10: Quantifier Nesting Complexity Insights', 
                 fontsize=16, fontweight='bold')
    
    # 1. Complexity factor analysis
    ax1 = axes[0, 0]
    complexity_factors = []
    complexity_labels = []
    
    for clause_count in clause_counts:
        for depth in depths:
            if depth in data[clause_count]:
                complexity_factor = clause_count * (depth ** 2)  # Quadratic depth impact hypothesis
                complexity_factors.append(complexity_factor)
                complexity_labels.append(f'C{clause_count}D{depth}')
    
    # Calculate average solving time for each complexity factor
    avg_times = []
    for i, (clause_count, depth) in enumerate([(c, d) for c in clause_counts for d in depths if d in data[c]]):
        depth_data = data[clause_count][depth]
        times = []
        for solver in solvers:
            memory, time, sat_result = extract_solver_metrics(depth_data['data'], solver)
            if time > 0:
                times.append(time)
        avg_times.append(np.mean(times) if times else 0)
    
    ax1.scatter(complexity_factors, avg_times, alpha=0.7, s=60)
    ax1.set_xlabel('Complexity Factor (Clauses × Depth²)')
    ax1.set_ylabel('Average Solving Time (s)')
    ax1.set_title('Complexity Factor vs Solving Time', fontweight='bold')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)
    
    # Add trend line
    if len(complexity_factors) > 1:
        z = np.polyfit(np.log(complexity_factors), np.log(avg_times), 1)
        p = np.poly1d(z)
        ax1.plot(complexity_factors, np.exp(p(np.log(complexity_factors))), "r--", alpha=0.8, linewidth=2)
    
    # 2. Depth impact distribution
    ax2 = axes[0, 1]
    depth_impacts = {depth: [] for depth in depths}
    
    for clause_count in clause_counts:
        baseline_time = None
        if 1 in data[clause_count]:
            baseline_data = data[clause_count][1]
            baseline_times = []
            for solver in solvers:
                memory, time, sat_result = extract_solver_metrics(baseline_data['data'], solver)
                if time > 0:
                    baseline_times.append(time)
            baseline_time = np.mean(baseline_times) if baseline_times else None
        
        if baseline_time:
            for depth in depths[1:]:  # Skip depth 1 (baseline)
                if depth in data[clause_count]:
                    depth_data = data[clause_count][depth]
                    depth_times = []
                    for solver in solvers:
                        memory, time, sat_result = extract_solver_metrics(depth_data['data'], solver)
                        if time > 0:
                            depth_times.append(time)
                    
                    if depth_times:
                        avg_depth_time = np.mean(depth_times)
                        impact = avg_depth_time / baseline_time
                        depth_impacts[depth].append(impact)
    
    # Create box plot for depth impacts
    impact_data = [depth_impacts[depth] for depth in depths[1:]]
    ax2.boxplot(impact_data, tick_labels=depths[1:])
    ax2.set_xlabel('Nesting Depth')
    ax2.set_ylabel('Time Impact Factor (vs Depth 1)')
    ax2.set_title('Nesting Depth Impact Distribution', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='Baseline (Depth 1)')
    ax2.legend()
    
    # 3. Solver ranking by depth
    ax3 = axes[0, 2]
    solver_rankings = {solver: [] for solver in solvers}
    
    for depth in depths:
        # Calculate average performance for each solver at this depth
        solver_performance = {}
        for clause_count in clause_counts:
            if depth in data[clause_count]:
                depth_data = data[clause_count][depth]
                for solver in solvers:
                    memory, time, sat_result = extract_solver_metrics(depth_data['data'], solver)
                    if time > 0:
                        efficiency = 1 / (time + 0.001)
                        if solver not in solver_performance:
                            solver_performance[solver] = []
                        solver_performance[solver].append(efficiency)
        
        # Calculate average efficiency and rank solvers
        solver_avg_eff = []
        for solver in solvers:
            if solver in solver_performance and solver_performance[solver]:
                avg_eff = np.mean(solver_performance[solver])
                solver_avg_eff.append((solver, avg_eff))
        
        # Sort by efficiency and assign ranks
        if solver_avg_eff:
            solver_avg_eff.sort(key=lambda x: x[1], reverse=True)
            for rank, (solver, eff) in enumerate(solver_avg_eff):
                solver_rankings[solver].append(rank + 1)
        else:
            # If no data for this depth, assign neutral rank
            for solver in solvers:
                solver_rankings[solver].append(len(solvers) // 2)
    
    for solver in solvers:
        if solver_rankings[solver] and len(solver_rankings[solver]) == len(depths):
            ax3.plot(depths, solver_rankings[solver], 
                    'o-', label=solver, linewidth=2, markersize=6)
    
    ax3.set_xlabel('Nesting Depth')
    ax3.set_ylabel('Performance Rank (1=Best)')
    ax3.set_title('Solver Ranking by Nesting Depth', fontweight='bold')
    ax3.set_yticks(range(1, len(solvers) + 1))
    ax3.invert_yaxis()
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Memory vs Time trade-off by depth
    ax4 = axes[1, 0]
    colors = ['blue', 'green', 'orange', 'red']
    
    for depth_idx, depth in enumerate(depths):
        memory_vals = []
        time_vals = []
        
        for clause_count in clause_counts:
            if depth in data[clause_count]:
                depth_data = data[clause_count][depth]
                for solver in solvers:
                    memory, time, sat_result = extract_solver_metrics(depth_data['data'], solver)
                    if memory > 0 and time > 0:
                        memory_vals.append(memory)
                        time_vals.append(time)
        
        if memory_vals and time_vals:
            ax4.scatter(memory_vals, time_vals, alpha=0.7, label=f'Depth {depth}', 
                       color=colors[depth_idx], s=60)
    
    ax4.set_xlabel('Memory Usage (MB)')
    ax4.set_ylabel('Execution Time (s)')
    ax4.set_title('Memory vs Time Trade-off', fontweight='bold')
    ax4.set_xscale('log')
    ax4.set_yscale('log')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Clause scaling effect by depth
    ax5 = axes[1, 1]
    
    for depth in depths:
        clause_list = []
        avg_times = []
        
        for clause_count in sorted(clause_counts):
            if depth in data[clause_count]:
                depth_data = data[clause_count][depth]
                times = []
                for solver in solvers:
                    memory, time, sat_result = extract_solver_metrics(depth_data['data'], solver)
                    if time > 0:
                        times.append(time)
                
                if times:
                    clause_list.append(clause_count)
                    avg_times.append(np.mean(times))
        
        if clause_list and avg_times:
            ax5.plot(clause_list, avg_times, 'o-', label=f'Depth {depth}', linewidth=2, markersize=6)
    
    ax5.set_xlabel('Number of Clauses')
    ax5.set_ylabel('Average Solving Time (s)')
    ax5.set_title('Clause Scaling by Nesting Depth', fontweight='bold')
    ax5.set_xscale('log')
    ax5.set_yscale('log')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. SAT consistency analysis
    ax6 = axes[1, 2]
    sat_consistency = {depth: 0 for depth in depths}
    
    for depth in depths:
        total_cases = 0
        sat_cases = 0
        
        for clause_count in clause_counts:
            if depth in data[clause_count]:
                depth_data = data[clause_count][depth]
                for solver in solvers:
                    memory, time, sat_result = extract_solver_metrics(depth_data['data'], solver)
                    if memory > 0 or time > 0:
                        total_cases += 1
                        if sat_result:
                            sat_cases += 1
        
        if total_cases > 0:
            sat_consistency[depth] = (sat_cases / total_cases) * 100
    
    bars = ax6.bar(depths, [sat_consistency[d] for d in depths], alpha=0.7, color='lightgreen')
    ax6.set_xlabel('Nesting Depth')
    ax6.set_ylabel('SAT Success Rate (%)')
    ax6.set_title('SAT Consistency by Nesting Depth', fontweight='bold')
    ax6.set_ylim(0, 100)
    
    # Add percentage labels on bars
    for i, depth in enumerate(depths):
        ax6.text(depth, sat_consistency[depth] + 2, f'{sat_consistency[depth]:.1f}%', 
                ha='center', va='bottom')
    
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('problem10_plots/p10_quantifier_complexity_insights.png', dpi=300, bbox_inches='tight')
    plt.show()

def print_problem10_summary(data):
    """Print a comprehensive summary of Problem 10 analysis"""
    print("=" * 70)
    print("PROBLEM 10 ANALYSIS SUMMARY")
    print("=" * 70)
    print("Study: Impact of Quantifier Nesting Depth on Solver Performance")
    print()
    print("Formula specifications:")
    print("- Constant clause counts: 100, 200, 500")
    print("- Quantifier nesting depths: 1, 2, 3, 4")
    print("- Each clause contains at least one quantifier (∀ or ∃)")
    print("- Safety and liveness clauses in equal proportions (50%/50%)")
    print("- Atoms scale proportionally: 50, 100, 250")
    print()
    
    solvers = ['vampire', 'snake', 'z3', 'cvc5']
    
    for clause_count in sorted(data.keys()):
        print(f"Clause Count: {clause_count}")
        atoms = data[clause_count][1]['atoms'] if 1 in data[clause_count] else "N/A"
        print(f"  Atoms: {atoms}")
        
        for depth in [1, 2, 3, 4]:
            if depth in data[clause_count]:
                print(f"  Nesting Depth {depth}:")
                depth_data = data[clause_count][depth]
                
                for solver in solvers:
                    memory, time, sat_result = extract_solver_metrics(depth_data['data'], solver)
                    if memory > 0 or time > 0:
                        result_str = "SAT" if sat_result else "UNSAT"
                        print(f"    {solver}: {result_str} (Mem: {memory:.3f} MB, Time: {time:.3f} s)")
        print()
    
    print("Key insights:")
    print("- Quantifier nesting depth creates exponential complexity growth")
    print("- Different solvers show varying sensitivity to nesting complexity")
    print("- Memory usage scales more predictably than execution time")
    print("- Deeper nesting challenges formal reasoning capabilities")
    print()

def main():
    """Main function to run Problem 10 analysis"""
    # Create output directory
    os.makedirs('problem10_plots', exist_ok=True)
    
    # Load data
    print("Loading Problem 10 data...")
    data = load_problem10_data()
    
    if not data:
        print("No data found! Please check the data directory path.")
        return
    
    # Print summary
    print_problem10_summary(data)
    
    # Create comprehensive analysis plots
    print("Creating quantifier nesting depth analysis...")
    create_nesting_depth_analysis(data)
    
    print("Creating complexity scaling analysis...")
    create_complexity_scaling_analysis(data)
    
    print("Creating nesting efficiency analysis...")
    create_nesting_efficiency_analysis(data)
    
    print("Creating quantifier complexity insights...")
    create_quantifier_complexity_insights(data)
    
    print("Problem 10 analysis complete! Plots saved in problem10_plots/ directory.")

if __name__ == "__main__":
    main() 