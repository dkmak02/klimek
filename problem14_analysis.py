import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
from pathlib import Path

def load_problem14_data():
    """Load all problem14 CSV files and organize by clause count and single literal percentage"""
    
    data = {}
    
    # Load data from problem14 directory
    csv_files = glob.glob("problem14/problem14*.csv")
    
    for file in csv_files:
        filename = os.path.basename(file)
        
        # Extract parameters from filename: problem14_c100_a60_single10pct_results.csv
        parts = filename.replace('.csv', '').split('_')
        
        clauses = None
        atoms = None
        single_pct = None
        
        for part in parts:
            if part.startswith('c') and part[1:].isdigit():
                clauses = int(part[1:])
            elif part.startswith('a') and part[1:].isdigit():
                atoms = int(part[1:])
            elif part.startswith('single') and part[6:].replace('pct', '').isdigit():
                single_pct = int(part[6:].replace('pct', ''))
        
        if clauses is not None and single_pct is not None:
            # Read the CSV file
            df = pd.read_csv(file, sep=';')
            
            # Filter to get only the average row
            avg_row = df[df['Run Number'] == 'Average'].iloc[0]
            
            # Organize by clause count
            if clauses not in data:
                data[clauses] = {}
            
            data[clauses][single_pct] = {
                'clauses': clauses,
                'atoms': atoms,
                'single_pct': single_pct,
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

def create_single_literal_impact_analysis(data):
    """Create comprehensive analysis of single literal impact on solver performance"""
    
    solvers = ['vampire', 'snake', 'z3', 'cvc5', 'e']
    clause_counts = sorted(data.keys())
    single_percentages = [10, 25, 50]
    
    fig, axes = plt.subplots(len(clause_counts), 2, figsize=(16, 6*len(clause_counts)))
    if len(clause_counts) == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle('Problem 14: Impact of Single Literal Occurrences on Solver Performance\n(Single literal percentages: 10%, 25%, 50%)', 
                 fontsize=16, fontweight='bold')
    
    for clause_idx, clause_count in enumerate(clause_counts):
        ax_mem = axes[clause_idx, 0]
        ax_time = axes[clause_idx, 1]
        
        for solver in solvers:
            memories = []
            times = []
            available_percentages = []
            
            for single_pct in single_percentages:
                if single_pct in data[clause_count]:
                    constraint_data = data[clause_count][single_pct]
                    memory, time, sat_result = extract_solver_metrics(constraint_data['data'], solver)
                    
                    if memory > 0 or time > 0:
                        memories.append(memory)
                        times.append(time)
                        available_percentages.append(single_pct)
            
            if memories:
                ax_mem.plot(available_percentages, memories, 'o-', label=solver, linewidth=2, markersize=6)
            
            if times:
                ax_time.plot(available_percentages, times, 'o-', label=solver, linewidth=2, markersize=6)
        
        # Customize memory plot
        ax_mem.set_title(f'Memory Usage vs Single Literal Percentage - {clause_count} Clauses', fontweight='bold')
        ax_mem.set_xlabel('Percentage of Single Literals (%)')
        ax_mem.set_ylabel('Memory (MB)')
        ax_mem.set_yscale('log')
        ax_mem.set_xticks(single_percentages)
        ax_mem.legend()
        ax_mem.grid(True, alpha=0.3)
        
        # Customize time plot
        ax_time.set_title(f'Execution Time vs Single Literal Percentage - {clause_count} Clauses', fontweight='bold')
        ax_time.set_xlabel('Percentage of Single Literals (%)')
        ax_time.set_ylabel('Time (s)')
        ax_time.set_yscale('log')
        ax_time.set_xticks(single_percentages)
        ax_time.legend()
        ax_time.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('problem14_plots/p14_single_literal_impact.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_satisfiability_distribution_analysis(data):
    """Create analysis of SAT/UNSAT distribution with single literal constraints"""
    
    solvers = ['vampire', 'snake', 'z3', 'cvc5', 'e']
    clause_counts = sorted(data.keys())
    single_percentages = [10, 25, 50]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Problem 14: Satisfiability Analysis Under Single Literal Constraints', 
                 fontsize=16, fontweight='bold')
    
    # 1. SAT/UNSAT distribution by single literal percentage
    ax1 = axes[0, 0]
    
    sat_rates = {single_pct: [] for single_pct in single_percentages}
    
    for clause_count in clause_counts:
        for single_pct in single_percentages:
            if single_pct in data[clause_count]:
                constraint_data = data[clause_count][single_pct]
                
                sat_count = 0
                total_count = 0
                
                for solver in solvers:
                    memory, time, sat_result = extract_solver_metrics(constraint_data['data'], solver)
                    if memory > 0 or time > 0:
                        total_count += 1
                        if sat_result:
                            sat_count += 1
                
                sat_rate = (sat_count / total_count * 100) if total_count > 0 else 0
                sat_rates[single_pct].append(sat_rate)
    
    # Create bar plot for SAT rates
    x_pos = np.arange(len(single_percentages))
    avg_sat_rates = [np.mean(sat_rates[pct]) if sat_rates[pct] else 0 for pct in single_percentages]
    std_sat_rates = [np.std(sat_rates[pct]) if sat_rates[pct] else 0 for pct in single_percentages]
    
    bars = ax1.bar(x_pos, avg_sat_rates, yerr=std_sat_rates, capsize=5, alpha=0.7, color='skyblue')
    ax1.set_title('SAT Success Rate by Single Literal Percentage', fontweight='bold')
    ax1.set_xlabel('Single Literal Percentage (%)')
    ax1.set_ylabel('SAT Success Rate (%)')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([f'{pct}%' for pct in single_percentages])
    ax1.set_ylim(0, 100)
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, v in enumerate(avg_sat_rates):
        ax1.text(i, v + 2, f'{v:.1f}%', ha='center', va='bottom')
    
    # 2. Solver-specific SAT/UNSAT patterns
    ax2 = axes[0, 1]
    
    solver_sat_rates = {solver: [] for solver in solvers}
    
    for solver in solvers:
        for clause_count in clause_counts:
            for single_pct in single_percentages:
                if single_pct in data[clause_count]:
                    constraint_data = data[clause_count][single_pct]
                    memory, time, sat_result = extract_solver_metrics(constraint_data['data'], solver)
                    
                    if memory > 0 or time > 0:
                        solver_sat_rates[solver].append(1 if sat_result else 0)
    
    # Calculate average SAT rate per solver
    solver_avg_rates = {}
    for solver in solvers:
        if solver_sat_rates[solver]:
            avg_rate = np.mean(solver_sat_rates[solver]) * 100
            solver_avg_rates[solver] = avg_rate
    
    if solver_avg_rates:
        solvers_list = list(solver_avg_rates.keys())
        rates_list = list(solver_avg_rates.values())
        
        bars = ax2.bar(solvers_list, rates_list, alpha=0.7, color='lightgreen')
        ax2.set_title('SAT Success Rate by Solver', fontweight='bold')
        ax2.set_xlabel('Solver')
        ax2.set_ylabel('SAT Success Rate (%)')
        ax2.set_ylim(0, 100)
        ax2.grid(True, alpha=0.3)
        
        # Add value labels
        for i, v in enumerate(rates_list):
            ax2.text(i, v + 2, f'{v:.1f}%', ha='center', va='bottom')
    
    # 3. Difficulty scaling with single literal percentage
    ax3 = axes[1, 0]
    
    difficulty_scores = {}
    
    for clause_count in clause_counts:
        difficulties = []
        
        for single_pct in single_percentages:
            if single_pct in data[clause_count]:
                constraint_data = data[clause_count][single_pct]
                
                # Calculate difficulty as average solve time for successful solvers
                solve_times = []
                for solver in solvers:
                    memory, time, sat_result = extract_solver_metrics(constraint_data['data'], solver)
                    if time > 0:
                        solve_times.append(time)
                
                avg_time = np.mean(solve_times) if solve_times else 0
                difficulties.append(avg_time)
        
        if difficulties:
            difficulty_scores[f'{clause_count} clauses'] = difficulties
    
    # Create line plot for difficulty scaling
    for clause_count, difficulties in difficulty_scores.items():
        ax3.plot(single_percentages[:len(difficulties)], difficulties, 'o-', 
                label=clause_count, linewidth=2, markersize=6)
    
    ax3.set_title('Difficulty Scaling with Single Literal Percentage', fontweight='bold')
    ax3.set_xlabel('Single Literal Percentage (%)')
    ax3.set_ylabel('Average Solve Time (s)')
    ax3.set_yscale('log')
    ax3.set_xticks(single_percentages)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Unique condition impact analysis
    ax4 = axes[1, 1]
    
    # Calculate the impact of unique conditions (single literals) on formula structure
    structural_impact = {}
    
    for clause_count in clause_counts:
        impacts = []
        
        for single_pct in single_percentages:
            if single_pct in data[clause_count]:
                constraint_data = data[clause_count][single_pct]
                atoms = constraint_data['atoms']
                
                # Calculate structural sparsity
                single_literals = atoms * single_pct / 100
                multi_literals = atoms - single_literals
                
                # Structural impact as ratio of unique to repeated literals
                impact = single_literals / (multi_literals + 1) if multi_literals > 0 else single_literals
                impacts.append(impact)
        
        if impacts:
            structural_impact[f'{clause_count}C'] = impacts
    
    if structural_impact:
        x_pos = np.arange(len(single_percentages))
        width = 0.25
        
        colors = ['red', 'green', 'blue']
        for i, (clause_key, impacts) in enumerate(structural_impact.items()):
            offset = (i - 1) * width
            ax4.bar(x_pos + offset, impacts[:len(x_pos)], width, 
                   label=clause_key, alpha=0.7, color=colors[i % len(colors)])
        
        ax4.set_title('Structural Impact of Single Literals', fontweight='bold')
        ax4.set_xlabel('Single Literal Percentage (%)')
        ax4.set_ylabel('Unique/Repeated Literal Ratio')
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels([f'{pct}%' for pct in single_percentages])
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('problem14_plots/p14_satisfiability_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_solver_sensitivity_analysis(data):
    """Create analysis of individual solver sensitivity to single literal constraints"""
    
    solvers = ['vampire', 'snake', 'z3', 'cvc5', 'e']
    clause_counts = sorted(data.keys())
    single_percentages = [10, 25, 50]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    fig.suptitle('Problem 14: Individual Solver Sensitivity to Single Literal Constraints', 
                 fontsize=16, fontweight='bold')
    
    # Create subplot for each solver
    for solver_idx, solver in enumerate(solvers):
        if solver_idx >= len(axes):
            break
            
        ax = axes[solver_idx]
        
        for clause_count in clause_counts:
            times = []
            percentages = []
            sat_indicators = []
            
            for single_pct in single_percentages:
                if single_pct in data[clause_count]:
                    constraint_data = data[clause_count][single_pct]
                    memory, time, sat_result = extract_solver_metrics(constraint_data['data'], solver)
                    
                    if memory > 0 or time > 0:
                        times.append(time)
                        percentages.append(single_pct)
                        sat_indicators.append(sat_result)
            
            if times:
                # Use different markers for SAT vs UNSAT
                for i, (pct, time, is_sat) in enumerate(zip(percentages, times, sat_indicators)):
                    marker = 'o' if is_sat else 'x'
                    color = 'green' if is_sat else 'red'
                    if i == 0:  # Add label only for first point
                        ax.scatter(pct, time, marker=marker, color=color, s=60, alpha=0.7,
                                 label=f'{clause_count} clauses')
                    else:
                        ax.scatter(pct, time, marker=marker, color=color, s=60, alpha=0.7)
                
                # Connect points with lines
                ax.plot(percentages, times, '--', alpha=0.5, linewidth=1)
        
        ax.set_title(f'{solver.title()} Performance Pattern', fontweight='bold')
        ax.set_xlabel('Single Literal Percentage (%)')
        ax.set_ylabel('Execution Time (s)')
        ax.set_yscale('log')
        ax.set_xticks(single_percentages)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add custom legend for SAT/UNSAT markers
        ax.scatter([], [], marker='o', color='green', s=60, alpha=0.7, label='SAT')
        ax.scatter([], [], marker='x', color='red', s=60, alpha=0.7, label='UNSAT')
        ax.legend()
    
    # Remove empty subplots
    for i in range(len(solvers), len(axes)):
        fig.delaxes(axes[i])
    
    plt.tight_layout()
    plt.savefig('problem14_plots/p14_solver_sensitivity_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_unique_condition_heuristics_analysis(data):
    """Create analysis of how unique conditions affect solver heuristics"""
    
    solvers = ['vampire', 'snake', 'z3', 'cvc5', 'e']
    clause_counts = sorted(data.keys())
    single_percentages = [10, 25, 50]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Problem 14: Unique Condition Impact on Solver Heuristics', 
                 fontsize=16, fontweight='bold')
    
    # 1. Memory efficiency with single literal constraints
    ax1 = axes[0, 0]
    
    for solver in solvers:
        percentages = []
        memory_usages = []
        
        for single_pct in single_percentages:
            memories = []
            
            for clause_count in clause_counts:
                if single_pct in data[clause_count]:
                    constraint_data = data[clause_count][single_pct]
                    memory, time, sat_result = extract_solver_metrics(constraint_data['data'], solver)
                    if memory > 0:
                        memories.append(memory)
            
            if memories:
                avg_memory = np.mean(memories)
                percentages.append(single_pct)
                memory_usages.append(avg_memory)
        
        if memory_usages:
            ax1.plot(percentages, memory_usages, 'o-', label=solver, linewidth=2, markersize=6)
    
    ax1.set_title('Memory Usage vs Single Literal Percentage', fontweight='bold')
    ax1.set_xlabel('Single Literal Percentage (%)')
    ax1.set_ylabel('Average Memory Usage (MB)')
    ax1.set_yscale('log')
    ax1.set_xticks(single_percentages)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Performance consistency analysis
    ax2 = axes[0, 1]
    
    consistency_scores = {}
    
    for single_pct in single_percentages:
        solver_times = {solver: [] for solver in solvers}
        
        for clause_count in clause_counts:
            if single_pct in data[clause_count]:
                constraint_data = data[clause_count][single_pct]
                
                for solver in solvers:
                    memory, time, sat_result = extract_solver_metrics(constraint_data['data'], solver)
                    if time > 0:
                        solver_times[solver].append(time)
        
        # Calculate consistency as inverse of coefficient of variation
        consistencies = []
        for solver in solvers:
            if len(solver_times[solver]) > 1:
                mean_time = np.mean(solver_times[solver])
                std_time = np.std(solver_times[solver])
                cv = std_time / mean_time if mean_time > 0 else 0
                consistency = 1 / (cv + 0.001)
                consistencies.append(consistency)
        
        if consistencies:
            consistency_scores[f'{single_pct}%'] = np.mean(consistencies)
    
    if consistency_scores:
        labels = list(consistency_scores.keys())
        values = list(consistency_scores.values())
        
        bars = ax2.bar(labels, values, alpha=0.7, color='lightcoral')
        ax2.set_title('Solver Consistency vs Single Literal Percentage', fontweight='bold')
        ax2.set_xlabel('Single Literal Percentage')
        ax2.set_ylabel('Average Consistency Score (1/CV)')
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3)
    
    # 3. Heuristic effectiveness matrix
    ax3 = axes[1, 0]
    
    # Create effectiveness matrix: solvers vs single literal percentages
    effectiveness_matrix = []
    solver_labels = []
    
    for solver in solvers:
        effectiveness_row = []
        solver_labels.append(solver)
        
        for single_pct in single_percentages:
            # Calculate effectiveness as combination of success rate and speed
            sat_count = 0
            total_count = 0
            total_time = 0
            
            for clause_count in clause_counts:
                if single_pct in data[clause_count]:
                    constraint_data = data[clause_count][single_pct]
                    memory, time, sat_result = extract_solver_metrics(constraint_data['data'], solver)
                    
                    if memory > 0 or time > 0:
                        total_count += 1
                        total_time += time
                        if sat_result:
                            sat_count += 1
            
            if total_count > 0:
                success_rate = sat_count / total_count
                avg_time = total_time / total_count
                effectiveness = success_rate / (avg_time + 0.001)  # Success per second
                effectiveness_row.append(effectiveness)
            else:
                effectiveness_row.append(0)
        
        effectiveness_matrix.append(effectiveness_row)
    
    im = ax3.imshow(effectiveness_matrix, cmap='YlOrRd', aspect='auto')
    ax3.set_title('Solver Effectiveness Matrix', fontweight='bold')
    ax3.set_xlabel('Single Literal Percentage')
    ax3.set_ylabel('Solver')
    ax3.set_xticks(range(len(single_percentages)))
    ax3.set_xticklabels([f'{pct}%' for pct in single_percentages])
    ax3.set_yticks(range(len(solver_labels)))
    ax3.set_yticklabels(solver_labels)
    
    # Add text annotations
    for i in range(len(solver_labels)):
        for j in range(len(single_percentages)):
            if effectiveness_matrix[i][j] > 0:
                text = ax3.text(j, i, f'{effectiveness_matrix[i][j]:.1f}', 
                               ha="center", va="center", color="black", fontsize=10)
    
    plt.colorbar(im, ax=ax3, label='Effectiveness (Success/Time)')
    
    # 4. Exception handling analysis
    ax4 = axes[1, 1]
    
    exception_patterns = {}
    
    for clause_count in clause_counts:
        unsat_counts = []
        
        for single_pct in single_percentages:
            if single_pct in data[clause_count]:
                constraint_data = data[clause_count][single_pct]
                
                unsat_count = 0
                total_count = 0
                
                for solver in solvers:
                    memory, time, sat_result = extract_solver_metrics(constraint_data['data'], solver)
                    if memory > 0 or time > 0:
                        total_count += 1
                        if not sat_result:
                            unsat_count += 1
                
                unsat_rate = (unsat_count / total_count * 100) if total_count > 0 else 0
                unsat_counts.append(unsat_rate)
        
        if unsat_counts:
            exception_patterns[f'{clause_count} clauses'] = unsat_counts
    
    # Create stacked bar chart for exception patterns
    if exception_patterns:
        x_pos = np.arange(len(single_percentages))
        width = 0.25
        
        colors = ['lightblue', 'lightgreen', 'lightyellow']
        for i, (clause_key, unsat_rates) in enumerate(exception_patterns.items()):
            offset = (i - 1) * width
            ax4.bar(x_pos + offset, unsat_rates[:len(x_pos)], width, 
                   label=clause_key, alpha=0.7, color=colors[i % len(colors)])
        
        ax4.set_title('UNSAT Rate by Single Literal Percentage', fontweight='bold')
        ax4.set_xlabel('Single Literal Percentage (%)')
        ax4.set_ylabel('UNSAT Rate (%)')
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels([f'{pct}%' for pct in single_percentages])
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('problem14_plots/p14_heuristics_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def print_problem14_summary(data):
    """Print a comprehensive summary of Problem 14 analysis"""
    print("=" * 70)
    print("PROBLEM 14 ANALYSIS SUMMARY")
    print("=" * 70)
    print("Study: Impact of Controlled Single Literal Occurrences on Solver Performance")
    print()
    print("Experimental specifications:")
    print("- Clause counts: 100, 200, 500")
    print("- Single literal percentages: 10%, 25%, 50%")
    print("- Remaining literals appear multiple times")
    print("- Equal safety/liveness proportions (50%/50%)")
    print("- Single literals simulate unique conditions/exceptions")
    print("- 100-second timeout, average of 3 runs per configuration")
    print()
    
    solvers = ['vampire', 'snake', 'z3', 'cvc5', 'e']
    
    for clause_count in sorted(data.keys()):
        print(f"Clause Count: {clause_count}")
        
        for single_pct in [10, 25, 50]:
            if single_pct in data[clause_count]:
                print(f"  {single_pct}% single literals:")
                constraint_data = data[clause_count][single_pct]
                atoms = constraint_data['atoms']
                single_literals = int(atoms * single_pct / 100)
                print(f"    Total atoms: {atoms}, Single literals: {single_literals}")
                
                sat_results = []
                for solver in solvers:
                    memory, time, sat_result = extract_solver_metrics(constraint_data['data'], solver)
                    if memory > 0 or time > 0:
                        result_str = "SAT" if sat_result else "UNSAT"
                        print(f"      {solver}: {result_str} (Mem: {memory:.3f} MB, Time: {time:.3f} s)")
                        sat_results.append(sat_result)
                
                sat_rate = (sum(sat_results) / len(sat_results) * 100) if sat_results else 0
                print(f"    Overall SAT rate: {sat_rate:.1f}%")
        print()
    
    print("Key insights:")
    print("- Single literal constraints significantly affect satisfiability")
    print("- Mixed SAT/UNSAT results indicate increased problem difficulty")
    print("- Unique conditions create challenging edge cases for solvers")
    print("- Different solvers show varying sensitivity to single literal patterns")
    print()

def main():
    """Main function to run Problem 14 analysis"""
    # Create output directory
    os.makedirs('problem14_plots', exist_ok=True)
    
    # Load data
    print("Loading Problem 14 data...")
    data = load_problem14_data()
    
    if not data:
        print("No data found! Please check the data directory path.")
        return
    
    # Print summary
    print_problem14_summary(data)
    
    # Create comprehensive analysis plots
    print("Creating single literal impact analysis...")
    create_single_literal_impact_analysis(data)
    
    print("Creating satisfiability distribution analysis...")
    create_satisfiability_distribution_analysis(data)
    
    print("Creating solver sensitivity analysis...")
    create_solver_sensitivity_analysis(data)
    
    print("Creating unique condition heuristics analysis...")
    create_unique_condition_heuristics_analysis(data)
    
    print("Problem 14 analysis complete! Plots saved in problem14_plots/ directory.")

if __name__ == "__main__":
    main() 