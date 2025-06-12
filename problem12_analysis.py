import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
from pathlib import Path

def load_problem12_data():
    """Load all problem12 CSV files and organize by clause count and max occurrence constraint"""
    
    data = {}
    
    # Load data from problem12 directory
    csv_files = glob.glob("problem12/problem12*.csv")
    
    for file in csv_files:
        filename = os.path.basename(file)
        
        # Extract parameters from filename: problem12_c100_a57_max5_results.csv
        parts = filename.replace('.csv', '').split('_')
        
        clauses = None
        atoms = None
        max_occurrences = None
        
        for part in parts:
            if part.startswith('c') and part[1:].isdigit():
                clauses = int(part[1:])
            elif part.startswith('a') and part[1:].isdigit():
                atoms = int(part[1:])
            elif part.startswith('max') and part[3:].isdigit():
                max_occurrences = int(part[3:])
        
        if clauses and max_occurrences:
            # Read the CSV file
            df = pd.read_csv(file, sep=';')
            
            # Filter to get only the average row
            avg_row = df[df['Run Number'] == 'Average'].iloc[0]
            
            # Organize by clause count
            if clauses not in data:
                data[clauses] = {}
            
            data[clauses][max_occurrences] = {
                'clauses': clauses,
                'atoms': atoms,
                'max_occurrences': max_occurrences,
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

def create_occurrence_constraint_analysis(data):
    """Create comprehensive analysis of variable occurrence constraint impact"""
    
    solvers = ['vampire', 'snake', 'z3', 'cvc5', 'e']
    clause_counts = sorted(data.keys())
    max_occurrences = [2, 3, 5]
    
    fig, axes = plt.subplots(len(clause_counts), 2, figsize=(16, 6*len(clause_counts)))
    if len(clause_counts) == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle('Problem 12: Variable Occurrence Constraint Impact on Solver Performance\n(Max occurrences per variable: 2, 3, 5)', 
                 fontsize=16, fontweight='bold')
    
    for clause_idx, clause_count in enumerate(clause_counts):
        ax_mem = axes[clause_idx, 0]
        ax_time = axes[clause_idx, 1]
        
        for solver in solvers:
            memories = []
            times = []
            available_constraints = []
            
            for max_occ in max_occurrences:
                if max_occ in data[clause_count]:
                    constraint_data = data[clause_count][max_occ]
                    memory, time, sat_result = extract_solver_metrics(constraint_data['data'], solver)
                    
                    if memory > 0 or time > 0:
                        memories.append(memory)
                        times.append(time)
                        available_constraints.append(max_occ)
            
            if memories:
                ax_mem.plot(available_constraints, memories, 'o-', label=solver, linewidth=2, markersize=6)
            
            if times:
                ax_time.plot(available_constraints, times, 'o-', label=solver, linewidth=2, markersize=6)
        
        # Customize memory plot
        ax_mem.set_title(f'Memory Usage vs Max Variable Occurrences - {clause_count} Clauses', fontweight='bold')
        ax_mem.set_xlabel('Max Variable Occurrences')
        ax_mem.set_ylabel('Memory (MB)')
        ax_mem.set_yscale('log')
        ax_mem.set_xticks(max_occurrences)
        ax_mem.legend()
        ax_mem.grid(True, alpha=0.3)
        
        # Customize time plot
        ax_time.set_title(f'Execution Time vs Max Variable Occurrences - {clause_count} Clauses', fontweight='bold')
        ax_time.set_xlabel('Max Variable Occurrences')
        ax_time.set_ylabel('Time (s)')
        ax_time.set_yscale('log')
        ax_time.set_xticks(max_occurrences)
        ax_time.legend()
        ax_time.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('problem12_plots/p12_occurrence_constraint_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_sparsity_density_analysis(data):
    """Create analysis of formula sparsity and density effects"""
    
    solvers = ['vampire', 'snake', 'z3', 'cvc5', 'e']
    clause_counts = sorted(data.keys())
    max_occurrences = [2, 3, 5]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Problem 12: Formula Sparsity and Density Analysis', 
                 fontsize=16, fontweight='bold')
    
    # 1. Variable density analysis
    ax1 = axes[0, 0]
    
    for clause_count in clause_counts:
        densities = []
        performance_scores = []
        
        for max_occ in max_occurrences:
            if max_occ in data[clause_count]:
                constraint_data = data[clause_count][max_occ]
                atoms = constraint_data['atoms']
                
                # Calculate density as clauses per variable
                density = clause_count / atoms
                densities.append(density)
                
                # Calculate average performance score
                performance_times = []
                for solver in solvers:
                    memory, time, sat_result = extract_solver_metrics(constraint_data['data'], solver)
                    if time > 0:
                        performance_times.append(time)
                
                avg_time = np.mean(performance_times) if performance_times else 0
                performance_score = 1 / (avg_time + 0.001)  # Inverse time for performance
                performance_scores.append(performance_score)
        
        if densities and performance_scores:
            ax1.plot(densities, performance_scores, 'o-', label=f'{clause_count} clauses', 
                    linewidth=2, markersize=8)
            
            # Add max occurrence labels
            for i, max_occ in enumerate(max_occurrences[:len(densities)]):
                ax1.annotate(f'max{max_occ}', (densities[i], performance_scores[i]),
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    ax1.set_title('Performance vs Formula Density', fontweight='bold')
    ax1.set_xlabel('Formula Density (Clauses/Variables)')
    ax1.set_ylabel('Performance Score (1/Time)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Variable count scaling
    ax2 = axes[0, 1]
    
    for max_occ in max_occurrences:
        clause_list = []
        atom_list = []
        
        for clause_count in clause_counts:
            if max_occ in data[clause_count]:
                constraint_data = data[clause_count][max_occ]
                clause_list.append(clause_count)
                atom_list.append(constraint_data['atoms'])
        
        if clause_list and atom_list:
            ax2.plot(clause_list, atom_list, 'o-', label=f'Max {max_occ} occurrences', 
                    linewidth=2, markersize=6)
    
    ax2.set_title('Variable Count Scaling by Constraint', fontweight='bold')
    ax2.set_xlabel('Number of Clauses')
    ax2.set_ylabel('Number of Variables')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Constraint efficiency heatmap
    ax3 = axes[1, 0]
    
    # Create matrix for heatmap
    efficiency_matrix = []
    clause_labels = []
    
    for clause_count in clause_counts:
        efficiency_row = []
        clause_labels.append(f'{clause_count}C')
        
        for max_occ in max_occurrences:
            if max_occ in data[clause_count]:
                constraint_data = data[clause_count][max_occ]
                
                # Calculate efficiency as average performance across solvers
                efficiencies = []
                for solver in solvers:
                    memory, time, sat_result = extract_solver_metrics(constraint_data['data'], solver)
                    if memory > 0 and time > 0:
                        efficiency = 1 / ((memory + 0.001) * (time + 0.001))
                        efficiencies.append(efficiency)
                
                avg_efficiency = np.mean(efficiencies) if efficiencies else 0
                efficiency_row.append(avg_efficiency)
            else:
                efficiency_row.append(0)
        
        efficiency_matrix.append(efficiency_row)
    
    im = ax3.imshow(efficiency_matrix, cmap='YlGnBu', aspect='auto')
    ax3.set_title('Solver Efficiency by Constraint', fontweight='bold')
    ax3.set_xlabel('Max Variable Occurrences')
    ax3.set_ylabel('Clause Count')
    ax3.set_xticks(range(len(max_occurrences)))
    ax3.set_xticklabels([f'Max {mo}' for mo in max_occurrences])
    ax3.set_yticks(range(len(clause_labels)))
    ax3.set_yticklabels(clause_labels)
    
    # Add text annotations
    for i in range(len(clause_labels)):
        for j in range(len(max_occurrences)):
            if efficiency_matrix[i][j] > 0:
                text = ax3.text(j, i, f'{efficiency_matrix[i][j]:.1f}', 
                               ha="center", va="center", color="black", fontsize=10)
    
    plt.colorbar(im, ax=ax3)
    
    # 4. SAT success rate analysis
    ax4 = axes[1, 1]
    
    sat_success_data = {}
    
    for max_occ in max_occurrences:
        success_rates = []
        
        for clause_count in clause_counts:
            if max_occ in data[clause_count]:
                constraint_data = data[clause_count][max_occ]
                
                sat_count = 0
                total_count = 0
                
                for solver in solvers:
                    memory, time, sat_result = extract_solver_metrics(constraint_data['data'], solver)
                    if memory > 0 or time > 0:
                        total_count += 1
                        if sat_result:
                            sat_count += 1
                
                success_rate = (sat_count / total_count * 100) if total_count > 0 else 0
                success_rates.append(success_rate)
        
        if success_rates:
            sat_success_data[f'Max {max_occ}'] = np.mean(success_rates)
    
    if sat_success_data:
        labels = list(sat_success_data.keys())
        values = list(sat_success_data.values())
        
        bars = ax4.bar(labels, values, alpha=0.7, color=['skyblue', 'lightgreen', 'lightcoral'])
        ax4.set_title('SAT Success Rate by Constraint', fontweight='bold')
        ax4.set_xlabel('Variable Occurrence Constraint')
        ax4.set_ylabel('SAT Success Rate (%)')
        ax4.set_ylim(0, 100)
        
        # Add value labels on bars
        for i, v in enumerate(values):
            ax4.text(i, v + 2, f'{v:.1f}%', ha='center', va='bottom')
        
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('problem12_plots/p12_sparsity_density_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_solver_constraint_sensitivity(data):
    """Create analysis of individual solver sensitivity to occurrence constraints"""
    
    solvers = ['vampire', 'snake', 'z3', 'cvc5', 'e']
    clause_counts = sorted(data.keys())
    max_occurrences = [2, 3, 5]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    fig.suptitle('Problem 12: Individual Solver Sensitivity to Variable Occurrence Constraints', 
                 fontsize=16, fontweight='bold')
    
    # Create subplot for each solver
    for solver_idx, solver in enumerate(solvers):
        if solver_idx >= len(axes):
            break
            
        ax = axes[solver_idx]
        
        for clause_count in clause_counts:
            memories = []
            times = []
            constraints = []
            
            for max_occ in max_occurrences:
                if max_occ in data[clause_count]:
                    constraint_data = data[clause_count][max_occ]
                    memory, time, sat_result = extract_solver_metrics(constraint_data['data'], solver)
                    
                    if memory > 0 or time > 0:
                        memories.append(memory)
                        times.append(time)
                        constraints.append(max_occ)
            
            if times:
                ax.plot(constraints, times, 'o-', label=f'{clause_count} clauses', linewidth=2, markersize=6)
        
        ax.set_title(f'{solver.title()} Time Sensitivity', fontweight='bold')
        ax.set_xlabel('Max Variable Occurrences')
        ax.set_ylabel('Execution Time (s)')
        ax.set_yscale('log')
        ax.set_xticks(max_occurrences)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Remove empty subplots
    for i in range(len(solvers), len(axes)):
        fig.delaxes(axes[i])
    
    plt.tight_layout()
    plt.savefig('problem12_plots/p12_solver_constraint_sensitivity.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_modular_behavior_analysis(data):
    """Create analysis of modular/diluted dependency effects"""
    
    solvers = ['vampire', 'snake', 'z3', 'cvc5', 'e']
    clause_counts = sorted(data.keys())
    max_occurrences = [2, 3, 5]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Problem 12: Modular Behavior and Diluted Dependencies Analysis', 
                 fontsize=16, fontweight='bold')
    
    # 1. Dependency strength analysis
    ax1 = axes[0, 0]
    
    for clause_count in clause_counts:
        dependency_strengths = []
        avg_solve_times = []
        
        for max_occ in max_occurrences:
            if max_occ in data[clause_count]:
                constraint_data = data[clause_count][max_occ]
                atoms = constraint_data['atoms']
                
                # Calculate dependency strength as max_occurrences / atoms ratio
                dependency_strength = max_occ / atoms
                dependency_strengths.append(dependency_strength)
                
                # Calculate average solve time
                solve_times = []
                for solver in solvers:
                    memory, time, sat_result = extract_solver_metrics(constraint_data['data'], solver)
                    if time > 0:
                        solve_times.append(time)
                
                avg_time = np.mean(solve_times) if solve_times else 0
                avg_solve_times.append(avg_time)
        
        if dependency_strengths and avg_solve_times:
            ax1.scatter(dependency_strengths, avg_solve_times, 
                       label=f'{clause_count} clauses', s=80, alpha=0.7)
            
            # Add constraint labels
            for i, max_occ in enumerate(max_occurrences[:len(dependency_strengths)]):
                ax1.annotate(f'max{max_occ}', 
                           (dependency_strengths[i], avg_solve_times[i]),
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    ax1.set_title('Solve Time vs Dependency Strength', fontweight='bold')
    ax1.set_xlabel('Dependency Strength (Max Occ / Variables)')
    ax1.set_ylabel('Average Solve Time (s)')
    ax1.set_yscale('log')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Modularity impact on memory usage
    ax2 = axes[0, 1]
    
    constraint_labels = [f'Max {mo}' for mo in max_occurrences]
    memory_by_constraint = {label: [] for label in constraint_labels}
    
    for clause_count in clause_counts:
        for i, max_occ in enumerate(max_occurrences):
            if max_occ in data[clause_count]:
                constraint_data = data[clause_count][max_occ]
                
                # Calculate average memory usage
                memories = []
                for solver in solvers:
                    memory, time, sat_result = extract_solver_metrics(constraint_data['data'], solver)
                    if memory > 0:
                        memories.append(memory)
                
                avg_memory = np.mean(memories) if memories else 0
                memory_by_constraint[constraint_labels[i]].append(avg_memory)
    
    # Create box plots
    memory_data = [memory_by_constraint[label] for label in constraint_labels if memory_by_constraint[label]]
    valid_labels = [label for label in constraint_labels if memory_by_constraint[label]]
    
    if memory_data:
        ax2.boxplot(memory_data, tick_labels=valid_labels)
        ax2.set_title('Memory Usage Distribution by Constraint', fontweight='bold')
        ax2.set_xlabel('Variable Occurrence Constraint')
        ax2.set_ylabel('Memory Usage (MB)')
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3)
    
    # 3. Constraint complexity vs performance
    ax3 = axes[1, 0]
    
    # Calculate theoretical complexity based on variable reuse
    complexity_scores = []
    performance_scores = []
    labels = []
    
    for clause_count in clause_counts:
        for max_occ in max_occurrences:
            if max_occ in data[clause_count]:
                constraint_data = data[clause_count][max_occ]
                atoms = constraint_data['atoms']
                
                # Complexity as interaction potential (lower max_occ = higher complexity due to more variables)
                complexity = atoms * clause_count / max_occ  # More variables and constraints = higher complexity
                complexity_scores.append(complexity)
                
                # Performance as inverse of average solve time
                solve_times = []
                for solver in solvers:
                    memory, time, sat_result = extract_solver_metrics(constraint_data['data'], solver)
                    if time > 0:
                        solve_times.append(time)
                
                avg_time = np.mean(solve_times) if solve_times else 1
                performance_score = 1 / avg_time
                performance_scores.append(performance_score)
                
                labels.append(f'C{clause_count}M{max_occ}')
    
    if complexity_scores and performance_scores:
        scatter = ax3.scatter(complexity_scores, performance_scores, 
                             c=range(len(complexity_scores)), cmap='viridis', s=80, alpha=0.7)
        
        # Add labels for some points
        for i, label in enumerate(labels[::2]):  # Every other label to avoid crowding
            ax3.annotate(label, (complexity_scores[i*2], performance_scores[i*2]),
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        ax3.set_title('Performance vs Theoretical Complexity', fontweight='bold')
        ax3.set_xlabel('Theoretical Complexity Score')
        ax3.set_ylabel('Performance Score (1/Time)')
        ax3.set_xscale('log')
        ax3.set_yscale('log')
        ax3.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax3, label='Configuration Index')
    
    # 4. Variable reuse efficiency
    ax4 = axes[1, 1]
    
    reuse_efficiency = []
    constraint_types = []
    
    for max_occ in max_occurrences:
        efficiencies = []
        
        for clause_count in clause_counts:
            if max_occ in data[clause_count]:
                constraint_data = data[clause_count][max_occ]
                atoms = constraint_data['atoms']
                
                # Efficiency as clauses per variable
                efficiency = clause_count / atoms
                efficiencies.append(efficiency)
        
        if efficiencies:
            avg_efficiency = np.mean(efficiencies)
            reuse_efficiency.append(avg_efficiency)
            constraint_types.append(f'Max {max_occ}')
    
    if reuse_efficiency:
        bars = ax4.bar(constraint_types, reuse_efficiency, alpha=0.7, 
                      color=['lightblue', 'lightgreen', 'lightcoral'])
        ax4.set_title('Variable Reuse Efficiency', fontweight='bold')
        ax4.set_xlabel('Max Variable Occurrences')
        ax4.set_ylabel('Clauses per Variable')
        
        # Add value labels on bars
        for i, v in enumerate(reuse_efficiency):
            ax4.text(i, v + max(reuse_efficiency) * 0.01, f'{v:.2f}', ha='center', va='bottom')
        
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('problem12_plots/p12_modular_behavior_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def print_problem12_summary(data):
    """Print a comprehensive summary of Problem 12 analysis"""
    print("=" * 70)
    print("PROBLEM 12 ANALYSIS SUMMARY")
    print("=" * 70)
    print("Study: Impact of Variable Occurrence Constraints on Solver Performance")
    print()
    print("Constraint specifications:")
    print("- Clause counts: 100, 200, 500")
    print("- Max variable occurrences: 2, 3, 5")
    print("- Full variable coverage ensured in all formulas")
    print("- Equal proportions of safety/liveness clauses (50%/50%)")
    print("- Random formula generation with occurrence constraints")
    print()
    
    solvers = ['vampire', 'snake', 'z3', 'cvc5', 'e']
    
    for clause_count in sorted(data.keys()):
        print(f"Clause Count: {clause_count}")
        
        for max_occ in [2, 3, 5]:
            if max_occ in data[clause_count]:
                print(f"  Max {max_occ} occurrences per variable:")
                constraint_data = data[clause_count][max_occ]
                atoms = constraint_data['atoms']
                print(f"    Variables required: {atoms}")
                print(f"    Variable density: {clause_count/atoms:.2f} clauses/variable")
                
                for solver in solvers:
                    memory, time, sat_result = extract_solver_metrics(constraint_data['data'], solver)
                    if memory > 0 or time > 0:
                        result_str = "SAT" if sat_result else "UNSAT"
                        print(f"      {solver}: {result_str} (Mem: {memory:.3f} MB, Time: {time:.3f} s)")
        print()
    
    print("Key insights:")
    print("- Stricter occurrence constraints require more variables for coverage")
    print("- Variable reuse patterns significantly affect solver performance")
    print("- Sparse formulas (max 2 occurrences) create different solving challenges")
    print("- Modular behavior emerges from limited variable interactions")
    print()

def main():
    """Main function to run Problem 12 analysis"""
    # Create output directory
    os.makedirs('problem12_plots', exist_ok=True)
    
    # Load data
    print("Loading Problem 12 data...")
    data = load_problem12_data()
    
    if not data:
        print("No data found! Please check the data directory path.")
        return
    
    # Print summary
    print_problem12_summary(data)
    
    # Create comprehensive analysis plots
    print("Creating variable occurrence constraint analysis...")
    create_occurrence_constraint_analysis(data)
    
    print("Creating sparsity and density analysis...")
    create_sparsity_density_analysis(data)
    
    print("Creating solver constraint sensitivity analysis...")
    create_solver_constraint_sensitivity(data)
    
    print("Creating modular behavior analysis...")
    create_modular_behavior_analysis(data)
    
    print("Problem 12 analysis complete! Plots saved in problem12_plots/ directory.")

if __name__ == "__main__":
    main() 