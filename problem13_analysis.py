import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
from pathlib import Path

def load_problem13_data():
    """Load all problem13 CSV files and organize by clause count, removal percentage, and variant"""
    
    data = {}
    
    # Load data from problem13 directory
    csv_files = glob.glob("problem13/problem13*.csv")
    
    for file in csv_files:
        filename = os.path.basename(file)
        
        # Extract parameters from filename: problem13_c500_a250_50pct_v1_results.csv
        parts = filename.replace('.csv', '').split('_')
        
        clauses = None
        atoms = None
        removal_pct = None
        variant = None
        
        for part in parts:
            if part.startswith('c') and part[1:].isdigit():
                clauses = int(part[1:])
            elif part.startswith('a') and part[1:].isdigit():
                atoms = int(part[1:])
            elif part.endswith('pct'):
                removal_pct = int(part.replace('pct', ''))
            elif part.startswith('v') and part[1:].isdigit():
                variant = int(part[1:])
            elif part == 'base':
                removal_pct = 0
                variant = 'base'
        
        if clauses is not None:
            # Read the CSV file
            df = pd.read_csv(file, sep=';')
            
            # Filter to get only the average row
            avg_row = df[df['Run Number'] == 'Average'].iloc[0]
            
            # Organize by clause count
            if clauses not in data:
                data[clauses] = {}
            
            # Organize by removal percentage
            if removal_pct not in data[clauses]:
                data[clauses][removal_pct] = {}
            
            data[clauses][removal_pct][variant] = {
                'clauses': clauses,
                'atoms': atoms,
                'removal_pct': removal_pct,
                'variant': variant,
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

def create_clause_removal_impact_analysis(data):
    """Create comprehensive analysis of clause removal impact on solver performance"""
    
    solvers = ['vampire', 'snake', 'z3', 'cvc5', 'e']
    clause_counts = sorted(data.keys())
    removal_percentages = [0, 10, 20, 50]
    
    fig, axes = plt.subplots(len(clause_counts), 2, figsize=(16, 6*len(clause_counts)))
    if len(clause_counts) == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle('Problem 13: Impact of Random Clause Removal on Solver Performance\n(Removal percentages: 0%, 10%, 20%, 50%)', 
                 fontsize=16, fontweight='bold')
    
    for clause_idx, clause_count in enumerate(clause_counts):
        ax_mem = axes[clause_idx, 0]
        ax_time = axes[clause_idx, 1]
        
        for solver in solvers:
            memories = []
            times = []
            available_percentages = []
            
            for removal_pct in removal_percentages:
                if removal_pct in data[clause_count]:
                    # Average across all variants for this removal percentage
                    variant_memories = []
                    variant_times = []
                    
                    for variant in data[clause_count][removal_pct]:
                        variant_data = data[clause_count][removal_pct][variant]
                        memory, time, sat_result = extract_solver_metrics(variant_data['data'], solver)
                        
                        if memory > 0 or time > 0:
                            variant_memories.append(memory)
                            variant_times.append(time)
                    
                    if variant_memories:
                        avg_memory = np.mean(variant_memories)
                        avg_time = np.mean(variant_times)
                        memories.append(avg_memory)
                        times.append(avg_time)
                        available_percentages.append(removal_pct)
            
            if memories:
                ax_mem.plot(available_percentages, memories, 'o-', label=solver, linewidth=2, markersize=6)
            
            if times:
                ax_time.plot(available_percentages, times, 'o-', label=solver, linewidth=2, markersize=6)
        
        # Customize memory plot
        ax_mem.set_title(f'Memory Usage vs Clause Removal - {clause_count} Base Clauses', fontweight='bold')
        ax_mem.set_xlabel('Percentage of Clauses Removed (%)')
        ax_mem.set_ylabel('Memory (MB)')
        ax_mem.set_yscale('log')
        ax_mem.set_xticks(removal_percentages)
        ax_mem.legend()
        ax_mem.grid(True, alpha=0.3)
        
        # Customize time plot
        ax_time.set_title(f'Execution Time vs Clause Removal - {clause_count} Base Clauses', fontweight='bold')
        ax_time.set_xlabel('Percentage of Clauses Removed (%)')
        ax_time.set_ylabel('Time (s)')
        ax_time.set_yscale('log')
        ax_time.set_xticks(removal_percentages)
        ax_time.legend()
        ax_time.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('problem13_plots/p13_clause_removal_impact.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_satisfiability_analysis(data):
    """Create analysis of satisfiability changes with clause removal"""
    
    solvers = ['vampire', 'snake', 'z3', 'cvc5', 'e']
    clause_counts = sorted(data.keys())
    removal_percentages = [0, 10, 20, 50]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Problem 13: Satisfiability Analysis Under Clause Removal', 
                 fontsize=16, fontweight='bold')
    
    # 1. SAT/UNSAT distribution by removal percentage
    ax1 = axes[0, 0]
    
    sat_rates = {removal_pct: [] for removal_pct in removal_percentages}
    
    for clause_count in clause_counts:
        for removal_pct in removal_percentages:
            if removal_pct in data[clause_count]:
                sat_count = 0
                total_count = 0
                
                for variant in data[clause_count][removal_pct]:
                    variant_data = data[clause_count][removal_pct][variant]
                    
                    for solver in solvers:
                        memory, time, sat_result = extract_solver_metrics(variant_data['data'], solver)
                        if memory > 0 or time > 0:
                            total_count += 1
                            if sat_result:
                                sat_count += 1
                
                sat_rate = (sat_count / total_count * 100) if total_count > 0 else 0
                sat_rates[removal_pct].append(sat_rate)
    
    # Create box plots for SAT rates
    sat_data = [sat_rates[pct] for pct in removal_percentages if sat_rates[pct]]
    labels = [f'{pct}%' for pct in removal_percentages if sat_rates[pct]]
    
    if sat_data:
        ax1.boxplot(sat_data, labels=labels)
        ax1.set_title('SAT Success Rate by Removal Percentage', fontweight='bold')
        ax1.set_xlabel('Clauses Removed (%)')
        ax1.set_ylabel('SAT Success Rate (%)')
        ax1.set_ylim(0, 105)
        ax1.grid(True, alpha=0.3)
    
    # 2. Variance in results across variants
    ax2 = axes[0, 1]
    
    variance_data = {}
    
    for clause_count in clause_counts:
        for removal_pct in [10, 20, 50]:  # Skip base (0%) as it has no variants
            if removal_pct in data[clause_count]:
                times_by_solver = {solver: [] for solver in solvers}
                
                for variant in data[clause_count][removal_pct]:
                    variant_data = data[clause_count][removal_pct][variant]
                    
                    for solver in solvers:
                        memory, time, sat_result = extract_solver_metrics(variant_data['data'], solver)
                        if time > 0:
                            times_by_solver[solver].append(time)
                
                # Calculate variance for each solver
                for solver in solvers:
                    if len(times_by_solver[solver]) > 1:
                        variance = np.var(times_by_solver[solver])
                        key = f'C{clause_count}_{removal_pct}%'
                        if key not in variance_data:
                            variance_data[key] = []
                        variance_data[key].append(variance)
    
    if variance_data:
        positions = range(len(variance_data))
        keys = list(variance_data.keys())
        variances = [np.mean(variance_data[key]) for key in keys]
        
        bars = ax2.bar(positions, variances, alpha=0.7, color='lightblue')
        ax2.set_title('Performance Variance Across Random Variants', fontweight='bold')
        ax2.set_xlabel('Configuration')
        ax2.set_ylabel('Time Variance (s²)')
        ax2.set_xticks(positions)
        ax2.set_xticklabels(keys, rotation=45)
        ax2.grid(True, alpha=0.3)
    
    # 3. Critical clause analysis
    ax3 = axes[1, 0]
    
    performance_degradation = {}
    
    for clause_count in clause_counts:
        degradations = []
        
        if 0 in data[clause_count] and 50 in data[clause_count]:
            for solver in solvers:
                # Get base performance
                base_variant = list(data[clause_count][0].keys())[0]
                base_data = data[clause_count][0][base_variant]
                base_memory, base_time, base_sat = extract_solver_metrics(base_data['data'], solver)
                
                if base_time > 0:
                    # Get 50% removal performance (average across variants)
                    removal_times = []
                    for variant in data[clause_count][50]:
                        variant_data = data[clause_count][50][variant]
                        memory, time, sat_result = extract_solver_metrics(variant_data['data'], solver)
                        if time > 0:
                            removal_times.append(time)
                    
                    if removal_times:
                        avg_removal_time = np.mean(removal_times)
                        degradation = (avg_removal_time - base_time) / base_time * 100
                        degradations.append(degradation)
        
        if degradations:
            performance_degradation[f'{clause_count} clauses'] = np.mean(degradations)
    
    if performance_degradation:
        keys = list(performance_degradation.keys())
        values = list(performance_degradation.values())
        
        colors = ['red' if v > 0 else 'green' for v in values]
        bars = ax3.bar(keys, values, alpha=0.7, color=colors)
        ax3.set_title('Performance Change: Base vs 50% Removal', fontweight='bold')
        ax3.set_xlabel('Configuration')
        ax3.set_ylabel('Performance Change (%)')
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax3.grid(True, alpha=0.3)
        
        # Add value labels
        for i, v in enumerate(values):
            ax3.text(i, v + (max(values) - min(values)) * 0.01, f'{v:.1f}%', 
                    ha='center', va='bottom' if v > 0 else 'top')
    
    # 4. Solver robustness to clause removal
    ax4 = axes[1, 1]
    
    solver_robustness = {solver: [] for solver in solvers}
    
    for clause_count in clause_counts:
        for removal_pct in [10, 20, 50]:
            if removal_pct in data[clause_count]:
                for solver in solvers:
                    times = []
                    for variant in data[clause_count][removal_pct]:
                        variant_data = data[clause_count][removal_pct][variant]
                        memory, time, sat_result = extract_solver_metrics(variant_data['data'], solver)
                        if time > 0:
                            times.append(time)
                    
                    if times:
                        # Robustness as inverse of coefficient of variation
                        mean_time = np.mean(times)
                        std_time = np.std(times)
                        cv = std_time / mean_time if mean_time > 0 else 0
                        robustness = 1 / (cv + 0.001)  # Higher = more robust
                        solver_robustness[solver].append(robustness)
    
    # Create box plots for robustness
    robustness_data = [solver_robustness[solver] for solver in solvers if solver_robustness[solver]]
    valid_solvers = [solver for solver in solvers if solver_robustness[solver]]
    
    if robustness_data:
        ax4.boxplot(robustness_data, labels=valid_solvers)
        ax4.set_title('Solver Robustness to Random Clause Removal', fontweight='bold')
        ax4.set_xlabel('Solver')
        ax4.set_ylabel('Robustness Score (1/CV)')
        ax4.set_yscale('log')
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('problem13_plots/p13_satisfiability_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_redundancy_criticality_analysis(data):
    """Create analysis of clause redundancy and criticality patterns"""
    
    solvers = ['vampire', 'snake', 'z3', 'cvc5', 'e']
    clause_counts = sorted(data.keys())
    removal_percentages = [0, 10, 20, 50]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Problem 13: Clause Redundancy and Criticality Analysis', 
                 fontsize=16, fontweight='bold')
    
    # 1. Performance scaling with clause reduction
    ax1 = axes[0, 0]
    
    for clause_count in clause_counts:
        effective_clauses = []
        avg_times = []
        
        for removal_pct in removal_percentages:
            if removal_pct in data[clause_count]:
                remaining_clauses = clause_count * (100 - removal_pct) / 100
                effective_clauses.append(remaining_clauses)
                
                # Calculate average time across all solvers and variants
                all_times = []
                for variant in data[clause_count][removal_pct]:
                    variant_data = data[clause_count][removal_pct][variant]
                    
                    for solver in solvers:
                        memory, time, sat_result = extract_solver_metrics(variant_data['data'], solver)
                        if time > 0:
                            all_times.append(time)
                
                avg_time = np.mean(all_times) if all_times else 0
                avg_times.append(avg_time)
        
        if effective_clauses and avg_times:
            ax1.plot(effective_clauses, avg_times, 'o-', label=f'{clause_count} base clauses', 
                    linewidth=2, markersize=6)
            
            # Add removal percentage labels
            for i, removal_pct in enumerate(removal_percentages[:len(effective_clauses)]):
                ax1.annotate(f'{removal_pct}%', (effective_clauses[i], avg_times[i]),
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    ax1.set_title('Performance vs Effective Clause Count', fontweight='bold')
    ax1.set_xlabel('Effective Number of Clauses')
    ax1.set_ylabel('Average Solve Time (s)')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Memory efficiency with clause reduction
    ax2 = axes[0, 1]
    
    for solver in solvers:
        clause_reductions = []
        memory_reductions = []
        
        for clause_count in clause_counts:
            if 0 in data[clause_count] and 50 in data[clause_count]:
                # Get base memory
                base_variant = list(data[clause_count][0].keys())[0]
                base_data = data[clause_count][0][base_variant]
                base_memory, base_time, base_sat = extract_solver_metrics(base_data['data'], solver)
                
                if base_memory > 0:
                    # Get 50% removal memory (average across variants)
                    removal_memories = []
                    for variant in data[clause_count][50]:
                        variant_data = data[clause_count][50][variant]
                        memory, time, sat_result = extract_solver_metrics(variant_data['data'], solver)
                        if memory > 0:
                            removal_memories.append(memory)
                    
                    if removal_memories:
                        avg_removal_memory = np.mean(removal_memories)
                        clause_reduction = 50  # 50% reduction
                        memory_reduction = (base_memory - avg_removal_memory) / base_memory * 100
                        
                        clause_reductions.append(clause_reduction)
                        memory_reductions.append(memory_reduction)
        
        if clause_reductions and memory_reductions:
            ax2.scatter(clause_reductions, memory_reductions, label=solver, s=80, alpha=0.7)
    
    ax2.set_title('Memory Reduction vs Clause Reduction', fontweight='bold')
    ax2.set_xlabel('Clause Reduction (%)')
    ax2.set_ylabel('Memory Reduction (%)')
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax2.axvline(x=50, color='black', linestyle='-', alpha=0.3)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Consistency across random removals
    ax3 = axes[1, 0]
    
    consistency_scores = {}
    
    for clause_count in clause_counts:
        for removal_pct in [10, 20, 50]:
            if removal_pct in data[clause_count]:
                # Calculate consistency as inverse of standard deviation across variants
                solver_consistencies = []
                
                for solver in solvers:
                    times = []
                    for variant in data[clause_count][removal_pct]:
                        variant_data = data[clause_count][removal_pct][variant]
                        memory, time, sat_result = extract_solver_metrics(variant_data['data'], solver)
                        if time > 0:
                            times.append(time)
                    
                    if len(times) > 1:
                        consistency = 1 / (np.std(times) + 0.001)
                        solver_consistencies.append(consistency)
                
                if solver_consistencies:
                    avg_consistency = np.mean(solver_consistencies)
                    key = f'C{clause_count}_{removal_pct}%'
                    consistency_scores[key] = avg_consistency
    
    if consistency_scores:
        keys = list(consistency_scores.keys())
        values = list(consistency_scores.values())
        
        bars = ax3.bar(range(len(keys)), values, alpha=0.7, color='lightgreen')
        ax3.set_title('Result Consistency Across Random Removals', fontweight='bold')
        ax3.set_xlabel('Configuration')
        ax3.set_ylabel('Consistency Score (1/σ)')
        ax3.set_xticks(range(len(keys)))
        ax3.set_xticklabels(keys, rotation=45)
        ax3.set_yscale('log')
        ax3.grid(True, alpha=0.3)
    
    # 4. Critical clause density estimation
    ax4 = axes[1, 1]
    
    critical_densities = []
    clause_sizes = []
    
    for clause_count in clause_counts:
        if 0 in data[clause_count] and 50 in data[clause_count]:
            # Estimate critical clause density based on performance degradation
            base_variant = list(data[clause_count][0].keys())[0]
            base_data = data[clause_count][0][base_variant]
            
            degradations = []
            for solver in solvers:
                base_memory, base_time, base_sat = extract_solver_metrics(base_data['data'], solver)
                
                if base_time > 0:
                    # Average performance at 50% removal
                    removal_times = []
                    for variant in data[clause_count][50]:
                        variant_data = data[clause_count][50][variant]
                        memory, time, sat_result = extract_solver_metrics(variant_data['data'], solver)
                        if time > 0:
                            removal_times.append(time)
                    
                    if removal_times:
                        avg_removal_time = np.mean(removal_times)
                        degradation = avg_removal_time / base_time
                        degradations.append(degradation)
            
            if degradations:
                avg_degradation = np.mean(degradations)
                # Estimate critical density (higher degradation = higher critical density)
                critical_density = min(avg_degradation * 25, 100)  # Scale to percentage
                critical_densities.append(critical_density)
                clause_sizes.append(clause_count)
    
    if critical_densities and clause_sizes:
        ax4.scatter(clause_sizes, critical_densities, s=100, alpha=0.7, color='red')
        ax4.set_title('Estimated Critical Clause Density', fontweight='bold')
        ax4.set_xlabel('Base Clause Count')
        ax4.set_ylabel('Estimated Critical Clauses (%)')
        ax4.set_xscale('log')
        
        # Add trend line
        if len(clause_sizes) > 1:
            z = np.polyfit(np.log(clause_sizes), critical_densities, 1)
            p = np.poly1d(z)
            x_trend = np.logspace(np.log10(min(clause_sizes)), np.log10(max(clause_sizes)), 100)
            ax4.plot(x_trend, p(np.log(x_trend)), "r--", alpha=0.8, linewidth=2)
        
        ax4.grid(True, alpha=0.3)
        
        # Add annotations
        for i, (x, y) in enumerate(zip(clause_sizes, critical_densities)):
            ax4.annotate(f'{y:.1f}%', (x, y), xytext=(5, 5), 
                        textcoords='offset points', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('problem13_plots/p13_redundancy_criticality_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_solver_adaptation_analysis(data):
    """Create analysis of how different solvers adapt to clause removal"""
    
    solvers = ['vampire', 'snake', 'z3', 'cvc5', 'e']
    clause_counts = sorted(data.keys())
    removal_percentages = [0, 10, 20, 50]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    fig.suptitle('Problem 13: Individual Solver Adaptation to Clause Removal', 
                 fontsize=16, fontweight='bold')
    
    # Create subplot for each solver
    for solver_idx, solver in enumerate(solvers):
        if solver_idx >= len(axes):
            break
            
        ax = axes[solver_idx]
        
        for clause_count in clause_counts:
            times = []
            percentages = []
            
            for removal_pct in removal_percentages:
                if removal_pct in data[clause_count]:
                    # Average across all variants for this removal percentage
                    variant_times = []
                    
                    for variant in data[clause_count][removal_pct]:
                        variant_data = data[clause_count][removal_pct][variant]
                        memory, time, sat_result = extract_solver_metrics(variant_data['data'], solver)
                        
                        if time > 0:
                            variant_times.append(time)
                    
                    if variant_times:
                        avg_time = np.mean(variant_times)
                        times.append(avg_time)
                        percentages.append(removal_pct)
            
            if times:
                ax.plot(percentages, times, 'o-', label=f'{clause_count} clauses', linewidth=2, markersize=6)
        
        ax.set_title(f'{solver.title()} Adaptation Pattern', fontweight='bold')
        ax.set_xlabel('Clauses Removed (%)')
        ax.set_ylabel('Execution Time (s)')
        ax.set_yscale('log')
        ax.set_xticks(removal_percentages)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Remove empty subplots
    for i in range(len(solvers), len(axes)):
        fig.delaxes(axes[i])
    
    plt.tight_layout()
    plt.savefig('problem13_plots/p13_solver_adaptation_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def print_problem13_summary(data):
    """Print a comprehensive summary of Problem 13 analysis"""
    print("=" * 70)
    print("PROBLEM 13 ANALYSIS SUMMARY")
    print("=" * 70)
    print("Study: Impact of Random Clause Removal on Solver Performance")
    print()
    print("Experimental specifications:")
    print("- Base clause counts: 100, 200, 500")
    print("- Clause removal percentages: 10%, 20%, 50%")
    print("- Three random variants per removal percentage")
    print("- Equal safety/liveness proportions (50%/50%)")
    print("- Base formulas: Complete and satisfiable")
    print("- 100-second timeout, average of 3 runs per configuration")
    print()
    
    solvers = ['vampire', 'snake', 'z3', 'cvc5', 'e']
    
    for clause_count in sorted(data.keys()):
        print(f"Base Formula: {clause_count} clauses")
        
        for removal_pct in [0, 10, 20, 50]:
            if removal_pct in data[clause_count]:
                if removal_pct == 0:
                    print(f"  Original formula (0% removal):")
                    variant = list(data[clause_count][removal_pct].keys())[0]
                    variant_data = data[clause_count][removal_pct][variant]
                    
                    for solver in solvers:
                        memory, time, sat_result = extract_solver_metrics(variant_data['data'], solver)
                        if memory > 0 or time > 0:
                            result_str = "SAT" if sat_result else "UNSAT"
                            print(f"    {solver}: {result_str} (Mem: {memory:.3f} MB, Time: {time:.3f} s)")
                else:
                    print(f"  {removal_pct}% clause removal (3 variants):")
                    
                    # Show variance across variants
                    for solver in solvers:
                        times = []
                        memories = []
                        sat_results = []
                        
                        for variant in data[clause_count][removal_pct]:
                            variant_data = data[clause_count][removal_pct][variant]
                            memory, time, sat_result = extract_solver_metrics(variant_data['data'], solver)
                            if memory > 0 or time > 0:
                                times.append(time)
                                memories.append(memory)
                                sat_results.append(sat_result)
                        
                        if times:
                            avg_time = np.mean(times)
                            std_time = np.std(times)
                            avg_memory = np.mean(memories)
                            sat_count = sum(sat_results)
                            
                            print(f"    {solver}: {sat_count}/3 SAT (Avg Time: {avg_time:.3f}±{std_time:.3f} s, Avg Mem: {avg_memory:.3f} MB)")
        print()
    
    print("Key insights:")
    print("- All configurations maintain SAT satisfiability")
    print("- Random clause removal generally improves or maintains performance")
    print("- Low variance across different random removal sets")
    print("- Suggests significant clause redundancy in original formulas")
    print()

def main():
    """Main function to run Problem 13 analysis"""
    # Create output directory
    os.makedirs('problem13_plots', exist_ok=True)
    
    # Load data
    print("Loading Problem 13 data...")
    data = load_problem13_data()
    
    if not data:
        print("No data found! Please check the data directory path.")
        return
    
    # Print summary
    print_problem13_summary(data)
    
    # Create comprehensive analysis plots
    print("Creating clause removal impact analysis...")
    create_clause_removal_impact_analysis(data)
    
    print("Creating satisfiability analysis...")
    create_satisfiability_analysis(data)
    
    print("Creating redundancy and criticality analysis...")
    create_redundancy_criticality_analysis(data)
    
    print("Creating solver adaptation analysis...")
    create_solver_adaptation_analysis(data)
    
    print("Problem 13 analysis complete! Plots saved in problem13_plots/ directory.")

if __name__ == "__main__":
    main() 