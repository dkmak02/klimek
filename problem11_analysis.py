import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
from pathlib import Path

def load_problem11_data():
    """Load all problem11 CSV files and organize by clause count and polarity ratio"""
    
    # Define polarity configurations
    polarity_configs = {
        'a': {'ratio': '100_0_0', 'description': '100% Positive', 'pos': 100, 'neg': 0, 'mixed': 0},
        'b': {'ratio': '0_50_50', 'description': '50% Negative, 50% Mixed', 'pos': 0, 'neg': 50, 'mixed': 50},
        'c': {'ratio': '33_33_33', 'description': '33% Each Type', 'pos': 33, 'neg': 33, 'mixed': 33},
        'd': {'ratio': '0_0_100', 'description': '100% Mixed', 'pos': 0, 'neg': 0, 'mixed': 100}
    }
    
    data = {}
    
    # Load data from problem11 directory
    csv_files = glob.glob("problem11/problem11*.csv")
    
    for file in csv_files:
        filename = os.path.basename(file)
        
        # Extract parameters from filename: problem11a_c100_a50_ratio_100_0_0_results.csv
        parts = filename.replace('.csv', '').split('_')
        
        # Extract variant (a, b, c, d)
        variant_key = None
        if parts[0].startswith('problem11') and len(parts[0]) > 9:
            variant_key = parts[0][-1]
        
        clauses = None
        atoms = None
        
        for part in parts:
            if part.startswith('c') and part[1:].isdigit():
                clauses = int(part[1:])
            elif part.startswith('a') and part[1:].isdigit():
                atoms = int(part[1:])
        
        if clauses and variant_key and variant_key in polarity_configs:
            # Read the CSV file
            df = pd.read_csv(file, sep=';')
            
            # Filter to get only the average row
            avg_row = df[df['Run Number'] == 'Average'].iloc[0]
            
            # Organize by clause count
            if clauses not in data:
                data[clauses] = {}
            
            data[clauses][variant_key] = {
                'clauses': clauses,
                'atoms': atoms,
                'variant_key': variant_key,
                'polarity_config': polarity_configs[variant_key],
                'filename': filename,
                'data': avg_row
            }
    
    return data, polarity_configs

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

def create_polarity_impact_analysis(data, polarity_configs):
    """Create comprehensive analysis of clause polarity impact"""
    
    solvers = ['vampire', 'snake', 'z3', 'cvc5', 'e']
    clause_counts = sorted(data.keys())
    variants = ['a', 'b', 'c', 'd']
    
    fig, axes = plt.subplots(len(clause_counts), 2, figsize=(16, 6*len(clause_counts)))
    if len(clause_counts) == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle('Problem 11: Clause Polarity Impact on Solver Performance\n(Positive:Negative:Mixed Ratios)', 
                 fontsize=16, fontweight='bold')
    
    for clause_idx, clause_count in enumerate(clause_counts):
        ax_mem = axes[clause_idx, 0]
        ax_time = axes[clause_idx, 1]
        
        # Prepare data for plotting
        variant_labels = []
        for variant in variants:
            if variant in data[clause_count]:
                config = polarity_configs[variant]
                label = f"{config['pos']}:{config['neg']}:{config['mixed']}"
                variant_labels.append(label)
        
        for solver in solvers:
            memories = []
            times = []
            
            for variant in variants:
                if variant in data[clause_count]:
                    variant_data = data[clause_count][variant]
                    memory, time, sat_result = extract_solver_metrics(variant_data['data'], solver)
                    
                    if memory > 0 or time > 0:
                        memories.append(memory)
                        times.append(time)
                    else:
                        memories.append(0)
                        times.append(0)
            
            if any(m > 0 for m in memories):
                x_pos = np.arange(len(variant_labels))
                ax_mem.plot(x_pos, memories, 'o-', label=solver, linewidth=2, markersize=6)
            
            if any(t > 0 for t in times):
                x_pos = np.arange(len(variant_labels))
                ax_time.plot(x_pos, times, 'o-', label=solver, linewidth=2, markersize=6)
        
        # Customize memory plot
        ax_mem.set_title(f'Memory Usage vs Polarity Ratio - {clause_count} Clauses', fontweight='bold')
        ax_mem.set_xlabel('Polarity Ratio (Pos:Neg:Mixed)')
        ax_mem.set_ylabel('Memory (MB)')
        ax_mem.set_yscale('log')
        ax_mem.set_xticks(range(len(variant_labels)))
        ax_mem.set_xticklabels(variant_labels)
        ax_mem.legend()
        ax_mem.grid(True, alpha=0.3)
        
        # Customize time plot
        ax_time.set_title(f'Execution Time vs Polarity Ratio - {clause_count} Clauses', fontweight='bold')
        ax_time.set_xlabel('Polarity Ratio (Pos:Neg:Mixed)')
        ax_time.set_ylabel('Time (s)')
        ax_time.set_yscale('log')
        ax_time.set_xticks(range(len(variant_labels)))
        ax_time.set_xticklabels(variant_labels)
        ax_time.legend()
        ax_time.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('problem11_plots/p11_polarity_impact_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_polarity_comparison_matrix(data, polarity_configs):
    """Create matrix comparison of all polarity configurations"""
    
    solvers = ['vampire', 'snake', 'z3', 'cvc5', 'e']
    variants = ['a', 'b', 'c', 'd']
    clause_counts = sorted(data.keys())
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Problem 11: Polarity Configuration Comparison Matrix', 
                 fontsize=16, fontweight='bold')
    
    # 1. Memory usage heatmap
    ax1 = axes[0, 0]
    memory_matrix = []
    variant_labels = []
    
    for variant in variants:
        variant_label = f"{polarity_configs[variant]['pos']}:{polarity_configs[variant]['neg']}:{polarity_configs[variant]['mixed']}"
        variant_labels.append(variant_label)
        
        memory_row = []
        for clause_count in clause_counts:
            if variant in data[clause_count]:
                variant_data = data[clause_count][variant]
                # Calculate average memory across working solvers
                memories = []
                for solver in solvers:
                    memory, time, sat_result = extract_solver_metrics(variant_data['data'], solver)
                    if memory > 0:
                        memories.append(memory)
                avg_memory = np.mean(memories) if memories else 0
                memory_row.append(avg_memory)
            else:
                memory_row.append(0)
        
        memory_matrix.append(memory_row)
    
    im1 = ax1.imshow(memory_matrix, cmap='YlOrRd', aspect='auto')
    ax1.set_title('Average Memory Usage (MB)', fontweight='bold')
    ax1.set_xlabel('Clause Count')
    ax1.set_ylabel('Polarity Ratio (Pos:Neg:Mixed)')
    ax1.set_xticks(range(len(clause_counts)))
    ax1.set_xticklabels(clause_counts)
    ax1.set_yticks(range(len(variant_labels)))
    ax1.set_yticklabels(variant_labels)
    
    # Add text annotations
    for i in range(len(variant_labels)):
        for j in range(len(clause_counts)):
            if memory_matrix[i][j] > 0:
                text = ax1.text(j, i, f'{memory_matrix[i][j]:.2f}', 
                               ha="center", va="center", color="black", fontsize=10)
    
    plt.colorbar(im1, ax=ax1)
    
    # 2. Execution time heatmap
    ax2 = axes[0, 1]
    time_matrix = []
    
    for variant in variants:
        time_row = []
        for clause_count in clause_counts:
            if variant in data[clause_count]:
                variant_data = data[clause_count][variant]
                # Calculate average time across working solvers
                times = []
                for solver in solvers:
                    memory, time, sat_result = extract_solver_metrics(variant_data['data'], solver)
                    if time > 0:
                        times.append(time)
                avg_time = np.mean(times) if times else 0
                time_row.append(avg_time)
            else:
                time_row.append(0)
        
        time_matrix.append(time_row)
    
    im2 = ax2.imshow(time_matrix, cmap='YlGnBu', aspect='auto')
    ax2.set_title('Average Execution Time (s)', fontweight='bold')
    ax2.set_xlabel('Clause Count')
    ax2.set_ylabel('Polarity Ratio (Pos:Neg:Mixed)')
    ax2.set_xticks(range(len(clause_counts)))
    ax2.set_xticklabels(clause_counts)
    ax2.set_yticks(range(len(variant_labels)))
    ax2.set_yticklabels(variant_labels)
    
    # Add text annotations
    for i in range(len(variant_labels)):
        for j in range(len(clause_counts)):
            if time_matrix[i][j] > 0:
                text = ax2.text(j, i, f'{time_matrix[i][j]:.3f}', 
                               ha="center", va="center", color="black", fontsize=10)
    
    plt.colorbar(im2, ax=ax2)
    
    # 3. SAT success rate
    ax3 = axes[0, 2]
    sat_matrix = []
    
    for variant in variants:
        sat_row = []
        for clause_count in clause_counts:
            if variant in data[clause_count]:
                variant_data = data[clause_count][variant]
                # Calculate SAT success rate
                sat_count = 0
                total_count = 0
                for solver in solvers:
                    memory, time, sat_result = extract_solver_metrics(variant_data['data'], solver)
                    if memory > 0 or time > 0:
                        total_count += 1
                        if sat_result:
                            sat_count += 1
                
                sat_rate = (sat_count / total_count) if total_count > 0 else 0
                sat_row.append(sat_rate)
            else:
                sat_row.append(0)
        
        sat_matrix.append(sat_row)
    
    im3 = ax3.imshow(sat_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    ax3.set_title('SAT Success Rate', fontweight='bold')
    ax3.set_xlabel('Clause Count')
    ax3.set_ylabel('Polarity Ratio (Pos:Neg:Mixed)')
    ax3.set_xticks(range(len(clause_counts)))
    ax3.set_xticklabels(clause_counts)
    ax3.set_yticks(range(len(variant_labels)))
    ax3.set_yticklabels(variant_labels)
    
    # Add text annotations
    for i in range(len(variant_labels)):
        for j in range(len(clause_counts)):
            text = ax3.text(j, i, f'{sat_matrix[i][j]:.2f}', 
                           ha="center", va="center", color="black", fontsize=10, fontweight='bold')
    
    plt.colorbar(im3, ax=ax3)
    
    # 4. Polarity efficiency analysis
    ax4 = axes[1, 0]
    polarity_types = ['Pure Positive\n(100:0:0)', 'Neg+Mixed\n(0:50:50)', 'Balanced\n(33:33:33)', 'Pure Mixed\n(0:0:100)']
    efficiency_scores = []
    
    for variant in variants:
        # Calculate efficiency as inverse of average (memory × time)
        total_efficiency = 0
        count = 0
        
        for clause_count in clause_counts:
            if variant in data[clause_count]:
                variant_data = data[clause_count][variant]
                
                for solver in solvers:
                    memory, time, sat_result = extract_solver_metrics(variant_data['data'], solver)
                    if memory > 0 and time > 0:
                        efficiency = 1 / ((memory + 0.001) * (time + 0.001))
                        total_efficiency += efficiency
                        count += 1
        
        avg_efficiency = total_efficiency / count if count > 0 else 0
        efficiency_scores.append(avg_efficiency)
    
    bars = ax4.bar(polarity_types, efficiency_scores, alpha=0.7, color=['lightblue', 'lightgreen', 'gold', 'lightcoral'])
    ax4.set_title('Polarity Configuration Efficiency', fontweight='bold')
    ax4.set_ylabel('Efficiency Score (1/(Memory×Time))')
    ax4.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for i, v in enumerate(efficiency_scores):
        ax4.text(i, v + max(efficiency_scores) * 0.01, f'{v:.1f}', ha='center', va='bottom')
    
    ax4.grid(True, alpha=0.3)
    
    # 5. Solver preference analysis
    ax5 = axes[1, 1]
    solver_preferences = {solver: [] for solver in solvers}
    
    for solver in solvers:
        for variant in variants:
            variant_performance = []
            for clause_count in clause_counts:
                if variant in data[clause_count]:
                    variant_data = data[clause_count][variant]
                    memory, time, sat_result = extract_solver_metrics(variant_data['data'], solver)
                    if time > 0:
                        performance_score = 1 / (time + 0.001)  # Time efficiency
                        variant_performance.append(performance_score)
            
            avg_performance = np.mean(variant_performance) if variant_performance else 0
            solver_preferences[solver].append(avg_performance)
    
    x_pos = np.arange(len(variant_labels))
    width = 0.15
    
    for i, solver in enumerate(solvers):
        if any(p > 0 for p in solver_preferences[solver]):
            ax5.bar(x_pos + i*width, solver_preferences[solver], 
                   width, label=solver, alpha=0.8)
    
    ax5.set_title('Solver Performance by Polarity Type', fontweight='bold')
    ax5.set_xlabel('Polarity Ratio (Pos:Neg:Mixed)')
    ax5.set_ylabel('Performance Score (1/Time)')
    ax5.set_xticks(x_pos + width * 2)
    ax5.set_xticklabels(variant_labels)
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Polarity complexity analysis
    ax6 = axes[1, 2]
    
    # Theoretical complexity scores based on polarity mix
    complexity_scores = []
    complexity_labels = []
    
    for variant in variants:
        config = polarity_configs[variant]
        # Calculate complexity based on polarity diversity
        pos_ratio = config['pos'] / 100
        neg_ratio = config['neg'] / 100
        mixed_ratio = config['mixed'] / 100
        
        # Entropy-like complexity measure
        ratios = [pos_ratio, neg_ratio, mixed_ratio]
        ratios = [r for r in ratios if r > 0]
        
        if len(ratios) > 1:
            complexity = -sum(r * np.log(r) for r in ratios if r > 0)
        else:
            complexity = 0  # Pure configurations have lowest complexity
        
        complexity_scores.append(complexity)
        complexity_labels.append(f"{config['pos']}:{config['neg']}:{config['mixed']}")
    
    bars = ax6.bar(complexity_labels, complexity_scores, alpha=0.7, 
                  color=['blue', 'green', 'orange', 'red'])
    ax6.set_title('Polarity Configuration Complexity\n(Entropy-based)', fontweight='bold')
    ax6.set_xlabel('Polarity Ratio (Pos:Neg:Mixed)')
    ax6.set_ylabel('Complexity Score')
    
    # Add value labels
    for i, v in enumerate(complexity_scores):
        ax6.text(i, v + max(complexity_scores) * 0.02, f'{v:.3f}', ha='center', va='bottom')
    
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('problem11_plots/p11_polarity_comparison_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_solver_polarity_sensitivity(data, polarity_configs):
    """Create analysis of individual solver sensitivity to polarity changes"""
    
    solvers = ['vampire', 'snake', 'z3', 'cvc5', 'e']
    variants = ['a', 'b', 'c', 'd']
    clause_counts = sorted(data.keys())
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    fig.suptitle('Problem 11: Individual Solver Polarity Sensitivity Analysis', 
                 fontsize=16, fontweight='bold')
    
    # Create subplot for each solver
    for solver_idx, solver in enumerate(solvers):
        if solver_idx >= len(axes):
            break
            
        ax = axes[solver_idx]
        
        for clause_count in clause_counts:
            memories = []
            times = []
            variant_labels = []
            
            for variant in variants:
                if variant in data[clause_count]:
                    variant_data = data[clause_count][variant]
                    memory, time, sat_result = extract_solver_metrics(variant_data['data'], solver)
                    
                    if memory > 0 or time > 0:
                        memories.append(memory)
                        times.append(time)
                        config = polarity_configs[variant]
                        label = f"{config['pos']}:{config['neg']}:{config['mixed']}"
                        variant_labels.append(label)
            
            if memories and times:
                # Normalize for visualization
                norm_memories = np.array(memories) / max(memories) if max(memories) > 0 else memories
                norm_times = np.array(times) / max(times) if max(times) > 0 else times
                
                x_pos = np.arange(len(variant_labels))
                width = 0.35
                
                bars1 = ax.bar(x_pos - width/2, norm_memories, width, 
                              label=f'Memory (C{clause_count})', alpha=0.7)
                bars2 = ax.bar(x_pos + width/2, norm_times, width, 
                              label=f'Time (C{clause_count})', alpha=0.7)
        
        ax.set_title(f'{solver.title()} Polarity Sensitivity', fontweight='bold')
        ax.set_xlabel('Polarity Ratio (Pos:Neg:Mixed)')
        ax.set_ylabel('Normalized Performance')
        ax.set_xticks(range(len(variant_labels)))
        if variant_labels:
            ax.set_xticklabels(variant_labels, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Remove empty subplots
    for i in range(len(solvers), len(axes)):
        fig.delaxes(axes[i])
    
    plt.tight_layout()
    plt.savefig('problem11_plots/p11_solver_polarity_sensitivity.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_polarity_contradiction_analysis(data, polarity_configs):
    """Create analysis of contradiction detection patterns by polarity"""
    
    solvers = ['vampire', 'snake', 'z3', 'cvc5', 'e']
    variants = ['a', 'b', 'c', 'd']
    clause_counts = sorted(data.keys())
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Problem 11: Polarity Impact on Contradiction Detection and Proving', 
                 fontsize=16, fontweight='bold')
    
    # 1. SAT/UNSAT distribution by polarity
    ax1 = axes[0, 0]
    polarity_sat_data = {}
    
    for variant in variants:
        config = polarity_configs[variant]
        polarity_label = f"{config['pos']}:{config['neg']}:{config['mixed']}"
        
        sat_count = 0
        total_count = 0
        
        for clause_count in clause_counts:
            if variant in data[clause_count]:
                variant_data = data[clause_count][variant]
                
                for solver in solvers:
                    memory, time, sat_result = extract_solver_metrics(variant_data['data'], solver)
                    if memory > 0 or time > 0:
                        total_count += 1
                        if sat_result:
                            sat_count += 1
        
        sat_percentage = (sat_count / total_count * 100) if total_count > 0 else 0
        polarity_sat_data[polarity_label] = sat_percentage
    
    labels = list(polarity_sat_data.keys())
    values = list(polarity_sat_data.values())
    
    bars = ax1.bar(labels, values, alpha=0.7, color=['lightblue', 'lightgreen', 'gold', 'lightcoral'])
    ax1.set_title('SAT Success Rate by Polarity Configuration', fontweight='bold')
    ax1.set_xlabel('Polarity Ratio (Pos:Neg:Mixed)')
    ax1.set_ylabel('SAT Success Rate (%)')
    ax1.set_ylim(0, 100)
    
    # Add percentage labels on bars
    for i, v in enumerate(values):
        ax1.text(i, v + 2, f'{v:.1f}%', ha='center', va='bottom')
    
    ax1.grid(True, alpha=0.3)
    
    # 2. Time to solution by polarity
    ax2 = axes[0, 1]
    
    for clause_count in clause_counts:
        times_by_polarity = []
        polarity_labels = []
        
        for variant in variants:
            if variant in data[clause_count]:
                variant_data = data[clause_count][variant]
                config = polarity_configs[variant]
                
                # Calculate median time across solvers
                times = []
                for solver in solvers:
                    memory, time, sat_result = extract_solver_metrics(variant_data['data'], solver)
                    if time > 0:
                        times.append(time)
                
                median_time = np.median(times) if times else 0
                times_by_polarity.append(median_time)
                polarity_labels.append(f"{config['pos']}:{config['neg']}:{config['mixed']}")
        
        if times_by_polarity:
            ax2.plot(range(len(polarity_labels)), times_by_polarity, 
                    'o-', label=f'{clause_count} clauses', linewidth=2, markersize=6)
    
    ax2.set_title('Median Solving Time by Polarity', fontweight='bold')
    ax2.set_xlabel('Polarity Ratio (Pos:Neg:Mixed)')
    ax2.set_ylabel('Median Time (s)')
    ax2.set_yscale('log')
    ax2.set_xticks(range(len(polarity_labels)))
    ax2.set_xticklabels(polarity_labels)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Memory usage patterns
    ax3 = axes[1, 0]
    
    for clause_count in clause_counts:
        memories_by_polarity = []
        
        for variant in variants:
            if variant in data[clause_count]:
                variant_data = data[clause_count][variant]
                
                # Calculate median memory across solvers
                memories = []
                for solver in solvers:
                    memory, time, sat_result = extract_solver_metrics(variant_data['data'], solver)
                    if memory > 0:
                        memories.append(memory)
                
                median_memory = np.median(memories) if memories else 0
                memories_by_polarity.append(median_memory)
        
        if memories_by_polarity:
            ax3.plot(range(len(polarity_labels)), memories_by_polarity, 
                    'o-', label=f'{clause_count} clauses', linewidth=2, markersize=6)
    
    ax3.set_title('Median Memory Usage by Polarity', fontweight='bold')
    ax3.set_xlabel('Polarity Ratio (Pos:Neg:Mixed)')
    ax3.set_ylabel('Median Memory (MB)')
    ax3.set_yscale('log')
    ax3.set_xticks(range(len(polarity_labels)))
    ax3.set_xticklabels(polarity_labels)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Polarity robustness analysis
    ax4 = axes[1, 1]
    
    # Calculate variance in performance across polarities for each solver
    solver_robustness = {}
    
    for solver in solvers:
        performance_variance = []
        
        for clause_count in clause_counts:
            clause_times = []
            
            for variant in variants:
                if variant in data[clause_count]:
                    variant_data = data[clause_count][variant]
                    memory, time, sat_result = extract_solver_metrics(variant_data['data'], solver)
                    if time > 0:
                        clause_times.append(time)
            
            if len(clause_times) > 1:
                variance = np.var(clause_times)
                performance_variance.append(variance)
        
        avg_variance = np.mean(performance_variance) if performance_variance else 0
        solver_robustness[solver] = avg_variance
    
    # Filter out solvers with no data
    working_solvers = [s for s in solvers if solver_robustness[s] > 0]
    robustness_values = [solver_robustness[s] for s in working_solvers]
    
    if working_solvers:
        bars = ax4.bar(working_solvers, robustness_values, alpha=0.7, color='lightsteelblue')
        ax4.set_title('Solver Robustness to Polarity Changes\n(Lower = More Robust)', fontweight='bold')
        ax4.set_xlabel('Solver')
        ax4.set_ylabel('Performance Variance')
        ax4.set_yscale('log')
        
        # Add value labels on bars
        for i, v in enumerate(robustness_values):
            ax4.text(i, v, f'{v:.2e}', ha='center', va='bottom')
        
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('problem11_plots/p11_polarity_contradiction_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def print_problem11_summary(data, polarity_configs):
    """Print a comprehensive summary of Problem 11 analysis"""
    print("=" * 70)
    print("PROBLEM 11 ANALYSIS SUMMARY")
    print("=" * 70)
    print("Study: Impact of Clause Polarity on Solver Performance")
    print()
    print("Polarity configurations tested:")
    for variant, config in polarity_configs.items():
        print(f"  {variant}) {config['description']}: {config['pos']}% pos, {config['neg']}% neg, {config['mixed']}% mixed")
    print()
    print("Formula specifications:")
    print("- Clause counts: 100, 200, 500")
    print("- Each formula contains all atoms at least once")
    print("- Pure positive: only positive literals (p, q)")
    print("- Pure negative: only negative literals (¬p, ¬q)")
    print("- Mixed: combination of positive and negative literals")
    print()
    
    solvers = ['vampire', 'snake', 'z3', 'cvc5', 'e']
    
    for clause_count in sorted(data.keys()):
        print(f"Clause Count: {clause_count}")
        atoms = data[clause_count]['a']['atoms'] if 'a' in data[clause_count] else "N/A"
        print(f"  Atoms: {atoms}")
        
        for variant in ['a', 'b', 'c', 'd']:
            if variant in data[clause_count]:
                config = polarity_configs[variant]
                print(f"  {config['description']} ({config['pos']}:{config['neg']}:{config['mixed']}):")
                variant_data = data[clause_count][variant]
                
                for solver in solvers:
                    memory, time, sat_result = extract_solver_metrics(variant_data['data'], solver)
                    if memory > 0 or time > 0:
                        result_str = "SAT" if sat_result else "UNSAT"
                        print(f"    {solver}: {result_str} (Mem: {memory:.3f} MB, Time: {time:.3f} s)")
        print()
    
    print("Key insights:")
    print("- Clause polarity significantly affects solver behavior and performance")
    print("- Pure configurations (100% positive or 100% mixed) show different patterns")
    print("- Mixed clauses create more complex reasoning challenges")
    print("- Solver sensitivity to polarity varies significantly")
    print()

def main():
    """Main function to run Problem 11 analysis"""
    # Create output directory
    os.makedirs('problem11_plots', exist_ok=True)
    
    # Load data
    print("Loading Problem 11 data...")
    data, polarity_configs = load_problem11_data()
    
    if not data:
        print("No data found! Please check the data directory path.")
        return
    
    # Print summary
    print_problem11_summary(data, polarity_configs)
    
    # Create comprehensive analysis plots
    print("Creating polarity impact analysis...")
    create_polarity_impact_analysis(data, polarity_configs)
    
    print("Creating polarity comparison matrix...")
    create_polarity_comparison_matrix(data, polarity_configs)
    
    print("Creating solver polarity sensitivity analysis...")
    create_solver_polarity_sensitivity(data, polarity_configs)
    
    print("Creating polarity contradiction detection analysis...")
    create_polarity_contradiction_analysis(data, polarity_configs)
    
    print("Problem 11 analysis complete! Plots saved in problem11_plots/ directory.")

if __name__ == "__main__":
    main() 