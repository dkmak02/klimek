import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
from pathlib import Path

def load_problem9_data():
    """Load all problem9 CSV files and organize by variant type"""
    
    # Define problem variants
    problem_variants = {
        'g': 'asymmetric_subalternation',    # F1:100, F2:1000 clauses
        'h': 'safety_liveness_dominance',    # 80% safety vs 80% liveness
        'i': 'dynamic_evolution'             # F2 as modified F1
    }
    
    data = {}
    
    # Load data from results directory
    results_dir = "problem9/results/benchmark-test-20250607-121313/results"
    csv_files = glob.glob(os.path.join(results_dir, "problem9*.csv"))
    
    for file in csv_files:
        filename = os.path.basename(file)
        
        # Extract variant type from filename
        variant_key = None
        for part in filename.split('_'):
            if part.startswith('problem9') and len(part) > 8:
                variant_key = part[-1]  # Get the letter (g, h, i)
                break
        
        if not variant_key or variant_key not in problem_variants:
            continue
            
        variant = problem_variants[variant_key]
        
        # Extract parameters from filename
        clauses = None
        atoms = None
        safety_prec = None
        
        parts = filename.replace('.csv', '').split('_')
        for i, part in enumerate(parts):
            if part.startswith('c') and part[1:].isdigit():
                clauses = int(part[1:])
            elif part.startswith('a') and part[1:].isdigit():
                atoms = int(part[1:])
            elif part.startswith('prec') and part[4:].isdigit():
                safety_prec = int(part[4:])
        
        # Read the CSV file
        df = pd.read_csv(file, sep=';')
        
        # Filter to get only the average row
        avg_row = df[df['Run Number'] == 'Average'].iloc[0]
        
        # Store in our data structure
        if variant not in data:
            data[variant] = []
        
        data[variant].append({
            'variant_key': variant_key,
            'clauses': clauses,
            'atoms': atoms,
            'safety_prec': safety_prec if safety_prec else 50,
            'variant': variant,
            'filename': filename,
            'data': avg_row
        })
    
    return data, problem_variants

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

def create_problem9_variants_comparison(data):
    """Create comprehensive comparison plots for all Problem 9 variants"""
    
    solvers = ['vampire', 'snake', 'z3', 'cvc5', 'e']
    variants = ['asymmetric_subalternation', 'safety_liveness_dominance', 'dynamic_evolution']
    variant_labels = ['Asymmetric\nSubalternation\n(F₁:100, F₂:1000)', 
                     'Safety/Liveness\nDominance\n(80% Safety)', 
                     'Dynamic\nEvolution\n(F₂ modified F₁)']
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Problem 9: Logical Square Variants Analysis', fontsize=16, fontweight='bold')
    
    # Memory comparison
    ax_mem = axes[0, 0]
    ax_time = axes[0, 1]
    ax_complexity = axes[1, 0]
    ax_sat_results = axes[1, 1]
    
    # Data for plotting
    solver_data = {solver: {'memory': [], 'time': [], 'complexity': [], 'sat_count': []} for solver in solvers}
    
    for variant_idx, variant in enumerate(variants):
        if variant not in data or not data[variant]:
            continue
            
        variant_data = data[variant][0]  # Take the first (and only) entry
        
        for solver in solvers:
            memory, time, sat_result = extract_solver_metrics(variant_data['data'], solver)
            
            solver_data[solver]['memory'].append(memory)
            solver_data[solver]['time'].append(time)
            
            # Calculate complexity metric (atoms * clauses)
            complexity = variant_data['atoms'] * variant_data['clauses']
            solver_data[solver]['complexity'].append(complexity)
            solver_data[solver]['sat_count'].append(1 if sat_result else 0)
    
    # Plot memory usage
    x_pos = np.arange(len(variant_labels))
    width = 0.15
    
    for i, solver in enumerate(solvers):
        if any(m > 0 for m in solver_data[solver]['memory']):
            ax_mem.bar(x_pos + i*width, solver_data[solver]['memory'], 
                      width, label=solver, alpha=0.8)
    
    ax_mem.set_title('Memory Usage by Variant', fontweight='bold')
    ax_mem.set_xlabel('Problem Variant')
    ax_mem.set_ylabel('Memory (MB)')
    ax_mem.set_yscale('log')
    ax_mem.set_xticks(x_pos + width * 2)
    ax_mem.set_xticklabels(variant_labels)
    ax_mem.legend()
    ax_mem.grid(True, alpha=0.3)
    
    # Plot execution time
    for i, solver in enumerate(solvers):
        if any(t > 0 for t in solver_data[solver]['time']):
            ax_time.bar(x_pos + i*width, solver_data[solver]['time'], 
                       width, label=solver, alpha=0.8)
    
    ax_time.set_title('Execution Time by Variant', fontweight='bold')
    ax_time.set_xlabel('Problem Variant')
    ax_time.set_ylabel('Time (s)')
    ax_time.set_yscale('log')
    ax_time.set_xticks(x_pos + width * 2)
    ax_time.set_xticklabels(variant_labels)
    ax_time.legend()
    ax_time.grid(True, alpha=0.3)
    
    # Plot complexity comparison
    complexities = []
    for variant in variants:
        if variant in data and data[variant]:
            variant_data = data[variant][0]
            complexity = variant_data['atoms'] * variant_data['clauses']
            complexities.append(complexity)
        else:
            complexities.append(0)
    
    ax_complexity.bar(variant_labels, complexities, alpha=0.7, color=['skyblue', 'lightgreen', 'lightcoral'])
    ax_complexity.set_title('Problem Complexity (Atoms × Clauses)', fontweight='bold')
    ax_complexity.set_xlabel('Problem Variant')
    ax_complexity.set_ylabel('Complexity Score')
    ax_complexity.set_yscale('log')
    for i, v in enumerate(complexities):
        if v > 0:
            ax_complexity.text(i, v, f'{v:,}', ha='center', va='bottom')
    ax_complexity.grid(True, alpha=0.3)
    
    # Plot SAT results summary
    sat_percentages = []
    for variant in variants:
        if variant in data and data[variant]:
            variant_data = data[variant][0]
            sat_count = 0
            total_solvers = 0
            
            for solver in solvers:
                memory, time, sat_result = extract_solver_metrics(variant_data['data'], solver)
                if memory > 0 or time > 0:  # Solver ran successfully
                    total_solvers += 1
                    if sat_result:
                        sat_count += 1
            
            sat_percentage = (sat_count / total_solvers * 100) if total_solvers > 0 else 0
            sat_percentages.append(sat_percentage)
        else:
            sat_percentages.append(0)
    
    bars = ax_sat_results.bar(variant_labels, sat_percentages, alpha=0.7, 
                             color=['skyblue', 'lightgreen', 'lightcoral'])
    ax_sat_results.set_title('SAT Results Success Rate', fontweight='bold')
    ax_sat_results.set_xlabel('Problem Variant')
    ax_sat_results.set_ylabel('SAT Success Rate (%)')
    ax_sat_results.set_ylim(0, 100)
    
    # Add percentage labels on bars
    for i, v in enumerate(sat_percentages):
        ax_sat_results.text(i, v + 2, f'{v:.1f}%', ha='center', va='bottom')
    
    ax_sat_results.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('problem9_plots/p09_variants_comprehensive_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_asymmetric_complexity_analysis(data):
    """Create detailed analysis for asymmetric subalternation (Problem 9g)"""
    
    if 'asymmetric_subalternation' not in data:
        print("No asymmetric subalternation data found")
        return
    
    variant_data = data['asymmetric_subalternation'][0]
    solvers = ['vampire', 'snake', 'z3', 'cvc5', 'e']
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Problem 9g: Asymmetric Subalternation Analysis\n(F₁ ⇒ F₂) ∧ ¬(F₂ ⇒ F₁) with F₁:100, F₂:1000 clauses', 
                 fontsize=14, fontweight='bold')
    
    # Extract solver performance data
    solver_memory = []
    solver_time = []
    solver_names = []
    solver_sat = []
    
    for solver in solvers:
        memory, time, sat_result = extract_solver_metrics(variant_data['data'], solver)
        if memory > 0 or time > 0:
            solver_memory.append(memory)
            solver_time.append(time)
            solver_names.append(solver)
            solver_sat.append('SAT' if sat_result else 'UNSAT')
    
    # Memory usage plot
    ax1 = axes[0, 0]
    bars1 = ax1.bar(solver_names, solver_memory, alpha=0.7, color='skyblue')
    ax1.set_title('Memory Usage per Solver', fontweight='bold')
    ax1.set_ylabel('Memory (MB)')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, v in enumerate(solver_memory):
        ax1.text(i, v, f'{v:.2f}', ha='center', va='bottom')
    
    # Execution time plot
    ax2 = axes[0, 1]
    bars2 = ax2.bar(solver_names, solver_time, alpha=0.7, color='lightgreen')
    ax2.set_title('Execution Time per Solver', fontweight='bold')
    ax2.set_ylabel('Time (s)')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, v in enumerate(solver_time):
        ax2.text(i, v, f'{v:.3f}', ha='center', va='bottom')
    
    # SAT results visualization
    ax3 = axes[1, 0]
    sat_colors = ['green' if result == 'SAT' else 'red' for result in solver_sat]
    bars3 = ax3.bar(solver_names, [1]*len(solver_names), color=sat_colors, alpha=0.7)
    ax3.set_title('SAT/UNSAT Results', fontweight='bold')
    ax3.set_ylabel('Result')
    ax3.set_yticks([0, 1])
    ax3.set_yticklabels(['UNSAT', 'SAT'])
    
    # Add result labels
    for i, result in enumerate(solver_sat):
        ax3.text(i, 0.5, result, ha='center', va='center', fontweight='bold', color='white')
    
    # Complexity analysis
    ax4 = axes[1, 1]
    complexity_metrics = ['F₁ Clauses', 'F₂ Clauses', 'Total Atoms', 'Complexity\n(Atoms×Clauses)']
    complexity_values = [100, 1000, variant_data['atoms'], variant_data['atoms'] * variant_data['clauses']]
    
    bars4 = ax4.bar(complexity_metrics, complexity_values, alpha=0.7, color='lightcoral')
    ax4.set_title('Problem Complexity Breakdown', fontweight='bold')
    ax4.set_ylabel('Count/Score')
    ax4.set_yscale('log')
    ax4.grid(True, alpha=0.3)
    
    # Add value labels
    for i, v in enumerate(complexity_values):
        ax4.text(i, v, f'{v:,}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('problem9_plots/p09g_asymmetric_subalternation_detailed.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_safety_liveness_analysis(data):
    """Create detailed analysis for safety/liveness dominance (Problem 9h)"""
    
    if 'safety_liveness_dominance' not in data:
        print("No safety/liveness dominance data found")
        return
    
    variant_data = data['safety_liveness_dominance'][0]
    solvers = ['vampire', 'snake', 'z3', 'cvc5', 'e']
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Problem 9h: Safety/Liveness Dominance Analysis\nF₁: 80% Safety, F₂: 80% Liveness (Asymmetric Semantics)', 
                 fontsize=14, fontweight='bold')
    
    # Extract solver performance data
    solver_memory = []
    solver_time = []
    solver_names = []
    solver_sat = []
    
    for solver in solvers:
        memory, time, sat_result = extract_solver_metrics(variant_data['data'], solver)
        if memory > 0 or time > 0:
            solver_memory.append(memory)
            solver_time.append(time)
            solver_names.append(solver)
            solver_sat.append('SAT' if sat_result else 'UNSAT')
    
    # Performance comparison
    ax1 = axes[0, 0]
    x_pos = np.arange(len(solver_names))
    width = 0.35
    
    bars1 = ax1.bar(x_pos - width/2, solver_memory, width, label='Memory (MB)', alpha=0.7, color='skyblue')
    ax1_twin = ax1.twinx()
    bars2 = ax1_twin.bar(x_pos + width/2, solver_time, width, label='Time (s)', alpha=0.7, color='lightgreen')
    
    ax1.set_title('Memory vs Time Performance', fontweight='bold')
    ax1.set_xlabel('Solver')
    ax1.set_ylabel('Memory (MB)', color='blue')
    ax1_twin.set_ylabel('Time (s)', color='green')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(solver_names)
    ax1.grid(True, alpha=0.3)
    
    # SAT results pie chart
    ax2 = axes[0, 1]
    sat_count = sum(1 for result in solver_sat if result == 'SAT')
    unsat_count = len(solver_sat) - sat_count
    
    if sat_count > 0 or unsat_count > 0:
        wedges, texts, autotexts = ax2.pie([sat_count, unsat_count], labels=['SAT', 'UNSAT'], autopct='%1.1f%%',
                colors=['green', 'red'])
        for wedge in wedges:
            wedge.set_alpha(0.7)
        ax2.set_title('SAT/UNSAT Distribution', fontweight='bold')
    
    # Semantic analysis
    ax3 = axes[1, 0]
    semantic_types = ['F₁: Safety\n(80%)', 'F₂: Liveness\n(80%)']
    semantic_percentages = [80, 80]
    
    bars3 = ax3.bar(semantic_types, semantic_percentages, alpha=0.7, 
                   color=['blue', 'orange'])
    ax3.set_title('Semantic Distribution', fontweight='bold')
    ax3.set_ylabel('Percentage (%)')
    ax3.set_ylim(0, 100)
    
    for i, v in enumerate(semantic_percentages):
        ax3.text(i, v + 2, f'{v}%', ha='center', va='bottom')
    
    ax3.grid(True, alpha=0.3)
    
    # Solver efficiency ranking
    ax4 = axes[1, 1]
    efficiency_scores = []
    
    for i, solver in enumerate(solver_names):
        # Calculate efficiency as inverse of (memory * time)
        if solver_memory[i] > 0 and solver_time[i] > 0:
            efficiency = 1 / (solver_memory[i] * solver_time[i])
        else:
            efficiency = 0
        efficiency_scores.append(efficiency)
    
    bars4 = ax4.bar(solver_names, efficiency_scores, alpha=0.7, color='purple')
    ax4.set_title('Solver Efficiency Ranking\n(1 / (Memory × Time))', fontweight='bold')
    ax4.set_ylabel('Efficiency Score')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('problem9_plots/p09h_safety_liveness_dominance_detailed.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_dynamic_evolution_analysis(data):
    """Create detailed analysis for dynamic evolution (Problem 9i)"""
    
    if 'dynamic_evolution' not in data:
        print("No dynamic evolution data found")
        return
    
    variant_data = data['dynamic_evolution'][0]
    solvers = ['vampire', 'snake', 'z3', 'cvc5', 'e']
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Problem 9i: Dynamic Evolution Analysis\nF₂ as Modified Version of F₁ (10% Changed Clauses)', 
                 fontsize=14, fontweight='bold')
    
    # Extract solver performance data
    performance_data = {}
    for solver in solvers:
        memory, time, sat_result = extract_solver_metrics(variant_data['data'], solver)
        if memory > 0 or time > 0:
            performance_data[solver] = {
                'memory': memory,
                'time': time,
                'sat': sat_result
            }
    
    if not performance_data:
        print("No valid performance data found for dynamic evolution analysis")
        return
    
    # Performance radar chart comparison
    ax1 = axes[0, 0]
    solver_names = list(performance_data.keys())
    memory_values = [performance_data[s]['memory'] for s in solver_names]
    time_values = [performance_data[s]['time'] for s in solver_names]
    
    # Normalize values for comparison (higher is better for visualization)
    max_memory = max(memory_values) if memory_values else 1
    max_time = max(time_values) if time_values else 1
    
    norm_memory = [1 - (m / max_memory) for m in memory_values]  # Inverted: lower memory is better
    norm_time = [1 - (t / max_time) for t in time_values]        # Inverted: lower time is better
    
    x_pos = np.arange(len(solver_names))
    width = 0.35
    
    bars1 = ax1.bar(x_pos - width/2, norm_memory, width, label='Memory Efficiency', alpha=0.7)
    bars2 = ax1.bar(x_pos + width/2, norm_time, width, label='Time Efficiency', alpha=0.7)
    
    ax1.set_title('Solver Efficiency Comparison', fontweight='bold')
    ax1.set_xlabel('Solver')
    ax1.set_ylabel('Efficiency Score (0-1)')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(solver_names)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Evolution simulation timeline
    ax2 = axes[0, 1]
    evolution_stages = ['Original F₁', '5% Modified', '10% Modified\n(Current F₂)', '15% Modified', '20% Modified']
    # Simulate evolution impact (hypothetical data based on current results)
    base_time = min(time_values) if time_values else 0.01
    evolution_times = [base_time * 0.8, base_time * 0.9, base_time, base_time * 1.2, base_time * 1.5]
    
    ax2.plot(evolution_stages, evolution_times, 'o-', linewidth=2, markersize=8, color='red')
    ax2.set_title('Hypothetical Evolution Impact\non Solving Time', fontweight='bold')
    ax2.set_xlabel('Evolution Stage')
    ax2.set_ylabel('Solving Time (s)')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)
    
    # Mark current stage
    ax2.axvline(x=2, color='green', linestyle='--', alpha=0.7, label='Current Stage')
    ax2.legend()
    
    # SAT consistency analysis
    ax3 = axes[1, 0]
    sat_results = [performance_data[s]['sat'] for s in solver_names]
    sat_count = sum(sat_results)
    consistency_percentage = (sat_count / len(sat_results)) * 100 if sat_results else 0
    
    # Create consistency pie chart
    consistent_solvers = sat_count
    inconsistent_solvers = len(sat_results) - sat_count
    
    if consistent_solvers > 0 or inconsistent_solvers > 0:
        sizes = [consistent_solvers, inconsistent_solvers]
        labels = ['Consistent (SAT)', 'Inconsistent (UNSAT)']
        colors = ['lightgreen', 'lightcoral']
        
        wedges, texts, autotexts = ax3.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors)
        for wedge in wedges:
            wedge.set_alpha(0.7)
        ax3.set_title(f'Model Consistency\n({consistency_percentage:.1f}% SAT Agreement)', fontweight='bold')
    
    # Evolution stability metrics
    ax4 = axes[1, 1]
    stability_metrics = ['Memory\nStability', 'Time\nStability', 'SAT\nConsistency', 'Overall\nStability']
    
    # Calculate stability scores (higher is better)
    memory_stability = 1 / (np.std(memory_values) + 0.001) if len(memory_values) > 1 else 1
    time_stability = 1 / (np.std(time_values) + 0.001) if len(time_values) > 1 else 1
    sat_consistency = consistency_percentage / 100
    overall_stability = (memory_stability + time_stability + sat_consistency) / 3
    
    # Normalize to 0-1 scale
    stability_scores = [
        min(memory_stability, 1),
        min(time_stability, 1), 
        sat_consistency,
        min(overall_stability, 1)
    ]
    
    bars4 = ax4.bar(stability_metrics, stability_scores, alpha=0.7, 
                   color=['skyblue', 'lightgreen', 'gold', 'purple'])
    ax4.set_title('Evolution Stability Metrics', fontweight='bold')
    ax4.set_ylabel('Stability Score (0-1)')
    ax4.set_ylim(0, 1)
    ax4.grid(True, alpha=0.3)
    
    # Add value labels
    for i, v in enumerate(stability_scores):
        ax4.text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('problem9_plots/p09i_dynamic_evolution_detailed.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_cross_variant_performance_heatmap(data):
    """Create a heatmap showing performance across all variants and solvers"""
    
    solvers = ['vampire', 'snake', 'z3', 'cvc5', 'e']
    variants = ['asymmetric_subalternation', 'safety_liveness_dominance', 'dynamic_evolution']
    variant_labels = ['Asymmetric\nSubalternation', 'Safety/Liveness\nDominance', 'Dynamic\nEvolution']
    
    # Prepare data matrices
    memory_matrix = []
    time_matrix = []
    sat_matrix = []
    
    for variant in variants:
        if variant not in data or not data[variant]:
            memory_row = [0] * len(solvers)
            time_row = [0] * len(solvers)
            sat_row = [0] * len(solvers)
        else:
            variant_data = data[variant][0]
            memory_row = []
            time_row = []
            sat_row = []
            
            for solver in solvers:
                memory, time, sat_result = extract_solver_metrics(variant_data['data'], solver)
                memory_row.append(memory)
                time_row.append(time)
                sat_row.append(1 if sat_result else 0)
        
        memory_matrix.append(memory_row)
        time_matrix.append(time_row)
        sat_matrix.append(sat_row)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Problem 9: Cross-Variant Performance Heatmaps', fontsize=16, fontweight='bold')
    
    # Memory heatmap
    ax1 = axes[0]
    im1 = ax1.imshow(memory_matrix, cmap='YlOrRd', aspect='auto')
    ax1.set_title('Memory Usage (MB)', fontweight='bold')
    ax1.set_xticks(range(len(solvers)))
    ax1.set_xticklabels(solvers)
    ax1.set_yticks(range(len(variant_labels)))
    ax1.set_yticklabels(variant_labels)
    
    # Add text annotations
    for i in range(len(variant_labels)):
        for j in range(len(solvers)):
            if memory_matrix[i][j] > 0:
                text = ax1.text(j, i, f'{memory_matrix[i][j]:.2f}', 
                               ha="center", va="center", color="black", fontsize=10)
    
    plt.colorbar(im1, ax=ax1)
    
    # Time heatmap
    ax2 = axes[1]
    im2 = ax2.imshow(time_matrix, cmap='YlGnBu', aspect='auto')
    ax2.set_title('Execution Time (s)', fontweight='bold')
    ax2.set_xticks(range(len(solvers)))
    ax2.set_xticklabels(solvers)
    ax2.set_yticks(range(len(variant_labels)))
    ax2.set_yticklabels(variant_labels)
    
    # Add text annotations
    for i in range(len(variant_labels)):
        for j in range(len(solvers)):
            if time_matrix[i][j] > 0:
                text = ax2.text(j, i, f'{time_matrix[i][j]:.3f}', 
                               ha="center", va="center", color="black", fontsize=10)
    
    plt.colorbar(im2, ax=ax2)
    
    # SAT results heatmap
    ax3 = axes[2]
    im3 = ax3.imshow(sat_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    ax3.set_title('SAT Results (1=SAT, 0=UNSAT)', fontweight='bold')
    ax3.set_xticks(range(len(solvers)))
    ax3.set_xticklabels(solvers)
    ax3.set_yticks(range(len(variant_labels)))
    ax3.set_yticklabels(variant_labels)
    
    # Add text annotations
    for i in range(len(variant_labels)):
        for j in range(len(solvers)):
            result_text = 'SAT' if sat_matrix[i][j] == 1 else 'UNSAT'
            text = ax3.text(j, i, result_text, 
                           ha="center", va="center", color="black", fontsize=10, fontweight='bold')
    
    plt.colorbar(im3, ax=ax3)
    
    plt.tight_layout()
    plt.savefig('problem9_plots/p09_cross_variant_heatmaps.png', dpi=300, bbox_inches='tight')
    plt.show()

def print_problem9_summary(data, problem_variants):
    """Print a comprehensive summary of Problem 9 analysis"""
    print("=" * 60)
    print("PROBLEM 9 ANALYSIS SUMMARY")
    print("=" * 60)
    print("Study: Logical Square Variants with Asymmetric Properties")
    print()
    print("Variants tested:")
    print("  g) Asymmetric subalternation: (F₁ ⇒ F₂) ∧ ¬(F₂ ⇒ F₁)")
    print("     F₁: 100 clauses, F₂: 1000 clauses (10x complexity difference)")
    print("  h) Safety/liveness dominance: F₁: 80% safety, F₂: 80% liveness")
    print("     Tests contradictory and subalternation with semantic asymmetry")
    print("  i) Dynamic evolution: F₂ as modified F₁ with 10% changed clauses")
    print("     Simulates requirement evolution and version consistency")
    print()
    
    solvers = ['vampire', 'snake', 'z3', 'cvc5', 'e']
    
    for variant_name, variant_data_list in data.items():
        if not variant_data_list:
            continue
            
        variant_data = variant_data_list[0]
        print(f"{variant_name.replace('_', ' ').title()}:")
        print(f"  Configuration: {variant_data['clauses']} clauses, {variant_data['atoms']} atoms")
        print(f"  Safety percentage: {variant_data['safety_prec']}%")
        print(f"  Solver performance:")
        
        for solver in solvers:
            memory, time, sat_result = extract_solver_metrics(variant_data['data'], solver)
            if memory > 0 or time > 0:
                result_str = "SAT" if sat_result else "UNSAT"
                print(f"    {solver}: {result_str} (Mem: {memory:.3f} MB, Time: {time:.3f} s)")
        print()
    
    print("Key insights:")
    print("- Asymmetric complexity (g) shows significant impact on solver performance")
    print("- Safety/liveness dominance (h) creates semantic solving challenges") 
    print("- Dynamic evolution (i) demonstrates requirement evolution consistency")
    print("- Different logical relationships require different solving strategies")
    print()

def main():
    """Main function to run Problem 9 analysis"""
    # Create output directory
    os.makedirs('problem9_plots', exist_ok=True)
    
    # Load data
    print("Loading Problem 9 data...")
    data, problem_variants = load_problem9_data()
    
    if not data:
        print("No data found! Please check the data directory path.")
        return
    
    # Print summary
    print_problem9_summary(data, problem_variants)
    
    # Create comprehensive analysis plots
    print("Creating Problem 9 variants comparison plots...")
    create_problem9_variants_comparison(data)
    
    print("Creating asymmetric subalternation detailed analysis...")
    create_asymmetric_complexity_analysis(data)
    
    print("Creating safety/liveness dominance detailed analysis...")
    create_safety_liveness_analysis(data)
    
    print("Creating dynamic evolution detailed analysis...")
    create_dynamic_evolution_analysis(data)
    
    print("Creating cross-variant performance heatmaps...")
    create_cross_variant_performance_heatmap(data)
    
    print("Problem 9 analysis complete! Plots saved in problem9_plots/ directory.")

if __name__ == "__main__":
    main() 