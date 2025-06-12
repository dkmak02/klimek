import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
from pathlib import Path

def load_problem7_data(data_dir="problem7/results/benchmark-test-20250606-130050/results"):
    """Load all problem7 CSV files and organize by variant type (disjunction vs conjunction)"""
    
    # Find all CSV files
    csv_files = glob.glob(os.path.join(data_dir, "problem7*.csv"))
    
    data = {
        'disjunction': [],  # problem7a - OR combination
        'conjunction': []   # problem7b - AND combination
    }
    
    for file in csv_files:
        filename = os.path.basename(file)
        
        # Extract parameters from filename
        parts = filename.replace('.csv', '').split('_')
        
        # Determine variant type
        if 'problem7a' in filename:
            variant = 'disjunction'
        elif 'problem7b' in filename:
            variant = 'conjunction'
        else:
            continue
            
        # Find clause count and atoms
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
        
        if clauses and variant:
            # Read the CSV file
            df = pd.read_csv(file, sep=';')
            
            # Filter to get only the average row
            avg_row = df[df['Run Number'] == 'Average'].iloc[0]
            
            # Store in our data structure
            data[variant].append({
                'clauses': clauses,
                'atoms': atoms,
                'safety_prec': safety_prec if safety_prec else 50,
                'variant': variant,
                'filename': filename,
                'data': avg_row
            })
    
    # Sort by number of clauses
    for variant in data:
        data[variant].sort(key=lambda x: x['clauses'])
    
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

def create_comparison_plots(data):
    """Create comparison plots between disjunction and conjunction variants"""
    
    # Define solvers to analyze
    solvers = ['vampire', 'snake', 'z3', 'prover9', 'cvc5', 'e', 'inkresat']
    solver_colors = {
        'vampire': '#1f77b4',
        'snake': '#ff7f0e', 
        'z3': '#2ca02c',
        'prover9': '#8c564b',
        'cvc5': '#d62728',
        'e': '#9467bd',
        'inkresat': '#17becf'
    }
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Problem 7: Disjunction vs Conjunction Formula Combination Impact', fontsize=16, fontweight='bold')
    
    # Memory comparison
    ax_mem = axes[0, 0]
    # Time comparison
    ax_time = axes[0, 1]
    # SAT Results comparison
    ax_sat = axes[1, 0]
    # Performance summary
    ax_summary = axes[1, 1]
    
    variants = ['disjunction', 'conjunction']
    variant_labels = ['Disjunction (OR)', 'Conjunction (AND)']
    
    solver_memories = {solver: [] for solver in solvers}
    solver_times = {solver: [] for solver in solvers}
    solver_sat_rates = {solver: [] for solver in solvers}
    
    for variant_idx, variant in enumerate(variants):
        if not data[variant]:
            continue
            
        variant_data = data[variant][0]  # Taking the first (and likely only) data point
        
        memories = []
        times = []
        sat_results = []
        solver_names = []
        
        for solver in solvers:
            memory, time, sat_result = extract_solver_metrics(variant_data['data'], solver)
            
            if memory > 0 or time > 0:  # Only include solvers with meaningful data
                memories.append(memory)
                times.append(time)
                sat_results.append(1 if sat_result else 0)
                solver_names.append(solver)
                
                solver_memories[solver].append(memory)
                solver_times[solver].append(time)
                solver_sat_rates[solver].append(1 if sat_result else 0)
        
        # Memory bar plot
        x_pos = np.arange(len(solver_names))
        width = 0.35
        offset = width * (variant_idx - 0.5)
        
        bars_mem = ax_mem.bar(x_pos + offset, memories, width, 
                             label=variant_labels[variant_idx], alpha=0.8)
        
        # Time bar plot
        bars_time = ax_time.bar(x_pos + offset, times, width,
                               label=variant_labels[variant_idx], alpha=0.8)
        
        # SAT results bar plot
        bars_sat = ax_sat.bar(x_pos + offset, sat_results, width,
                             label=variant_labels[variant_idx], alpha=0.8)
    
    # Customize memory plot
    ax_mem.set_title('Memory Usage Comparison', fontweight='bold')
    ax_mem.set_xlabel('Solvers')
    ax_mem.set_ylabel('Memory (MB)')
    ax_mem.set_yscale('log')
    ax_mem.set_xticks(x_pos)
    ax_mem.set_xticklabels(solver_names, rotation=45)
    ax_mem.legend()
    ax_mem.grid(True, alpha=0.3)
    
    # Customize time plot
    ax_time.set_title('Execution Time Comparison', fontweight='bold')
    ax_time.set_xlabel('Solvers')
    ax_time.set_ylabel('Time (s)')
    ax_time.set_yscale('log')
    ax_time.set_xticks(x_pos)
    ax_time.set_xticklabels(solver_names, rotation=45)
    ax_time.legend()
    ax_time.grid(True, alpha=0.3)
    
    # Customize SAT results plot
    ax_sat.set_title('Satisfiability Results', fontweight='bold')
    ax_sat.set_xlabel('Solvers')
    ax_sat.set_ylabel('SAT Success (1=True, 0=False)')
    ax_sat.set_xticks(x_pos)
    ax_sat.set_xticklabels(solver_names, rotation=45)
    ax_sat.legend()
    ax_sat.grid(True, alpha=0.3)
    
    # Create performance summary table
    ax_summary.axis('off')
    summary_data = []
    
    for solver in solvers:
        if solver_memories[solver] and solver_times[solver]:
            disj_mem = solver_memories[solver][0] if len(solver_memories[solver]) > 0 else 0
            conj_mem = solver_memories[solver][1] if len(solver_memories[solver]) > 1 else 0
            disj_time = solver_times[solver][0] if len(solver_times[solver]) > 0 else 0
            conj_time = solver_times[solver][1] if len(solver_times[solver]) > 1 else 0
            
            if disj_mem > 0 and conj_mem > 0:
                mem_ratio = conj_mem / disj_mem
                time_ratio = conj_time / disj_time if disj_time > 0 else float('inf')
                
                summary_data.append([solver.upper(), f"{disj_mem:.3f}", f"{conj_mem:.3f}", 
                                   f"{mem_ratio:.2f}x", f"{disj_time:.3f}", f"{conj_time:.3f}", 
                                   f"{time_ratio:.2f}x"])
    
    if summary_data:
        headers = ['Solver', 'Disj Mem', 'Conj Mem', 'Mem Ratio', 'Disj Time', 'Conj Time', 'Time Ratio']
        table = ax_summary.table(cellText=summary_data, colLabels=headers, 
                               cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        ax_summary.set_title('Performance Comparison Summary', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('problem7_plots/p07_disjunction_vs_conjunction.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_solver_detail_plots(data):
    """Create detailed plots for each solver showing disjunction vs conjunction performance"""
    
    solvers = ['vampire', 'snake', 'z3', 'prover9', 'cvc5', 'e']
    
    for solver in solvers:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle(f'Problem 7: {solver.upper()} - Disjunction vs Conjunction Analysis', fontsize=14, fontweight='bold')
        
        variants = ['disjunction', 'conjunction']
        variant_labels = ['Disjunction (OR)', 'Conjunction (AND)']
        colors = ['#1f77b4', '#ff7f0e']
        
        memories = []
        times = []
        labels = []
        
        for variant_idx, variant in enumerate(variants):
            if not data[variant]:
                continue
                
            variant_data = data[variant][0]
            memory, time, sat_result = extract_solver_metrics(variant_data['data'], solver)
            
            if memory > 0 or time > 0:
                memories.append(memory)
                times.append(time)
                labels.append(variant_labels[variant_idx])
        
        # Memory comparison
        if memories:
            bars1 = ax1.bar(range(len(labels)), memories, color=colors[:len(labels)], alpha=0.8)
            ax1.set_title('Memory Usage')
            ax1.set_ylabel('Memory (MB)')
            ax1.set_xticks(range(len(labels)))
            ax1.set_xticklabels(labels)
            ax1.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, mem in zip(bars1, memories):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{mem:.3f}', ha='center', va='bottom')
        
        # Time comparison
        if times:
            bars2 = ax2.bar(range(len(labels)), times, color=colors[:len(labels)], alpha=0.8)
            ax2.set_title('Execution Time')
            ax2.set_ylabel('Time (s)')
            ax2.set_xticks(range(len(labels)))
            ax2.set_xticklabels(labels)
            ax2.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, time in zip(bars2, times):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{time:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(f'problem7_plots/p07_{solver}_detailed_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()

def print_data_summary(data):
    """Print a summary of the loaded data"""
    print("=== Problem 7 Data Summary ===")
    print("Study: Impact of formula combination methods on solver performance")
    print("Combination methods:")
    print("  a) Disjunction (OR): G = (F_1) ∨ (F_2) ∨ (F_3)")
    print("  b) Conjunction (AND): G = (F_1) ∧ (F_2) ∧ (F_3)")
    print("Formula structure: G ⇒ R (implication converted to ¬G ∨ R)")
    print("Individual formulas F_1, F_2, F_3: 50% liveness + 50% safety clauses")
    print()
    
    for variant, variant_data in data.items():
        if variant_data:
            print(f"{variant.title()} variant:")
            clause_counts = [item['clauses'] for item in variant_data]
            print(f"  Clause counts: {clause_counts}")
            print(f"  Number of test cases: {len(variant_data)}")
            print()

def analyze_sat_results(data):
    """Analyze and compare SAT results between variants"""
    print("=== SAT Results Analysis ===")
    
    solvers = ['vampire', 'snake', 'z3', 'prover9', 'cvc5', 'e', 'inkresat']
    
    for variant, variant_data in data.items():
        if not variant_data:
            continue
            
        print(f"\n{variant.title()} variant SAT results:")
        test_case = variant_data[0]
        
        for solver in solvers:
            memory, time, sat_result = extract_solver_metrics(test_case['data'], solver)
            if memory > 0 or time > 0:
                print(f"  {solver}: {sat_result} (Memory: {memory:.3f} MB, Time: {time:.3f} s)")

def main():
    """Main function to run the analysis"""
    # Create output directory
    os.makedirs('problem7_plots', exist_ok=True)
    
    # Load data
    print("Loading Problem 7 data...")
    data = load_problem7_data()
    
    if not any(data.values()):
        print("No data found! Please check the data directory path.")
        return
    
    # Print summary
    print_data_summary(data)
    
    # Analyze SAT results
    analyze_sat_results(data)
    
    # Create plots
    print("Creating comparison plots...")
    create_comparison_plots(data)
    
    print("Creating detailed solver plots...")
    create_solver_detail_plots(data)
    
    print("Analysis complete! Plots saved in problem7_plots/ directory.")

if __name__ == "__main__":
    main() 