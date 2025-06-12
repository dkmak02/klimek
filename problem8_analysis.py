import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
from pathlib import Path

def load_problem8_data():
    """Load all problem8 CSV files and organize by logical relationship and distribution type"""
    
    # Define logical relationships
    logical_relationships = {
        'a': 'contradictory',    # (F₁ ⇒ ¬F₂) ∧ (¬F₁ ⇒ F₂)
        'b': 'subcontrary',      # ¬(¬F₁ ∧ ¬F₂)
        'c': 'subalternated',    # (F₁ ⇒ F₂) ∧ ¬(F₂ ⇒ F₁)
        'd': 'contradictory_poisson',    # Poisson variant of a
        'e': 'subcontrary_poisson',      # Poisson variant of b
        'f': 'subalternated_poisson'     # Poisson variant of c
    }
    
    data = {}
    
    # Load regular distribution data
    regular_dir = "problem8/results/benchmark-test-20250606-130433/results"
    csv_files = glob.glob(os.path.join(regular_dir, "problem8*.csv"))
    
    # Load Poisson distribution data
    poisson_dir = "problem8poisson/results/benchmark-test-20250606-142348/results"
    csv_files.extend(glob.glob(os.path.join(poisson_dir, "problem8*.csv")))
    
    for file in csv_files:
        filename = os.path.basename(file)
        
        # Extract parameters from filename
        parts = filename.replace('.csv', '').split('_')
        
        # Find relationship type
        relationship_key = None
        for part in parts:
            if part.startswith('problem8') and len(part) > 8:
                relationship_key = part[-1]  # Get the letter (a, b, c, d, e, f)
                break
        
        if not relationship_key or relationship_key not in logical_relationships:
            continue
            
        relationship = logical_relationships[relationship_key]
        
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
        
        if clauses and relationship:
            # Read the CSV file
            df = pd.read_csv(file, sep=';')
            
            # Filter to get only the average row
            avg_row = df[df['Run Number'] == 'Average'].iloc[0]
            
            # Store in our data structure
            if relationship not in data:
                data[relationship] = []
            
            data[relationship].append({
                'clauses': clauses,
                'atoms': atoms,
                'safety_prec': safety_prec if safety_prec else 50,
                'relationship': relationship,
                'filename': filename,
                'data': avg_row
            })
    
    # Sort by number of clauses
    for relationship in data:
        data[relationship].sort(key=lambda x: x['clauses'])
    
    return data, logical_relationships

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

def create_logical_relationship_comparison(data):
    """Create comparison plots between different logical relationships"""
    
    solvers = ['vampire', 'snake', 'z3', 'cvc5', 'e']
    relationships = ['contradictory', 'subcontrary', 'subalternated']
    
    # Get clause counts
    clause_counts = []
    for rel in relationships:
        if rel in data:
            clause_counts.extend([item['clauses'] for item in data[rel]])
    clause_counts = sorted(set(clause_counts))
    
    # Create subplots for each clause count
    fig, axes = plt.subplots(len(clause_counts), 2, figsize=(16, 6*len(clause_counts)))
    if len(clause_counts) == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle('Problem 8: Logical Square Relationships Impact on Solver Performance', fontsize=16, fontweight='bold')
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    for clause_idx, clause_count in enumerate(clause_counts):
        ax_mem = axes[clause_idx, 0]
        ax_time = axes[clause_idx, 1]
        
        for solver in solvers:
            memories = []
            times = []
            labels = []
            
            for rel_idx, relationship in enumerate(relationships):
                if relationship not in data:
                    continue
                    
                # Find data for this clause count
                rel_data = [item for item in data[relationship] if item['clauses'] == clause_count]
                if rel_data:
                    memory, time, sat_result = extract_solver_metrics(rel_data[0]['data'], solver)
                    if memory > 0 or time > 0:
                        memories.append(memory)
                        times.append(time)
                        labels.append(relationship.title())
            
            if memories:
                x_pos = np.arange(len(labels))
                ax_mem.plot(x_pos, memories, 'o-', label=solver, linewidth=2, markersize=6)
                ax_time.plot(x_pos, times, 'o-', label=solver, linewidth=2, markersize=6)
        
        # Customize plots
        ax_mem.set_title(f'Memory Usage - {clause_count} Clauses', fontweight='bold')
        ax_mem.set_xlabel('Logical Relationship')
        ax_mem.set_ylabel('Memory (MB)')
        ax_mem.set_yscale('log')
        ax_mem.set_xticks(range(len(relationships)))
        ax_mem.set_xticklabels([rel.title() for rel in relationships], rotation=45)
        ax_mem.legend()
        ax_mem.grid(True, alpha=0.3)
        
        ax_time.set_title(f'Execution Time - {clause_count} Clauses', fontweight='bold')
        ax_time.set_xlabel('Logical Relationship')
        ax_time.set_ylabel('Time (s)')
        ax_time.set_yscale('log')
        ax_time.set_xticks(range(len(relationships)))
        ax_time.set_xticklabels([rel.title() for rel in relationships], rotation=45)
        ax_time.legend()
        ax_time.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('problem8_plots/p08_logical_relationships_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_poisson_comparison_plots(data):
    """Create comparison plots between regular and Poisson distributions"""
    
    solvers = ['vampire', 'snake', 'z3', 'cvc5', 'e']
    regular_relationships = ['contradictory', 'subcontrary', 'subalternated']
    poisson_relationships = ['contradictory_poisson', 'subcontrary_poisson', 'subalternated_poisson']
    
    for i, (regular, poisson) in enumerate(zip(regular_relationships, poisson_relationships)):
        if regular not in data or poisson not in data:
            continue
            
        # Get clause counts
        clause_counts = []
        clause_counts.extend([item['clauses'] for item in data[regular]])
        clause_counts.extend([item['clauses'] for item in data[poisson]])
        clause_counts = sorted(set(clause_counts))
        
        fig, axes = plt.subplots(len(clause_counts), 2, figsize=(16, 6*len(clause_counts)))
        if len(clause_counts) == 1:
            axes = axes.reshape(1, -1)
        
        fig.suptitle(f'Problem 8: Regular vs Poisson Distribution - {regular.title()}', fontsize=16, fontweight='bold')
        
        for clause_idx, clause_count in enumerate(clause_counts):
            ax_mem = axes[clause_idx, 0]
            ax_time = axes[clause_idx, 1]
            
            for solver in solvers:
                regular_data = [item for item in data[regular] if item['clauses'] == clause_count]
                poisson_data = [item for item in data[poisson] if item['clauses'] == clause_count]
                
                if regular_data and poisson_data:
                    reg_memory, reg_time, reg_sat = extract_solver_metrics(regular_data[0]['data'], solver)
                    pois_memory, pois_time, pois_sat = extract_solver_metrics(poisson_data[0]['data'], solver)
                    
                    if reg_memory > 0 or pois_memory > 0:
                        ax_mem.bar([f'{solver}_Regular', f'{solver}_Poisson'], 
                                  [reg_memory, pois_memory], alpha=0.8)
                    
                    if reg_time > 0 or pois_time > 0:
                        ax_time.bar([f'{solver}_Regular', f'{solver}_Poisson'], 
                                   [reg_time, pois_time], alpha=0.8)
            
            ax_mem.set_title(f'Memory Usage - {clause_count} Clauses', fontweight='bold')
            ax_mem.set_ylabel('Memory (MB)')
            ax_mem.set_yscale('log')
            ax_mem.tick_params(axis='x', rotation=45)
            ax_mem.grid(True, alpha=0.3)
            
            ax_time.set_title(f'Execution Time - {clause_count} Clauses', fontweight='bold')
            ax_time.set_ylabel('Time (s)')
            ax_time.set_yscale('log')
            ax_time.tick_params(axis='x', rotation=45)
            ax_time.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'problem8_plots/p08_{regular}_regular_vs_poisson.png', dpi=300, bbox_inches='tight')
        plt.show()

def create_formula_size_scaling_plots(data):
    """Create plots showing how performance scales with formula size for each logical relationship"""
    
    solvers = ['vampire', 'snake', 'z3', 'cvc5', 'e']
    relationships = ['contradictory', 'subcontrary', 'subalternated']
    
    for relationship in relationships:
        if relationship not in data:
            continue
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle(f'Problem 8: Formula Size Scaling - {relationship.title()}', fontsize=14, fontweight='bold')
        
        clause_counts = [item['clauses'] for item in data[relationship]]
        
        for solver in solvers:
            memories = []
            times = []
            
            for item in data[relationship]:
                memory, time, sat_result = extract_solver_metrics(item['data'], solver)
                memories.append(memory)
                times.append(time)
            
            if any(m > 0 for m in memories):
                ax1.plot(clause_counts, memories, 'o-', label=solver, linewidth=2, markersize=6)
            
            if any(t > 0 for t in times):
                ax2.plot(clause_counts, times, 'o-', label=solver, linewidth=2, markersize=6)
        
        ax1.set_title('Memory Usage Scaling')
        ax1.set_xlabel('Number of Clauses')
        ax1.set_ylabel('Memory (MB)')
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.set_title('Execution Time Scaling')
        ax2.set_xlabel('Number of Clauses')
        ax2.set_ylabel('Time (s)')
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'problem8_plots/p08_{relationship}_scaling.png', dpi=300, bbox_inches='tight')
        plt.show()

def print_data_summary(data, logical_relationships):
    """Print a summary of the loaded data"""
    print("=== Problem 8 Data Summary ===")
    print("Study: Impact of logical square relationships on solver performance")
    print("Logical relationships tested:")
    print("  a) Contradictory: (F₁ ⇒ ¬F₂) ∧ (¬F₁ ⇒ F₂)")
    print("  b) Subcontrary: ¬(¬F₁ ∧ ¬F₂)")
    print("  c) Subalternated: (F₁ ⇒ F₂) ∧ ¬(F₂ ⇒ F₁)")
    print("  d-f) Poisson variants of a-c")
    print("Individual formulas F₁, F₂: 50% liveness + 50% safety clauses")
    print()
    
    for relationship, rel_data in data.items():
        if rel_data:
            print(f"{relationship.title()}:")
            clause_counts = [item['clauses'] for item in rel_data]
            print(f"  Clause counts: {clause_counts}")
            print(f"  Number of test cases: {len(rel_data)}")
            print()

def analyze_sat_results(data):
    """Analyze SAT results across different logical relationships"""
    print("=== SAT Results Analysis by Logical Relationship ===")
    
    solvers = ['vampire', 'snake', 'z3', 'prover9', 'cvc5', 'e', 'inkresat']
    
    for relationship, rel_data in data.items():
        if not rel_data:
            continue
            
        print(f"\n{relationship.title()} SAT Results:")
        
        for clause_item in rel_data:
            print(f"  {clause_item['clauses']} clauses:")
            
            for solver in solvers:
                memory, time, sat_result = extract_solver_metrics(clause_item['data'], solver)
                if memory > 0 or time > 0:
                    result_str = "SAT" if sat_result else "UNSAT"
                    print(f"    {solver}: {result_str} (Mem: {memory:.3f} MB, Time: {time:.3f} s)")

def main():
    """Main function to run the analysis"""
    # Create output directory
    os.makedirs('problem8_plots', exist_ok=True)
    
    # Load data
    print("Loading Problem 8 data...")
    data, logical_relationships = load_problem8_data()
    
    if not data:
        print("No data found! Please check the data directory path.")
        return
    
    # Print summary
    print_data_summary(data, logical_relationships)
    
    # Analyze SAT results
    analyze_sat_results(data)
    
    # Create plots
    print("Creating logical relationship comparison plots...")
    create_logical_relationship_comparison(data)
    
    print("Creating Poisson vs regular comparison plots...")
    create_poisson_comparison_plots(data)
    
    print("Creating formula size scaling plots...")
    create_formula_size_scaling_plots(data)
    
    print("Analysis complete! Plots saved in problem8_plots/ directory.")

if __name__ == "__main__":
    main() 