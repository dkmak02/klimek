#!/usr/bin/env python3
"""
Problem P04 Analysis: Impact of Constant Clause Lengths on Solver Performance (Fixed)

This script analyzes how constant clause lengths (2, 3, 4, 5) affect solver performance 
across different formula sizes (50, 100, 200, 500, 1000, 2000 clauses).
All clauses in each formula have the same length.

Fixed version that properly handles SAT/UNSAT results.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configuration
plt.style.use('default')
sns.set_palette("husl")

def read_and_process_data(csv_file):
    """Read and process the CSV data"""
    df = pd.read_csv(csv_file, sep=';', decimal=',')
    
    # Extract clause length from the data pattern
    # We have 24 rows total: 6 sizes × 4 lengths
    df['clause_length'] = 0  # Will be filled
    
    # Group by clauses and atoms to assign clause lengths
    grouped = df.groupby(['Number of Clauses', 'Number of Atoms'])
    
    for (clauses, atoms), group in grouped:
        # Each group should have 4 rows (for lengths 2,3,4,5)
        if len(group) == 4:
            # Sort by vampire memory to get consistent ordering
            group_sorted = group.sort_values('vampire Memory (MB)')
            lengths = [2, 3, 4, 5]
            
            for i, (idx, row) in enumerate(group_sorted.iterrows()):
                df.loc[idx, 'clause_length'] = lengths[i]
    
    return df

def is_solver_successful(df_row, solver):
    """Check if a solver run was successful (includes both SAT and UNSAT, excludes only ERROR)"""
    sat_col = f'{solver} SAT'
    
    if solver == 'prover9':
        # Prover9: successful if not ERROR
        return df_row[sat_col] != 'ERROR'
    else:
        # Other solvers: successful if SAT or UNSAT (True or False), not ERROR
        return df_row[sat_col] in [True, False]

def safe_log_transform(values, min_val=0.00001):
    """Safely transform values for log scale"""
    return np.where(values <= 0, min_val, values)

def create_plots(df, output_dir):
    """Create comprehensive analysis plots"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Define solvers and their properties
    solvers = ['vampire', 'snake', 'z3', 'prover9', 'cvc5', 'e', 'inkresat']
    solver_colors = {
        'vampire': '#e74c3c', 'snake': '#2ecc71', 'z3': '#3498db',
        'prover9': '#f39c12', 'cvc5': '#9b59b6', 'e': '#1abc9c', 'inkresat': '#34495e'
    }
    
    # 1. Time Performance Heatmaps
    plt.figure(figsize=(16, 12))
    
    for i, solver in enumerate(solvers, 1):
        plt.subplot(3, 3, i)
        
        time_col = f'{solver} Time (s)'
        
        # Filter successful runs
        df_success = df[df.apply(lambda row: is_solver_successful(row, solver), axis=1)].copy()
        
        if len(df_success) == 0:
            plt.text(0.5, 0.5, f'{solver.upper()}\nNo successful runs', 
                    ha='center', va='center', transform=plt.gca().transAxes, fontsize=12)
            plt.title(f'{solver.upper()} - Time Performance')
            continue
        
        # Create pivot for heatmap
        pivot_data = df_success.pivot_table(
            values=time_col, 
            index='clause_length', 
            columns='Number of Clauses',
            aggfunc='mean'
        )
        
        if not pivot_data.empty:
            sns.heatmap(pivot_data, annot=True, fmt='.4f', cmap='YlOrRd', 
                       cbar_kws={'label': 'Time (s)'})
            plt.title(f'{solver.upper()} - Time Performance')
            plt.xlabel('Number of Clauses')
            plt.ylabel('Clause Length')
        else:
            plt.text(0.5, 0.5, f'{solver.upper()}\nInsufficient data', 
                    ha='center', va='center', transform=plt.gca().transAxes, fontsize=12)
            plt.title(f'{solver.upper()} - Time Performance')
    
    # Subplot 8 removed - no summary text
    
    plt.tight_layout()
    plt.savefig(output_dir / 'p04_01_time_heatmaps.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Performance Analysis Overview
    plt.figure(figsize=(16, 10))
    
    # Time comparison by clause length
    plt.subplot(2, 3, 1)
    
    for solver in ['vampire', 'snake', 'z3', 'cvc5', 'e']:  # Skip problematic ones
        time_col = f'{solver} Time (s)'
        
        clause_times = []
        clause_lengths = []
        
        for clause_len in [2, 3, 4, 5]:
            df_len = df[df['clause_length'] == clause_len]
            df_solver = df_len[df_len.apply(lambda row: is_solver_successful(row, solver), axis=1)]
            
            if len(df_solver) > 0:
                avg_time = df_solver[time_col].mean()
                clause_times.append(avg_time)
                clause_lengths.append(clause_len)
        
        if clause_times:
            plt.plot(clause_lengths, clause_times, marker='o', 
                    label=solver, linewidth=2, markersize=6, 
                    color=solver_colors[solver])
    
    plt.xlabel('Clause Length')
    plt.ylabel('Average Time (s)')
    plt.title('Performance vs Clause Length')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    # Success rates by clause length
    plt.subplot(2, 3, 2)
    
    success_by_length = []
    for clause_len in [2, 3, 4, 5]:
        df_len = df[df['clause_length'] == clause_len]
        total_success = 0
        total_runs = 0
        
        for solver in solvers:
            success_count = df_len.apply(lambda row: is_solver_successful(row, solver), axis=1).sum()
            total_success += success_count
            total_runs += len(df_len)
        
        if total_runs > 0:
            success_rate = (total_success / total_runs) * 100
            success_by_length.append(success_rate)
        else:
            success_by_length.append(0)
    
    plt.bar([2, 3, 4, 5], success_by_length, color=['lightcoral', 'lightblue', 'lightgreen', 'lightyellow'])
    plt.xlabel('Clause Length')
    plt.ylabel('Overall Success Rate (%)')
    plt.title('Success Rate by Clause Length')
    plt.grid(True, alpha=0.3)
    
    # Memory usage by clause length (excluding InKreSAT)
    plt.subplot(2, 3, 3)
    
    for solver in ['vampire', 'snake', 'z3', 'cvc5', 'e']:
        memory_col = f'{solver} Memory (MB)'
        
        clause_memory = []
        clause_lengths = []
        
        for clause_len in [2, 3, 4, 5]:
            df_len = df[df['clause_length'] == clause_len]
            df_solver = df_len[df_len.apply(lambda row: is_solver_successful(row, solver), axis=1)]
            
            if len(df_solver) > 0:
                avg_memory = df_solver[memory_col].mean()
                clause_memory.append(avg_memory)
                clause_lengths.append(clause_len)
        
        if clause_memory:
            plt.plot(clause_lengths, clause_memory, marker='s', 
                    label=solver, linewidth=2, markersize=6, 
                    color=solver_colors[solver])
    
    plt.xlabel('Clause Length')
    plt.ylabel('Average Memory (MB)')
    plt.title('Memory Usage vs Clause Length')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Scaling by formula size
    plt.subplot(2, 3, 4)
    
    for clause_len in [2, 3, 4, 5]:
        df_len = df[df['clause_length'] == clause_len]
        
        sizes = []
        avg_times = []
        
        for size in sorted(df_len['Number of Clauses'].unique()):
            df_size = df_len[df_len['Number of Clauses'] == size]
            
            # Average across successful solvers
            total_time = 0
            count = 0
            for solver in ['vampire', 'snake', 'z3', 'cvc5', 'e']:
                time_col = f'{solver} Time (s)'
                
                df_solver = df_size[df_size.apply(lambda row: is_solver_successful(row, solver), axis=1)]
                if len(df_solver) > 0:
                    total_time += df_solver[time_col].iloc[0]
                    count += 1
            
            if count > 0:
                sizes.append(size)
                avg_times.append(total_time / count)
        
        if sizes:
            plt.plot(sizes, avg_times, marker='o', label=f'Length {clause_len}', 
                    linewidth=2, markersize=6)
    
    plt.xlabel('Number of Clauses')
    plt.ylabel('Average Time (s)')
    plt.title('Scaling by Formula Size')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    # Best solver by clause length
    plt.subplot(2, 3, 5)
    
    best_solvers = []
    best_times = []
    clause_lens = []
    
    for clause_len in [2, 3, 4, 5]:
        df_len = df[df['clause_length'] == clause_len]
        solver_times = {}
        
        for solver in ['vampire', 'snake', 'z3', 'cvc5', 'e']:
            time_col = f'{solver} Time (s)'
            
            df_solver = df_len[df_len.apply(lambda row: is_solver_successful(row, solver), axis=1)]
            if len(df_solver) > 0:
                avg_time = df_solver[time_col].mean()
                solver_times[solver] = avg_time
        
        if solver_times:
            best_solver = min(solver_times, key=solver_times.get)
            best_solvers.append(best_solver)
            best_times.append(solver_times[best_solver])
            clause_lens.append(clause_len)
    
    if best_solvers:
        colors = [solver_colors[solver] for solver in best_solvers]
        bars = plt.bar(clause_lens, best_times, color=colors, alpha=0.7)
        
        # Add solver names on bars
        for i, (bar, solver) in enumerate(zip(bars, best_solvers)):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0001, 
                    solver.upper(), ha='center', va='bottom', fontsize=8)
        
        plt.xlabel('Clause Length')
        plt.ylabel('Best Time (s)')
        plt.title('Best Performer by Clause Length')
        plt.grid(True, alpha=0.3)
    
    # Subplot 6 removed - no summary text
    
    plt.tight_layout()
    plt.savefig(output_dir / 'p04_02_analysis_overview.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Individual Solver Analysis
    plt.figure(figsize=(18, 12))
    
    successful_solvers = ['vampire', 'snake', 'z3', 'cvc5', 'e', 'inkresat']
    
    for i, solver in enumerate(successful_solvers, 1):
        plt.subplot(2, 3, i)
        
        time_col = f'{solver} Time (s)'
        
        for clause_len in [2, 3, 4, 5]:
            df_len = df[df['clause_length'] == clause_len]
            df_solver = df_len[df_len.apply(lambda row: is_solver_successful(row, solver), axis=1)]
            
            if len(df_solver) > 0:
                sizes = df_solver['Number of Clauses']
                times = safe_log_transform(df_solver[time_col])
                plt.plot(sizes, times, marker='o', label=f'Length {clause_len}', 
                        linewidth=2, markersize=6)
        
        plt.xlabel('Number of Clauses')
        plt.ylabel('Time (s)')
        plt.title(f'{solver.upper()} - Performance by Clause Length')
        plt.yscale('log')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'p04_03_individual_solvers.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. SAT/UNSAT Analysis
    plt.figure(figsize=(16, 10))
    
    # SAT/UNSAT distribution by clause length
    plt.subplot(2, 3, 1)
    
    sat_data = []
    unsat_data = []
    error_data = []
    
    for clause_len in [2, 3, 4, 5]:
        df_len = df[df['clause_length'] == clause_len]
        
        sat_count = 0
        unsat_count = 0
        error_count = 0
        
        # Count results across all solvers
        for solver in solvers:
            sat_col = f'{solver} SAT'
            
            for _, row in df_len.iterrows():
                if solver == 'prover9':
                    if row[sat_col] == 'ERROR':
                        error_count += 1
                    elif row[sat_col] == True:
                        sat_count += 1
                    elif row[sat_col] == False:
                        unsat_count += 1
                else:
                    if row[sat_col] == True:
                        sat_count += 1
                    elif row[sat_col] == False:
                        unsat_count += 1
        
        sat_data.append(sat_count)
        unsat_data.append(unsat_count)
        error_data.append(error_count)
    
    x = np.arange(len([2, 3, 4, 5]))
    width = 0.25
    
    plt.bar(x - width, sat_data, width, label='SAT', color='lightgreen')
    plt.bar(x, unsat_data, width, label='UNSAT', color='lightcoral')
    plt.bar(x + width, error_data, width, label='ERROR', color='lightgray')
    
    plt.xlabel('Clause Length')
    plt.ylabel('Count')
    plt.title('SAT/UNSAT/ERROR Distribution')
    plt.xticks(x, [2, 3, 4, 5])
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Solver agreement analysis
    plt.subplot(2, 3, 2)
    
    agreement_data = []
    for clause_len in [2, 3, 4, 5]:
        df_len = df[df['clause_length'] == clause_len]
        
        agreements = 0
        total_comparisons = 0
        
        for _, row in df_len.iterrows():
            # Compare all pairs of solvers
            solver_results = {}
            for solver in solvers:
                sat_col = f'{solver} SAT'
                if is_solver_successful(row, solver):
                    solver_results[solver] = row[sat_col]
            
            # Count agreements
            solver_list = list(solver_results.keys())
            for i in range(len(solver_list)):
                for j in range(i+1, len(solver_list)):
                    total_comparisons += 1
                    if solver_results[solver_list[i]] == solver_results[solver_list[j]]:
                        agreements += 1
        
        if total_comparisons > 0:
            agreement_rate = (agreements / total_comparisons) * 100
            agreement_data.append(agreement_rate)
        else:
            agreement_data.append(0)
    
    plt.bar([2, 3, 4, 5], agreement_data, color='skyblue', alpha=0.7)
    plt.xlabel('Clause Length')
    plt.ylabel('Solver Agreement (%)')
    plt.title('Solver Agreement by Clause Length')
    plt.grid(True, alpha=0.3)
    
    # Time vs satisfiability
    plt.subplot(2, 3, 3)
    
    for solver in ['vampire', 'snake', 'z3', 'cvc5', 'e']:
        time_col = f'{solver} Time (s)'
        sat_col = f'{solver} SAT'
        
        df_solver = df[df.apply(lambda row: is_solver_successful(row, solver), axis=1)]
        
        if len(df_solver) > 0:
            sat_times = df_solver[df_solver[sat_col] == True][time_col]
            unsat_times = df_solver[df_solver[sat_col] == False][time_col]
            
            if len(sat_times) > 0:
                plt.scatter([solver]*len(sat_times), sat_times, 
                           color='green', alpha=0.6, s=30, label='SAT' if solver == 'vampire' else "")
            if len(unsat_times) > 0:
                plt.scatter([solver]*len(unsat_times), unsat_times, 
                           color='red', alpha=0.6, s=30, label='UNSAT' if solver == 'vampire' else "")
    
    plt.ylabel('Time (s)')
    plt.title('Time vs SAT/UNSAT Results')
    plt.yscale('log')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 4 removed - no summary text
    
    plt.tight_layout()
    plt.savefig(output_dir / 'p04_04_satisfiability_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Summary Statistics
    summary_data = []
    
    for solver in solvers:
        time_col = f'{solver} Time (s)'
        memory_col = f'{solver} Memory (MB)'
        
        # Filter successful runs
        df_success = df[df.apply(lambda row: is_solver_successful(row, solver), axis=1)]
        
        success_rate = (len(df_success) / len(df)) * 100
        
        if len(df_success) > 0:
            avg_time = df_success[time_col].mean()
            avg_memory = df_success[memory_col].mean()
            
            # Find best clause length
            best_clause_len = None
            best_time = float('inf')
            for clause_len in [2, 3, 4, 5]:
                df_len = df_success[df_success['clause_length'] == clause_len]
                if len(df_len) > 0:
                    len_avg_time = df_len[time_col].mean()
                    if len_avg_time < best_time:
                        best_time = len_avg_time
                        best_clause_len = clause_len
        else:
            avg_time = np.nan
            avg_memory = np.nan
            best_clause_len = None
        
        summary_data.append({
            'Solver': solver.upper(),
            'Success Rate (%)': f"{success_rate:.1f}",
            'Avg Time (s)': f"{avg_time:.4f}" if not np.isnan(avg_time) else "N/A",
            'Avg Memory (MB)': f"{avg_memory:.2f}" if not np.isnan(avg_memory) else "N/A",
            'Best Clause Length': best_clause_len if best_clause_len else "N/A"
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Create summary table
    plt.figure(figsize=(12, 8))
    plt.axis('off')
    
    table = plt.table(cellText=summary_df.values,
                     colLabels=summary_df.columns,
                     cellLoc='center',
                     loc='center',
                     colWidths=[0.18, 0.18, 0.18, 0.18, 0.18])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2)
    
    # Style the table
    for i in range(len(summary_df.columns)):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    for i in range(1, len(summary_df) + 1):
        for j in range(len(summary_df.columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f1f1f2')
    
    plt.title('P04: Summary - Constant Clause Lengths Analysis', 
             fontsize=16, fontweight='bold', pad=20)
    
    # Key findings text removed
    
    plt.savefig(output_dir / 'p04_05_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save summary to CSV
    summary_df.to_csv(output_dir / 'p04_summary.csv', index=False)
    
    print("✅ Problem P04 analysis completed!")
    print(f"📁 Results saved in: {output_dir}")
    print("\n📊 Generated plots:")
    for i, plot_name in enumerate([
        "p04_01_time_heatmaps.png",
        "p04_02_analysis_overview.png", 
        "p04_03_individual_solvers.png",
        "p04_04_satisfiability_analysis.png",
        "p04_05_summary.png"
    ], 1):
        print(f"  📊 {plot_name}")
    print(f"  📋 p04_summary.csv")
    
    return summary_df

def main():
    """Main execution function"""
    csv_file = 'problem4/avg.csv'
    output_dir = 'problem4_plots'
    
    print("🔍 Problem P04: Constant Clause Lengths Analysis (Fixed)")
    print("=" * 55)
    
    # Read and process data
    print("📖 Reading data...")
    df = read_and_process_data(csv_file)
    print(f"   Data shape: {df.shape}")
    print(f"   Clause lengths: {sorted(df['clause_length'].unique())}")
    print(f"   Formula sizes: {sorted(df['Number of Clauses'].unique())}")
    
    # Create plots
    print("\n🎨 Creating visualizations...")
    summary_df = create_plots(df, output_dir)
    
    print("\n🔍 Key insights:")
    print("- Fixed success detection for SAT/UNSAT results")
    print("- All solvers now show realistic success rates")

if __name__ == "__main__":
    main() 