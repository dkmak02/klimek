import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path

# Set style for better looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("Set2")  # More distinguishable colors

def load_data(csv_file):
    """Load and preprocess the CSV data"""
    # Read CSV with semicolon delimiter
    df = pd.read_csv(csv_file, delimiter=';')
    
    # Convert comma decimal separators to dots for proper float conversion
    numeric_columns = [col for col in df.columns if 'Memory' in col or 'Time' in col]
    
    for col in numeric_columns:
        if df[col].dtype == 'object':
            # Replace comma with dot and convert to float
            df[col] = df[col].astype(str).str.replace(',', '.').astype(float)
    
    # Calculate atom-to-clause ratio
    df['Atom_Clause_Ratio'] = df['Number of Atoms'] / df['Number of Clauses']
    
    # Sort by number of clauses and then by ratio for proper plotting
    df = df.sort_values(['Number of Clauses', 'Atom_Clause_Ratio'])
    
    return df

def filter_valid_data(df, solver, exclude_errors=True):
    """Filter data to exclude ERROR values and invalid measurements"""
    if exclude_errors:
        sat_col = f'{solver} SAT'
        # Exclude rows where SAT column contains 'ERROR'
        mask = df[sat_col] != 'ERROR'
        return df[mask]
    return df

def create_focused_time_analysis(df, output_dir):
    """Create focused time analysis with clear separation by solver performance"""
    
    # Divide solvers into performance categories
    fast_solvers = ['vampire', 'snake', 'z3', 'inkresat']
    medium_solvers = ['cvc5', 'e']
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    
    # Plot 1: Fast solvers - Time vs Ratio for different clause counts
    ax1 = axes[0, 0]
    clause_counts = [50, 100, 200, 500, 1000, 2000]
    colors = plt.cm.tab10(np.linspace(0, 1, len(clause_counts)))
    
    for solver in fast_solvers:
        time_col = f'{solver} Time (s)'
        for i, clauses in enumerate(clause_counts):
            clause_data = df[df['Number of Clauses'] == clauses]
            filtered_data = filter_valid_data(clause_data, solver)
            
            if len(filtered_data) > 0:
                time_data = filtered_data[time_col].replace(0, 0.00001)
                ax1.plot(filtered_data['Atom_Clause_Ratio'], time_data, 
                        marker='o', linewidth=2, markersize=6, color=colors[i],
                        label=f'{clauses} klauzul' if solver == fast_solvers[0] else "",
                        linestyle='-' if solver in ['vampire', 'snake'] else '--')
    
    ax1.set_xlabel('Stosunek atomów do klauzul', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Czas wykonania (s)', fontsize=12, fontweight='bold')
    ax1.set_title('Szybkie solvery: Vampire, Snake, Z3, InKreSAT\n(linie ciągłe: Vampire/Snake, przerywane: Z3/InKreSAT)', 
                 fontsize=14, fontweight='bold')
    ax1.set_yscale('log')
    ax1.legend(title='Liczba klauzul', title_fontsize=12, fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Medium solvers - Time vs Ratio
    ax2 = axes[0, 1]
    
    for solver in medium_solvers:
        time_col = f'{solver} Time (s)'
        for i, clauses in enumerate(clause_counts):
            clause_data = df[df['Number of Clauses'] == clauses]
            filtered_data = filter_valid_data(clause_data, solver)
            
            if len(filtered_data) > 0:
                time_data = filtered_data[time_col].replace(0, 0.00001)
                style = '-' if solver == 'cvc5' else '--'
                ax2.plot(filtered_data['Atom_Clause_Ratio'], time_data, 
                        marker='s', linewidth=2, markersize=6, color=colors[i],
                        label=f'{clauses} klauzul' if solver == medium_solvers[0] else "",
                        linestyle=style)
    
    ax2.set_xlabel('Stosunek atomów do klauzul', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Czas wykonania (s)', fontsize=12, fontweight='bold')
    ax2.set_title('Średnie solvery: CVC5, E\n(linia ciągła: CVC5, przerywana: E)', 
                 fontsize=14, fontweight='bold')
    ax2.set_yscale('log')
    ax2.legend(title='Liczba klauzul', title_fontsize=12, fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Direct solver comparison for specific clause count
    ax3 = axes[1, 0]
    focus_clauses = 1000  # Focus on 1000 clauses for clear comparison
    
    clause_data = df[df['Number of Clauses'] == focus_clauses]
    all_solvers = ['vampire', 'snake', 'z3', 'cvc5', 'e', 'inkresat']
    solver_colors = plt.cm.Set1(np.linspace(0, 1, len(all_solvers)))
    
    for i, solver in enumerate(all_solvers):
        time_col = f'{solver} Time (s)'
        filtered_data = filter_valid_data(clause_data, solver)
        
        if len(filtered_data) > 0:
            time_data = filtered_data[time_col].replace(0, 0.00001)
            ax3.plot(filtered_data['Atom_Clause_Ratio'], time_data, 
                    marker='o', linewidth=3, markersize=8, color=solver_colors[i],
                    label=solver.capitalize())
    
    ax3.set_xlabel('Stosunek atomów do klauzul', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Czas wykonania (s)', fontsize=12, fontweight='bold')
    ax3.set_title(f'Porównanie wszystkich solverów - {focus_clauses} klauzul', 
                 fontsize=14, fontweight='bold')
    ax3.set_yscale('log')
    ax3.legend(title='Solvery', title_fontsize=12, fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Performance scaling with clause count
    ax4 = axes[1, 1]
    best_solvers = ['vampire', 'snake', 'cvc5']
    ratio_focus = 2.0  # Focus on 2:1 ratio
    
    ratio_data = df[df['Atom_Clause_Ratio'] == ratio_focus]
    
    for i, solver in enumerate(best_solvers):
        time_col = f'{solver} Time (s)'
        filtered_data = filter_valid_data(ratio_data, solver)
        
        if len(filtered_data) > 0:
            time_data = filtered_data[time_col].replace(0, 0.00001)
            ax4.loglog(filtered_data['Number of Clauses'], time_data, 
                      marker='o', linewidth=3, markersize=8, 
                      label=solver.capitalize())
    
    ax4.set_xlabel('Liczba klauzul', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Czas wykonania (s)', fontsize=12, fontweight='bold')
    ax4.set_title(f'Skalowanie z liczbą klauzul (stosunek {ratio_focus}:1)', 
                 fontsize=14, fontweight='bold')
    ax4.legend(title='Najlepsze solvery', title_fontsize=12, fontsize=10)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'p03_improved_time_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def create_memory_usage_analysis(df, output_dir):
    """Create clear memory usage analysis"""
    
    # Exclude InKreSAT from memory analysis
    memory_solvers = ['vampire', 'snake', 'z3', 'cvc5', 'e']
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Memory vs Clause count for different ratios
    ax1 = axes[0, 0]
    ratios_to_show = [2, 3, 5, 10]
    ratio_colors = plt.cm.viridis(np.linspace(0, 1, len(ratios_to_show)))
    
    # Focus on Snake for clarity
    solver = 'snake'
    memory_col = f'{solver} Memory (MB)'
    
    for i, ratio in enumerate(ratios_to_show):
        ratio_data = df[df['Atom_Clause_Ratio'] == ratio]
        filtered_data = filter_valid_data(ratio_data, solver)
        
        if len(filtered_data) > 0:
            memory_data = filtered_data[memory_col].replace(0, 0.00001)
            ax1.plot(filtered_data['Number of Clauses'], memory_data, 
                    marker='o', linewidth=3, markersize=8, color=ratio_colors[i],
                    label=f'Stosunek {int(ratio)}:1')
    
    ax1.set_xlabel('Liczba klauzul', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Zużycie pamięci (MB)', fontsize=12, fontweight='bold')
    ax1.set_title('Snake - Wpływ stosunku atomów/klauzul na pamięć', 
                 fontsize=14, fontweight='bold')
    ax1.legend(title='Stosunek A:K', title_fontsize=12, fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Memory comparison between solvers
    ax2 = axes[0, 1]
    focus_clauses = 500
    clause_data = df[df['Number of Clauses'] == focus_clauses]
    
    solver_colors = plt.cm.Set1(np.linspace(0, 1, len(memory_solvers)))
    
    for i, solver in enumerate(memory_solvers):
        memory_col = f'{solver} Memory (MB)'
        filtered_data = filter_valid_data(clause_data, solver)
        
        if len(filtered_data) > 0:
            memory_data = filtered_data[memory_col].replace(0, 0.00001)
            ax2.plot(filtered_data['Atom_Clause_Ratio'], memory_data, 
                    marker='s', linewidth=3, markersize=8, color=solver_colors[i],
                    label=solver.capitalize())
    
    ax2.set_xlabel('Stosunek atomów do klauzul', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Zużycie pamięci (MB)', fontsize=12, fontweight='bold')
    ax2.set_title(f'Porównanie pamięci - {focus_clauses} klauzul', 
                 fontsize=14, fontweight='bold')
    ax2.legend(title='Solvery', title_fontsize=12, fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Memory efficiency (Memory per clause)
    ax3 = axes[1, 0]
    
    for i, solver in enumerate(['vampire', 'snake', 'z3']):  # Focus on most different
        memory_col = f'{solver} Memory (MB)'
        filtered_data = filter_valid_data(df, solver)
        
        if len(filtered_data) > 0:
            memory_per_clause = filtered_data[memory_col] / filtered_data['Number of Clauses']
            memory_per_clause = memory_per_clause.replace(0, 0.00001)
            
            scatter = ax3.scatter(filtered_data['Atom_Clause_Ratio'], memory_per_clause,
                                 c=filtered_data['Number of Clauses'], cmap='viridis',
                                 s=60, alpha=0.7, label=solver.capitalize())
    
    ax3.set_xlabel('Stosunek atomów do klauzul', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Pamięć na klauzulę (MB)', fontsize=12, fontweight='bold')
    ax3.set_title('Efektywność pamięci (kolor = liczba klauzul)', 
                 fontsize=14, fontweight='bold')
    ax3.legend(title='Solvery', title_fontsize=12, fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax3)
    cbar.set_label('Liczba klauzul')
    
    # Plot 4: Memory growth rate
    ax4 = axes[1, 1]
    
    # Calculate memory growth rate for each solver
    growth_data = []
    for solver in memory_solvers:
        memory_col = f'{solver} Memory (MB)'
        filtered_data = filter_valid_data(df, solver)
        
        # Group by ratio and calculate growth
        for ratio in [2, 3, 5]:
            ratio_data = filtered_data[filtered_data['Atom_Clause_Ratio'] == ratio]
            if len(ratio_data) >= 2:
                clauses = ratio_data['Number of Clauses'].values
                memory = ratio_data[memory_col].values
                
                # Calculate average growth rate
                if len(clauses) > 1:
                    growth_rate = np.mean(np.diff(memory) / np.diff(clauses))
                    growth_data.append({
                        'Solver': solver.capitalize(),
                        'Ratio': ratio,
                        'Growth_Rate': max(growth_rate, 0.001)
                    })
    
    growth_df = pd.DataFrame(growth_data)
    
    if len(growth_df) > 0:
        pivot_growth = growth_df.pivot(index='Solver', columns='Ratio', values='Growth_Rate')
        
        # Create bar plot
        x = np.arange(len(pivot_growth.index))
        width = 0.25
        
        for i, ratio in enumerate(pivot_growth.columns):
            if ratio in pivot_growth.columns:
                ax4.bar(x + i*width, pivot_growth[ratio].values, width, 
                       label=f'Stosunek {int(ratio)}:1')
        
        ax4.set_xlabel('Solvery', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Wzrost pamięci (MB/klauzula)', fontsize=12, fontweight='bold')
        ax4.set_title('Tempo wzrostu zużycia pamięci', fontsize=14, fontweight='bold')
        ax4.set_xticks(x + width)
        ax4.set_xticklabels(pivot_growth.index)
        ax4.legend(title='Stosunek A:K', title_fontsize=12, fontsize=10)
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'p03_improved_memory_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def create_solver_performance_comparison(df, output_dir):
    """Create comprehensive solver performance comparison"""
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    
    # Plot 1: Success rate analysis
    ax1 = axes[0, 0]
    
    solvers = ['vampire', 'snake', 'z3', 'prover9', 'cvc5', 'e', 'inkresat']
    success_rates = []
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
    
    for solver in solvers:
        filtered_data = filter_valid_data(df, solver)
        sat_col = f'{solver} SAT'
        
        if len(filtered_data) > 0:
            success_rate = (filtered_data[sat_col] == True).sum() / len(df) * 100
        else:
            success_rate = 0
        success_rates.append(success_rate)
    
    bars = ax1.bar([s.capitalize() for s in solvers], success_rates, color=colors)
    ax1.set_ylabel('Wskaźnik sukcesu (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Wskaźnik sukcesu solverów', fontsize=14, fontweight='bold')
    ax1.set_ylim(0, 105)
    
    # Add value labels on bars
    for bar, rate in zip(bars, success_rates):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Average time comparison
    ax2 = axes[0, 1]
    
    avg_times = []
    for solver in solvers:
        filtered_data = filter_valid_data(df, solver)
        time_col = f'{solver} Time (s)'
        
        if len(filtered_data) > 0:
            avg_time = filtered_data[time_col].mean()
            avg_times.append(max(avg_time, 0.00001))
        else:
            avg_times.append(0.00001)
    
    bars = ax2.bar([s.capitalize() for s in solvers], avg_times, color=colors)
    ax2.set_ylabel('Średni czas (s)', fontsize=12, fontweight='bold')
    ax2.set_title('Średni czas wykonania', fontsize=14, fontweight='bold')
    ax2.set_yscale('log')
    
    # Add value labels
    for bar, time_val in zip(bars, avg_times):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height * 1.1,
                f'{time_val:.3f}s', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Memory usage comparison (excluding InKreSAT)
    ax3 = axes[0, 2]
    
    memory_solvers = [s for s in solvers if s != 'inkresat']
    avg_memories = []
    memory_colors = colors[:-1]  # Exclude last color for InKreSAT
    
    for solver in memory_solvers:
        filtered_data = filter_valid_data(df, solver)
        memory_col = f'{solver} Memory (MB)'
        
        if len(filtered_data) > 0:
            avg_memory = filtered_data[memory_col].mean()
            avg_memories.append(max(avg_memory, 0.001))
        else:
            avg_memories.append(0.001)
    
    bars = ax3.bar([s.capitalize() for s in memory_solvers], avg_memories, color=memory_colors)
    ax3.set_ylabel('Średnia pamięć (MB)', fontsize=12, fontweight='bold')
    ax3.set_title('Średnie zużycie pamięci', fontsize=14, fontweight='bold')
    
    # Add value labels
    for bar, mem_val in zip(bars, avg_memories):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + max(avg_memories)*0.01,
                f'{mem_val:.1f}MB', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Performance vs complexity (bubble chart)
    ax4 = axes[1, 0]
    
    # Calculate complexity score (clauses * ratio)
    df['Complexity'] = df['Number of Clauses'] * df['Atom_Clause_Ratio']
    
    for i, solver in enumerate(['vampire', 'snake', 'cvc5']):  # Focus on best
        time_col = f'{solver} Time (s)'
        filtered_data = filter_valid_data(df, solver)
        
        if len(filtered_data) > 0:
            time_data = filtered_data[time_col].replace(0, 0.00001)
            sizes = filtered_data['Number of Clauses'] / 10  # Scale for bubble size
            
            ax4.scatter(filtered_data['Complexity'], time_data, 
                       s=sizes, alpha=0.6, c=colors[i], label=solver.capitalize())
    
    ax4.set_xlabel('Złożoność (klauzule × stosunek)', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Czas wykonania (s)', fontsize=12, fontweight='bold')
    ax4.set_title('Wydajność vs złożoność problemu', fontsize=14, fontweight='bold')
    ax4.set_yscale('log')
    ax4.set_xscale('log')
    ax4.legend(title='Solvery', title_fontsize=12, fontsize=10)
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Optimal ratios for each solver
    ax5 = axes[1, 1]
    
    optimal_ratios = []
    for solver in ['vampire', 'snake', 'z3', 'cvc5', 'e']:
        time_col = f'{solver} Time (s)'
        filtered_data = filter_valid_data(df, solver)
        
        if len(filtered_data) > 0:
            # Find ratio with minimum average time
            ratio_times = filtered_data.groupby('Atom_Clause_Ratio')[time_col].mean()
            optimal_ratio = ratio_times.idxmin()
            optimal_ratios.append(optimal_ratio)
        else:
            optimal_ratios.append(0)
    
    solver_names = ['Vampire', 'Snake', 'Z3', 'CVC5', 'E']
    bars = ax5.bar(solver_names, optimal_ratios, color=colors[:5])
    ax5.set_ylabel('Optymalny stosunek atomów/klauzul', fontsize=12, fontweight='bold')
    ax5.set_title('Optymalne stosunki dla każdego solvera', fontsize=14, fontweight='bold')
    
    # Add value labels
    for bar, ratio in zip(bars, optimal_ratios):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{ratio:.1f}:1', ha='center', va='bottom', fontweight='bold')
    
    ax5.grid(True, alpha=0.3, axis='y')
    
    # Plot 6: Error analysis for Prover9
    ax6 = axes[1, 2]
    
    prover9_all = df[['Number of Clauses', 'Atom_Clause_Ratio', 'prover9 SAT']].copy()
    prover9_errors = prover9_all[prover9_all['prover9 SAT'] == 'ERROR']
    prover9_success = prover9_all[prover9_all['prover9 SAT'] == True]
    
    if len(prover9_errors) > 0:
        ax6.scatter(prover9_errors['Atom_Clause_Ratio'], prover9_errors['Number of Clauses'],
                   c='red', s=100, alpha=0.7, marker='x', label='ERROR', linewidth=3)
    
    if len(prover9_success) > 0:
        ax6.scatter(prover9_success['Atom_Clause_Ratio'], prover9_success['Number of Clauses'],
                   c='green', s=60, alpha=0.7, marker='o', label='SUCCESS')
    
    ax6.set_xlabel('Stosunek atomów do klauzul', fontsize=12, fontweight='bold')
    ax6.set_ylabel('Liczba klauzul', fontsize=12, fontweight='bold')
    ax6.set_title('Prover9 - Analiza błędów vs sukcesów', fontsize=14, fontweight='bold')
    ax6.legend(title='Status', title_fontsize=12, fontsize=10)
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'p03_improved_solver_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def create_summary_table(df, output_dir):
    """Create an improved summary table with key insights"""
    
    solvers = ['vampire', 'snake', 'z3', 'prover9', 'cvc5', 'e', 'inkresat']
    
    summary_data = []
    
    for solver in solvers:
        time_col = f'{solver} Time (s)'
        memory_col = f'{solver} Memory (MB)'
        sat_col = f'{solver} SAT'
        
        filtered_df = filter_valid_data(df, solver)
        
        if len(filtered_df) > 0:
            # Find best performance ratio
            if filtered_df[time_col].min() > 0:
                best_ratio_idx = filtered_df[time_col].idxmin()
                best_ratio = filtered_df.loc[best_ratio_idx, 'Atom_Clause_Ratio']
                best_clauses = filtered_df.loc[best_ratio_idx, 'Number of Clauses']
            else:
                best_ratio = 'N/A'
                best_clauses = 'N/A'
            
            solver_stats = {
                'Solver': solver.capitalize(),
                'Avg Time (s)': f"{filtered_df[time_col].mean():.4f}",
                'Min Time (s)': f"{filtered_df[time_col].min():.4f}",
                'Max Time (s)': f"{filtered_df[time_col].max():.4f}",
                'Avg Memory (MB)': f"{filtered_df[memory_col].mean():.2f}" if solver != 'inkresat' else 'N/A',
                'Success Rate (%)': f"{(filtered_df[sat_col] == True).sum() / len(df) * 100:.1f}",
                'Valid Tests': len(filtered_df),
                'Best Ratio': f"{best_ratio:.1f}:1" if best_ratio != 'N/A' else 'N/A',
                'Best Config': f"{best_clauses}k/{best_ratio:.1f}" if best_ratio != 'N/A' else 'N/A'
            }
        else:
            solver_stats = {
                'Solver': solver.capitalize(),
                'Avg Time (s)': 'N/A',
                'Min Time (s)': 'N/A',
                'Max Time (s)': 'N/A',
                'Avg Memory (MB)': 'N/A',
                'Success Rate (%)': '0.0',
                'Valid Tests': 0,
                'Best Ratio': 'N/A',
                'Best Config': 'N/A'
            }
        summary_data.append(solver_stats)
    
    summary_df = pd.DataFrame(summary_data)
    
    # Create enhanced table plot
    fig, ax = plt.subplots(figsize=(18, 8))
    ax.axis('tight')
    ax.axis('off')
    
    # Create table
    table = ax.table(cellText=summary_df.values,
                     colLabels=summary_df.columns,
                     cellLoc='center',
                     loc='center')
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.3, 2)
    
    # Enhanced styling
    header_color = '#4CAF50'
    success_color = '#E8F5E8'
    warning_color = '#FFF3CD'
    error_color = '#F8D7DA'
    
    # Style header
    for i in range(len(summary_df.columns)):
        table[(0, i)].set_facecolor(header_color)
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Style data rows based on performance
    for i in range(1, len(summary_df) + 1):
        success_rate = float(summary_df.iloc[i-1]['Success Rate (%)'])
        
        if success_rate == 100.0:
            row_color = success_color
        elif success_rate > 0:
            row_color = warning_color
        else:
            row_color = error_color
        
        for j in range(len(summary_df.columns)):
            table[(i, j)].set_facecolor(row_color)
    
    plt.title('Szczegółowe porównanie wydajności solverów - Problem P03\n' +
              'Różne stosunki atomów do klauzul (50-2000 klauzul, stosunki 2:1 do 10:1)\n' +
              'Kolor: Zielony=100% sukces, Żółty=Częściowy sukces, Czerwony=Brak sukcesu', 
              fontsize=16, fontweight='bold', pad=30)
    
    plt.savefig(output_dir / 'p03_improved_summary_table.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Save to CSV
    summary_df.to_csv(output_dir / 'p03_improved_summary.csv', index=False)
    
    return fig, summary_df

def main():
    """Main function to generate all improved plots"""
    # Setup
    csv_file = Path('problem3/avg.csv')
    output_dir = Path('problem3_plots')
    output_dir.mkdir(exist_ok=True)
    
    # Load data
    print("Loading and analyzing data...")
    df = load_data(csv_file)
    
    print(f"Dataset overview:")
    print(f"- Total configurations: {len(df)}")
    print(f"- Clause counts: {sorted(df['Number of Clauses'].unique())}")
    print(f"- Atom-to-clause ratios: {sorted(df['Atom_Clause_Ratio'].unique())}")
    
    # Check data quality
    error_count = 0
    for col in df.columns:
        if 'SAT' in col:
            errors = (df[col] == 'ERROR').sum()
            if errors > 0:
                print(f"- ERROR values in {col}: {errors}")
                error_count += errors
    
    print(f"- Total ERROR values: {error_count}")
    print(f"- InKreSAT memory data: always 0.0 MB (excluded from memory analysis)")
    
    # Generate improved plots
    print("\n🎯 Generating improved visualizations...")
    
    print("1. Creating focused time analysis...")
    create_focused_time_analysis(df, output_dir)
    
    print("2. Creating memory usage analysis...")
    create_memory_usage_analysis(df, output_dir)
    
    print("3. Creating comprehensive solver comparison...")
    create_solver_performance_comparison(df, output_dir)
    
    print("4. Creating enhanced summary table...")
    fig, summary_df = create_summary_table(df, output_dir)
    
    print(f"\n✅ All improved plots saved to: {output_dir}")
    print("Generated files:")
    for file in sorted(output_dir.glob('p03_improved_*.png')):
        print(f"  📊 {file.name}")
    print(f"  📋 p03_improved_summary.csv")
    
    # Print key insights
    print(f"\n🔍 Key insights:")
    successful_solvers = summary_df[summary_df['Success Rate (%)'] == '100.0']['Solver'].tolist()
    print(f"- Fully successful solvers: {', '.join(successful_solvers)}")
    
    fastest_solver = summary_df.loc[summary_df['Avg Time (s)'] != 'N/A']['Avg Time (s)'].astype(float).idxmin()
    fastest_name = summary_df.iloc[fastest_solver]['Solver']
    print(f"- Fastest solver on average: {fastest_name}")
    
    print(f"- Prover9 has significant issues with larger problems")
    print(f"- Lower atom-to-clause ratios generally perform better")

if __name__ == "__main__":
    main() 