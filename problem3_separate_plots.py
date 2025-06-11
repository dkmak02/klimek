import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path

# Set style for better looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("Set2")

def load_data(csv_file):
    """Load and preprocess the CSV data"""
    df = pd.read_csv(csv_file, delimiter=';')
    
    # Convert comma decimal separators to dots
    numeric_columns = [col for col in df.columns if 'Memory' in col or 'Time' in col]
    
    for col in numeric_columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype(str).str.replace(',', '.').astype(float)
    
    # Calculate atom-to-clause ratio
    df['Atom_Clause_Ratio'] = df['Number of Atoms'] / df['Number of Clauses']
    
    # Sort data
    df = df.sort_values(['Number of Clauses', 'Atom_Clause_Ratio'])
    
    return df

def filter_valid_data(df, solver, exclude_errors=True):
    """Filter data to exclude ERROR values"""
    if exclude_errors:
        sat_col = f'{solver} SAT'
        mask = df[sat_col] != 'ERROR'
        return df[mask]
    return df

def plot_01_fast_solvers_time(df, output_dir):
    """Plot 1: Fast solvers time analysis"""
    
    fast_solvers = ['vampire', 'snake', 'z3', 'inkresat']
    clause_counts = [50, 100, 200, 500, 1000, 2000]
    colors = plt.cm.tab10(np.linspace(0, 1, len(clause_counts)))
    
    plt.figure(figsize=(14, 8))
    
    for solver in fast_solvers:
        time_col = f'{solver} Time (s)'
        for i, clauses in enumerate(clause_counts):
            clause_data = df[df['Number of Clauses'] == clauses]
            filtered_data = filter_valid_data(clause_data, solver)
            
            if len(filtered_data) > 0:
                time_data = filtered_data[time_col].replace(0, 0.00001)
                linestyle = '-' if solver in ['vampire', 'snake'] else '--'
                plt.plot(filtered_data['Atom_Clause_Ratio'], time_data, 
                        marker='o', linewidth=2.5, markersize=7, color=colors[i],
                        label=f'{clauses} klauzul' if solver == fast_solvers[0] else "",
                        linestyle=linestyle)
    
    plt.xlabel('Stosunek atomów do klauzul', fontsize=14, fontweight='bold')
    plt.ylabel('Czas wykonania (s)', fontsize=14, fontweight='bold')
    plt.title('Szybkie solvery: Vampire, Snake, Z3, InKreSAT\n(linie ciągłe: Vampire/Snake, przerywane: Z3/InKreSAT)', 
             fontsize=16, fontweight='bold')
    plt.yscale('log')
    plt.legend(title='Liczba klauzul', title_fontsize=14, fontsize=12, loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'p03_01_fast_solvers_time.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_02_medium_solvers_time(df, output_dir):
    """Plot 2: Medium solvers time analysis"""
    
    medium_solvers = ['cvc5', 'e']
    clause_counts = [50, 100, 200, 500, 1000, 2000]
    colors = plt.cm.tab10(np.linspace(0, 1, len(clause_counts)))
    
    plt.figure(figsize=(14, 8))
    
    for solver in medium_solvers:
        time_col = f'{solver} Time (s)'
        for i, clauses in enumerate(clause_counts):
            clause_data = df[df['Number of Clauses'] == clauses]
            filtered_data = filter_valid_data(clause_data, solver)
            
            if len(filtered_data) > 0:
                time_data = filtered_data[time_col].replace(0, 0.00001)
                linestyle = '-' if solver == 'cvc5' else '--'
                plt.plot(filtered_data['Atom_Clause_Ratio'], time_data, 
                        marker='s', linewidth=2.5, markersize=7, color=colors[i],
                        label=f'{clauses} klauzul' if solver == medium_solvers[0] else "",
                        linestyle=linestyle)
    
    plt.xlabel('Stosunek atomów do klauzul', fontsize=14, fontweight='bold')
    plt.ylabel('Czas wykonania (s)', fontsize=14, fontweight='bold')
    plt.title('Średnie solvery: CVC5, E\n(linia ciągła: CVC5, przerywana: E)', 
             fontsize=16, fontweight='bold')
    plt.yscale('log')
    plt.legend(title='Liczba klauzul', title_fontsize=14, fontsize=12, loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'p03_02_medium_solvers_time.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_03_solver_comparison_1000(df, output_dir):
    """Plot 3: All solvers comparison for 1000 clauses"""
    
    focus_clauses = 1000
    clause_data = df[df['Number of Clauses'] == focus_clauses]
    all_solvers = ['vampire', 'snake', 'z3', 'prover9', 'cvc5', 'e', 'inkresat']
    solver_colors = plt.cm.Set1(np.linspace(0, 1, len(all_solvers)))
    
    plt.figure(figsize=(14, 8))
    
    for i, solver in enumerate(all_solvers):
        time_col = f'{solver} Time (s)'
        filtered_data = filter_valid_data(clause_data, solver)
        
        if len(filtered_data) > 0:
            time_data = filtered_data[time_col].replace(0, 0.00001)
            plt.plot(filtered_data['Atom_Clause_Ratio'], time_data, 
                    marker='o', linewidth=3, markersize=8, color=solver_colors[i],
                    label=solver.capitalize())
    
    plt.xlabel('Stosunek atomów do klauzul', fontsize=14, fontweight='bold')
    plt.ylabel('Czas wykonania (s)', fontsize=14, fontweight='bold')
    plt.title(f'Porównanie wszystkich solverów - {focus_clauses} klauzul', 
             fontsize=16, fontweight='bold')
    plt.yscale('log')
    plt.legend(title='Solvery', title_fontsize=14, fontsize=12, loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'p03_03_all_solvers_1000clauses.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_04_scaling_with_clauses(df, output_dir):
    """Plot 4: Performance scaling with clause count"""
    
    best_solvers = ['vampire', 'snake', 'prover9', 'cvc5']
    ratio_focus = 2.0
    ratio_data = df[df['Atom_Clause_Ratio'] == ratio_focus]
    
    plt.figure(figsize=(14, 8))
    
    for i, solver in enumerate(best_solvers):
        time_col = f'{solver} Time (s)'
        filtered_data = filter_valid_data(ratio_data, solver)
        
        if len(filtered_data) > 0:
            time_data = filtered_data[time_col].replace(0, 0.00001)
            plt.loglog(filtered_data['Number of Clauses'], time_data, 
                      marker='o', linewidth=3, markersize=8, 
                      label=solver.capitalize())
    
    plt.xlabel('Liczba klauzul', fontsize=14, fontweight='bold')
    plt.ylabel('Czas wykonania (s)', fontsize=14, fontweight='bold')
    plt.title(f'Skalowanie z liczbą klauzul (stosunek {ratio_focus}:1)', 
             fontsize=16, fontweight='bold')
    plt.legend(title='Najlepsze solvery', title_fontsize=14, fontsize=12, loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'p03_04_scaling_with_clauses.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_05_snake_memory_analysis(df, output_dir):
    """Plot 5: Snake memory analysis by ratio"""
    
    ratios_to_show = [2, 3, 5, 10]
    ratio_colors = plt.cm.viridis(np.linspace(0, 1, len(ratios_to_show)))
    
    plt.figure(figsize=(14, 8))
    
    solver = 'snake'
    memory_col = f'{solver} Memory (MB)'
    
    for i, ratio in enumerate(ratios_to_show):
        ratio_data = df[df['Atom_Clause_Ratio'] == ratio]
        filtered_data = filter_valid_data(ratio_data, solver)
        
        if len(filtered_data) > 0:
            memory_data = filtered_data[memory_col].replace(0, 0.00001)
            plt.plot(filtered_data['Number of Clauses'], memory_data, 
                    marker='o', linewidth=3, markersize=8, color=ratio_colors[i],
                    label=f'Stosunek {int(ratio)}:1')
    
    plt.xlabel('Liczba klauzul', fontsize=14, fontweight='bold')
    plt.ylabel('Zużycie pamięci (MB)', fontsize=14, fontweight='bold')
    plt.title('Snake - Wpływ stosunku atomów/klauzul na zużycie pamięci', 
             fontsize=16, fontweight='bold')
    plt.legend(title='Stosunek A:K', title_fontsize=14, fontsize=12, loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'p03_05_snake_memory_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_06_memory_comparison_500(df, output_dir):
    """Plot 6: Memory comparison for 500 clauses"""
    
    focus_clauses = 500
    clause_data = df[df['Number of Clauses'] == focus_clauses]
    memory_solvers = ['vampire', 'snake', 'z3', 'cvc5', 'e']
    solver_colors = plt.cm.Set1(np.linspace(0, 1, len(memory_solvers)))
    
    plt.figure(figsize=(14, 8))
    
    for i, solver in enumerate(memory_solvers):
        memory_col = f'{solver} Memory (MB)'
        filtered_data = filter_valid_data(clause_data, solver)
        
        if len(filtered_data) > 0:
            memory_data = filtered_data[memory_col].replace(0, 0.00001)
            plt.plot(filtered_data['Atom_Clause_Ratio'], memory_data, 
                    marker='s', linewidth=3, markersize=8, color=solver_colors[i],
                    label=solver.capitalize())
    
    plt.xlabel('Stosunek atomów do klauzul', fontsize=14, fontweight='bold')
    plt.ylabel('Zużycie pamięci (MB)', fontsize=14, fontweight='bold')
    plt.title(f'Porównanie zużycia pamięci - {focus_clauses} klauzul', 
             fontsize=16, fontweight='bold')
    plt.legend(title='Solvery', title_fontsize=14, fontsize=12, loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'p03_06_memory_comparison_500clauses.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_07_success_rates(df, output_dir):
    """Plot 7: Success rates for all solvers"""
    
    solvers = ['vampire', 'snake', 'z3', 'prover9', 'cvc5', 'e', 'inkresat']
    success_rates = []
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
    
    plt.figure(figsize=(14, 8))
    
    for solver in solvers:
        filtered_data = filter_valid_data(df, solver)
        sat_col = f'{solver} SAT'
        
        if len(filtered_data) > 0:
            success_rate = (filtered_data[sat_col].isin([True, False])).sum() / len(df) * 100
        else:
            success_rate = 0
        success_rates.append(success_rate)
    
    bars = plt.bar([s.capitalize() for s in solvers], success_rates, color=colors)
    plt.ylabel('Wskaźnik sukcesu (%)', fontsize=14, fontweight='bold')
    plt.title('Wskaźnik sukcesu solverów - Problem P03', fontsize=16, fontweight='bold')
    plt.ylim(0, 105)
    
    # Add value labels on bars
    for bar, rate in zip(bars, success_rates):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_dir / 'p03_07_success_rates.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_08_average_times(df, output_dir):
    """Plot 8: Average execution times"""
    
    solvers = ['vampire', 'snake', 'z3', 'prover9', 'cvc5', 'e', 'inkresat']
    avg_times = []
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
    
    plt.figure(figsize=(14, 8))
    
    for solver in solvers:
        filtered_data = filter_valid_data(df, solver)
        time_col = f'{solver} Time (s)'
        
        if len(filtered_data) > 0:
            avg_time = filtered_data[time_col].mean()
            avg_times.append(max(avg_time, 0.00001))
        else:
            avg_times.append(0.00001)
    
    bars = plt.bar([s.capitalize() for s in solvers], avg_times, color=colors)
    plt.ylabel('Średni czas wykonania (s)', fontsize=14, fontweight='bold')
    plt.title('Średni czas wykonania - Problem P03', fontsize=16, fontweight='bold')
    plt.yscale('log')
    
    # Add value labels
    for bar, time_val in zip(bars, avg_times):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height * 1.5,
                f'{time_val:.4f}s', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_dir / 'p03_08_average_times.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_09_memory_usage(df, output_dir):
    """Plot 9: Average memory usage"""
    
    memory_solvers = ['vampire', 'snake', 'z3', 'cvc5', 'e']
    avg_memories = []
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd', '#8c564b']
    
    plt.figure(figsize=(14, 8))
    
    for solver in memory_solvers:
        filtered_data = filter_valid_data(df, solver)
        memory_col = f'{solver} Memory (MB)'
        
        if len(filtered_data) > 0:
            avg_memory = filtered_data[memory_col].mean()
            avg_memories.append(max(avg_memory, 0.001))
        else:
            avg_memories.append(0.001)
    
    bars = plt.bar([s.capitalize() for s in memory_solvers], avg_memories, color=colors)
    plt.ylabel('Średnie zużycie pamięci (MB)', fontsize=14, fontweight='bold')
    plt.title('Średnie zużycie pamięci - Problem P03', fontsize=16, fontweight='bold')
    
    # Add value labels
    for bar, mem_val in zip(bars, avg_memories):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + max(avg_memories)*0.02,
                f'{mem_val:.1f} MB', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_dir / 'p03_09_memory_usage.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_10_optimal_ratios(df, output_dir):
    """Plot 10: Optimal ratios for each solver"""
    
    plt.figure(figsize=(14, 8))
    
    optimal_ratios = []
    solver_names = ['Vampire', 'Snake', 'Z3', 'Prover9', 'CVC5', 'E']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    for solver in ['vampire', 'snake', 'z3', 'prover9', 'cvc5', 'e']:
        time_col = f'{solver} Time (s)'
        filtered_data = filter_valid_data(df, solver)
        
        if len(filtered_data) > 0:
            # Find ratio with minimum average time
            ratio_times = filtered_data.groupby('Atom_Clause_Ratio')[time_col].mean()
            optimal_ratio = ratio_times.idxmin()
            optimal_ratios.append(optimal_ratio)
        else:
            optimal_ratios.append(0)
    
    bars = plt.bar(solver_names, optimal_ratios, color=colors)
    plt.ylabel('Optymalny stosunek atomów/klauzul', fontsize=14, fontweight='bold')
    plt.title('Optymalne stosunki atomów/klauzul dla każdego solvera', fontsize=16, fontweight='bold')
    
    # Add value labels
    for bar, ratio in zip(bars, optimal_ratios):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{ratio:.1f}:1', ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_dir / 'p03_10_optimal_ratios.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_11_prover9_error_analysis(df, output_dir):
    """Plot 11: Prover9 error analysis"""
    
    plt.figure(figsize=(14, 8))
    
    prover9_all = df[['Number of Clauses', 'Atom_Clause_Ratio', 'prover9 SAT']].copy()
    prover9_errors = prover9_all[prover9_all['prover9 SAT'] == 'ERROR']
    prover9_success = prover9_all[prover9_all['prover9 SAT'].isin([True, False])]
    
    if len(prover9_errors) > 0:
        plt.scatter(prover9_errors['Atom_Clause_Ratio'], prover9_errors['Number of Clauses'],
                   c='red', s=150, alpha=0.8, marker='x', label='ERROR', linewidth=4)
    
    if len(prover9_success) > 0:
        plt.scatter(prover9_success['Atom_Clause_Ratio'], prover9_success['Number of Clauses'],
                   c='green', s=100, alpha=0.8, marker='o', label='SUCCESS')
    
    plt.xlabel('Stosunek atomów do klauzul', fontsize=14, fontweight='bold')
    plt.ylabel('Liczba klauzul', fontsize=14, fontweight='bold')
    plt.title('Prover9 - Analiza błędów vs sukcesów', fontsize=16, fontweight='bold')
    plt.legend(title='Status', title_fontsize=14, fontsize=12, loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'p03_11_prover9_errors.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_12_performance_vs_complexity(df, output_dir):
    """Plot 12: Performance vs problem complexity"""
    
    plt.figure(figsize=(14, 8))
    
    # Calculate complexity score
    df['Complexity'] = df['Number of Clauses'] * df['Atom_Clause_Ratio']
    
    best_solvers = ['vampire', 'snake', 'prover9', 'cvc5']
    colors = ['#1f77b4', '#ff7f0e', '#d62728', '#9467bd']
    
    for i, solver in enumerate(best_solvers):
        time_col = f'{solver} Time (s)'
        filtered_data = filter_valid_data(df, solver)
        
        if len(filtered_data) > 0:
            time_data = filtered_data[time_col].replace(0, 0.00001)
            sizes = filtered_data['Number of Clauses'] / 10  # Scale for bubble size
            
            plt.scatter(filtered_data['Complexity'], time_data, 
                       s=sizes, alpha=0.7, c=colors[i], label=solver.capitalize())
    
    plt.xlabel('Złożoność problemu (klauzule × stosunek)', fontsize=14, fontweight='bold')
    plt.ylabel('Czas wykonania (s)', fontsize=14, fontweight='bold')
    plt.title('Wydajność vs złożoność problemu\n(rozmiar bąbelka = liczba klauzul)', 
             fontsize=16, fontweight='bold')
    plt.yscale('log')
    plt.xscale('log')
    plt.legend(title='Solvery', title_fontsize=14, fontsize=12, loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'p03_12_performance_vs_complexity.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_summary_table(df, output_dir):
    """Create enhanced summary table"""
    
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
                'Success Rate (%)': f"{(filtered_df[sat_col].isin([True, False])).sum() / len(df) * 100:.1f}",
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
    
    # Create enhanced table
    fig, ax = plt.subplots(figsize=(20, 10))
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(cellText=summary_df.values,
                     colLabels=summary_df.columns,
                     cellLoc='center',
                     loc='center')
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.4, 2.5)
    
    # Enhanced styling
    header_color = '#4CAF50'
    success_color = '#E8F5E8'
    warning_color = '#FFF3CD'
    error_color = '#F8D7DA'
    
    # Style header
    for i in range(len(summary_df.columns)):
        table[(0, i)].set_facecolor(header_color)
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Style data rows
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
              fontsize=18, fontweight='bold', pad=40)
    
    plt.savefig(output_dir / 'p03_13_summary_table.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Save CSV
    summary_df.to_csv(output_dir / 'p03_separate_summary.csv', index=False)
    
    return summary_df

def main():
    """Main function to generate all separate plots"""
    
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
    
    # Generate all plots separately
    print("\n🎯 Generating separate plots for maximum readability...")
    
    print("📊 1/13: Fast solvers time analysis...")
    plot_01_fast_solvers_time(df, output_dir)
    
    print("📊 2/13: Medium solvers time analysis...")
    plot_02_medium_solvers_time(df, output_dir)
    
    print("📊 3/13: All solvers comparison (1000 clauses)...")
    plot_03_solver_comparison_1000(df, output_dir)
    
    print("📊 4/13: Performance scaling with clauses...")
    plot_04_scaling_with_clauses(df, output_dir)
    
    print("📊 5/13: Snake memory analysis...")
    plot_05_snake_memory_analysis(df, output_dir)
    
    print("📊 6/13: Memory comparison (500 clauses)...")
    plot_06_memory_comparison_500(df, output_dir)
    
    print("📊 7/13: Success rates...")
    plot_07_success_rates(df, output_dir)
    
    print("📊 8/13: Average execution times...")
    plot_08_average_times(df, output_dir)
    
    print("📊 9/13: Memory usage...")
    plot_09_memory_usage(df, output_dir)
    
    print("📊 10/13: Optimal ratios...")
    plot_10_optimal_ratios(df, output_dir)
    
    print("📊 11/13: Prover9 error analysis...")
    plot_11_prover9_error_analysis(df, output_dir)
    
    print("📊 12/13: Performance vs complexity...")
    plot_12_performance_vs_complexity(df, output_dir)
    
    print("📊 13/13: Summary table...")
    summary_df = create_summary_table(df, output_dir)
    
    print(f"\n✅ All 13 separate plots saved to: {output_dir}")
    print("Generated files:")
    for i in range(1, 14):
        filename = f'p03_{i:02d}_*.png'
        files = list(output_dir.glob(filename))
        if files:
            print(f"  📊 {files[0].name}")
    print(f"  📋 p03_separate_summary.csv")
    
    # Print insights
    print(f"\n🔍 Key insights:")
    successful_solvers = summary_df[summary_df['Success Rate (%)'] == '100.0']['Solver'].tolist()
    print(f"- Fully successful solvers: {', '.join(successful_solvers)}")
    
    fastest_solver = summary_df.loc[summary_df['Avg Time (s)'] != 'N/A']['Avg Time (s)'].astype(float).idxmin()
    fastest_name = summary_df.iloc[fastest_solver]['Solver']
    print(f"- Fastest solver on average: {fastest_name}")
    
    print(f"- Each plot is now a separate, large, readable figure!")

if __name__ == "__main__":
    main() 