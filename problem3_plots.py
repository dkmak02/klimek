import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D

# Set style for better looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

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

def create_ratio_analysis_plots(df, output_dir):
    """Create plots showing how atom-to-clause ratio affects performance"""
    
    # Get unique clause counts for analysis
    clause_counts = sorted(df['Number of Clauses'].unique())
    
    # Solvers to analyze (excluding InKreSAT from memory plots later)
    solvers = {
        'Vampire': 'vampire Time (s)',
        'Snake': 'snake Time (s)', 
        'Z3': 'z3 Time (s)',
        'Prover9': 'prover9 Time (s)',
        'CVC5': 'cvc5 Time (s)',
        'E': 'e Time (s)',
        'InKreSAT': 'inkresat Time (s)'
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Time vs Ratio for different clause counts
    ax1 = axes[0, 0]
    for clauses in clause_counts:
        clause_data = df[df['Number of Clauses'] == clauses]
        for solver_name, time_col in solvers.items():
            solver_key = solver_name.lower().replace('sat', 'sat')
            if solver_key == 'inkresat':
                solver_key = 'inkresat'
            
            filtered_data = filter_valid_data(clause_data, solver_key)
            if len(filtered_data) > 0:
                # Replace zeros for better visualization
                time_data = filtered_data[time_col].replace(0, 0.00001)
                ax1.plot(filtered_data['Atom_Clause_Ratio'], time_data, 
                        marker='o', linewidth=1.5, markersize=4, 
                        label=f'{solver_name} ({clauses} klauzul)' if clauses == clause_counts[0] else "")
    
    ax1.set_xlabel('Stosunek atomów do klauzul', fontsize=12)
    ax1.set_ylabel('Czas wykonania (s)', fontsize=12)
    ax1.set_title('Wpływ stosunku atomów/klauzul na czas wykonania', fontsize=14, fontweight='bold')
    ax1.set_yscale('log')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Memory vs Ratio (excluding InKreSAT)
    ax2 = axes[0, 1]
    memory_solvers = {k: v.replace('Time (s)', 'Memory (MB)') for k, v in solvers.items() if k != 'InKreSAT'}
    
    for clauses in clause_counts:
        clause_data = df[df['Number of Clauses'] == clauses]
        for solver_name, memory_col in memory_solvers.items():
            solver_key = solver_name.lower()
            
            filtered_data = filter_valid_data(clause_data, solver_key)
            if len(filtered_data) > 0:
                # Replace zeros for better visualization
                memory_data = filtered_data[memory_col].replace(0, 0.00001)
                ax2.plot(filtered_data['Atom_Clause_Ratio'], memory_data, 
                        marker='s', linewidth=1.5, markersize=4,
                        label=f'{solver_name} ({clauses} klauzul)' if clauses == clause_counts[0] else "")
    
    ax2.set_xlabel('Stosunek atomów do klauzul', fontsize=12)
    ax2.set_ylabel('Zużycie pamięci (MB)', fontsize=12)
    ax2.set_title('Wpływ stosunku atomów/klauzul na zużycie pamięci', fontsize=14, fontweight='bold')
    ax2.set_yscale('log')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Success rate by ratio
    ax3 = axes[1, 0]
    ratios = sorted(df['Atom_Clause_Ratio'].unique())
    
    for solver_name in solvers.keys():
        solver_key = solver_name.lower().replace('sat', 'sat')
        if solver_key == 'inkresat':
            solver_key = 'inkresat'
        
        sat_col = f'{solver_key} SAT'
        success_rates = []
        
        for ratio in ratios:
            ratio_data = df[df['Atom_Clause_Ratio'] == ratio]
            filtered_data = filter_valid_data(ratio_data, solver_key)
            
            if len(filtered_data) > 0:
                success_rate = (filtered_data[sat_col] == True).sum() / len(filtered_data) * 100
                success_rates.append(success_rate)
            else:
                success_rates.append(0)
        
        ax3.plot(ratios, success_rates, marker='o', linewidth=2, markersize=6, label=solver_name)
    
    ax3.set_xlabel('Stosunek atomów do klauzul', fontsize=12)
    ax3.set_ylabel('Wskaźnik sukcesu (%)', fontsize=12)
    ax3.set_title('Wskaźnik sukcesu vs stosunek atomów/klauzul', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 105)
    
    # Plot 4: Average performance by clause count
    ax4 = axes[1, 1]
    best_solvers = ['Snake', 'Vampire', 'CVC5']  # Focus on best performing
    
    for solver_name in best_solvers:
        solver_key = solver_name.lower()
        time_col = f'{solver_key} Time (s)'
        
        avg_times = []
        for clauses in clause_counts:
            clause_data = df[df['Number of Clauses'] == clauses]
            filtered_data = filter_valid_data(clause_data, solver_key)
            
            if len(filtered_data) > 0:
                avg_time = filtered_data[time_col].mean()
                avg_times.append(max(avg_time, 0.00001))  # Replace 0 with small value
            else:
                avg_times.append(0.00001)
        
        ax4.loglog(clause_counts, avg_times, marker='o', linewidth=2, markersize=6, label=solver_name)
    
    ax4.set_xlabel('Liczba klauzul', fontsize=12)
    ax4.set_ylabel('Średni czas wykonania (s)', fontsize=12)
    ax4.set_title('Średni czas vs liczba klauzul (najlepsze solvery)', fontsize=14, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'p03_ratio_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def create_heatmap_analysis(df, output_dir):
    """Create heatmaps showing performance across different clause counts and ratios"""
    
    # Focus on successful solvers for heatmap
    successful_solvers = ['snake', 'vampire', 'cvc5']
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for idx, solver in enumerate(successful_solvers):
        time_col = f'{solver} Time (s)'
        
        # Filter valid data
        filtered_df = filter_valid_data(df, solver)
        
        if len(filtered_df) > 0:
            # Create pivot table for heatmap
            pivot_data = filtered_df.pivot_table(
                values=time_col, 
                index='Number of Clauses', 
                columns='Atom_Clause_Ratio', 
                aggfunc='mean'
            )
            
            # Replace zeros with small values for log scale
            pivot_data = pivot_data.replace(0, 0.00001)
            
            # Create heatmap with log scale
            im = axes[idx].imshow(np.log10(pivot_data.values), cmap='viridis', aspect='auto')
            
            # Set labels
            axes[idx].set_title(f'{solver.capitalize()} - Czas wykonania (log10)', fontweight='bold')
            axes[idx].set_xlabel('Stosunek atomów/klauzul')
            axes[idx].set_ylabel('Liczba klauzul')
            
            # Set ticks
            axes[idx].set_xticks(range(len(pivot_data.columns)))
            axes[idx].set_xticklabels([f'{ratio:.1f}' for ratio in pivot_data.columns], rotation=45)
            axes[idx].set_yticks(range(len(pivot_data.index)))
            axes[idx].set_yticklabels(pivot_data.index)
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=axes[idx])
            cbar.set_label('log10(Czas) [s]')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'p03_heatmap_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def create_3d_analysis(df, output_dir):
    """Create 3D plots showing relationship between clauses, ratio, and performance"""
    
    # Focus on best performing solvers
    best_solvers = ['vampire', 'snake', 'cvc5', 'e']
    colors = ['blue', 'red', 'green', 'orange']
    
    fig = plt.figure(figsize=(20, 10))
    
    # 3D Time plot with all solvers
    ax1 = fig.add_subplot(121, projection='3d')
    
    for i, solver in enumerate(best_solvers):
        time_col = f'{solver} Time (s)'
        filtered_df = filter_valid_data(df, solver)
        
        if len(filtered_df) > 0:
            time_data = filtered_df[time_col].replace(0, 0.00001)
            
            ax1.scatter(filtered_df['Number of Clauses'], 
                       filtered_df['Atom_Clause_Ratio'], 
                       time_data,
                       c=colors[i], s=50, alpha=0.7, label=solver.capitalize())
    
    ax1.set_xlabel('Liczba klauzul')
    ax1.set_ylabel('Stosunek atomów/klauzul')
    ax1.set_zlabel('Czas wykonania (s)')
    ax1.set_title('Wszystkie solvery - Czas wykonania 3D', fontweight='bold')
    ax1.set_zscale('log')
    ax1.legend()
    
    # 3D Memory plot with all solvers (excluding InKreSAT)
    ax2 = fig.add_subplot(122, projection='3d')
    
    for i, solver in enumerate(best_solvers):
        memory_col = f'{solver} Memory (MB)'
        filtered_df = filter_valid_data(df, solver)
        
        if len(filtered_df) > 0:
            memory_data = filtered_df[memory_col].replace(0, 0.00001)
            
            ax2.scatter(filtered_df['Number of Clauses'], 
                       filtered_df['Atom_Clause_Ratio'], 
                       memory_data,
                       c=colors[i], s=50, alpha=0.7, label=solver.capitalize())
    
    ax2.set_xlabel('Liczba klauzul')
    ax2.set_ylabel('Stosunek atomów/klauzul')
    ax2.set_zlabel('Zużycie pamięci (MB)')
    ax2.set_title('Wszystkie solvery - Zużycie pamięci 3D', fontweight='bold')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'p03_3d_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def create_individual_3d_plots(df, output_dir):
    """Create individual 3D plots for each solver"""
    
    # All successful solvers
    solvers = ['vampire', 'snake', 'z3', 'cvc5', 'e', 'inkresat']
    
    fig = plt.figure(figsize=(24, 16))
    
    for idx, solver in enumerate(solvers):
        time_col = f'{solver} Time (s)'
        memory_col = f'{solver} Memory (MB)'
        
        filtered_df = filter_valid_data(df, solver)
        
        if len(filtered_df) > 0:
            # Time plot
            ax1 = fig.add_subplot(3, 4, idx*2 + 1, projection='3d')
            time_data = filtered_df[time_col].replace(0, 0.00001)
            
            scatter1 = ax1.scatter(filtered_df['Number of Clauses'], 
                                  filtered_df['Atom_Clause_Ratio'], 
                                  time_data,
                                  c=time_data, cmap='viridis', s=40)
            
            ax1.set_xlabel('Klauzule')
            ax1.set_ylabel('Stosunek A/K')
            ax1.set_zlabel('Czas (s)')
            ax1.set_title(f'{solver.capitalize()} - Czas', fontweight='bold', fontsize=10)
            ax1.set_zscale('log')
            
            # Memory plot (skip for InKreSAT)
            if solver != 'inkresat':
                ax2 = fig.add_subplot(3, 4, idx*2 + 2, projection='3d')
                memory_data = filtered_df[memory_col].replace(0, 0.00001)
                
                scatter2 = ax2.scatter(filtered_df['Number of Clauses'], 
                                      filtered_df['Atom_Clause_Ratio'], 
                                      memory_data,
                                      c=memory_data, cmap='plasma', s=40)
                
                ax2.set_xlabel('Klauzule')
                ax2.set_ylabel('Stosunek A/K')
                ax2.set_zlabel('Pamięć (MB)')
                ax2.set_title(f'{solver.capitalize()} - Pamięć', fontweight='bold', fontsize=10)
            else:
                # For InKreSAT, create a text plot explaining no memory data
                ax2 = fig.add_subplot(3, 4, idx*2 + 2)
                ax2.text(0.5, 0.5, f'{solver.capitalize()}\nBrak danych\no pamięci', 
                        ha='center', va='center', fontsize=12, fontweight='bold')
                ax2.set_xlim(0, 1)
                ax2.set_ylim(0, 1)
                ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'p03_individual_3d_plots.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def create_comparison_plots(df, output_dir):
    """Create comparison plots showing different aspects of P03"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Time vs Clause count for different ratios (Snake only)
    ax1 = axes[0, 0]
    ratios_to_show = [2, 3, 4, 5, 10]
    
    for ratio in ratios_to_show:
        ratio_data = df[df['Atom_Clause_Ratio'] == ratio]
        filtered_data = filter_valid_data(ratio_data, 'snake')
        
        if len(filtered_data) > 0:
            time_data = filtered_data['snake Time (s)'].replace(0, 0.00001)
            ax1.loglog(filtered_data['Number of Clauses'], time_data, 
                      marker='o', linewidth=2, markersize=6, label=f'Stosunek {int(ratio)}:1')
    
    ax1.set_xlabel('Liczba klauzul')
    ax1.set_ylabel('Czas wykonania (s)')
    ax1.set_title('Snake - Wpływ stosunku atomów/klauzul', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Memory vs Clause count for different ratios (Snake only)
    ax2 = axes[0, 1]
    
    for ratio in ratios_to_show:
        ratio_data = df[df['Atom_Clause_Ratio'] == ratio]
        filtered_data = filter_valid_data(ratio_data, 'snake')
        
        if len(filtered_data) > 0:
            memory_data = filtered_data['snake Memory (MB)'].replace(0, 0.00001)
            ax2.semilogy(filtered_data['Number of Clauses'], memory_data, 
                        marker='s', linewidth=2, markersize=6, label=f'Stosunek {int(ratio)}:1')
    
    ax2.set_xlabel('Liczba klauzul')
    ax2.set_ylabel('Zużycie pamięci (MB)')
    ax2.set_title('Snake - Zużycie pamięci vs stosunek atomów/klauzul', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Prover9 analysis (shows significant time differences)
    ax3 = axes[1, 0]
    prover9_data = filter_valid_data(df, 'prover9')
    
    if len(prover9_data) > 0:
        time_data = prover9_data['prover9 Time (s)'].replace(0, 0.00001)
        scatter = ax3.scatter(prover9_data['Atom_Clause_Ratio'], time_data, 
                             c=prover9_data['Number of Clauses'], cmap='viridis', 
                             s=60, alpha=0.7)
        
        ax3.set_xlabel('Stosunek atomów do klauzul')
        ax3.set_ylabel('Czas wykonania (s)')
        ax3.set_title('Prover9 - Czas vs stosunek (kolor = liczba klauzul)', fontweight='bold')
        ax3.set_yscale('log')
        ax3.grid(True, alpha=0.3)
        
        cbar = plt.colorbar(scatter, ax=ax3)
        cbar.set_label('Liczba klauzul')
    
    # Plot 4: All solvers performance comparison (average time by solver)
    ax4 = axes[1, 1]
    
    solvers = ['vampire', 'snake', 'z3', 'prover9', 'cvc5', 'e', 'inkresat']
    avg_times = []
    success_rates = []
    
    for solver in solvers:
        filtered_data = filter_valid_data(df, solver)
        time_col = f'{solver} Time (s)'
        sat_col = f'{solver} SAT'
        
        if len(filtered_data) > 0:
            avg_time = filtered_data[time_col].mean()
            success_rate = (filtered_data[sat_col] == True).sum() / len(df) * 100
            
            avg_times.append(max(avg_time, 0.00001))
            success_rates.append(success_rate)
        else:
            avg_times.append(0.00001)
            success_rates.append(0)
    
    # Create bubble plot
    sizes = [sr * 10 for sr in success_rates]  # Scale success rates for bubble size
    colors = plt.cm.viridis(np.linspace(0, 1, len(solvers)))
    
    for i, solver in enumerate(solvers):
        ax4.scatter(avg_times[i], success_rates[i], s=sizes[i], 
                   c=[colors[i]], alpha=0.7, label=solver.capitalize())
    
    ax4.set_xlabel('Średni czas wykonania (s)')
    ax4.set_ylabel('Wskaźnik sukcesu (%)')
    ax4.set_title('Porównanie solverów - Problem P03', fontweight='bold')
    ax4.set_xscale('log')
    ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'p03_comparison_plots.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def create_solver_comparison_table(df, output_dir):
    """Create a summary table comparing solvers across different formula configurations"""
    
    solvers = ['vampire', 'snake', 'z3', 'prover9', 'cvc5', 'e', 'inkresat']
    
    # Create summary statistics
    summary_data = []
    
    for solver in solvers:
        time_col = f'{solver} Time (s)'
        memory_col = f'{solver} Memory (MB)'
        sat_col = f'{solver} SAT'
        
        # Filter out ERROR values for this solver
        filtered_df = filter_valid_data(df, solver)
        
        if len(filtered_df) > 0:
            solver_stats = {
                'Solver': solver.capitalize(),
                'Avg Time (s)': filtered_df[time_col].mean(),
                'Max Time (s)': filtered_df[time_col].max(),
                'Avg Memory (MB)': filtered_df[memory_col].mean() if solver != 'inkresat' else 'N/A',
                'Max Memory (MB)': filtered_df[memory_col].max() if solver != 'inkresat' else 'N/A',
                'Success Rate (%)': (filtered_df[sat_col] == True).sum() / len(df) * 100,
                'Valid Tests': len(filtered_df),
                'Best Ratio': filtered_df.loc[filtered_df[time_col].idxmin(), 'Atom_Clause_Ratio'] if filtered_df[time_col].min() > 0 else 'N/A'
            }
        else:
            solver_stats = {
                'Solver': solver.capitalize(),
                'Avg Time (s)': 'N/A',
                'Max Time (s)': 'N/A', 
                'Avg Memory (MB)': 'N/A',
                'Max Memory (MB)': 'N/A',
                'Success Rate (%)': 0.0,
                'Valid Tests': 0,
                'Best Ratio': 'N/A'
            }
        summary_data.append(solver_stats)
    
    summary_df = pd.DataFrame(summary_data)
    
    # Create table plot
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(cellText=summary_df.values,
                     colLabels=summary_df.columns,
                     cellLoc='center',
                     loc='center')
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    
    # Style the table
    for i in range(len(summary_df.columns)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    plt.title('Porównanie wydajności solverów - Problem P03\n(różne stosunki atomów do klauzul)', 
              fontsize=14, fontweight='bold', pad=20)
    plt.savefig(output_dir / 'p03_solver_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig, summary_df

def main():
    """Main function to generate all plots"""
    # Setup
    csv_file = Path('problem3/avg.csv')
    output_dir = Path('problem3_plots')
    output_dir.mkdir(exist_ok=True)
    
    # Load data
    print("Loading data...")
    df = load_data(csv_file)
    print(f"Loaded data for {len(df)} test configurations")
    print(f"Clause counts: {sorted(df['Number of Clauses'].unique())}")
    print(f"Atom-to-clause ratios: {sorted(df['Atom_Clause_Ratio'].unique())}")
    
    # Check for ERROR values
    error_count = 0
    for col in df.columns:
        if 'SAT' in col:
            errors = (df[col] == 'ERROR').sum()
            if errors > 0:
                print(f"Found {errors} ERROR value(s) in {col}")
                error_count += errors
    
    if error_count > 0:
        print(f"Total ERROR values found: {error_count} - these will be excluded from plots")
    
    print(f"InKreSAT memory data: {df['inkresat Memory (MB)'].unique()} - excluded from memory plots")
    
    # Generate plots
    print("\nGenerating ratio analysis plots...")
    create_ratio_analysis_plots(df, output_dir)
    
    print("Generating heatmap analysis...")
    create_heatmap_analysis(df, output_dir)
    
    print("Generating 3D analysis plots...")
    create_3d_analysis(df, output_dir)
    
    print("Generating individual 3D plots for each solver...")
    create_individual_3d_plots(df, output_dir)
    
    print("Generating comparison plots...")
    create_comparison_plots(df, output_dir)
    
    print("Generating solver comparison table...")
    fig, summary_df = create_solver_comparison_table(df, output_dir)
    
    # Save summary to CSV
    summary_df.to_csv(output_dir / 'p03_solver_summary.csv', index=False)
    
    print(f"\nAll plots saved to: {output_dir}")
    print("Generated files:")
    for file in output_dir.glob('p03_*.png'):
        print(f"  - {file.name}")
    print(f"  - p03_solver_summary.csv")

if __name__ == "__main__":
    main() 