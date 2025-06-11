import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path

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
    
    # Sort by number of clauses for proper plotting
    df = df.sort_values('Number of Clauses')
    
    return df

def filter_valid_data(df, solver, exclude_errors=True):
    """Filter data to exclude ERROR values and invalid measurements"""
    if exclude_errors:
        sat_col = f'{solver} SAT'
        # Exclude rows where SAT column contains 'ERROR'
        mask = df[sat_col] != 'ERROR'
        return df[mask]
    return df

def create_time_plots(df, output_dir):
    """Create plots showing time vs number of clauses for different solvers"""
    
    # Solvers and their time columns
    solvers = {
        'Vampire': 'vampire Time (s)',
        'Snake': 'snake Time (s)', 
        'Z3': 'z3 Time (s)',
        'Prover9': 'prover9 Time (s)',
        'CVC5': 'cvc5 Time (s)',
        'E': 'e Time (s)',
        'InKreSAT': 'inkresat Time (s)'
    }
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Linear scale plot
    for solver_name, time_col in solvers.items():
        solver_key = solver_name.lower().replace('sat', 'sat')  # Handle InKreSAT -> inkresat
        if solver_key == 'inkresat':
            solver_key = 'inkresat'
        
        # Filter out ERROR values for this solver
        filtered_df = filter_valid_data(df, solver_key)
        
        if len(filtered_df) > 0:
            ax1.plot(filtered_df['Number of Clauses'], filtered_df[time_col], 
                    marker='o', linewidth=2, markersize=6, label=solver_name)
    
    ax1.set_xlabel('Liczba klauzul', fontsize=12)
    ax1.set_ylabel('Czas wykonania (s)', fontsize=12)
    ax1.set_title('Wpływ liczby klauzul na czas wykonania\n(rozkład Poissona długości klauzul)', 
                 fontsize=14, fontweight='bold')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Log scale plot
    for solver_name, time_col in solvers.items():
        solver_key = solver_name.lower().replace('sat', 'sat')
        if solver_key == 'inkresat':
            solver_key = 'inkresat'
            
        # Filter out ERROR values for this solver
        filtered_df = filter_valid_data(df, solver_key)
        
        # Replace zero values with small number for log scale
        if len(filtered_df) > 0:
            log_data = filtered_df.copy()
            log_data[time_col] = log_data[time_col].replace(0, 0.00001)
            ax2.loglog(log_data['Number of Clauses'], log_data[time_col],
                      marker='o', linewidth=2, markersize=6, label=solver_name)
    
    ax2.set_xlabel('Liczba klauzul (skala log)', fontsize=12)
    ax2.set_ylabel('Czas wykonania (s, skala log)', fontsize=12)
    ax2.set_title('Wpływ liczby klauzul na czas wykonania (skala logarytmiczna)\n(rozkład Poissona długości klauzul)', 
                 fontsize=14, fontweight='bold')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'p02_time_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def create_memory_plots(df, output_dir):
    """Create plots showing memory usage vs number of clauses for different solvers"""
    
    # Solvers and their memory columns (excluding InKreSAT due to no memory data)
    solvers = {
        'Vampire': 'vampire Memory (MB)',
        'Snake': 'snake Memory (MB)',
        'Z3': 'z3 Memory (MB)',
        'Prover9': 'prover9 Memory (MB)',
        'CVC5': 'cvc5 Memory (MB)',
        'E': 'e Memory (MB)'
        # Excluded InKreSAT as it reports 0.0 MB consistently
    }
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Linear scale plot
    for solver_name, memory_col in solvers.items():
        solver_key = solver_name.lower()
        
        # Filter out ERROR values for this solver
        filtered_df = filter_valid_data(df, solver_key)
        
        if len(filtered_df) > 0:
            ax1.plot(filtered_df['Number of Clauses'], filtered_df[memory_col], 
                    marker='s', linewidth=2, markersize=6, label=solver_name)
    
    ax1.set_xlabel('Liczba klauzul', fontsize=12)
    ax1.set_ylabel('Zużycie pamięci (MB)', fontsize=12)
    ax1.set_title('Wpływ liczby klauzul na zużycie pamięci\n(rozkład Poissona długości klauzul)', 
                 fontsize=14, fontweight='bold')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Log scale plot
    for solver_name, memory_col in solvers.items():
        solver_key = solver_name.lower()
        
        # Filter out ERROR values for this solver
        filtered_df = filter_valid_data(df, solver_key)
        
        # Replace zero values with small number for log scale
        if len(filtered_df) > 0:
            log_data = filtered_df.copy()
            log_data[memory_col] = log_data[memory_col].replace(0, 0.00001)
            ax2.loglog(log_data['Number of Clauses'], log_data[memory_col],
                      marker='s', linewidth=2, markersize=6, label=solver_name)
    
    ax2.set_xlabel('Liczba klauzul (skala log)', fontsize=12)
    ax2.set_ylabel('Zużycie pamięci (MB, skala log)', fontsize=12)
    ax2.set_title('Wpływ liczby klauzul na zużycie pamięci (skala logarytmiczna)\n(rozkład Poissona długości klauzul)', 
                 fontsize=14, fontweight='bold')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'p02_memory_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def create_combined_performance_plot(df, output_dir):
    """Create a combined plot showing time vs memory for different formula sizes"""
    
    # Exclude InKreSAT from this plot due to no memory data
    solvers = ['vampire', 'snake', 'z3', 'prover9', 'cvc5', 'e']
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Color map for different clause numbers
    colors = plt.cm.viridis(np.linspace(0, 1, len(df)))
    
    for i, (_, row) in enumerate(df.iterrows()):
        clause_count = int(row['Number of Clauses'])
        
        for solver in solvers:
            time_col = f'{solver} Time (s)'
            memory_col = f'{solver} Memory (MB)'
            sat_col = f'{solver} SAT'
            
            # Skip if ERROR in SAT column
            if row[sat_col] == 'ERROR':
                continue
                
            # Replace zeros with small values for log plot
            time_val = max(row[time_col], 0.00001)
            memory_val = max(row[memory_col], 0.00001)
            
            ax.scatter(memory_val, time_val, 
                      s=100, alpha=0.7, c=[colors[i]], 
                      label=f'{solver.capitalize()} ({clause_count} klauzul)' if i == 0 else "")
    
    ax.set_xlabel('Zużycie pamięci (MB)', fontsize=12)
    ax.set_ylabel('Czas wykonania (s)', fontsize=12)
    ax.set_title('Zależność między czasem a pamięcią dla różnych rozmiarów formuł\n(rozkład Poissona długości klauzul, wykluczono InKreSAT)', 
                fontsize=14, fontweight='bold')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    # Create custom legend for clause counts
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                 markerfacecolor=colors[i], markersize=10,
                                 label=f'{int(row["Number of Clauses"])} klauzul')
                      for i, (_, row) in enumerate(df.iterrows())]
    ax.legend(handles=legend_elements, title='Liczba klauzul', 
             bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'p02_time_vs_memory.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def create_comparison_with_p01(output_dir):
    """Create a comparison plot between P01 (equal clause lengths) and P02 (Poisson distribution)"""
    
    # Load both datasets
    try:
        df_p01 = load_data(Path('problem1/avg.csv'))
        df_p02 = load_data(Path('problem2/avg.csv'))
        
        # Focus on solvers that work well: Snake and select few others
        solvers_to_compare = ['snake', 'vampire', 'cvc5']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Time comparison
        for solver in solvers_to_compare:
            time_col = f'{solver} Time (s)'
            
            # Filter valid data for both problems
            p01_filtered = filter_valid_data(df_p01, solver)
            p02_filtered = filter_valid_data(df_p02, solver)
            
            # Find common clause counts
            common_clauses = set(p01_filtered['Number of Clauses']) & set(p02_filtered['Number of Clauses'])
            common_clauses = sorted(list(common_clauses))
            
            if common_clauses:
                p01_times = []
                p02_times = []
                
                for clauses in common_clauses:
                    p01_time = p01_filtered[p01_filtered['Number of Clauses'] == clauses][time_col].iloc[0]
                    p02_time = p02_filtered[p02_filtered['Number of Clauses'] == clauses][time_col].iloc[0]
                    
                    p01_times.append(p01_time)
                    p02_times.append(p02_time)
                
                ax1.plot(common_clauses, p01_times, 'o-', linewidth=2, markersize=6, 
                        label=f'{solver.capitalize()} - P01 (równe długości)')
                ax1.plot(common_clauses, p02_times, 's--', linewidth=2, markersize=6, 
                        label=f'{solver.capitalize()} - P02 (Poisson)')
        
        ax1.set_xlabel('Liczba klauzul', fontsize=12)
        ax1.set_ylabel('Czas wykonania (s)', fontsize=12)
        ax1.set_title('Porównanie czasu: P01 vs P02', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Memory comparison (excluding InKreSAT)
        for solver in [s for s in solvers_to_compare if s != 'inkresat']:
            memory_col = f'{solver} Memory (MB)'
            
            # Filter valid data for both problems
            p01_filtered = filter_valid_data(df_p01, solver)
            p02_filtered = filter_valid_data(df_p02, solver)
            
            # Find common clause counts
            common_clauses = set(p01_filtered['Number of Clauses']) & set(p02_filtered['Number of Clauses'])
            common_clauses = sorted(list(common_clauses))
            
            if common_clauses:
                p01_memory = []
                p02_memory = []
                
                for clauses in common_clauses:
                    p01_mem = p01_filtered[p01_filtered['Number of Clauses'] == clauses][memory_col].iloc[0]
                    p02_mem = p02_filtered[p02_filtered['Number of Clauses'] == clauses][memory_col].iloc[0]
                    
                    p01_memory.append(p01_mem)
                    p02_memory.append(p02_mem)
                
                ax2.plot(common_clauses, p01_memory, 'o-', linewidth=2, markersize=6, 
                        label=f'{solver.capitalize()} - P01 (równe długości)')
                ax2.plot(common_clauses, p02_memory, 's--', linewidth=2, markersize=6, 
                        label=f'{solver.capitalize()} - P02 (Poisson)')
        
        ax2.set_xlabel('Liczba klauzul', fontsize=12)
        ax2.set_ylabel('Zużycie pamięci (MB)', fontsize=12)
        ax2.set_title('Porównanie pamięci: P01 vs P02', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'p02_comparison_with_p01.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig
        
    except Exception as e:
        print(f"Could not create P01 vs P02 comparison: {e}")
        return None

def create_solver_comparison_table(df, output_dir):
    """Create a summary table comparing solvers across different formula sizes"""
    
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
                'Success Rate': (filtered_df[sat_col].isin([True, False])).sum() / len(df) * 100,
                'Valid Tests': len(filtered_df)
            }
        else:
            solver_stats = {
                'Solver': solver.capitalize(),
                'Avg Time (s)': 'N/A',
                'Max Time (s)': 'N/A', 
                'Avg Memory (MB)': 'N/A',
                'Max Memory (MB)': 'N/A',
                'Success Rate': 0.0,
                'Valid Tests': 0
            }
        summary_data.append(solver_stats)
    
    summary_df = pd.DataFrame(summary_data)
    
    # Create table plot
    fig, ax = plt.subplots(figsize=(14, 6))
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
    
    plt.title('Porównanie wydajności solverów - Problem P02\n(rozkład Poissona długości klauzul)', 
              fontsize=14, fontweight='bold', pad=20)
    plt.savefig(output_dir / 'p02_solver_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig, summary_df

def main():
    """Main function to generate all plots"""
    # Setup
    csv_file = Path('problem2/avg.csv')
    output_dir = Path('problem2_plots')
    output_dir.mkdir(exist_ok=True)
    
    # Load data
    print("Loading data...")
    df = load_data(csv_file)
    print(f"Loaded data for {len(df)} formula sizes: {sorted(df['Number of Clauses'].tolist())}")
    
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
    print("\nGenerating time analysis plots...")
    create_time_plots(df, output_dir)
    
    print("Generating memory analysis plots (excluding InKreSAT)...")
    create_memory_plots(df, output_dir)
    
    print("Generating combined performance plot (excluding InKreSAT)...")
    create_combined_performance_plot(df, output_dir)
    
    print("Generating comparison with Problem P01...")
    create_comparison_with_p01(output_dir)
    
    print("Generating solver comparison table...")
    fig, summary_df = create_solver_comparison_table(df, output_dir)
    
    # Save summary to CSV
    summary_df.to_csv(output_dir / 'p02_solver_summary.csv', index=False)
    
    print(f"\nAll plots saved to: {output_dir}")
    print("Generated files:")
    for file in output_dir.glob('p02_*.png'):
        print(f"  - {file.name}")
    print(f"  - p02_solver_summary.csv")

if __name__ == "__main__":
    main() 