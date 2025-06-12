import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import glob
import re
import math
import argparse

def extract_problem_info(filename):
    # Extract information from filename pattern: problem{num}_c{clauses}_a{atoms}_prec{prec}_lengths_{length}
    match = re.search(r'problem(\d+)_c(\d+)_a(\d+)_prec(\d+)_lengths_(\d+)', filename)
    if match:
        problem_num = int(match.group(1))
        clauses = int(match.group(2))
        atoms = int(match.group(3))
        prec = int(match.group(4))
        length = int(match.group(5))
        return {
            'Problem Number': problem_num,
            'Clauses': clauses,
            'Atoms': atoms,
            'Precision': prec,
            'Clause Length': length
        }
    return None

def load_and_combine_csvs(input_dir):
    all_data = []
    for filename in os.listdir(input_dir):
        if filename.endswith('.csv'):
            filepath = os.path.join(input_dir, filename)
            info = extract_problem_info(filename)
            
            # Read CSV with semicolon separator
            df = pd.read_csv(filepath, sep=';')
            
            # Convert comma decimal separator to dot for numeric columns
            numeric_columns = df.columns[df.columns.str.contains('Memory|Time')]
            for col in numeric_columns:
                if df[col].dtype == 'object':
                    df[col] = df[col].str.replace(',', '.').astype(float)
                else:
                    df[col] = df[col].astype(float)
            
            if info is not None:
                for key, value in info.items():
                    df[key] = value
            all_data.append(df)
    
    if not all_data:
        raise ValueError(f"No CSV files found in {input_dir}")
    
    return pd.concat(all_data, ignore_index=True)

def get_prover_colors(provers):
    """Create a consistent color mapping for provers"""
    # Create a colormap for different provers
    colors = plt.cm.tab10(np.linspace(0, 1, len(provers)))
    return dict(zip(provers, colors))

def plot_prover_timings(df, output_dir, prover_colors, fixed_clause_length=False):
    # Filter out 'Average' rows and get only timing columns
    timing_cols = [col for col in df.columns if 'Time' in col and 'SAT' not in col and 'Timeout' not in col]
    provers = [col.split()[0] for col in timing_cols]
    
    # Create a new dataframe with just the timing data
    timing_data = df[df['Run Number'] != 'Average'][timing_cols].copy()
    timing_data.columns = provers
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    
    if fixed_clause_length and 'Clause Length' in df.columns:
        # Get unique clause lengths
        clause_lengths = sorted(df['Clause Length'].unique())
        all_data = []
        x_ticks = []
        x_labels = []
        current_offset = 0
        
        # Process each clause length
        for clause_len in clause_lengths:
            # Get data for this clause length
            mask = df['Clause Length'] == clause_len
            clause_df = timing_data[mask].copy()
            
            if not clause_df.empty:
                # Add offset to the index to create separation
                if current_offset > 0:
                    clause_df.index = clause_df.index + current_offset
                
                all_data.append(clause_df)
                x_ticks.append(clause_df.index[len(clause_df)//2])  # Middle of this clause length's data
                x_labels.append(f'Length {clause_len}')
                
                # Update offset for next clause length
                current_offset = clause_df.index[-1] + 50
        
        # Combine all data
        if all_data:
            timing_data = pd.concat(all_data)
            
            # Create boxplot with custom colors
            boxplot = plt.boxplot([timing_data[prover] for prover in provers],
                                tick_labels=[prover.capitalize() for prover in provers],
                                patch_artist=True)
            
            # Set x-axis ticks and labels
            plt.xticks(x_ticks, x_labels)
            
            # Add vertical lines to separate clause lengths
            for tick in x_ticks:
                plt.axvline(x=tick, color='gray', linestyle=':', alpha=0.3)
    else:
        # Create boxplot with custom colors
        boxplot = plt.boxplot([timing_data[prover] for prover in provers],
                            tick_labels=[prover.capitalize() for prover in provers],
                            patch_artist=True)
    
    # Set colors for boxes
    for patch, prover in zip(boxplot['boxes'], provers):
        patch.set_facecolor(prover_colors[prover])
    
    plt.title('Prover Timings Comparison')
    plt.ylabel('Time (seconds)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'combined_prover_timings.png'))
    plt.close()

def plot_prover_timings_log(df, output_dir, prover_colors, fixed_clause_length=False):
    # Tylko rekordy Average
    df = df[df['Run Number'] == 'Average']
    timing_cols = [col for col in df.columns if 'Time' in col and 'SAT' not in col and 'Timeout' not in col]
    provers = [col.split()[0] for col in timing_cols]
    timing_data = df[timing_cols].copy()
    timing_data.columns = provers
    plt.figure(figsize=(12, 6))
    
    if fixed_clause_length and 'Clause Length' in df.columns:
        # Get unique clause lengths
        clause_lengths = sorted(df['Clause Length'].unique())
        all_data = []
        x_ticks = []
        x_labels = []
        current_offset = 0
        
        # Process each clause length
        for clause_len in clause_lengths:
            # Get data for this clause length
            mask = df['Clause Length'] == clause_len
            clause_df = timing_data[mask].copy()
            
            if not clause_df.empty:
                # Add offset to the index to create separation
                if current_offset > 0:
                    clause_df.index = clause_df.index + current_offset
                
                all_data.append(clause_df)
                x_ticks.append(clause_df.index[len(clause_df)//2])  # Middle of this clause length's data
                x_labels.append(f'Length {clause_len}')
                
                # Update offset for next clause length
                current_offset = clause_df.index[-1] + 50
        
        # Combine all data
        if all_data:
            timing_data = pd.concat(all_data)
            
            # Create boxplot with custom colors
            boxplot = plt.boxplot([timing_data[prover] for prover in provers],
                                tick_labels=[prover.capitalize() for prover in provers],
                                patch_artist=True)
            
            # Set x-axis ticks and labels
            plt.xticks(x_ticks, x_labels)
            
            # Add vertical lines to separate clause lengths
            for tick in x_ticks:
                plt.axvline(x=tick, color='gray', linestyle=':', alpha=0.3)
    else:
        # Create boxplot with custom colors
        boxplot = plt.boxplot([timing_data[prover] for prover in provers],
                            tick_labels=[prover.capitalize() for prover in provers],
                            patch_artist=True)
    
    # Set colors for boxes
    for patch, prover in zip(boxplot['boxes'], provers):
        patch.set_facecolor(prover_colors[prover])
    
    plt.yscale('log')
    plt.title('Prover Timings Comparison (Log Scale)')
    plt.ylabel('Time (seconds) - Log Scale')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'combined_prover_timings_log.png'))
    plt.close()

def plot_memory_usage(df, output_dir, prover_colors, fixed_clause_length=False):
    # Tylko rekordy Average
    df = df[df['Run Number'] == 'Average']
    # Get memory columns for all provers (excluding inkresat)
    memory_cols = [col for col in df.columns if 'Memory' in col and 'inkresat' not in col]
    provers = [col.split()[0] for col in memory_cols]
    memory_data = df[memory_cols].copy()
    memory_data.columns = provers
    plt.figure(figsize=(12, 6))
    
    if fixed_clause_length and 'Clause Length' in df.columns:
        # Get unique clause lengths
        clause_lengths = sorted(df['Clause Length'].unique())
        all_data = []
        x_ticks = []
        x_labels = []
        current_offset = 0
        
        # Process each clause length
        for clause_len in clause_lengths:
            # Get data for this clause length
            mask = df['Clause Length'] == clause_len
            clause_df = memory_data[mask].copy()
            
            if not clause_df.empty:
                # Add offset to the index to create separation
                if current_offset > 0:
                    clause_df.index = clause_df.index + current_offset
                
                all_data.append(clause_df)
                x_ticks.append(clause_df.index[len(clause_df)//2])  # Middle of this clause length's data
                x_labels.append(f'Length {clause_len}')
                
                # Update offset for next clause length
                current_offset = clause_df.index[-1] + 50
        
        # Combine all data
        if all_data:
            memory_data = pd.concat(all_data)
            
            # Create boxplot with custom colors
            boxplot = plt.boxplot([memory_data[prover] for prover in provers],
                                tick_labels=[prover.capitalize() for prover in provers],
                                patch_artist=True)
            
            # Set x-axis ticks and labels
            plt.xticks(x_ticks, x_labels)
            
            # Add vertical lines to separate clause lengths
            for tick in x_ticks:
                plt.axvline(x=tick, color='gray', linestyle=':', alpha=0.3)
    else:
        # Create boxplot with custom colors
        boxplot = plt.boxplot([memory_data[prover] for prover in provers],
                            tick_labels=[prover.capitalize() for prover in provers],
                            patch_artist=True)
    
    # Set colors for boxes
    for patch, prover in zip(boxplot['boxes'], provers):
        patch.set_facecolor(prover_colors[prover])
    
    plt.title('Memory Usage Comparison')
    plt.ylabel('Memory (MB)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'combined_memory_usage.png'))
    plt.close()

def plot_combined_time_all_provers(df, output_dir, prover_colors, fixed_clause_length=False):
    # Get only Average rows
    df = df[df['Run Number'] == 'Average']
    
    # Get timing columns for all provers
    timing_cols = [col for col in df.columns if 'Time' in col and 'SAT' not in col and 'Timeout' not in col]
    provers = [col.split()[0] for col in timing_cols]
    
    # Sort by number of clauses
    df = df.sort_values('Number of Clauses')
    
    plt.figure(figsize=(12, 6))
    
    if fixed_clause_length and 'Clause Length' in df.columns:
        # Get unique clause lengths
        clause_lengths = sorted(df['Clause Length'].unique())
        all_x = []
        all_y = []
        x_ticks = []
        x_labels = []
        
        for prover in provers:
            prover_x = []
            prover_y = []
            
            # Process each clause length
            for clause_len in clause_lengths:
                clause_df = df[df['Clause Length'] == clause_len]
                subdf = clause_df[clause_df[f'{prover} SAT'] != 'ERROR'].sort_values('Number of Clauses')
                x = subdf['Number of Clauses']
                y = subdf[f'{prover} Time (s)']
                
                # Offset x values to create separation between clause lengths
                if prover_x:  # If not the first clause length
                    x = x + max(prover_x) + 50  # Add offset
                
                prover_x.extend(x)
                prover_y.extend(y)
                
                # Add tick marks and labels (only for the first prover)
                if prover == provers[0]:
                    x_ticks.append(x.iloc[len(x)//2])  # Middle of this clause length's data
                    x_labels.append(f'Length {clause_len}')
            
            all_x.append(prover_x)
            all_y.append(prover_y)
        
        # Plot each prover's data
        for i, prover in enumerate(provers):
            plt.plot(all_x[i], all_y[i], color=prover_colors[prover], 
                    linewidth=2, label=prover.capitalize())
        
        # Set x-axis ticks and labels
        plt.xticks(x_ticks, x_labels)
        
        # Add vertical lines to separate clause lengths
        for tick in x_ticks:
            plt.axvline(x=tick, color='gray', linestyle=':', alpha=0.3)
    else:
        for prover in provers:
            timing_col = f'{prover} Time (s)'
            sat_col = f'{prover} SAT'
            # Only plot data points where this specific prover doesn't have an error
            valid_data = df[df[sat_col] != 'ERROR'].copy()
            plt.plot(valid_data['Number of Clauses'], valid_data[timing_col], 
                    color=prover_colors[prover], 
                    linewidth=2, 
                    label=prover.capitalize())
    
    plt.xlabel('Number of clauses')
    plt.ylabel('Time (seconds)')
    plt.title('Combined Prover Timings')
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'combined_time_all_provers.png'))
    plt.close()

def plot_combined_time_all_provers_log(df, output_dir, prover_colors, fixed_clause_length=False):
    # Get only Average rows
    df = df[df['Run Number'] == 'Average']
    
    # Get timing columns for all provers
    timing_cols = [col for col in df.columns if 'Time' in col and 'SAT' not in col and 'Timeout' not in col]
    provers = [col.split()[0] for col in timing_cols]
    
    # Sort by number of clauses
    df = df.sort_values('Number of Clauses')
    
    plt.figure(figsize=(12, 6))
    
    if fixed_clause_length and 'Clause Length' in df.columns:
        # Get unique clause lengths
        clause_lengths = sorted(df['Clause Length'].unique())
        all_x = []
        all_y = []
        x_ticks = []
        x_labels = []
        
        for prover in provers:
            prover_x = []
            prover_y = []
            
            # Process each clause length
            for clause_len in clause_lengths:
                clause_df = df[df['Clause Length'] == clause_len]
                subdf = clause_df[clause_df[f'{prover} SAT'] != 'ERROR'].sort_values('Number of Clauses')
                x = subdf['Number of Clauses']
                y = subdf[f'{prover} Time (s)']
                
                # Replace zero values with a small positive number for log scale
                y = y.replace(0, 0.001)
                
                # Offset x values to create separation between clause lengths
                if prover_x:  # If not the first clause length
                    x = x + max(prover_x) + 50  # Add offset
                
                prover_x.extend(x)
                prover_y.extend(y)
                
                # Add tick marks and labels (only for the first prover)
                if prover == provers[0]:
                    x_ticks.append(x.iloc[len(x)//2])  # Middle of this clause length's data
                    x_labels.append(f'Length {clause_len}')
            
            all_x.append(prover_x)
            all_y.append(prover_y)
        
        # Plot each prover's data
        for i, prover in enumerate(provers):
            plt.plot(all_x[i], all_y[i], color=prover_colors[prover], 
                    linewidth=2, label=prover.capitalize())
        
        # Set x-axis ticks and labels
        plt.xticks(x_ticks, x_labels)
        
        # Add vertical lines to separate clause lengths
        for tick in x_ticks:
            plt.axvline(x=tick, color='gray', linestyle=':', alpha=0.3)
    else:
        for prover in provers:
            timing_col = f'{prover} Time (s)'
            sat_col = f'{prover} SAT'
            # Only plot data points where this specific prover doesn't have an error
            valid_data = df[df[sat_col] != 'ERROR'].copy()
            # Replace zero values with a small positive number for log scale
            valid_data[timing_col] = valid_data[timing_col].replace(0, 0.001)
            plt.plot(valid_data['Number of Clauses'], valid_data[timing_col], 
                    color=prover_colors[prover], 
                    linewidth=2, 
                    label=prover.capitalize())
    
    plt.xlabel('Number of clauses')
    plt.ylabel('Time (seconds)')
    plt.title('Combined Prover Timings (Log Scale)')
    plt.grid(True)
    plt.yscale('log')  # Set y-axis to logarithmic scale
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'combined_time_all_provers_log.png'))
    plt.close()

def plot_combined_memory_all_provers(df, output_dir, prover_colors, fixed_clause_length=False):
    # Get only Average rows
    df = df[df['Run Number'] == 'Average']
    
    # Get memory columns for all provers (excluding inkresat)
    memory_cols = [col for col in df.columns if 'Memory' in col and 'inkresat' not in col]
    provers = [col.split()[0] for col in memory_cols]
    
    # Sort by number of clauses
    df = df.sort_values('Number of Clauses')
    
    plt.figure(figsize=(12, 6))
    
    if fixed_clause_length and 'Clause Length' in df.columns:
        # Get unique clause lengths
        clause_lengths = sorted(df['Clause Length'].unique())
        all_x = []
        all_y = []
        x_ticks = []
        x_labels = []
        
        for prover in provers:
            prover_x = []
            prover_y = []
            
            # Process each clause length
            for clause_len in clause_lengths:
                clause_df = df[df['Clause Length'] == clause_len]
                subdf = clause_df[clause_df[f'{prover} SAT'] != 'ERROR'].sort_values('Number of Clauses')
                x = subdf['Number of Clauses']
                y = subdf[f'{prover} Memory (MB)']
                
                # Offset x values to create separation between clause lengths
                if prover_x:  # If not the first clause length
                    x = x + max(prover_x) + 50  # Add offset
                
                prover_x.extend(x)
                prover_y.extend(y)
                
                # Add tick marks and labels (only for the first prover)
                if prover == provers[0]:
                    x_ticks.append(x.iloc[len(x)//2])  # Middle of this clause length's data
                    x_labels.append(f'Length {clause_len}')
            
            all_x.append(prover_x)
            all_y.append(prover_y)
        
        # Plot each prover's data
        for i, prover in enumerate(provers):
            plt.plot(all_x[i], all_y[i], color=prover_colors[prover], 
                    linewidth=2, label=prover.capitalize())
        
        # Set x-axis ticks and labels
        plt.xticks(x_ticks, x_labels)
        
        # Add vertical lines to separate clause lengths
        for tick in x_ticks:
            plt.axvline(x=tick, color='gray', linestyle=':', alpha=0.3)
    else:
        for prover in provers:
            memory_col = f'{prover} Memory (MB)'
            sat_col = f'{prover} SAT'
            # Only plot data points where this specific prover doesn't have an error
            valid_data = df[df[sat_col] != 'ERROR'].copy()
            plt.plot(valid_data['Number of Clauses'], valid_data[memory_col], 
                    color=prover_colors[prover], 
                    linewidth=2, 
                    label=prover.capitalize())
    
    plt.xlabel('Number of clauses')
    plt.ylabel('Memory (MB)')
    plt.title('Combined Prover Memory Usage')
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'combined_memory_all_provers.png'))
    plt.close()

def plot_combined_memory_all_provers_log(df, output_dir, prover_colors, fixed_clause_length=False):
    # Get only Average rows
    df = df[df['Run Number'] == 'Average']
    
    # Get memory columns for all provers (excluding inkresat)
    memory_cols = [col for col in df.columns if 'Memory' in col and 'inkresat' not in col]
    provers = [col.split()[0] for col in memory_cols]
    
    # Sort by number of clauses
    df = df.sort_values('Number of Clauses')
    
    plt.figure(figsize=(12, 6))
    
    if fixed_clause_length and 'Clause Length' in df.columns:
        # Get unique clause lengths
        clause_lengths = sorted(df['Clause Length'].unique())
        all_x = []
        all_y = []
        x_ticks = []
        x_labels = []
        
        for prover in provers:
            prover_x = []
            prover_y = []
            
            # Process each clause length
            for clause_len in clause_lengths:
                clause_df = df[df['Clause Length'] == clause_len]
                subdf = clause_df[clause_df[f'{prover} SAT'] != 'ERROR'].sort_values('Number of Clauses')
                x = subdf['Number of Clauses']
                y = subdf[f'{prover} Memory (MB)']
                
                # Replace zero values with a small positive number for log scale
                y = y.replace(0, 0.001)
                
                # Offset x values to create separation between clause lengths
                if prover_x:  # If not the first clause length
                    x = x + max(prover_x) + 50  # Add offset
                
                prover_x.extend(x)
                prover_y.extend(y)
                
                # Add tick marks and labels (only for the first prover)
                if prover == provers[0]:
                    x_ticks.append(x.iloc[len(x)//2])  # Middle of this clause length's data
                    x_labels.append(f'Length {clause_len}')
            
            all_x.append(prover_x)
            all_y.append(prover_y)
        
        # Plot each prover's data
        for i, prover in enumerate(provers):
            plt.plot(all_x[i], all_y[i], color=prover_colors[prover], 
                    linewidth=2, label=prover.capitalize())
        
        # Set x-axis ticks and labels
        plt.xticks(x_ticks, x_labels)
        
        # Add vertical lines to separate clause lengths
        for tick in x_ticks:
            plt.axvline(x=tick, color='gray', linestyle=':', alpha=0.3)
    else:
        for prover in provers:
            memory_col = f'{prover} Memory (MB)'
            sat_col = f'{prover} SAT'
            # Only plot data points where this specific prover doesn't have an error
            valid_data = df[df[sat_col] != 'ERROR'].copy()
            # Replace zero values with a small positive number for log scale
            valid_data[memory_col] = valid_data[memory_col].replace(0, 0.001)
            plt.plot(valid_data['Number of Clauses'], valid_data[memory_col], 
                    color=prover_colors[prover], 
                    linewidth=2, 
                    label=prover.capitalize())
    
    plt.xlabel('Number of clauses')
    plt.ylabel('Memory (MB)')
    plt.title('Combined Prover Memory Usage (Log Scale)')
    plt.grid(True)
    plt.yscale('log')  # Set y-axis to logarithmic scale
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'combined_memory_all_provers_log.png'))
    plt.close()

def plot_timing_atoms_size(df, output_dir, prover_name, global_atoms_min=None, global_atoms_max=None):
    # Filter only rows where this specific prover doesn't have an error
    df = df[df[f'{prover_name} SAT'] != 'ERROR']
    # Wykres: X=liczba klauzul, Y=czas, rozmiar punktu=liczba atomów, linia łącząca punkty
    df = df[df['Run Number'] == 'Average']
    df = df.sort_values('Clauses')
    x = df['Clauses']
    y = df[f'{prover_name} Time (s)']
    atoms = df['Atoms']
    # Normalizacja rozmiaru punktu względem globalnego min/max
    min_size = 50
    max_size = 600
    atoms_min = atoms.min() if global_atoms_min is None else global_atoms_min
    atoms_max = atoms.max() if global_atoms_max is None else global_atoms_max
    if atoms_max > atoms_min:
        sizes = min_size + (atoms - atoms_min) / (atoms_max - atoms_min) * (max_size - min_size)
    else:
        sizes = np.full_like(atoms, (min_size + max_size) / 2)
    plt.figure(figsize=(10, 6))
    # Linia łącząca punkty
    plt.plot(x, y, color='royalblue', linewidth=2, alpha=0.7, zorder=1)
    # Punkty z rozmiarem zależnym od liczby atomów
    scatter = plt.scatter(x, y, s=sizes, c='royalblue', alpha=0.7, edgecolor='k', zorder=2)
    plt.xlabel('Number of clauses')
    plt.ylabel('Time (s)')
    plt.title(f'{prover_name.capitalize()} timing (point size = number of atoms)')
    plt.grid(True)
    # Przykładowa legenda rozmiarów
    for val in np.linspace(atoms_min, atoms_max, 3, dtype=int):
        plt.scatter([], [], s=min_size + (val - atoms_min) / (atoms_max - atoms_min) * (max_size - min_size),
                    c='royalblue', alpha=0.7, edgecolor='k', label=f'Atoms: {val}')
    plt.legend(scatterpoints=1, frameon=True, labelspacing=1, title='Number of atoms', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{prover_name}_timing_atoms_size.png'))
    plt.close()

def plot_dual_axis_timing(df, output_dir, prover_name):
    import matplotlib.ticker as ticker
    # Przygotuj dane
    df = df[df['Run Number'] == 'Average']
    df = df.sort_values('Clauses')
    x = df['Clauses']
    atoms = df['Atoms']
    timing = df[f'{prover_name} Time (s)']

    fig, ax1 = plt.subplots(figsize=(8, 5))
    # Słupki: liczba atomów
    ax1.bar(x, atoms, color='lightgrey', label='Number of atoms')
    ax1.set_xlabel('Number of clauses')
    ax1.set_ylabel('Number of atoms', color='grey')
    ax1.tick_params(axis='y', labelcolor='grey')

    # Druga oś Y: timing
    ax2 = ax1.twinx()
    ax2.plot(x, timing, color='purple', linewidth=3, label=f'{prover_name.capitalize()} timing')
    ax2.set_ylabel('Time (s)', color='purple')
    ax2.tick_params(axis='y', labelcolor='purple')
    ax2.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.2f}'))

    # Tytuł i legenda
    plt.title(f'{prover_name.capitalize()} timing vs atoms/clauses')
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f'{prover_name}_timing_dual_axis.png'))
    plt.close()

def plot_individual_prover_timing(df, prover_name, output_dir):
    # Get timing column for specific prover
    timing_col = f'{prover_name} Time (s)'
    
    # Filter only 'Average' rows
    df = df[df['Run Number'] == 'Average']
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    
    # Get unique number of atoms and sort them
    unique_atoms = sorted(df['Atoms'].unique())
    
    # Create a colormap for different numbers of atoms
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_atoms)))
    
    # Plot boxplot for each number of atoms
    for atoms, color in zip(unique_atoms, colors):
        atoms_data = df[df['Atoms'] == atoms]
        
        # Prepare data for boxplot
        boxplot_data = []
        boxplot_positions = []
        max_values = []
        
        for clauses in sorted(atoms_data['Clauses'].unique()):
            clause_data = atoms_data[atoms_data['Clauses'] == clauses][timing_col]
            if not clause_data.empty:
                boxplot_data.append(clause_data)
                boxplot_positions.append(clauses)
                max_values.append(clause_data.max())
        
        if boxplot_data:
            # Plot boxplot
            bp = plt.boxplot(boxplot_data, positions=boxplot_positions,
                           widths=min(np.diff(boxplot_positions))/2 if len(boxplot_positions) > 1 else 0.5,
                           patch_artist=True)
            
            # Set boxplot colors
            for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
                plt.setp(bp[element], color=color)
            for patch in bp['boxes']:
                patch.set(facecolor=color, alpha=0.3)
            
            # Plot line connecting maximum values
            plt.plot(boxplot_positions, max_values, color=color, linestyle='--', 
                    linewidth=2, label=f'Atoms: {atoms}')
    
    plt.title(f'{prover_name} Timing by Number of Clauses')
    plt.xlabel('Number of Clauses')
    plt.ylabel('Time (seconds)')
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'combined_{prover_name.lower()}_timing.png'))
    plt.close()

def plot_all_timing_atoms_size(df, output_dir, provers, global_atoms_min=None, global_atoms_max=None, prover_colors=None, fixed_clause_length=False):
    # Filter only Average rows
    df = df[df['Run Number'] == 'Average']
    
    # Create a figure with 2x3 subplots
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes = axes.flatten()
    
    # Get unique clause lengths
    clause_lengths = sorted(df['Clause Length'].unique())
    
    # Process each clause length
    for idx, clause_len in enumerate(clause_lengths):
        if idx >= len(axes):  # Skip if we run out of subplots
            break
            
        clause_df = df[df['Clause Length'] == clause_len].copy()
        
        # Create scatter plot for atoms
        scatter = axes[idx].scatter(clause_df['Number of Clauses'], 
                                  clause_df['Number of Atoms'],
                                  c='gray', alpha=0.3, s=30, label='Number of Atoms')
        
        # Plot timing data for each prover
        for prover in provers:
            timing_col = f'{prover} Time (s)'
            sat_col = f'{prover} SAT'
            # Only plot data points where this specific prover doesn't have an error
            valid_data = clause_df[clause_df[sat_col] != 'ERROR'].copy()
            # Replace zero values with a small positive number for log scale
            valid_data[timing_col] = valid_data[timing_col].replace(0, 0.001)
            axes[idx].plot(valid_data['Number of Clauses'], valid_data[timing_col],
                         color=prover_colors[prover],
                         linewidth=2,
                         label=prover.capitalize())
        
        # Set y-axis to logarithmic scale
        axes[idx].set_yscale('log')
        
        # Set labels and title
        axes[idx].set_xlabel('Length')
        axes[idx].set_ylabel('Time (seconds) / Number of Atoms')
        axes[idx].set_title(f'Clause Length {clause_len}')
        
        # Add grid
        axes[idx].grid(True, which="both", ls="-", alpha=0.2)
        
        # Add legend only to the first subplot
        if idx == 0:
            # Create a single legend for both scatter and lines
            lines = axes[idx].get_lines()
            handles = [scatter] + lines
            labels = ['Number of Atoms'] + [line.get_label() for line in lines]
            axes[idx].legend(handles=handles, labels=labels, 
                           bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Remove any unused subplots
    for idx in range(len(clause_lengths), len(axes)):
        fig.delaxes(axes[idx])
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(os.path.join(output_dir, 'all_provers_timing_clause_lengths.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Plot results from CSV files')
    parser.add_argument('input_dir', help='Input directory containing CSV files')
    parser.add_argument('output_dir', help='Output directory for plots')
    parser.add_argument('--fixed-clause-length', action='store_true', 
                       help='Use fixed clause length plot layout (single row)')
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Load and combine all CSV files
    combined_df = load_and_combine_csvs(args.input_dir)
    
    # Get all provers from timing columns
    timing_cols = [col for col in combined_df.columns if 'Time' in col and 'SAT' not in col and 'Timeout' not in col]
    all_provers = [col.split()[0] for col in timing_cols]
    
    # Create a single color mapping for all provers
    prover_colors = get_prover_colors(all_provers)
    
    # Get memory provers (excluding inkresat)
    memory_cols = [col for col in combined_df.columns if 'Memory' in col and 'inkresat' not in col]
    memory_provers = [col.split()[0] for col in memory_cols]
    
    # Get global min and max atoms for consistent sizing
    global_atoms_min = combined_df['Atoms'].min()
    global_atoms_max = combined_df['Atoms'].max()

    # Generate plots
    plot_prover_timings(combined_df, args.output_dir, prover_colors, args.fixed_clause_length)
    plot_prover_timings_log(combined_df, args.output_dir, prover_colors, args.fixed_clause_length)
    plot_memory_usage(combined_df, args.output_dir, prover_colors, args.fixed_clause_length)
    plot_combined_time_all_provers(combined_df, args.output_dir, prover_colors, args.fixed_clause_length)
    plot_combined_time_all_provers_log(combined_df, args.output_dir, prover_colors, args.fixed_clause_length)
    plot_combined_memory_all_provers(combined_df, args.output_dir, prover_colors, args.fixed_clause_length)
    plot_combined_memory_all_provers_log(combined_df, args.output_dir, prover_colors, args.fixed_clause_length)
    plot_all_timing_atoms_size(combined_df, args.output_dir, all_provers, global_atoms_min, global_atoms_max, prover_colors, args.fixed_clause_length)

if __name__ == '__main__':
    main() 