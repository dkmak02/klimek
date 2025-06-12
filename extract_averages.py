import csv
import os
import sys
import argparse
from pathlib import Path


def extract_averages_from_csv(csv_file_path):
    """Extract average rows from a single CSV file"""
    averages = []
    
    try:
        with open(csv_file_path, 'r', encoding='utf-8') as file:
            # Use semicolon as delimiter based on the provided format
            reader = csv.reader(file, delimiter=';')
            
            for row in reader:
                if row and row[0].strip().lower() == 'average':
                    averages.append(row)
    
    except Exception as e:
        print(f"Error processing file {csv_file_path}: {e}")
    
    return averages


def process_folder(input_folder, output_file):
    """Process all CSV files in the input folder and extract averages"""
    input_path = Path(input_folder)
    
    if not input_path.exists():
        print(f"Error: Input folder '{input_folder}' does not exist.")
        return False
    
    if not input_path.is_dir():
        print(f"Error: '{input_folder}' is not a directory.")
        return False
    
    # Find all CSV files in the folder
    csv_files = list(input_path.glob("*.csv"))
    
    if not csv_files:
        print(f"No CSV files found in '{input_folder}'")
        return False
    
    print(f"Found {len(csv_files)} CSV files to process...")
    
    all_averages = []
    header_written = False
    
    # Process each CSV file
    for csv_file in csv_files:
        print(f"Processing: {csv_file.name}")
        averages = extract_averages_from_csv(csv_file)
        
        if averages:
            all_averages.extend(averages)
            print(f"  Found {len(averages)} average row(s)")
        else:
            print(f"  No average rows found")
    
    if not all_averages:
        print("No average data found in any of the CSV files.")
        return False
    
    # Write results to output file
    try:
        with open(output_file, 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file, delimiter=';')
            
            # Write header (assuming all files have the same structure)
            # Read the first CSV file to get the header
            first_csv = csv_files[0]
            with open(first_csv, 'r', encoding='utf-8') as header_file:
                header_reader = csv.reader(header_file, delimiter=';')
                header = next(header_reader)
                writer.writerow(header)
            
            # Write all average rows
            for avg_row in all_averages:
                writer.writerow(avg_row)
        
        print(f"\nSuccess! Extracted {len(all_averages)} average rows.")
        print(f"Results saved to: {output_file}")
        return True
        
    except Exception as e:
        print(f"Error writing to output file '{output_file}': {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Extract average data from CSV files in a folder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python extract_averages.py "C:\\Users\\dkmak\\Desktop\\solvers-provers-results\\problem1\\results\\benchmark-test-20250605-194428\\results" averages_output.csv
        """
    )
    
    parser.add_argument(
        'input_folder',
        help='Path to the folder containing CSV files'
    )
    
    parser.add_argument(
        'output_file',
        help='Path to the output CSV file where averages will be saved'
    )
    
    args = parser.parse_args()
    
    print(f"Input folder: {args.input_folder}")
    print(f"Output file: {args.output_file}")
    print("-" * 50)
    
    success = process_folder(args.input_folder, args.output_file)
    
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main() 