#!/usr/bin/env python3
"""
Script to analyze the sampling rate of the magnetic field data
"""

try:
    import pandas as pd
    import numpy as np
    PANDAS_AVAILABLE = True
except ImportError:
    print("❌ Required libraries not found!")
    print("💡 Please install pandas and numpy:")
    print("   pip install pandas numpy")
    print("   or")
    print("   conda install pandas numpy")
    exit(1)

import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading

def analyze_sampling_rate(csv_file_path):
    """Analyze the sampling rate of a CSV file with magnetic field data"""
    
    print(f"Analyzing: {os.path.basename(csv_file_path)}")
    print("=" * 60)
    
    # Read the CSV file
    df = pd.read_csv(csv_file_path)
    
    # Convert timestamps to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Basic info
    print(f"Total records: {len(df):,}")
    print(f"Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    duration = df['timestamp'].max() - df['timestamp'].min()
    print(f"Duration: {duration}")
    print(f"Unique sensors: {df['sensor_id'].nunique()}")
    print(f"Sensor IDs: {sorted(df['sensor_id'].unique())}")
    print()
    
    # Calculate time differences
    time_diffs = df['timestamp'].diff().dropna()
    time_diffs_seconds = time_diffs.dt.total_seconds()
    
    # Calculate statistics
    mean_interval = time_diffs_seconds.mean()
    median_interval = time_diffs_seconds.median()
    std_interval = time_diffs_seconds.std()
    min_interval = time_diffs_seconds.min()
    max_interval = time_diffs_seconds.max()
    
    # Calculate sampling rate
    sampling_rate = 1.0 / mean_interval
    
    print("Time interval statistics:")
    print(f"  Mean interval: {mean_interval:.6f} seconds")
    print(f"  Median interval: {median_interval:.6f} seconds")
    print(f"  Std deviation: {std_interval:.6f} seconds")
    print(f"  Min interval: {min_interval:.6f} seconds")
    print(f"  Max interval: {max_interval:.6f} seconds")
    print()
    
    print("Sampling rate analysis:")
    print(f"  Calculated sampling rate: {sampling_rate:.6f} Hz")
    print(f"  Expected from filename (Downsample_60): {1/60:.6f} Hz")
    print(f"  Difference: {abs(sampling_rate - 1/60):.6f} Hz")
    print()
    
    # Check for irregular sampling
    irregular_count = np.sum(np.abs(time_diffs_seconds - mean_interval) > std_interval * 2)
    irregular_percentage = (irregular_count / len(time_diffs_seconds)) * 100
    
    print("Sampling regularity:")
    print(f"  Irregular intervals (>2σ from mean): {irregular_count} ({irregular_percentage:.2f}%)")
    
    if irregular_percentage < 5:
        print("  ✅ Sampling appears regular")
    elif irregular_percentage < 20:
        print("  ⚠️  Sampling has some irregularities")
    else:
        print("  ❌ Sampling appears highly irregular")
    
    print()
    
    # Analyze by sensor
    print("Sampling rate by sensor:")
    for sensor_id in sorted(df['sensor_id'].unique()):
        sensor_data = df[df['sensor_id'] == sensor_id].sort_values('timestamp')
        if len(sensor_data) > 1:
            sensor_diffs = sensor_data['timestamp'].diff().dropna()
            sensor_diffs_seconds = sensor_diffs.dt.total_seconds()
            sensor_rate = 1.0 / sensor_diffs_seconds.mean()
            print(f"  {sensor_id}: {sensor_rate:.6f} Hz ({len(sensor_data)} records)")
    
    return sampling_rate

class SamplingRateAnalyzerGUI:
    """GUI application for analyzing sampling rate of magnetic field data"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("🔍 Magnetic Field Data Sampling Rate Analyzer")
        self.root.geometry("800x600")
        self.root.configure(bg='#f0f0f0')
        
        # Configure style
        style = ttk.Style()
        style.theme_use('clam')
        
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the user interface"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="🔍 Magnetic Field Data Sampling Rate Analyzer", 
                               font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # Description
        desc_label = ttk.Label(main_frame, 
                               text="Select a CSV file containing magnetic field data to analyze its sampling rate.\n"
                                    "The file should have columns: timestamp, b_x, b_y, b_z, and optionally sensor_id.",
                               font=('Arial', 10))
        desc_label.grid(row=1, column=0, columnspan=3, pady=(0, 20))
        
        # File selection section
        file_frame = ttk.LabelFrame(main_frame, text="📁 File Selection", padding="10")
        file_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 20))
        file_frame.columnconfigure(1, weight=1)
        
        # File path display
        self.file_path_var = tk.StringVar()
        self.file_path_var.set("No file selected")
        
        ttk.Label(file_frame, text="Selected file:").grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        self.file_path_label = ttk.Label(file_frame, textvariable=self.file_path_var, 
                                        foreground='blue', font=('Arial', 9))
        self.file_path_label.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(0, 10))
        
        # Browse button
        self.browse_button = ttk.Button(file_frame, text="📁 Browse for CSV File", 
                                       command=self.browse_file)
        self.browse_button.grid(row=0, column=2, padx=(10, 0))
        
        # Analyze button
        self.analyze_button = ttk.Button(file_frame, text="🔍 Analyze Sampling Rate", 
                                        command=self.analyze_file, state='disabled')
        self.analyze_button.grid(row=1, column=0, columnspan=3, pady=(10, 0))
        
        # Progress bar
        self.progress = ttk.Progressbar(main_frame, mode='indeterminate')
        self.progress.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 20))
        
        # Results section
        results_frame = ttk.LabelFrame(main_frame, text="📊 Analysis Results", padding="10")
        results_frame.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 20))
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(0, weight=1)
        main_frame.rowconfigure(4, weight=1)
        
        # Results text area with scrollbar
        self.results_text = tk.Text(results_frame, height=15, width=80, wrap=tk.WORD, 
                                   font=('Courier', 9), bg='#f8f8f8')
        scrollbar = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, command=self.results_text.yview)
        self.results_text.configure(yscrollcommand=scrollbar.set)
        
        self.results_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        # Control buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=5, column=0, columnspan=3, pady=(10, 0))
        
        self.clear_button = ttk.Button(button_frame, text="🗑️ Clear Results", 
                                      command=self.clear_results)
        self.clear_button.grid(row=0, column=0, padx=(0, 10))
        
        self.export_button = ttk.Button(button_frame, text="💾 Export Results", 
                                       command=self.export_results, state='disabled')
        self.export_button.grid(row=0, column=1, padx=(0, 10))
        
        self.quit_button = ttk.Button(button_frame, text="❌ Exit", command=self.root.quit)
        self.quit_button.grid(row=0, column=2)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready to analyze files")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.grid(row=6, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(10, 0))
        
    def browse_file(self):
        """Open file dialog to select CSV file"""
        file_path = filedialog.askopenfilename(
            title="Select CSV File",
            filetypes=[
                ("CSV files", "*.csv"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            self.file_path_var.set(file_path)
            self.analyze_button.config(state='normal')
            self.status_var.set(f"File selected: {os.path.basename(file_path)}")
            
    def analyze_file(self):
        """Analyze the selected file in a separate thread"""
        file_path = self.file_path_var.get()
        if not file_path or file_path == "No file selected":
            messagebox.showerror("Error", "Please select a file first!")
            return
            
        # Disable buttons during analysis
        self.analyze_button.config(state='disabled')
        self.browse_button.config(state='disabled')
        self.progress.start()
        self.status_var.set("Analyzing file...")
        
        # Run analysis in separate thread to prevent GUI freezing
        thread = threading.Thread(target=self.run_analysis, args=(file_path,))
        thread.daemon = True
        thread.start()
        
    def run_analysis(self, file_path):
        """Run the analysis in a separate thread"""
        try:
            # Clear previous results
            self.results_text.delete(1.0, tk.END)
            
            # Capture the analysis output
            import io
            import sys
            from contextlib import redirect_stdout
            
            # Create a string buffer to capture output
            output_buffer = io.StringIO()
            
            with redirect_stdout(output_buffer):
                sampling_rate = analyze_sampling_rate(file_path)
            
            # Get the captured output
            analysis_output = output_buffer.getvalue()
            
            # Update GUI in main thread
            self.root.after(0, self.display_results, analysis_output, sampling_rate)
            
        except Exception as e:
            error_msg = f"❌ Error during analysis: {str(e)}"
            self.root.after(0, self.display_error, error_msg)
    
    def display_results(self, output, sampling_rate):
        """Display analysis results in the GUI"""
        self.progress.stop()
        self.analyze_button.config(state='normal')
        self.browse_button.config(state='normal')
        self.export_button.config(state='normal')
        
        # Display results
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, output)
        self.results_text.insert(tk.END, f"\n\n🎯 SUMMARY:\n")
        self.results_text.insert(tk.END, f"📈 Sampling Rate: {sampling_rate:.6f} Hz\n")
        self.results_text.insert(tk.END, f"📊 Nyquist Frequency: {sampling_rate/2:.6f} Hz\n")
        
        self.status_var.set(f"Analysis complete - Sampling rate: {sampling_rate:.6f} Hz")
        
    def display_error(self, error_msg):
        """Display error message in the GUI"""
        self.progress.stop()
        self.analyze_button.config(state='normal')
        self.browse_button.config(state='normal')
        
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, error_msg)
        self.status_var.set("Analysis failed")
        
        messagebox.showerror("Analysis Error", error_msg)
    
    def clear_results(self):
        """Clear the results text area"""
        self.results_text.delete(1.0, tk.END)
        self.export_button.config(state='disabled')
        self.status_var.set("Results cleared")
    
    def export_results(self):
        """Export results to a text file"""
        if not self.results_text.get(1.0, tk.END).strip():
            messagebox.showwarning("Warning", "No results to export!")
            return
            
        file_path = filedialog.asksaveasfilename(
            title="Save Results",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                with open(file_path, 'w') as f:
                    f.write(self.results_text.get(1.0, tk.END))
                messagebox.showinfo("Success", f"Results exported to {file_path}")
                self.status_var.set(f"Results exported to {os.path.basename(file_path)}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export results: {str(e)}")
    
    def run(self):
        """Start the GUI application"""
        self.root.mainloop()

def main():
    """Main function to run the GUI sampling rate analyzer"""
    try:
        # Create and run the GUI application
        app = SamplingRateAnalyzerGUI()
        app.run()
    except Exception as e:
        print(f"❌ Error starting GUI application: {e}")
        print("💡 Make sure tkinter is available on your system")
        print("   On some systems, you may need to install python3-tk")

if __name__ == "__main__":
    main()
