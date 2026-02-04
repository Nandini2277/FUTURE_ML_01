"""
Sales Forecasting Application - GUI Version
Easy-to-use desktop application for sales forecasting
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.ticker import FuncFormatter
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from datetime import datetime, timedelta
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import threading
import os

class SalesForecastApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sales Forecasting System")
        self.root.geometry("1200x800")
        
        # Variables
        self.df = None
        self.model = None
        self.forecast_days = tk.IntVar(value=90)
        self.data_loaded = False
        
        self.create_widgets()

    def format_currency(self, x, pos):
        if x >= 1e6:
            return f'${x/1e6:.1f}M'
        elif x >= 1e3:
            return f'${x/1e3:.0f}K'
        else:
            return f'${x:.0f}'
        
    def create_widgets(self):
        # Title
        title = tk.Label(self.root, text="📊 Sales Forecasting System", 
                        font=("Arial", 24, "bold"), fg="#2E86AB")
        title.pack(pady=20)
        
        # Control Panel
        control_frame = tk.Frame(self.root, bg="#f0f0f0", relief=tk.RAISED, bd=2)
        control_frame.pack(fill=tk.X, padx=20, pady=10)
        
        # Buttons
        btn_frame = tk.Frame(control_frame, bg="#f0f0f0")
        btn_frame.pack(pady=15)
        
        self.load_btn = tk.Button(btn_frame, text="📁 Load CSV Data", 
                                  command=self.load_data, bg="#2E86AB", fg="white",
                                  font=("Arial", 12, "bold"), padx=20, pady=10)
        self.load_btn.grid(row=0, column=0, padx=10)
        
        self.generate_btn = tk.Button(btn_frame, text="🎲 Generate Sample Data", 
                                     command=self.generate_sample_data, bg="#F18F01", fg="white",
                                     font=("Arial", 12, "bold"), padx=20, pady=10)
        self.generate_btn.grid(row=0, column=1, padx=10)
        
        self.train_btn = tk.Button(btn_frame, text="🤖 Train Model", 
                                   command=self.train_model, bg="#C73E1D", fg="white",
                                   font=("Arial", 12, "bold"), padx=20, pady=10,
                                   state=tk.DISABLED)
        self.train_btn.grid(row=0, column=2, padx=10)
        
        self.forecast_btn = tk.Button(btn_frame, text="📈 Generate Forecast", 
                                     command=self.generate_forecast, bg="#6A994E", fg="white",
                                     font=("Arial", 12, "bold"), padx=20, pady=10,
                                     state=tk.DISABLED)
        self.forecast_btn.grid(row=0, column=3, padx=10)
        
        # Forecast days selector
        days_frame = tk.Frame(control_frame, bg="#f0f0f0")
        days_frame.pack(pady=5)
        
        tk.Label(days_frame, text="Forecast Days:", bg="#f0f0f0", 
                font=("Arial", 10)).pack(side=tk.LEFT, padx=5)
        
        days_spinbox = tk.Spinbox(days_frame, from_=30, to=365, 
                                 textvariable=self.forecast_days, width=10,
                                 font=("Arial", 10))
        days_spinbox.pack(side=tk.LEFT)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready. Load data to begin.")
        status_bar = tk.Label(self.root, textvariable=self.status_var, 
                            relief=tk.SUNKEN, anchor=tk.W, bg="#d0d0d0")
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Main content area with tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Tab 1: Data Preview
        self.data_tab = tk.Frame(self.notebook)
        self.notebook.add(self.data_tab, text="📋 Data Preview")
        
        # Tab 2: Visualizations
        self.viz_tab = tk.Frame(self.notebook)
        self.notebook.add(self.viz_tab, text="📊 Visualizations")
        
        # Tab 3: Forecast
        self.forecast_tab = tk.Frame(self.notebook)
        self.notebook.add(self.forecast_tab, text="📈 Forecast Results")
        
        # Tab 4: Report
        self.report_tab = tk.Frame(self.notebook)
        self.notebook.add(self.report_tab, text="📝 Business Report")
        
        # Initialize tabs
        self.init_data_tab()
        self.init_viz_tab()
        self.init_forecast_tab()
        self.init_report_tab()
        
    def init_data_tab(self):
        # Data info
        self.data_info = tk.Text(self.data_tab, height=30, width=100, font=("Courier", 10))
        self.data_info.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        scrollbar = tk.Scrollbar(self.data_tab, command=self.data_info.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.data_info.config(yscrollcommand=scrollbar.set)
        
    def init_viz_tab(self):
        self.viz_canvas_frame = tk.Frame(self.viz_tab)
        self.viz_canvas_frame.pack(fill=tk.BOTH, expand=True)
        
    def init_forecast_tab(self):
        self.forecast_canvas_frame = tk.Frame(self.forecast_tab)
        self.forecast_canvas_frame.pack(fill=tk.BOTH, expand=True)
        
    def init_report_tab(self):
        self.report_text = tk.Text(self.report_tab, height=30, width=100, 
                                  font=("Courier", 10), wrap=tk.WORD)
        self.report_text.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        scrollbar = tk.Scrollbar(self.report_tab, command=self.report_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.report_text.config(yscrollcommand=scrollbar.set)
        
    def load_data(self):
        filename = filedialog.askopenfilename(
            title="Select Sales Data CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
    
        if filename:
            try:
                # Load CSV
                self.df = pd.read_csv(filename)
            
                # Show column selection dialog if needed
                if 'Date' not in self.df.columns or 'Sales' not in self.df.columns:
                    self.select_columns_dialog()
                    if not hasattr(self, 'date_col') or not hasattr(self, 'sales_col'):
                        return  # User cancelled
                else:
                    self.date_col = 'Date'
                    self.sales_col = 'Sales'
            
                # Rename columns to standard names
                self.df = self.df.rename(columns={
                    self.date_col: 'Date',
                    self.sales_col: 'Sales'
                })
            
                # Keep only Date and Sales columns
                self.df = self.df[['Date', 'Sales']]
            
                # Convert Date column
                self.df['Date'] = pd.to_datetime(self.df['Date'], errors='coerce')
            
                # Remove rows with invalid dates or sales
                self.df = self.df.dropna()
            
                # Ensure Sales is numeric
                self.df['Sales'] = pd.to_numeric(self.df['Sales'], errors='coerce')
                self.df = self.df.dropna()
            
                # Sort by date
                self.df = self.df.sort_values('Date').reset_index(drop=True)
            
                if len(self.df) < 100:
                    messagebox.showwarning("Warning", 
                        f"Only {len(self.df)} valid records found. Need at least 100 records for reliable forecasting.")
                    return
            
                self.data_loaded = True
                self.train_btn.config(state=tk.NORMAL)
                self.status_var.set(f"✅ Data loaded: {len(self.df)} records")
            
                # Show data preview
                self.show_data_preview()
            
                messagebox.showinfo("Success", 
                    f"Loaded {len(self.df)} records successfully!\n\n"
                    f"Date range: {self.df['Date'].min().date()} to {self.df['Date'].max().date()}")
            
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load data:\n{str(e)}\n\n"
                    "Please ensure your CSV has 'Date' and 'Sales' columns.")
                self.status_var.set("❌ Error loading data")

    def select_columns_dialog(self):
        """Dialog to let user select which columns to use"""
        dialog = tk.Toplevel(self.root)
        dialog.title("Select Columns")
        dialog.geometry("400x200")
        dialog.transient(self.root)
        dialog.grab_set()
    
        tk.Label(dialog, text="Your CSV doesn't have 'Date' and 'Sales' columns.\n", "Please select which columns to use:", font=("Arial", 11)).pack(pady=15)
    
        # Date column selector
        date_frame = tk.Frame(dialog)
        date_frame.pack(pady=10)
        tk.Label(date_frame, text="Date Column:", font=("Arial", 10)).pack(side=tk.LEFT, padx=5)
        date_var = tk.StringVar(value=self.df.columns[0])
        date_dropdown = ttk.Combobox(date_frame, textvariable=date_var, values=list(self.df.columns), width=20)
        date_dropdown.pack(side=tk.LEFT)
    
        # Sales column selector
        sales_frame = tk.Frame(dialog)
        sales_frame.pack(pady=10)
        tk.Label(sales_frame, text="Sales Column:", font=("Arial", 10)).pack(side=tk.LEFT, padx=5)
        sales_var = tk.StringVar(value=self.df.columns[1] if len(self.df.columns) > 1 else self.df.columns[0])
        sales_dropdown = ttk.Combobox(sales_frame, textvariable=sales_var, values=list(self.df.columns), width=20)
        sales_dropdown.pack(side=tk.LEFT)
    
        def on_ok():
            self.date_col = date_var.get()
            self.sales_col = sales_var.get()
            dialog.destroy()
    
        def on_cancel():
            dialog.destroy()
    
        btn_frame = tk.Frame(dialog)
        btn_frame.pack(pady=15)
        tk.Button(btn_frame, text="OK", command=on_ok, bg="#2E86AB", fg="white",
                 font=("Arial", 10), padx=20).pack(side=tk.LEFT, padx=10)
        tk.Button(btn_frame, text="Cancel", command=on_cancel, bg="#C73E1D", fg="white",
                 font=("Arial", 10), padx=20).pack(side=tk.LEFT, padx=10)
    
        dialog.wait_window()
    
    def generate_sample_data(self):
        try:
            self.status_var.set("Generating sample data...")
            
            # Generate sample data
            dates = pd.date_range(start='2020-01-01', periods=1095, freq='D')
            trend = np.linspace(5000, 12000, 1095)
            seasonal = 2000 * np.sin(2 * np.pi * np.arange(1095) / 365.25)
            noise = np.random.normal(0, 500, 1095)
            sales = np.maximum(trend + seasonal + noise, 1000)
            
            self.df = pd.DataFrame({'Date': dates, 'Sales': sales})
            
            self.data_loaded = True
            self.train_btn.config(state=tk.NORMAL)
            self.status_var.set(f"✅ Sample data generated: {len(self.df)} records")
            
            self.show_data_preview()
            
            messagebox.showinfo("Success", "Generated 3 years of sample sales data!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate data:\n{str(e)}")
            self.status_var.set("❌ Error generating data")
    
    def show_data_preview(self):
        self.data_info.delete(1.0, tk.END)
        
        info = f"""
{'='*80}
DATA PREVIEW
{'='*80}

Dataset Information:
- Records: {len(self.df):,}
- Date Range: {self.df['Date'].min().date()} to {self.df['Date'].max().date()}
- Columns: {', '.join(self.df.columns)}

Sales Statistics:
- Total Sales: ${self.df['Sales'].sum():,.2f}
- Average Daily Sales: ${self.df['Sales'].mean():,.2f}
- Minimum: ${self.df['Sales'].min():,.2f}
- Maximum: ${self.df['Sales'].max():,.2f}
- Std Dev: ${self.df['Sales'].std():,.2f}

First 10 Records:
{'-'*80}
{self.df.head(10).to_string()}

Last 10 Records:
{'-'*80}
{self.df.tail(10).to_string()}
"""
        
        self.data_info.insert(1.0, info)
    
    def train_model(self):
        if not self.data_loaded:
            messagebox.showwarning("Warning", "Please load data first!")
            return
        
        def train_thread():
            try:
                self.status_var.set("🤖 Training model... Please wait...")
                self.train_btn.config(state=tk.DISABLED)
                
                # Feature engineering
                df = self.df.copy()
                df['Month'] = df['Date'].dt.month
                df['DayOfWeek'] = df['Date'].dt.dayofweek
                df['DayOfYear'] = df['Date'].dt.dayofyear
                df['DaysSinceStart'] = (df['Date'] - df['Date'].min()).dt.days
                df['Sales_Lag_7'] = df['Sales'].shift(7)
                df['Sales_Lag_30'] = df['Sales'].shift(30)
                df['Sales_Rolling_7'] = df['Sales'].rolling(7, min_periods=1).mean()
                df['Sales_Rolling_30'] = df['Sales'].rolling(30, min_periods=1).mean()
                
                df = df.dropna().reset_index(drop=True)
                
                # Prepare features
                self.feature_cols = ['Month', 'DayOfWeek', 'DayOfYear', 'DaysSinceStart',
                                    'Sales_Lag_7', 'Sales_Lag_30', 'Sales_Rolling_7', 
                                    'Sales_Rolling_30']
                
                X = df[self.feature_cols].values.astype(np.float64)
                y = df['Sales'].values.astype(np.float64)
                
                # Train/test split
                split_idx = int(len(X) * 0.8)
                X_train, X_test = X[:split_idx], X[split_idx:]
                y_train, y_test = y[:split_idx], y[split_idx:]
                
                # Train model
                self.model = GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
                self.model.fit(X_train, y_train)
                
                # Evaluate
                y_pred = self.model.predict(X_test)
                self.mae = mean_absolute_error(y_test, y_pred)
                self.r2 = r2_score(y_test, y_pred)
                
                # Store for visualization
                self.df_processed = df
                self.X_test = X_test
                self.y_test = y_test
                self.y_pred = y_pred
                
                self.status_var.set(f"✅ Model trained! MAE: ${self.mae:,.2f}, R²: {self.r2:.4f}")
                self.forecast_btn.config(state=tk.NORMAL)
                
                # Create visualizations
                self.root.after(0, self.create_visualizations) 
                
            except Exception as e:
                messagebox.showerror("Error", f"Training failed:\n{str(e)}")
                self.status_var.set("❌ Training failed")
            finally:
                self.train_btn.config(state=tk.NORMAL)
        
        # Run in thread to prevent GUI freezing
        thread = threading.Thread(target=train_thread)
        thread.start()
    
    def create_visualizations(self):
        # Clear previous
        for widget in self.viz_canvas_frame.winfo_children():
            widget.destroy()
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(9, 6))
        fig.suptitle('Sales Analysis Dashboard', fontsize=14, fontweight='bold')
        
        # 1. Historical trend
        # Sales Trend
        axes[0, 0].plot(self.df_processed['Date'], self.df_processed['Sales'], color='#2E86AB', linewidth=1.5, alpha=0.7)
        axes[0, 0].set_title('Sales Trend', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('Date', fontsize=10)
        axes[0, 0].set_ylabel('Sales ($)', fontsize=10)
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].tick_params(axis='x', rotation=45, labelsize=8)  # Smaller labels
        axes[0, 0].xaxis.set_major_locator(plt.MaxNLocator(6))  # Fewer date labels
        
        # 2. Monthly pattern
        monthly = self.df_processed.groupby('Month')['Sales'].mean()
        axes[0, 1].bar(range(1, 13), monthly.values, color='#2E86AB', alpha=0.8)
        axes[0, 1].set_title('Average Sales by Month')
        axes[0, 1].set_xlabel('Month')
        axes[0, 1].set_ylabel('Avg Sales ($)')
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        
        # 3. Actual vs Predicted
        # Actual vs Predicted
        test_dates = self.df_processed.iloc[len(self.df_processed) - len(self.y_test):]['Date']
        axes[1, 0].plot(test_dates, self.y_test, label='Actual', color='#2E86AB', linewidth=2)
        axes[1, 0].plot(test_dates, self.y_pred, label='Predicted', color='#C73E1D', linewidth=2, linestyle='--')
        axes[1, 0].set_title('Actual vs Predicted', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Date', fontsize=10)
        axes[1, 0].set_ylabel('Sales ($)', fontsize=10)
        axes[1, 0].legend(fontsize=9)
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].tick_params(axis='x', rotation=45, labelsize=8)
        axes[1, 0].xaxis.set_major_locator(plt.MaxNLocator(6))  # Fewer date labels
        
        # 4. Error distribution
        errors = self.y_test - self.y_pred
        axes[1, 1].hist(errors, bins=30, color='#2E86AB', alpha=0.7, edgecolor='black')
        axes[1, 1].axvline(0, color='#C73E1D', linestyle='--', linewidth=2)
        axes[1, 1].set_title('Prediction Errors')
        axes[1, 1].set_xlabel('Error ($)')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].grid(True, alpha=0.3, axis='y')

        plt.subplots_adjust(left=0.08, right=0.97, top=0.94, bottom=0.10,hspace=0.35, wspace=0.25)
        
        plt.tight_layout(pad=2.0)  # More padding between plots
        
        # Embed in tkinter
        canvas = FigureCanvasTkAgg(fig, master=self.viz_canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def generate_forecast(self):
        if self.model is None:
            messagebox.showwarning("Warning", "Please train the model first!")
            return
        
        try:
            self.status_var.set("📈 Generating forecast...")
            
            days = self.forecast_days.get()
            last_date = self.df_processed['Date'].max()
            future_dates = pd.date_range(start=last_date + timedelta(days=1), 
                                        periods=days, freq='D')
            
            # Prepare future features
            future_df = pd.DataFrame({'Date': future_dates})
            future_df['Month'] = future_df['Date'].dt.month
            future_df['DayOfWeek'] = future_df['Date'].dt.dayofweek
            future_df['DayOfYear'] = future_df['Date'].dt.dayofyear
            future_df['DaysSinceStart'] = (future_df['Date'] - self.df_processed['Date'].min()).dt.days
            
            # Use recent values for lag features
            recent = self.df_processed['Sales'].tail(30).values
            future_df['Sales_Lag_7'] = recent[-7:].mean()
            future_df['Sales_Lag_30'] = recent.mean()
            future_df['Sales_Rolling_7'] = recent[-7:].mean()
            future_df['Sales_Rolling_30'] = recent.mean()
            
            # Predict
            X_future = future_df[self.feature_cols].values.astype(np.float64)
            predictions = self.model.predict(X_future)
            future_df['Forecast'] = predictions
            
            self.future_df = future_df
            
            # Show forecast
            self.show_forecast()
            
            # Generate report
            self.generate_report()
            
            self.status_var.set(f"✅ Forecast generated for {days} days")
            
            # Save option
            save = messagebox.askyesno("Save Forecast", 
                "Would you like to save the forecast to CSV?")
            if save:
                filename = filedialog.asksaveasfilename(
                    defaultextension=".csv",
                    filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
                )
                if filename:
                    future_df[['Date', 'Forecast']].to_csv(filename, index=False)
                    messagebox.showinfo("Success", f"Forecast saved to {filename}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Forecast failed:\n{str(e)}")
            self.status_var.set("❌ Forecast failed")
    
    def show_forecast(self):
        # Clear previous
        for widget in self.forecast_canvas_frame.winfo_children():
            widget.destroy()
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Historical + Forecast
        historical = self.df_processed.tail(180)
        ax1.plot(historical['Date'], historical['Sales'], 
                label='Historical', color='#2E86AB', linewidth=2)
        ax1.plot(self.future_df['Date'], self.future_df['Forecast'], 
                label='Forecast', color='#C73E1D', linewidth=2, linestyle='--')
        ax1.axvline(self.df_processed['Date'].max(), color='gray', 
                   linestyle=':', linewidth=2, label='Today')
        ax1.fill_between(self.future_df['Date'], 
                        self.future_df['Forecast'] * 0.9,
                        self.future_df['Forecast'] * 1.1,
                        color='#C73E1D', alpha=0.2, label='Confidence ±10%')
        ax1.set_title(f'{self.forecast_days.get()}-Day Sales Forecast', 
                     fontsize=10, fontweight='bold')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Sales ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)
        
        # Monthly summary
        # Monthly summary
        self.future_df['Month'] = self.future_df['Date'].dt.to_period('M')
        monthly = self.future_df.groupby('Month')['Forecast'].sum()
        ax2.bar(range(len(monthly)), monthly.values, color='#2E86AB', alpha=0.8)
        ax2.set_title('Monthly Forecast Summary', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Month', fontsize=10)
        ax2.set_ylabel('Total Sales ($)', fontsize=10)
        ax2.set_xticks(range(len(monthly)))
        ax2.set_xticklabels([str(m) for m in monthly.index], rotation=45, ha='right', fontsize=9)
        ax2.grid(True, alpha=0.3, axis='y')

        # Add value labels - positioned better  
        for i, v in enumerate(monthly.values):
            ax2.text(i, v + v*0.01, f'${v:,.0f}', ha='center', va='bottom', fontsize=9, fontweight='bold')  # Smaller font

        # Make room for labels at top
        ax2.set_ylim(0, max(monthly.values) * 1.15)  # 15% extra space at top

        plt.subplots_adjust(left=0.08, right=0.97, top=0.94, bottom=0.12,hspace=0.30)
        
        plt.tight_layout(pad=2.0)  # More padding between plots
        
        # Embed in tkinter
        canvas = FigureCanvasTkAgg(fig, master=self.forecast_canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def generate_report(self):
        self.report_text.delete(1.0, tk.END)
        
        total_forecast = self.future_df['Forecast'].sum()
        avg_forecast = self.future_df['Forecast'].mean()
        
        report = f"""
{'='*80}
SALES FORECAST REPORT
{'='*80}

Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

MODEL PERFORMANCE
{'-'*80}
Mean Absolute Error: ${self.mae:,.2f}
R² Score: {self.r2:.4f}
Accuracy: {(1 - self.mae/self.df['Sales'].mean())*100:.1f}%

FORECAST SUMMARY
{'-'*80}
Forecast Period: {self.forecast_days.get()} days
Start Date: {self.future_df['Date'].min().date()}
End Date: {self.future_df['Date'].max().date()}

Total Forecasted Revenue: ${total_forecast:,.2f}
Average Daily Sales: ${avg_forecast:,.2f}

MONTHLY BREAKDOWN
{'-'*80}
"""
        
        monthly = self.future_df.groupby('Month')['Forecast'].agg(['sum', 'mean', 'count'])
        for month, row in monthly.iterrows():
            report += f"{month}:\n"
            report += f"  Total: ${row['sum']:,.2f}\n"
            report += f"  Daily Avg: ${row['mean']:,.2f}\n"
            report += f"  Days: {int(row['count'])}\n\n"
        
        report += f"""
BUSINESS RECOMMENDATIONS
{'-'*80}
1. INVENTORY PLANNING
   • Plan procurement 2-3 weeks before projected demand
   • Expected revenue: ${total_forecast:,.2f}
   • Maintain safety stock for ±10% variance

2. STAFFING
   • Average daily sales: ${avg_forecast:,.2f}
   • Plan staffing levels accordingly
   • Consider seasonal patterns

3. CASH FLOW
   • Predicted accuracy: {(1 - self.mae/self.df['Sales'].mean())*100:.1f}%
   • Budget within confidence range (±10%)
   • Monitor actual vs forecast weekly

{'='*80}
END OF REPORT
{'='*80}
"""
        
        self.report_text.insert(1.0, report)

def main():
    root = tk.Tk()
    app = SalesForecastApp(root)
    root.mainloop()

if __name__ == "__main__":

    main()

