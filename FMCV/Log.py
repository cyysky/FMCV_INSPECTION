import os
from datetime import datetime,timedelta
import cv2
from pathlib import Path
import sqlite3
import pandas as pd
import tkinter as tk
from tkinter import filedialog

from FMCV.Logger import default_style
from FMCV import Logging

def init(s):
    global self
    self = s

def add_column_if_not_exists(conn, table_name, column_name, column_type):
    cursor = conn.cursor()
    
    # Get the table information
    cursor.execute(f"PRAGMA table_info({table_name})")
    table_info = cursor.fetchall()

    # Check if the column exists
    column_exists = any(info[1] == column_name for info in table_info)

    # Add the column if it does not exist
    if not column_exists:
        cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}")
        conn.commit()
        Logging.debug(f"Column '{column_name}' added to table '{table_name}'.")
    else:
        pass
        #print(f"Column '{column_name}' already exists in table '{table_name}'.")

def write_excel():
    global self
    db_path = Path(self.Config.results_path,'log.db')

    # Create database connection
    conn = sqlite3.connect(db_path)

    # Add a column named 'XYZ' with TEXT data type to the 'your_table' table if it does not exist
    table_name = "RESULTS_LOG"
    column_name = "PROFILE"
    column_type = "TEXT"
    add_column_if_not_exists(conn, table_name, column_name, column_type)

    # Read the data from the RESULTS_LOG table into a DataFrame
    df = pd.read_sql_query("SELECT * FROM RESULTS_LOG where ROI_PASS=0", conn)

    # Close the connection
    conn.close()

    # Process the data
    df['#'] = df['ID']
    df['Date'] = pd.to_datetime(df['LOG_TIMESTAMP']).dt.strftime('%Y%m%d')
    df['Log_file'] = pd.to_datetime(df['LOG_TIMESTAMP']).dt.strftime('%Y%m%d_%H%M%S')
    df['False call location'] = None
    df['QA'] = None

    # Group data by log_timestamp and check if any ROI_PASS is 0
    def process_group(group):
        group_filtered = df.loc[group.index]
        steps = '\n '.join([f"Step {row['STEP']}, ROI: {row['ROI_NAME']}" for _, row in group_filtered[group_filtered['ROI_PASS'] == 0].iterrows()])
        return steps

    # Group data by log_timestamp and check if any ROI_PASS is 0
    df_grouped = df.groupby('LOG_TIMESTAMP').agg({
        '#': 'first',
        'Date': 'first',
        'PROFILE':'first',
        'BARCODE': 'first',
        'Log_file': 'first',
        'ROI_PASS': lambda x: False if any(v == 0 for v in x) else True,
        'STEP': process_group,
        'False call location':'first',
        'QA' :'first'
    }).reset_index().rename(columns={'ROI_PASS': 'Results', 'STEP': 'Results_list'})

    #df_grouped.rename(columns={'row_number': 'new_column_name'}, inplace=True)

    # Select the required columns
    df = df_grouped[['#', 'Date', 'PROFILE','BARCODE', 'Log_file', 'Results', 'Results_list','False call location','QA']]


    # Generate the current timestamp and format it as yymmdd_HHmmss
    timestamp = datetime.now().strftime('%y%m%d_%H%M%S')

    # Prompt a dialog box for selecting a folder and filename to save the Excel file
    #root = tk.Tk()
    root = self.MainUi.top
    #root.withdraw()
    file_path = filedialog.asksaveasfilename(title="Select folder and filename to save the Excel file",
                                         initialfile=f"fail_summary_{timestamp}.xlsx",
                                         defaultextension=".xlsx",
                                         filetypes=[("Excel files", "*.xlsx")])


    if file_path:
        # Export the DataFrame to an Excel file in the selected folder with the chosen filename
        df.to_excel(file_path, index=False)



def write_log():
    global barcode, results_path, images_path, log_datetime, results, result_frame
    
    global current_datetime, log_datetime_iso8601
    
    global mes_path

    global self
    s = self
    
    Logging.info("Writing_log")
    
    try:
        
        barcode = s.Main.barcode
        results_path = s.Config.results_path
        images_path = s.Config.images_path
        mes_path = s.Config.mes_path
        #log_datetime = datetime.utcnow() + timedelta(hours=+8)
        
        current_datetime = datetime.now()
        log_datetime = current_datetime.strftime("%Y%m%d_%H%M%S") #https://www.w3schools.com/python/python_datetime.asp
        # Get the current date-time in the ISO 8601 format
        log_datetime_iso8601 = current_datetime.strftime('%Y-%m-%d %H:%M:%S')
        results = s.Main.results
        result_frame = s.Main.result_frame
        
        
        if str(results_path) != '.':
            os.makedirs(results_path.parents[0], exist_ok=True)
            db_path = Path(results_path,'log.db')
            results_path.mkdir(parents=True, exist_ok=True)
            try:
                # Create database connection
                conn = sqlite3.connect(db_path)
                c = conn.cursor()
                
                # Create users table
                c.execute('''CREATE TABLE IF NOT EXISTS "RESULTS_LOG" (
                                "ID"	INTEGER NOT NULL,
                                "LOG_TIMESTAMP"	TEXT NOT NULL,
                                "PROFILE"	TEXT NOT NULL,
                                "BARCODE"	TEXT,
                                "CAM"	INTEGER NOT NULL,
                                "STEP"	INTEGER NOT NULL,
                                "ROI_NAME"	TEXT NOT NULL,
                                "ROI_TYPE"	TEXT NOT NULL,
                                "ROI_PASS"	INTEGER NOT NULL,
                                "result_class"	TEXT,
                                "result_score"	NUMERIC,
                                "CODE"	TEXT,
                                "score"	NUMERIC,
                                "search_score"	NUMERIC,
                                "offset_x"	INTEGER,
                                "offset_y"	INTEGER,
                                "offset_x_mm"	NUMERIC,
                                "offset_y_mm"	NUMERIC,
                                "angle"	NUMERIC,
                                "OCR"	TEXT,
                                "3d_x"	NUMERIC,
                                "3d_y"	NUMERIC,
                                "3d_z"	NUMERIC,
                                "3d_rx"	NUMERIC,
                                "3d_ry"	NUMERIC,
                                "3d_rz"	NUMERIC
                            );''')
                conn.commit()
                
                
                # Add a column named 'PROFILE' with TEXT data type to the 'RESULTS_LOG' table if it does not exist
                table_name = "RESULTS_LOG"
                column_name = "PROFILE"
                column_type = "TEXT"
                add_column_if_not_exists(conn, table_name, column_name, column_type)
                
                # Add a column named 'PROFILE' with TEXT data type to the 'RESULTS_LOG' table if it does not exist
                table_name = "RESULTS_LOG"
                column_name = "angle"
                column_type = "NUMERIC"
                add_column_if_not_exists(conn, table_name, column_name, column_type)
                
                # Write the SQL SELECT statement to get the largest ID
                select_sql = 'SELECT MAX(ID) FROM RESULTS_LOG'

                # Execute the SELECT statement using the cursor and fetch the result
                c.execute(select_sql)
                try:
                    largest_id = c.fetchone()[0]
                    if largest_id is None:
                        largest_id = 0
                    # Print the largest ID
                    Logging.info("Largest ID:", largest_id)
                except:
                    Logging.log_traceback()
                    largest_id = 0 
                largest_id = largest_id + 1
                for src_n, src in enumerate(results):     
                    for step_n, step in enumerate(results[src_n]):
                        for roi_n, roi_result in enumerate(results[src_n][step_n]):
                            # Define the data to be inserted
                            data = (
                                largest_id, log_datetime_iso8601, s.Profile.name, barcode, src_n, step_n, roi_result["name"], roi_result["type"], roi_result.get("PASS"), roi_result.get("result_class"),
                                roi_result.get("result_score"), roi_result.get("CODE"), roi_result.get("score"), roi_result.get("search_score"),roi_result.get("offset_x"), roi_result.get("offset_y"),
                                roi_result.get("offset_x_mm"), roi_result.get("offset_y_mm"),roi_result.get("angle"),
                                roi_result.get("OCR"), roi_result.get("3d_x"),roi_result.get("3d_y"), roi_result.get("3d_z"), 
                                roi_result.get("3d_rx"), roi_result.get("3d_ry"), roi_result.get("3d_rz")
                            )
                            Logging.debug(data)
                            # Write the SQL INSERT statement with placeholders for the values
                            insert_sql = '''
                            INSERT INTO RESULTS_LOG (
                                ID, LOG_TIMESTAMP, PROFILE, BARCODE, CAM, STEP, ROI_NAME, ROI_TYPE, ROI_PASS, result_class,
                                result_score, CODE, score, search_score, offset_x, offset_y, offset_x_mm, offset_y_mm, angle,
                                OCR, "3d_x", "3d_y", "3d_z", "3d_rx", "3d_ry", "3d_rz"
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            '''

                            # Execute the INSERT statement using the cursor and commit the changes
                            c.execute(insert_sql, data)
                conn.commit()
                conn.close()
            except:
                Logging.log_traceback()
        
        
        if s.Config.log_type == "NORMAL":            
            default_style.init(s)
            default_style.write_log()
            
        if s.Config.log_type == "FLEX":
            flex_style.init(s)
            flex_style.write_log()
            
        if s.Config.log_type == "VS":
            vs_style.init(s)
            vs_style.write_log()
            
        if s.Config.log_type == "KAIFA":
            kaifa_style.init(s)
            kaifa_style.write_log()
            
        if s.Config.log_type == "PLEXUS":
            plexus_style.init(s)
            plexus_style.write_log()

        self.Config.write_total()
    except:
        Logging.log_traceback()

