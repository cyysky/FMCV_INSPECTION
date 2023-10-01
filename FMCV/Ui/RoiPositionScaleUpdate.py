from tkinter import *
from tkinter import ttk
from tkinter import filedialog
from tkinter import messagebox
from tkinter.messagebox import askokcancel, showinfo, WARNING


def init(in_start):
    global start
    start = in_start
    #if in_start.Config.log_type in ( "PLEXUS","KAIFA"):
    start.sub("tool/RoiPositionScaleUpdate/open_dialog", new_window)
    
#https://pythonprogramming.altervista.org/tkinter-open-a-new-window-and-just-one/
def new_window():
    global window_opened_root
    global operation_results_window
    try:
        if window_opened_root.state() == "normal": window_opened_root.focus()
    except:
        #traceback.print_exc()
        start.log.info("Creating New OperationResultsWindow")
        window_opened_root = tk.Toplevel()
        operation_results_window = OperationResultsWindow(start, window_opened_root)

class OperationResultsWindow:
    def __init__(self, start, root):
        self.start = start
        self.root = root
        root.geometry("1024x768-150+150")  
        self.main_frame = OperationResultsFrame(start, root)
        self.main_frame.pack(fill=BOTH, expand=True)
        self.root.attributes('-topmost',True)

class OperationResultsFrame(ttk.Frame):

    def __init__(self,start, parent, *args, **kwargs):
        #https://stackoverflow.com/questions/17466561/best-way-to-structure-a-tkinter-application/17470842
        ttk.Frame.__init__(self, parent, *args, **kwargs)
        self.parent = parent
        self.start = start
        self.current_roi = None

    def validate_input(input_str):
        try:
            int(input_str)
            return True
        except ValueError:
            return False

    def submit_values():
        if all(validate_input(entry.get()) for entry in entries):
            integers = [int(entry.get()) for entry in entries]
            messagebox.showinfo("Success", f"Values entered: {', '.join(map(str, integers))}")
            for entry in entries:
                entry.delete(0, tk.END)
        else:
            messagebox.showerror("Error", "Please enter valid integer values only.")

