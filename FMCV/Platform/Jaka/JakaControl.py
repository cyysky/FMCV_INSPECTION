#from tkinter import *
import time
import tkinter as tk
from tkinter import ttk
import threading
import math

def init(in_start):
    global start
    start = in_start
    
    if start.config.CONTROL.tcp_type.casefold() == "jaka":
        start.sub("platform/jaka/show", new_window)

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
        operation_results_window = JakaControlWindow(start, window_opened_root)

def refresh():
    global window_opened_root
    global operation_results_window
    try:
        operation_results_window.main_frame.refresh()
    except:
        pass
        #traceback.print_exc()
        #print("operation_results_window didnt opened")
        
 
class JakaControlWindow:
    def __init__(self, start, root):
        self.start = start
        self.root = root
        #root.geometry("1024x768-150+150")  
        self.main_frame = JakaControlFrame(start, root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        self.root.attributes('-topmost',True)
        self.root.title("JAKA Cobot Jogging Interface")
        self.main_frame.update_status()


class JakaControlFrame(ttk.Frame):

    def __init__(self,start, parent, *args, **kwargs):
        #https://stackoverflow.com/questions/17466561/best-way-to-structure-a-tkinter-application/17470842
        ttk.Frame.__init__(self, parent, *args, **kwargs)
        self.parent = parent
        self.start = start

        # Initialize the connection with the robot
        self.robot = self.start.Com.tcp_jaka
        ret = self.robot.get_sdk_version()
        print("SDK version is:", ret[1])
        #ret = robot.get_controller_ip()
        #print("Available robot ip:", ret[1])

        # Create the Tkinter GUI
        #root = tk.Tk()
        #root.title("JAKA Cobot Jogging Interface")

        # Global variables thread control,  track jogging state, jog speed, and step sizes
        self.jog_lock = threading.Lock()
        self.jog_event = threading.Event()
        self.jogging = False
        self.jog_speed  = tk.DoubleVar()
        self.linear_step  = tk.DoubleVar()
        self.degree_step = tk.DoubleVar()

        self.drag_mode = tk.BooleanVar()
        self.collision_status  = tk.StringVar()
        self.current_position  = tk.StringVar()
        self.current_joint =  tk.StringVar()
        
        # Create the main frame
        mainframe = self

        # Create the speed selection combo box
        self.speeds = [1, 10, 20, 50]
        self.jog_speed.set(self.speeds[1])  # Set the default speed
        speed_combobox = ttk.Combobox(mainframe, textvariable=self.jog_speed, values=self.speeds, state='readonly', width=5)
        speed_combobox.grid(column=0, row=0, padx=(20, 0), pady=(5, 0))


        ttk.Label(mainframe, text="<Speed|Step>").grid(column=1, row=0, padx=(20, 0), pady=(5, 0))


        # Create the step selection combo boxes
        self.linear_steps = [0.02, 0.1, 1, 10, 50 , 100]
        self.linear_step.set(self.linear_steps[2])  # Set the default linear step
        linear_step_combobox = ttk.Combobox(mainframe, textvariable=self.linear_step, values=self.linear_steps, state='readonly', width=5)
        linear_step_combobox.grid(column=2, row=0, padx=(20, 0), pady=(5, 0))


        # Create the labels and buttons for each axis
        j = 0
        for i, axis in enumerate(["X", "Y", "Z"],1):
            ttk.Label(mainframe, text=axis).grid(column=0, row=i, padx=(10, 0), pady=(5, 0))

            button = ttk.Button(mainframe, text="+")
            button.grid(column=1, row=i, padx=(0, 0), pady=(0, 0),ipady=10, ipadx=10)
            button.bind('<ButtonPress-1>', lambda event, a=j: self.start_jogging(event, a, 1))
            button.bind('<ButtonRelease-1>', self.stop_jogging)

            button = ttk.Button(mainframe, text="-")
            button.grid(column=2, row=i, padx=(0, 0), pady=(0, 0),ipady=10, ipadx=10)
            button.bind('<ButtonPress-1>', lambda event, a=j: self.start_jogging(event, a, -1))
            button.bind('<ButtonRelease-1>', self.stop_jogging)
            j+=1


        # Create the step selection combo boxes
        self.degree_steps = [0.01, 0.1, 1, 10]
        self.degree_step.set(self.degree_steps[1])  # Set the default linear step
        ttk.Label(mainframe, text="Step degree").grid(column=1, row=4, padx=(20, 0), pady=(5, 0))
        radius_step_combobox = ttk.Combobox(mainframe, textvariable=self.degree_step, values=self.degree_steps, state='readonly', width=5)
        radius_step_combobox.grid(column=2, row=4, padx=(20, 0), pady=(5, 0))

        r = 3
        for i, axis in enumerate(["Roll (RX)", "Pitch (RY)", "Yaw (RZ)"],5):
            ttk.Label(mainframe, text=axis).grid(column=0, row=i, padx=(10, 0), pady=(5, 0))

            button = ttk.Button(mainframe, text="+")
            button.grid(column=1, row=i, padx=(0, 0), pady=(0, 0),ipady=10, ipadx=10)
            button.bind('<ButtonPress-1>', lambda event, a=r: self.start_jogging(event, a, 1))
            button.bind('<ButtonRelease-1>', self.stop_jogging)

            button = ttk.Button(mainframe, text="-")
            button.grid(column=2, row=i, padx=(0, 0), pady=(0, 0),ipady=10, ipadx=10)
            button.bind('<ButtonPress-1>', lambda event, a=r: self.start_jogging(event, a, -1))
            button.bind('<ButtonRelease-1>', self.stop_jogging)
            r+=1

        # Combobox to enable/disable the jogging variable
        ttk.Label(mainframe, text="Jogging").grid(column=1, row=8, padx=(20, 0), pady=(5, 0))
        self.jogging_var = tk.BooleanVar()
        self.jogging_var.set(True)
        jogging_checkbox = ttk.Checkbutton(mainframe, variable=self.jogging_var, command=lambda: print(self.jogging_var.get()))
        jogging_checkbox.grid(column=2, row=8, padx=(20, 0), pady=(5, 0))
        
        button = ttk.Button(mainframe, text="Homing", command=self.robot.power_off)
        button.grid(column=0, row=9, padx=(20, 0), pady=(5, 0))
        button.bind('<ButtonPress-1>', self.start_homing)
        button.bind('<ButtonRelease-1>', self.stop_jogging)
        
        ttk.Button(mainframe, text="World", command=self.set_world_coordinate).grid(column=1, row=9, padx=(20, 0), pady=(5, 0))
        ttk.Button(mainframe, text="FMCV", command=self.set_FMCV_coordinate).grid(column=2, row=9, padx=(20, 0), pady=(5, 0))
        
        # Create the status labels
        ttk.Label(mainframe, textvariable=self.collision_status).grid(column=3, row=0, padx=(20, 0), pady=(5, 0))
        ttk.Label(mainframe, textvariable=self.current_position).grid(column=3, row=1, padx=(20, 0), pady=(5, 0))
        ttk.Label(mainframe, textvariable=self.current_joint).grid(column=3, row=2, padx=(20, 0), pady=(5, 0))

        # Create the robot control buttons
        ttk.Button(mainframe, text="Enable Robot", command=self.robot.enable_robot).grid(column=3, row=3, padx=(20, 0), pady=(5, 0))
        ttk.Button(mainframe, text="Disable Robot", command=self.robot.disable_robot).grid(column=3, row=4, padx=(20, 0), pady=(5, 0))
        ttk.Button(mainframe, text="Power On", command=self.robot.power_on).grid(column=3, row=5, padx=(20, 0), pady=(5, 0))
        ttk.Button(mainframe, text="Power Off", command=self.robot.power_off).grid(column=3, row=6, padx=(20, 0), pady=(5, 0))
        ttk.Button(mainframe, text="Teach Mode", command=self.toggle_drag_mode).grid(column=3, row=7, padx=(20, 0), pady=(5, 0))
        ttk.Button(mainframe, text="Recover Collision", command=self.recover_collision).grid(column=3, row=8, padx=(20, 0), pady=(5, 0))



        button = ttk.Button(mainframe, text="Zero Pose", command=self.robot.power_off)
        button.grid(column=3, row=9, padx=(20, 0), pady=(5, 0))
        button.bind('<ButtonPress-1>', self.start_reset_pos)
        button.bind('<ButtonRelease-1>', self.stop_jogging)
        
        button = ttk.Button(mainframe, text="STOP", command=self.robot.power_off)
        button.grid(column=0, columnspan = 4 ,sticky='nesw' ,row=10, padx=(0, 0), pady=(0, 0))
        button.bind('<ButtonPress-1>', lambda a:self.robot.motion_abort())
        button.bind('<ButtonRelease-1>', lambda a:self.robot.motion_abort())
        button.rowconfigure(10, minsize=100)

        # Create a dictionary to store the names and indices of the specific status items
        status_dict = {
            1: "Errcode",
            2: "Inpos",
            3: "Powered On",
            4: "Enabled",
            7: "Teaching Status",
            8: "Soft Limit",
            9: "User ID",
            10: "Tool ID",
            23: "Socket Connect",
            24: "Emergency Stop",
        }

        # Create labels to display the robot status
        self.status_labels = status_labels ={}
        for i, (index, name) in enumerate(status_dict.items()):
            label = ttk.Label(mainframe, text=f"{name}:")
            label.grid(column=5, row=i, padx=(20, 0), pady=(5, 0))
            status_labels[index] = ttk.Label(mainframe, text="")
            status_labels[index].grid(column=6, row=i, padx=(20, 0), pady=(5, 0))
    
    # Set world coordinate and end flage cooordinate system
    def set_world_coordinate(self):
        self.robot.set_user_frame_id(0)
        self.robot.set_tool_id(0)
        
    # Set world coordinate and end flage cooordinate system
    def set_FMCV_coordinate(self):
        self.robot.set_user_frame_id(10)
        self.robot.set_tool_id(10)
        
    # Enable or disable drag mode
    def toggle_drag_mode(self):
        self.drag_mode.set(not self.drag_mode.get())
        self.robot.drag_mode_enable(self.drag_mode.get())

    # Recover from collision
    def recover_collision(self):
        self.robot.collision_recover()

    # Jogging function
    def jog(self, axis, direction):
        self.jogging = True
        self.jog_event.set()
        if axis < 3:
            step = self.linear_step.get()
        else:
            step = self.degree_step.get() * math.pi/180 #degree to radius
        
        if not self.jogging_var.get():
            if self.robot.is_in_pos()[1] :
                ret = self.robot.get_tcp_position()
                if ret[0] == 0:
                    current_position_list = ret[1]
                    current_position_list[axis] += step * direction
                    self.robot.linear_move_extend(end_pos=current_position_list, move_mode=0, is_block=False, speed=self.jog_speed.get(), acc=self.jog_speed.get()*10, tol=0)
                    while self.jogging:
                        time.sleep(0.01)
        else:
            while self.jogging:
                if self.robot.is_in_pos()[1] :
                    ret = self.robot.get_tcp_position()
                    if ret[0] == 0:
                        current_position_list = ret[1]
                        current_position_list[axis] += step * direction
                        self.robot.linear_move_extend(end_pos=current_position_list, move_mode=0, is_block=False, speed=self.jog_speed.get(), acc=self.jog_speed.get()*10, tol=0)
                        print("instructed")
                    else:
                        print("Error occurred, error code:", ret[0])
                        break
        print("stopping")
        self.robot.motion_abort()
        self.jog_event.clear()
        print("done")
        
    # Jogging function
    def reset_pose(self):
        self.jogging = True
        
        while self.jogging:
            self.robot.joint_move(joint_pos=[0,0,0,0,0,0],move_mode= 0, is_block=False, speed=self.jog_speed.get() * 3.14/180 )
        self.robot.motion_abort()

    def start_reset_pos(self,event):
        if not self.jog_event.is_set():
            if not self.jogging:
                jog_thread = threading.Thread(target=self.reset_pose, args=())
                jog_thread.start()
            
    # Jogging function
    def user_home(self):
        self.jogging = True
        
        while self.jogging:
            self.robot.linear_move_extend(end_pos=[0,0,0,0,0,0],move_mode= 0, is_block=False, speed=self.jog_speed.get(), acc=self.jog_speed.get()*10, tol=0)
        self.robot.motion_abort()

    def start_homing(self,event):
        if not self.jog_event.is_set():
            if not self.jogging:
                jog_thread = threading.Thread(target=self.user_home, args=())
                jog_thread.start()

    def start_jogging(self,vevent, axis, direction):
        if not self.jog_event.is_set():
            if not self.jogging:
                print("start")
                jog_thread = threading.Thread(target=self.jog, args=(axis, direction))
                jog_thread.start()

    def stop_jogging(self,event):
        self.jogging = False
        self.robot.motion_abort()
        #self.jog_event.clear()
        print("instruct stop")
        
    # Update collision status and current position
    def update_status(self):
        ret = self.robot.get_robot_status()
        if ret[0] == 0:
            status = ret[1]
            for index, label in self.status_labels.items():
                value = status[index - 1]
                label.config(text=f"{value}")
        else:
            print("Error occurred, error code:", ret[0])
            
        self.collision_status.set("In Collision" if status[6 - 1] else "No Collision")
        
        tcp = list(status[19 - 1])
        tcp[3] = status[19 - 1][3]*180/math.pi
        tcp[4] = status[19 - 1][4]*180/math.pi
        tcp[5] = status[19 - 1][5]*180/math.pi
        
        self.current_position.set(f"Position: {[round(x, 3) for x in tcp]}")

        self.current_joint.set(f"Joint: {[round(x, 3) for x in status[20 - 1]]}")

        self.parent.after(1000, self.update_status)

    