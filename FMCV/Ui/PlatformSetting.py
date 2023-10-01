import tkinter as tk
from tkinter import ttk
from FMCV.Ui.ScrollableFrm  import ScrollableFrame
from FMCV.Platform.Platform import Platform
from FMCV.Ui.ImageViewFrm import ImageView
#from FMCV.Ui import MainUi
from FMCV import Profile
import json
import traceback


class PlatformSettingFrame(tk.Toplevel):
    is_platform_connected = False
    is_platform_enable = False

    def __init__(self, start, master=None):
        super().__init__(master=master)
        self.start = start
        self.title("Platform Setting")
        self.geometry("640x600")
        #self.resizable(False, False)
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.default_x = 350
        self.default_y = 0
        self.default_z = 0
        self.default_roll = 0
        
        self.actual_x = self.default_x
        self.actual_y = self.default_y
        self.actual_z = self.default_z
        self.actual_roll = self.default_roll

        self.is_closed = False
        self.platform = None

        # get the platform instance from "start" module
        if (start.MovingPlatform is not None):

            self.platform = start.MovingPlatform
            self.is_platform_connected = self.platform.is_connected
            self.original_feedback = self.platform.feedback_callback
            self.platform.feedback_callback = self.feedback_handler
            #print(start.MovingPlatform.feedback_callback)
            #print(dir(start.MovingPlatform.feedback_callback))

        self.create_widget(start, self)
        #self.transient(master)
        self.grab_set()
        #master.wait_window(self)

        # load profile step
        self.load_step()

    def create_widget(self, start, parent_frame):
        parent_frame.columnconfigure(0, weight=1)
        parent_frame.columnconfigure(1, weight=1)
        parent_frame.columnconfigure(2, weight=1)
        parent_frame.columnconfigure(3, weight=1)
        parent_frame.columnconfigure(4, weight=1)
        parent_frame.columnconfigure(5, weight=1)
        parent_frame.columnconfigure(6, weight=1)
        parent_frame.columnconfigure(7, weight=1)

        parent_frame.rowconfigure(0, weight=1)
        parent_frame.rowconfigure(1, weight=1)
        parent_frame.rowconfigure(2, weight=1)
        parent_frame.rowconfigure(3, weight=1)
        parent_frame.rowconfigure(4, weight=1)
        parent_frame.rowconfigure(5, weight=1)
        parent_frame.rowconfigure(6, weight=1)
        parent_frame.rowconfigure(7, weight=1)
        parent_frame.rowconfigure(8, weight=1)

        # Combobox to select from available platform
        tk.Label(parent_frame, text="Model:", bg="#FFFFFF").grid(row=0, column=0, padx=5, pady=5, sticky=tk.E+tk.W)

        # Select platform model
        self.selected_model = tk.StringVar()
        model_combobox = ttk.Combobox(parent_frame, textvariable=self.selected_model, state="readonly")
        available_model_list = Platform.available_models(["Dobot"])
        model_combobox['values'] = available_model_list
        model_combobox.bind('<<ComboboxSelected>>', self.model_selected)
        model_combobox.grid(row=0, column=1, columnspan=3, padx=5, pady=5, sticky=tk.E + tk.W)
        if(len(available_model_list) > 0):
            #print(f"set platform model to {available_model_list[0]}")
            self.selected_model.set(available_model_list[0])

        # ipaddress
        tk.Label(parent_frame, text="IP Address:", bg="#FFFFFF").grid(row=0, column=4, padx=5, pady=5, sticky=tk.E+tk.W)

        self.ipaddress = tk.StringVar()
        self.ipaddress.set("192.168.1.6")
        self.ip_entry = tk.Entry(parent_frame, textvariable=self.ipaddress)
        self.ip_entry.grid(row=0, column=5, columnspan=2, padx=5, pady=5, sticky=tk.E+tk.W)
        #self.update_button = tk.Button(parent_frame, text="Update", command=self.update_config)
        #self.update_button.grid(row=0, column=7, padx=5, pady=5, sticky=tk.E+tk.W)

        # Status and platform control
        status_label_frame = tk.LabelFrame(parent_frame, text="Mode and Status")
        status_label_frame.grid(row=1, rowspan=3, column=0, columnspan=8, padx=5, pady=5, sticky=tk.E + tk.W)
        status_label_frame.columnconfigure(0, weight=1)
        status_label_frame.columnconfigure(1, weight=1)
        status_label_frame.columnconfigure(2, weight=6)

        # Enable/disable button
        self.platform_mode = tk.StringVar()
        self.platform_mode_label = tk.Label(status_label_frame, textvariable=self.platform_mode)
        self.platform_mode_label.grid(row=0, column=0, padx=5, pady=5, sticky=tk.E + tk.W)
        self.enable_button = tk.Button(status_label_frame, text="Enable Platform", command=self.toggle_platform_enable)
        self.enable_button.grid(row=0, column=1, padx=5, pady=5, sticky=tk.E+tk.W)

        # Clear Error button
        self.operating_error = tk.StringVar()
        self.error_label = tk.Label(status_label_frame, textvariable=self.operating_error)
        self.error_label.grid(row=1, column=0, padx=5, pady=5, sticky=tk.E + tk.W)
        self.clear_error_button = tk.Button(status_label_frame, text="Clear Error", command=self.clear_error)
        self.clear_error_button.grid(row=1, column=1, padx=5, pady=5, sticky=tk.E+tk.W)

        # Reset Platform
        self.reset_platform_button = tk.Button(status_label_frame, text="Reset Platform", command=self.reset_platform)
        self.reset_platform_button.grid(row=2, column=1, padx=5, pady=5, sticky=tk.E+tk.W)

        # status Platform
        self.status_text = tk.Text(status_label_frame, width=10, height=4)
        self.status_text.grid(row=0, rowspan=3, column=2, padx=5, pady=5, sticky='news')

        # Status and platform control
        master_offset_frame = tk.LabelFrame(parent_frame, text="Master Offset")
        master_offset_frame.grid(row=4, column=0, columnspan=8, padx=5, pady=5, sticky='news')
        master_offset_frame.columnconfigure(0, weight=1)
        master_offset_frame.columnconfigure(1, weight=1)
        master_offset_frame.columnconfigure(2, weight=1)
        master_offset_frame.columnconfigure(3, weight=1)
        master_offset_frame.columnconfigure(4, weight=1)

        # Camera view
        # TODO: camera view
        #self.camera_view_canvas = tk.Canvas(master_offset_frame, width=100, height=300, scrollregion=(0, 0, 320, 240))
        #self.camera_view_canvas.grid(row=0, column=0, sticky='news')

        # master offset value
        tk.Label(master_offset_frame, text="Offset x, y, z, roll:").grid(row=0, column=0, padx=5, pady=5, sticky='ew')
        self.offset_x = tk.StringVar()
        self.offset_y = tk.StringVar()
        self.offset_z = tk.StringVar()
        self.offset_roll = tk.StringVar()
        offset_x_entry = tk.Entry(master_offset_frame, textvariable=self.offset_x, width=6, justify='center')
        offset_y_entry = tk.Entry(master_offset_frame, textvariable=self.offset_y, width=6, justify='center')
        offset_z_entry = tk.Entry(master_offset_frame, textvariable=self.offset_z, width=6, justify='center')
        offset_roll_entry = tk.Entry(master_offset_frame, textvariable=self.offset_roll, width=6, justify='center')
        offset_x_entry.grid(row=0, column=1, padx=5, pady=5, sticky='ew')
        offset_y_entry.grid(row=0, column=2, padx=5, pady=5, sticky='ew')
        offset_z_entry.grid(row=0, column=3, padx=5, pady=5, sticky='ew')
        offset_roll_entry.grid(row=0, column=4, padx=5, pady=5, sticky='ew')
        self.offset_x.set(0)
        self.offset_y.set(0)
        self.offset_z.set(0)
        self.offset_roll.set(0)

        # Profile Setting Frame
        profile_step_frame = tk.LabelFrame(parent_frame, text="Profile Setting")
        profile_step_frame.grid(row=5, rowspan=3, column=0, columnspan=8, padx=5, pady=5, sticky=tk.E + tk.W)
        profile_step_frame.columnconfigure(0, weight=1)
        profile_step_frame.columnconfigure(1, weight=7)

        # Profile selection listbox
        self.profile_listbox = tk.Listbox(profile_step_frame)
        self.profile_listbox.configure(exportselection=False)   # this prevent listbox lost selection when mouse click on other place
        self.profile_listbox.bind('<<ListboxSelect>>', self.select_step)
        self.profile_listbox.grid(row=0, column=0, padx=5, pady=5, sticky='news')


        # Profile Edit Frame
        profile_edit_frame = tk.Frame(profile_step_frame)
        profile_edit_frame.grid(row=0, column=1, padx=5, pady=5, sticky=tk.E+tk.W)
        profile_edit_frame.columnconfigure(0, weight=1)
        profile_edit_frame.columnconfigure(1, weight=1)
        profile_edit_frame.columnconfigure(2, weight=1)
        profile_edit_frame.columnconfigure(3, weight=1)

        profile_edit_frame.rowconfigure(0, weight=1)
        profile_edit_frame.rowconfigure(1, weight=1)
        profile_edit_frame.rowconfigure(2, weight=1)
        profile_edit_frame.rowconfigure(3, weight=1)
        profile_edit_frame.rowconfigure(4, weight=1)
        profile_edit_frame.rowconfigure(5, weight=1)
        profile_edit_frame.rowconfigure(6, weight=1)
        profile_edit_frame.rowconfigure(7, weight=1)

        # Cartesian coordinate header
        tk.Label(profile_edit_frame, text="x", bg="#FFFFFF").grid(row=0, column=0, padx=5, pady=5, sticky=tk.E + tk.W)
        tk.Label(profile_edit_frame, text="y", bg="#FFFFFF").grid(row=0, column=1, padx=5, pady=5, sticky=tk.E + tk.W)
        tk.Label(profile_edit_frame, text="z", bg="#FFFFFF").grid(row=0, column=2, padx=5, pady=5, sticky=tk.E + tk.W)
        tk.Label(profile_edit_frame, text="roll", bg="#FFFFFF").grid(row=0, column=3, padx=5, pady=5, sticky=tk.E + tk.W)

        # Up/Down button for x, y, z, roll
        x_up_button = tk.Button(profile_edit_frame, text="x+", command=lambda: self.update_new_position('x+'))
        y_up_button = tk.Button(profile_edit_frame, text="y+", command=lambda: self.update_new_position('y+'))
        z_up_button = tk.Button(profile_edit_frame, text="z+", command=lambda: self.update_new_position('z+'))
        roll_up_button = tk.Button(profile_edit_frame, text="roll+", command=lambda: self.update_new_position('roll+'))
        x_down_button = tk.Button(profile_edit_frame, text="x-", command=lambda: self.update_new_position('x-'))
        y_down_button = tk.Button(profile_edit_frame, text="y-", command=lambda: self.update_new_position('y-'))
        z_down_button = tk.Button(profile_edit_frame, text="z-", command=lambda: self.update_new_position('z-'))
        roll_down_button = tk.Button(profile_edit_frame, text="roll-", command=lambda: self.update_new_position('roll-'))
        x_up_button.grid(row=1, column=0, padx=5, pady=5, sticky=tk.E + tk.W)
        y_up_button.grid(row=1, column=1, padx=5, pady=5, sticky=tk.E + tk.W)
        z_up_button.grid(row=1, column=2, padx=5, pady=5, sticky=tk.E + tk.W)
        roll_up_button.grid(row=1, column=3, padx=5, pady=5, sticky=tk.E + tk.W)
        x_down_button.grid(row=4, column=0, padx=5, pady=5, sticky=tk.E + tk.W)
        y_down_button.grid(row=4, column=1, padx=5, pady=5, sticky=tk.E + tk.W)
        z_down_button.grid(row=4, column=2, padx=5, pady=5, sticky=tk.E + tk.W)
        roll_down_button.grid(row=4, column=3, padx=5, pady=5, sticky=tk.E + tk.W)


        # Edit/move x, y, z, roll
        self.new_x = tk.StringVar()
        self.new_y = tk.StringVar()
        self.new_z = tk.StringVar()
        self.new_roll = tk.StringVar()
        self.new_x_label = tk.Entry(profile_edit_frame, textvariable=self.new_x, width=6, justify='center')
        self.new_y_label = tk.Entry(profile_edit_frame, textvariable=self.new_y, width=6, justify='center')
        self.new_z_label = tk.Entry(profile_edit_frame, textvariable=self.new_z, width=6, justify='center')
        self.new_roll_label = tk.Entry(profile_edit_frame, textvariable=self.new_roll, width=6, justify='center')
        self.new_x_label.grid(row=2, column=0, padx=5, pady=5, sticky=tk.E+tk.W)
        self.new_y_label.grid(row=2, column=1, padx=5, pady=5, sticky=tk.E+tk.W)
        self.new_z_label.grid(row=2, column=2, padx=5, pady=5, sticky=tk.E+tk.W)
        self.new_roll_label.grid(row=2, column=3, padx=5, pady=5, sticky=tk.E+tk.W)
        self.new_x_label.bind('<Return>', self.set_position)
        self.new_y_label.bind('<Return>', self.set_position)
        self.new_z_label.bind('<Return>', self.set_position)
        self.new_roll_label.bind('<Return>', self.set_position)
        self.new_x.set(self.default_x)
        self.new_y.set(self.default_y)
        self.new_z.set(self.default_z)
        self.new_roll.set(self.default_roll)

        # Show xyz and roll position
        self.actual_x = tk.StringVar()
        self.actual_y = tk.StringVar()
        self.actual_z = tk.StringVar()
        self.actual_roll = tk.StringVar()
        self.actual_x_label = tk.Label(profile_edit_frame, textvariable=self.actual_x, bg="#66ff66")
        self.actual_y_label = tk.Label(profile_edit_frame, textvariable=self.actual_y, bg="#66ff66")
        self.actual_z_label = tk.Label(profile_edit_frame, textvariable=self.actual_z, bg="#66ff66")
        self.actual_roll_label = tk.Label(profile_edit_frame, textvariable=self.actual_roll, bg="#66ff66")
        self.actual_x_label.grid(row=3, column=0, padx=5, pady=5, sticky=tk.E+tk.W)
        self.actual_y_label.grid(row=3, column=1, padx=5, pady=5, sticky=tk.E+tk.W)
        self.actual_z_label.grid(row=3, column=2, padx=5, pady=5, sticky=tk.E+tk.W)
        self.actual_roll_label.grid(row=3, column=3, padx=5, pady=5, sticky=tk.E+tk.W)

        # set new position button
        set_position_button = tk.Button(profile_edit_frame, text="Set Position", command=self.set_position)
        set_position_button.grid(row=5, column=0, columnspan=2, padx=5, pady=5, sticky=tk.E+tk.W)

        # back to home position button
        home_position_button = tk.Button(profile_edit_frame, text="Home", command=self.home_position)
        home_position_button.grid(row=5, column=2, columnspan=2, padx=5, pady=5, sticky=tk.E + tk.W)

        # learn from actual position
        learn_profile_button = tk.Button(profile_edit_frame, text="Learn from actual position", command=self.learn_profile_position)
        learn_profile_button.grid(row=6, column=0, columnspan=2, padx=5, pady=5, sticky=tk.E + tk.W)

        # Reset Profile Position button
        reset_profile_button = tk.Button(profile_edit_frame, text="Reset Position", command=self.reset_profile_position)
        reset_profile_button.grid(row=6, column=2, columnspan=2, padx=5, pady=5, sticky=tk.E + tk.W)

        # Update Profile Step button
        update_profile_button = tk.Button(profile_edit_frame, text="Update Profile", command=self.update_profile)
        update_profile_button.grid(row=7, column=0, columnspan=4, padx=5, pady=5, sticky=tk.E + tk.W)

        pass    # end create_widget()

    def on_closing(self):
        if tk.messagebox.askokcancel("Quit", "Exit Platform Position Setting?"):

            # return to original feedback
            #if (self.original_feedback is not None):
            #    self.platform.feedback_callback = self.original_feedback
            try: #capture exception when platform is not available that prevent closing
                self.platform.feedback_callback = None
            except:
                traceback.print_exc() 

            self.destroy()
        pass

    def load_step(self):
        try:
            step_list = []
            #print(f"profile: {Profile.loaded_profile[0]}")
            for index, step in enumerate(Profile.loaded_profile[self.start.MainUi.cam_pos]):
                #print(f"step: {step['platform']}")
                step_list.append(index + 1)

            # Insert each step's index onto listbox
            [self.profile_listbox.insert(tk.END, i) for i in step_list]
        except Exception as e:
            traceback.print_exc()
            print("Empty step of source")

    def model_selected(self, event):
        model_name = self.selected_model.get()

        #todo: user select another model, take effect immediately
        pass

    def update_config(self) -> bool:
        """Update platform model and ipaddress"""
        # todo
        self.start.Config.config['PLATFORM']['model'] = self.selected_model.get()
        self.start.Config.config['PLATFORM']['ip_address'] = self.ipaddress.get()
        pass

    def toggle_platform_enable(self) -> bool:
        if(self.is_platform_enable == True):
            # disable platform
            if (self.platform is not None):
                self.platform.disable_platform()

            self.is_platform_enable = False
            self.enable_button.config(text="Enable Platform")
        else:
            # enable platform
            if (self.platform is not None):
                self.platform.enable_platform()

            self.is_platform_enable = True
            self.enable_button.config(text="Disable Platform")
        return self.is_platform_enable

    def clear_error(self):
        self.platform.clear_error()

    def reset_platform(self):
        self.platform.reset()

    def select_step(self, event):
        """
        User select step from listbox, fetch the platform position and put them onto their corresponding input entry
        """
        # Exact same function as reset_profile_position, just call it
        self.reset_profile_position()

    def update_new_position(self, command=""):
        self.increment_step = float(1)
        if (command == "x+"):
            self.new_x.set(float(self.new_x.get()) + self.increment_step)
        elif (command == "y+"):
            self.new_y.set(float(self.new_y.get()) + self.increment_step)
        elif (command == "z+"):
            self.new_z.set(float(self.new_z.get()) + self.increment_step)
        elif (command == "roll+"):
            self.new_roll.set(float(self.new_roll.get()) + self.increment_step)
        elif (command == "x-"):
            self.new_x.set(float(self.new_x.get()) - self.increment_step)
        elif (command == "y-"):
            self.new_y.set(float(self.new_y.get()) - self.increment_step)
        elif (command == "z-"):
            self.new_z.set(float(self.new_z.get()) - self.increment_step)
        elif (command == "roll-"):
            self.new_roll.set(float(self.new_roll.get()) - self.increment_step)

        # direct output to platform
        if (self.is_platform_connected):
            self.set_position()

    def set_position(self, *args):
        self.platform.move_to_point_async(self.new_x.get(),
                                          self.new_y.get(),
                                          self.new_z.get(),
                                          self.new_roll.get())

    def home_position(self):
        self.new_x.set(self.default_x)
        self.new_y.set(self.default_y)
        self.new_z.set(self.default_z)
        self.new_roll.set(self.default_roll)
        self.platform.move_to_point_async(self.default_x, self.default_y, self.default_z, self.default_roll)

    def update_profile(self):
        """
        Update new x, y, z and roll back to profile step
        """
        try:
            w = self.profile_listbox
            step_index = int(w.curselection()[0])

            Profile.loaded_profile[self.start.MainUi.cam_pos][step_index]["platform"]["x"] = float(self.new_x.get())
            Profile.loaded_profile[self.start.MainUi.cam_pos][step_index]["platform"]["y"] = float(self.new_y.get())
            Profile.loaded_profile[self.start.MainUi.cam_pos][step_index]["platform"]["z"] = float(self.new_z.get())
            Profile.loaded_profile[self.start.MainUi.cam_pos][step_index]["platform"]["roll"] = float(self.new_roll.get())

            # Set the flag to ask the Profile module saves this setting during exit the application
            Profile.flag_save = True

            # reset the platform
            self.platform.reset()
        except Exception as e:
            traceback.print_exc()
            print("Error during update platform position into profile")
        pass

    def reset_profile_position(self):
        """
        Reset "new" x, y, z and roll back to their original position
        """
        try:
            # extract the selected index
            w = self.profile_listbox
            step_index = int(w.curselection()[0])

            # get selected position from profile
            self.current_platform_position = Profile.loaded_profile[self.start.MainUi.cam_pos][step_index]["platform"]
            print(f"Position: {self.current_platform_position}")

            self.new_x.set(self.current_platform_position['x'])
            self.new_y.set(self.current_platform_position['y'])
            self.new_z.set(self.current_platform_position['z'])
            self.new_roll.set(self.current_platform_position['roll'])

            if (self.is_platform_connected):
                self.set_position()
        except Exception as e:
            traceback.print_exc()
            print("Error during reset platform position")
        pass

    def learn_profile_position(self):
        """
        Update the "new" x, y, z and roll value from actual value from platform hardware
        """
        self.new_x.set(float(self.actual_x.get()))
        self.new_y.set(float(self.actual_y.get()))
        self.new_z.set(float(self.actual_z.get()))
        self.new_roll.set(float(self.actual_roll.get()))
        pass

    def feedback_handler(self, data):
        try:
            self.actual_x.set(f"{self.platform.x(): .2f}")
            self.actual_y.set(f"{self.platform.y(): .2f}")
            self.actual_z.set(f"{self.platform.z(): .2f}")
            self.actual_roll.set(f"{self.platform.roll(): .2f}")
            self.platform_mode.set(f"Mode: {self.platform.operating_mode()}")

            self.status_text.delete('1.0', tk.END)
            self.status_text.insert(tk.END, f"{self.platform.get_log()}")
            self.status_text.see(tk.END)

            if(self.platform.is_error()):
                self.operating_error.set("Error!")
                self.error_label.config(bg="#ff0000", fg="#ffffff")
            else:
                self.operating_error.set("No Error")
                self.error_label.config(bg="#23ff23", fg="#000000")

            if(self.platform.operating_mode() == "MODE_DISABLED"):
                self.is_platform_enable = False
                self.platform_mode_label.config(bg="#ff0000", fg="#ffffff")
                self.enable_button.config(text="Enable Platform")
            elif(self.platform.operating_mode() == "MODE_ENABLE"):
                self.is_platform_enable = True
                self.platform_mode_label.config(bg="#23ff23", fg="#000000")
                self.enable_button.config(text="Disable Platform")
        except:
            traceback.print_exc()
        # pass
