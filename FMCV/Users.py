import sqlite3
import os
from pathlib import Path
import traceback

import tkinter as tk
from tkinter import *
from tkinter import ttk
from tkinter import filedialog
from tkinter import messagebox
from tkinter.messagebox import askokcancel, showinfo, WARNING, showwarning

db_path = Path('Profile','users.db')

print(db_path.parents[0])

os.makedirs(db_path.parents[0], exist_ok=True)

# Create database connection
conn = sqlite3.connect(db_path)
c = conn.cursor()

# Create users table
c.execute('''CREATE TABLE IF NOT EXISTS users 
             (username text, password text, usergroup text)''')
conn.commit()

is_admin = False
is_user = False

username = ""

logged_in = False

message_showed = False

def init(in_start):
    global start
    global logged_in
    global is_admin
    global is_user
    
    start = in_start
    
    if in_start.Config.log_type in ( "PLEXUS","KAIFA"):
        start.sub("ui/users/show", new_window)
    else:
        logged_in = True
        is_admin = True
        is_user = True
    
def login_admin():
    global message_showed
    global start
    global is_admin
    if hasattr(start, 'MainUi'): # use by profile write() during initialize when MainUi is not yet created
        if not is_admin or not logged_in:
            if not message_showed:
                new_window()
                start.MainUi.top.wait_window(window_opened_root)
            
        if not is_admin or not logged_in:
            if not message_showed:
                messagebox.showwarning("Users", "Please login as admin or all setting is not save")
                #message_showed = True
        return is_admin
    else:
        return True # use by profile write() during initialize
    
def login_user():
    global message_showed
    global is_user
    if not logged_in or not is_user:
        if not message_showed:
           new_window()
           start.MainUi.top.wait_window(window_opened_root)

    if not logged_in or not is_user: 
        if not message_showed:
            messagebox.showwarning("Users", "Please login")
            #message_showed = True
    return is_user

#https://pythonprogramming.altervista.org/tkinter-open-a-new-window-and-just-one/
def new_window():
    global message_showed
    global window_opened_root
    global users_managements_window
    try:
        if window_opened_root.state() == "normal": window_opened_root.focus()
    except:
        #traceback.print_exc()
        #message_showed = False
        start.log.info("Creating New User Management Window")
        window_opened_root = tk.Toplevel()
        users_managements_window = UsersManagementWindow(start, window_opened_root)

def refresh():
    global window_opened_root
    global users_managements_window
    try:
        users_managements_window.main_frame.refresh()
    except:
        traceback.print_exc()
        
class UsersManagementWindow:
    def __init__(self, start, root):
        self.start = start
        self.root = root
        #root.geometry("1024x768-150+150")  
        self.main_frame = UsersManagementFrame(start, root)
        #self.main_frame.pack(fill=BOTH, expand=True)
        #self.root.attributes('-topmost',True)
        self.root.lift()

class UsersManagementFrame(ttk.Frame):
    def __init__(self,start, root, *args, **kwargs):
        #https://stackoverflow.com/questions/17466561/best-way-to-structure-a-tkinter-application/17470842
        ttk.Frame.__init__(self, root, *args, **kwargs)
        self.root = root
        self.start = start
        self.c = start.Users.c
        self.conn = start.Users.conn
        
        # Create main window
        root.title("Login Management")

        # Create username label and entry widget
        self.username_label = tk.Label(root, text="Username:")
        self.username_label.grid(row=0, column=0)
        self.username_entry = tk.Entry(root)
        self.username_entry.grid(row=0, column=1)

        # Create password label and entry widget
        self.password_label = tk.Label(root, text="Password:")
        self.password_label.grid(row=1, column=0)
        self.password_entry = tk.Entry(root, show="*")
        self.password_entry.grid(row=1, column=1)

        # Create login button and label
        self.login_button = tk.Button(root, text="Login", command=self.login)
        self.login_button.grid(row=2, column=0)
        self.login_label = tk.Label(root)
        self.login_label.grid(row=2, column=1)

        # Create user group label and dropdown menu
        self.usergroup_label = tk.Label(root, text="User group:")
        self.usergroup_label.grid(row=3, column=0)
        self.usergroup_var = tk.StringVar()
        self.usergroup_var.set("user")
        self.usergroup_menu = tk.OptionMenu(root, self.usergroup_var, "user", "admin")
        self.usergroup_menu.grid(row=3, column=1)

        
        # Create register button and label
        self.register_button = tk.Button(root, text="Register", command=self.register, state = self.check_has_admin())
        self.register_button.grid(row=4, column=0)
        self.register_label = tk.Label(root)
        self.register_label.grid(row=4, column=1)

        # Create edit password button and entry widget
        self.edit_password_button = tk.Button(root, text="Edit password", command=self.edit_password, state="disabled")
        self.edit_password_button.grid(row=5, column=0)
        self.new_password_label = tk.Label(root, text="New password:")
        self.new_password_label.grid(row=6, column=0)
        self.new_password_entry = tk.Entry(root, show="*",state="disable")
        self.new_password_entry.grid(row=6, column=1)
        self.edit_password_label = tk.Label(root)
        self.edit_password_label.grid(row=7, column=1)
        
        #Create logout button
        self.logout_button = tk.Button(root, text="Logout", command=self.logout)
        
        if self.start.Users.logged_in:
            if not self.start.Users.is_admin:   
                self.register_button.grid_remove()
            # Show logout button and hide login and register buttons
            self.edit_password_button.config(state="disabled")
            self.logout_button.grid(row=2, column=0)
            self.login_button.grid_remove()
            self.username_entry.config(state="disable")
            self.password_entry.config(state="disable")
            self.login_label.config(text=self.start.Users.username)
            print("logged in")
        else:
            
            print("logged out")
            self.logout()
        
    def refresh(self):
        pass
    
    # Define login function
    def login(self):
        # Get username and password from entry widgets
        username = self.username_entry.get()
        password = self.password_entry.get()
        
        # Check if user exists in the database
        self.c.execute("SELECT * FROM users WHERE username=? AND password=?", (username, password))
        result = self.c.fetchone()
        if result:
            self.start.Users.username = result[0]
            self.start.Users.logged_in = True
            
            self.login_label.config(text="Login successful")
            # Check user group and enable/disable edit password button
            if result[2] == "admin":
                self.start.Users.is_admin = True
                self.start.Users.is_user = True
                self.register_button.grid_remove()
                self.register_button.config(state="normal")
                self.register_button.grid(row=4, column=0)
                self.username_entry.config(state="normal")
                self.password_entry.config(state="normal")
            else:
                self.register_button.grid_remove()
                self.start.Users.is_admin = False
                self.start.Users.is_user = True
                self.username_entry.config(state="disable")
                self.password_entry.config(state="disable")
                
            # Show logout button and hide login and register buttons
            self.edit_password_button.config(state="normal")
            self.new_password_entry.config(state="normal")
            self.logout_button.grid(row=2, column=0)
            self.login_button.grid_remove()

        else:
            self.login_label.config(text="Incorrect username or password")

    # Define register function
    def register(self):
        # Get username, password, and user group from entry widgets
        username =self.username_entry.get()
        password = self.password_entry.get()
        usergroup = self.usergroup_var.get()
        
        # Check if user already exists in the database
        self.c.execute("SELECT * FROM users WHERE username=?", (username,))
        result = self.c.fetchone()
        if result:
            self.register_label.config(text="Username already taken")
        else:
            # Add user to the database
            self.c.execute("INSERT INTO users VALUES (?, ?, ?)", (username, password, usergroup))
            self.conn.commit()
            self.register_label.config(text="Registration successful")

    # Define edit password function
    def edit_password(self):
        # Get username and new password from entry widgets
        username = self.username_entry.get()
        new_password =self.new_password_entry.get()
        
        # Update password in the database
        self.c.execute("UPDATE users SET password=? WHERE username=?", (new_password, username))
        self.conn.commit()
        self.edit_password_label.config(text="Password updated")

    # Define logout function
    def logout(self):
        self.start.Users.is_admin = False
        self.start.Users.is_user = False
        self.start.Users.username = ""
        self.start.Users.logged_in = False
    
        # Clear entry widgets and login and edit password labels
        self.username_entry.delete(0, tk.END)
        self.password_entry.delete(0, tk.END)
        self.new_password_entry.delete(0, tk.END)
        self.login_label.config(text="")
        self.edit_password_label.config(text="")
        
        # Hide logout button and show login and register buttons
        self.logout_button.grid_remove()
        self.login_button.grid(row=2, column=0)
        
        self.register_button.config(state=self.check_has_admin())
        self.register_button.grid(row=4, column=0)
        
        # Disable edit password button
        self.edit_password_button.config(state="disabled")
        self.new_password_entry.config(state="disabled")
        
        self.username_entry.config(state="normal")
        self.password_entry.config(state="normal")
        
    def check_has_admin(self):
        # Check if admin registered
        self.c.execute("SELECT * FROM users WHERE usergroup ='admin'")

        # Fetch all records as a list of tuples
        records = self.c.fetchall()

        has_admin = "disabled"
        # Process each record
        for record in records:
            # Extract the fields from the record tuple
            username, password, usergroup = record
            # Do something with the fields, e.g. print them
            #print(f"Username: {username}, User group: {usergroup}")
            
            has_admin = "disabled"
            
        if len(records) == 0:
            has_admin = "normal"
            
        return has_admin