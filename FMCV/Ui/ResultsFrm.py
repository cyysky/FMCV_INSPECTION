from tkinter import *
from tkinter import ttk
from PIL import ImageTk, Image
import cv2
from FMCV.Ui.ScrollableFrm  import ScrollableFrame
from FMCV import Util

class ResultsFrame(ttk.Frame):

    def __init__(self, start, parent, *args, **kwargs):
        #https://stackoverflow.com/questions/17466561/best-way-to-structure-a-tkinter-application/17470842
        ttk.Frame.__init__(self, parent, *args, **kwargs)
        self.parent = parent
        self.start = start
        
        m1 = ttk.PanedWindow(self,orient=HORIZONTAL)
        m1.pack(fill=BOTH, expand=True)
        
        tree = ttk.Treeview(m1, column=("Items", "Values"), show='headings' , height=5)
        tree.column("# 1", anchor=CENTER, width=20)
        tree.heading("# 1", text="Items")
        tree.column("# 2", anchor=CENTER,  width=40)
        tree.heading("# 2", text="Values")
        
        for i in range(50):
            tree.insert('', 'end', text="1", values=("", ""))
            
        tree.pack(expand=True,fill=Y)
        # Constructing vertical scrollbar
        # with treeview
        verscrlbar = ttk.Scrollbar(tree, orient ="vertical", command = tree.yview)
         
        # Calling pack method w.r.to vertical
        # scrollbar
        verscrlbar.pack(side ='right', fill ='y')
        tree.configure(yscrollcommand = verscrlbar.set)
        
        self.result_frame = ScrollableFrame(m1)
        self.content = content = self.result_frame.scrollable_frame
        
        self.lbl_result_overall = Label(self.content,width = 20 , height = 20, text = 'Overall Result')
        self.lbl_result_overall.pack(side=RIGHT,pady=15,padx=5)
        
        self.lbl_result = Label(self.content,width = 20 , height = 8, text = 'RESULT')
        self.lbl_result.pack()
        

        self.btn_open_log_folder = Button(content,text ="Open Results", command = start.ActionUi.open_log_folder)
        self.btn_open_log_folder.pack()
        
        self.btn_open_image_log_folder = Button(content,text ="Open Image Results", command = start.ActionUi.open_image_log_folder)
        self.btn_open_image_log_folder.pack()
        
        
        self.lbl_stat = Label(self.content,width = 15 , height = 4, text = "Total PASS : 0 \nTotal FAIL : 0")
        self.lbl_stat.pack()
        
        self.update_total()
        
        self.btn_reset_total_count = btn_reset_total_count = Button(content,text ="Reset Total Count")
        btn_reset_total_count.pack()
        btn_reset_total_count.configure(command=self.reset_total_count)


        
        Label(content, text = 'Folder').pack()
        self.folder_cmb = folder_cmb = ttk.Combobox(content)
        folder_cmb['values'] = tuple(self.start.Profile.get_image_folders_list())
        folder_cmb.pack()
        
        self.btn_save_result_roi_image = btn_save_result_roi_image = Button(content,text ="Save Result Image To Folder")
        btn_save_result_roi_image.pack()
        btn_save_result_roi_image.configure(command=self.save_result_roi_image)
        
        self.btn_open_folder = Button(content,text ="Open Folder", command = lambda:start.ActionUi.open_folder(folder_cmb.get()))
        self.btn_open_folder.pack()
        
        self.tree = tree
 
        m1.add(tree,weight=1)
        m1.add(self.result_frame,weight=1)
        
        self.result_roi = {}
     
    def save_result_roi_image(self):
        if self.start.Users.login_user():
            img = self.result_roi.get('result_image')
            if img is not None:
                #self.start.Profile.create_image_folder(self.folder_cmb.get())
                pth = self.start.Profile.write_image(self.folder_cmb.get(),img)
                print(self.folder_cmb.get())
                self.start.MainUi.write(f'Result Image Saved to {pth}')
            self.folder_cmb['values'] = tuple(self.start.Profile.get_image_folders_list())
            self.folder_cmb.pack()


    def update_results(self,roi_results):  
        
        self.tree.delete(*self.tree.get_children())   
        
        # for widgets in self.result_frame.scrollable_frame.winfo_children():
            # widgets.destroy()

        if roi_results.get("name") is not None:
            self.tree.insert('', 'end', text="1", values=("Name", roi_results.get("name")))
        if roi_results.get("PASS") is not None:
            self.tree.insert('', 'end', text="1", values=("PASS", roi_results.get("PASS")), \
                                                    tags=('green' if roi_results.get("PASS") else 'red',))
            
        if roi_results.get("type") is not None and roi_results.get('distance') is not None:
            if roi_results["type"] == "DIST" :
                roi = roi_results
                distance_max = roi['distance_max']
                distance_min = roi['distance_min']
                distance_y_max = roi['distance_y_max']
                distance_y_min = roi['distance_y_min']
                distance_x_max = roi['distance_x_max']
                distance_x_min = roi['distance_x_min']
                distance_x = roi['distance_x_mm']
                distance_y = roi['distance_y_mm']
                distance = roi['distance_mm']

                self.tree.insert('', 'end', text="1", values=("distance_mm", Util.convert_5_decimal_string(distance)),
                                    tags=('green' if (distance_min <= distance <= distance_max) else 'red',))
                                    
                self.tree.insert('', 'end', text="1", values=("distance_x_mm", Util.convert_5_decimal_string(distance_x)),
                                    tags=('green' if (distance_x_min <= distance_x <= distance_x_max) else 'red',))
                                    
                self.tree.insert('', 'end', text="1", values=("distance_y_mm", Util.convert_5_decimal_string(distance_y)),
                                    tags=('green' if (distance_y_min <= distance_y  <= distance_y_max) else 'red',))
                                    
                self.tree.insert('', 'end', text="1", values=("score", Util.convert_5_decimal_string(roi.get('score'))),
                                    tags=('green' if (roi.get('minimal') <= roi.get('score')) else 'red',))
                
        for roi_name, roi_value in reversed(Util.without_keys(roi_results,{"img","image","result_image","mask","mask_64","ano_result_image","result_blended_heatmap","2D_result_image","source_image","PASS","name"}).items()):
            if isinstance(roi_value, float):
                number = roi_value
                value = f"{number:.5f}".rstrip('0').rstrip('.') if '.' in f"{number:.5f}" else f"{number:.5f}"
                self.tree.insert('', 'end', text="1", values=(roi_name, value))
            else:
                self.tree.insert('', 'end', text="1", values=(roi_name, roi_value))
            #ttk.Label(self.content, text=str(roi_attrib)).pack()
            
        self.tree.tag_configure('green', background='green2')
        self.tree.tag_configure('red', background='red2')
        
        self.folder_cmb['values'] = tuple(self.start.Profile.get_image_folders_list())
        self.folder_cmb.pack()
    
    def set_result(self,is_pass):
        if is_pass is None:
            self.lbl_result.config(bg='SystemButtonFace',text="RESET")
            #self.update_total()
        else:
            if is_pass:
                color = 'green'
                msg = 'PASS'
            else:
                color = 'red'
                msg = 'FAIL'
            self.lbl_result.config(bg=color,text=msg)
        self.update_total()

    def set_overall_result(self,is_pass):
        if is_pass is None:
            self.lbl_result_overall.config(bg='SystemButtonFace',text="RESET")
            #self.update_total()
        else:
            if is_pass:
                color = 'green'
                msg = 'PASS'
            else:
                color = 'red'
                msg = 'FAIL'
            self.lbl_result_overall.config(bg=color,text=msg)
        self.update_total()

    def set_running(self):
        self.lbl_result_overall.config(bg='yellow',text="Running")
        
    def update_total(self):
        total_pass = self.start.Config.class_total.get("PASS")
        total_fail = self.start.Config.class_total.get("FAIL")
        self.lbl_stat.config(text = f"Total PASS : {total_pass}\nTotal FAIL : {total_fail}")
        
    def reset_total_count(self):
        if self.start.Users.login_user():
            self.start.Main.reset_total_count()
            self.update_total()
        
    def reset(self):
        self.lbl_result_overall.config(bg='SystemButtonFace',text="RESET")
        text = "FMCV"
        self.lbl_result.config(bg='SystemButtonFace',text=text)
            
        self.tree.delete(*self.tree.get_children())   