from tkinter import filedialog
import customtkinter
import os
from tkVideoPlayer import TkinterVideo
from PIL import Image, ImageTk
import cv2
import argparse
from persondetection import DetectorAPI
import matplotlib.pyplot as plt
from fpdf import FPDF
import numpy as np

customtkinter.set_appearance_mode("System")
customtkinter.set_default_color_theme("blue")


def argsParser():
    arg_parse = argparse.ArgumentParser()
    arg_parse.add_argument("-v", "--video", default=None, help="path to Video File ")
    arg_parse.add_argument("-i", "--image", default=None, help="path to Image File ")
    arg_parse.add_argument("-c", "--camera", default=False, help="Set true if you want to use the camera.")
    arg_parse.add_argument("-o", "--output", type=str, help="path to optional output video file")
    args = vars(arg_parse.parse_args())
    return args


global filename
global max_count, framex, county, maxim, avg_acc_list, max_avg_acc_list, max_acc, max_avg_acc, flag
global current_status, crowd_number, max_accuracy, crowd_status, video_file

global last_frame
last_frame = np.zeros((480, 640, 3), dtype=np.uint8)
global cap
cap = cv2.VideoCapture(0)
flag = 0


class App(customtkinter.CTk):

    def __init__(self):
        super().__init__()
        global flag
        flag = 0
        # configure window
        self.title("Real Time People Counting")
        self.geometry(f"{1280}x{720}")
        self.resizable(width=False, height=False)

        # configure grid layout (4x4)
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure((2, 3), weight=0)
        self.grid_rowconfigure((0, 1, 2), weight=1)

        # creating frames
        self.sidebar_frame = customtkinter.CTkFrame(self, width=140, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, rowspan=4, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(4, weight=1)

        self.display_frame = customtkinter.CTkFrame(self, width=600, corner_radius=10)
        self.display_frame.grid(row=0, column=2, rowspan=2, sticky="nsew", padx=(10, 20), pady=10)
        self.display_frame.grid_rowconfigure(4, weight=0)

        self.graph1_frame = customtkinter.CTkFrame(self, width=650, corner_radius=10)
        self.graph1_frame.grid(row=0, column=1, rowspan=1, sticky="nsew", padx=(20, 10), pady=10)
        self.graph1_frame.grid_rowconfigure(4, weight=0)

        self.output_frame = customtkinter.CTkFrame(self, width=1000, height=20, corner_radius=10)
        self.output_frame.grid(row=2, column=1, columnspan=2, sticky="nsew", padx=20, pady=10)

        # elements of sidebar_frame go here
        self.logo_label = customtkinter.CTkLabel(self.sidebar_frame, text="Crowd Detector",
                                                 font=customtkinter.CTkFont(size=20, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))

        self.input_label = customtkinter.CTkLabel(self.sidebar_frame, text="Input Options",
                                                  font=customtkinter.CTkFont(size=15, weight="bold"))
        self.input_label.grid(row=1, column=0, padx=20, pady=(20, 10))

        self.function_label = customtkinter.CTkLabel(self.sidebar_frame, text="Function Options",
                                                     font=customtkinter.CTkFont(size=15, weight="bold"))
        self.function_label.grid(row=5, column=0, padx=20)

        self.red_label = customtkinter.CTkLabel(self.sidebar_frame, text=" ",
                                                font=customtkinter.CTkFont(size=15, weight="bold"))
        self.red_label.grid(row=7, column=0, padx=20, pady=(200, 0))

        self.sidebar_button_1 = customtkinter.CTkButton(self.sidebar_frame, text="image",
                                                        command=self.image_button_event)
        self.sidebar_button_1.grid(row=2, column=0, padx=20, pady=10)

        self.sidebar_button_2 = customtkinter.CTkButton(self.sidebar_frame, text="Camera",
                                                        command=self.camera_button_event)
        self.sidebar_button_2.grid(row=4, column=0, padx=20, pady=10)

        self.sidebar_button_3 = customtkinter.CTkButton(self.sidebar_frame, text="video",
                                                        command=self.video_button_event)
        self.sidebar_button_3.grid(row=3, column=0, padx=20, pady=5)

        self.sidebar_button_4 = customtkinter.CTkButton(self.sidebar_frame, text="Solve",
                                                        command=self.solve_button_event)
        self.sidebar_button_4.grid(row=6, column=0, padx=20, pady=5)

        self.sidebar_button_5 = customtkinter.CTkButton(self.sidebar_frame, text="Generate PDF",
                                                        command=self.pdf_button_event)
        self.sidebar_button_5.grid(row=7, column=0, padx=20, pady=10)

        self.appearance_mode_label = customtkinter.CTkLabel(self.sidebar_frame, text="Appearance Mode:", anchor="w")
        self.appearance_mode_label.grid(row=9, column=0, padx=20, pady=(10, 0))
        self.appearance_mode_optionmenu = customtkinter.CTkOptionMenu(self.sidebar_frame,
                                                                      values=["Dark", "Light", "System"],
                                                                      command=self.change_appearance_mode_event)
        self.appearance_mode_optionmenu.grid(row=10, column=0, padx=20, pady=(10, 10))

        # Dashboard stuff goes here ------------------------------------------------------------------------------------

        # elements of preview_frame go here
        self.preview_label = customtkinter.CTkLabel(self.display_frame, text="Preview",
                                                    font=customtkinter.CTkFont(size=20, weight="bold"))
        self.preview_label.grid(row=0, column=0, padx=250)

        # elements of tabview go here
        self.tabview = customtkinter.CTkTabview(self.graph1_frame, width=400, height=350)
        self.tabview.grid(row=1, column=0, padx=(25, 0), pady=(20, 20), sticky="nsew")
        self.tabview.add("Enumeration")
        self.tabview.add("Accuracy")
        self.tabview.tab("Enumeration").grid_columnconfigure(0, weight=1)  # configure grid of individual tabs
        self.tabview.tab("Accuracy").grid_columnconfigure(0, weight=1)

        # enumeration tab
        graph1_path = 'Figure_2.png'
        graph1 = customtkinter.CTkImage(light_image=Image.open(graph1_path), dark_image=Image.open(graph1_path),
                                        size=(350, 250))

        self.graph1_button = customtkinter.CTkButton(self.tabview.tab("Enumeration"), text=" ", width=350, height=250,
                                                     fg_color="ORANGE", hover_color="ORANGE", image=graph1,
                                                     command=lambda: self.enumeration_button_event())
        self.graph1_button.grid(row=1, column=0, padx=5, pady=10)

        # accuracy tab
        graph2_path = 'Figure_1.png'
        graph2 = customtkinter.CTkImage(light_image=Image.open(graph2_path), dark_image=Image.open(graph2_path),
                                        size=(350, 250))

        self.graph1_button = customtkinter.CTkButton(self.tabview.tab("Accuracy"), text=" ", width=350, height=250,
                                                     fg_color="ORANGE", hover_color="ORANGE", image=graph2,
                                                     command=lambda: self.accuracy_button_event())
        self.graph1_button.grid(row=1, column=0, padx=5, pady=10)

        self.graph1_label = customtkinter.CTkLabel(self.graph1_frame, text="   Graphical Representation",
                                                   font=customtkinter.CTkFont(size=20, weight="bold"))
        self.graph1_label.grid(row=0, column=0, padx=5)

        # elements of output_frame go here
        self.output_label = customtkinter.CTkLabel(self.output_frame, text="Output",
                                                   font=customtkinter.CTkFont(size=20, weight="bold"))
        self.output_label.grid(row=0, column=0, padx=140)
        global current_status, crowd_number, max_accuracy, crowd_status
        current_status = "running program"
        crowd_number = "crowd number"
        max_accuracy = "max Confidence"
        crowd_status = "crowd status"

        self.text_area = customtkinter.CTkTextbox(self.output_frame, width=1000,
                                                  font=customtkinter.CTkFont(size=20, weight="bold"))
        self.text_area.insert("0.0",
                              f"{current_status}\n\n{crowd_number}\n\n{max_accuracy}\n\n{crowd_status}")
        self.text_area.grid(row=1, column=0, padx=(20, 0), pady=(20, 10), sticky="nsew")

    def change_appearance_mode_event(self, new_appearance_mode: str):
        customtkinter.set_appearance_mode(new_appearance_mode)

    def image_button_event(self):
        global filename, flag
        if flag != 0:
            for widget in self.display_frame.winfo_children():
                widget.destroy()
        flag = 1
        print("File works")
        filename = filedialog.askopenfilename(title="Select Image file")
        abs_path = os.path.abspath(filename)
        self.preview_label = customtkinter.CTkLabel(self.display_frame, text="Preview",
                                                    font=customtkinter.CTkFont(size=20, weight="bold"))
        self.preview_label.grid(row=0, column=0, padx=250)
        self.preview_image(abs_path)

    def preview_video(self, video_file):

        def open_video(video_file):
            if video_file:
                try:
                    vid_player.load(video_file)
                    vid_player.play()
                    progress_slider.set(-1)
                    play_pause_btn.configure(text="Pause ||")
                except:
                    print("Unable to load the file")

        def update_duration(event):
            try:
                duration = int(vid_player.video_info()["duration"])
                progress_slider.configure(from_=-1, to=duration, number_of_steps=duration)
            except:
                pass

        def seek(value):
            if video_file:
                try:
                    vid_player.seek(int(value))
                    vid_player.play()
                    vid_player.after(50, vid_player.pause)
                    play_pause_btn.configure(text="Play ►")
                except:
                    pass

        def update_scale(event):
            try:
                progress_slider.set(int(vid_player.current_duration()))
            except:
                pass

        def play_pause():
            if video_file:
                if vid_player.is_paused():
                    vid_player.play()
                    play_pause_btn.configure(text="Pause ||")

                else:
                    vid_player.pause()
                    play_pause_btn.configure(text="Play ►")

        def video_ended(event):
            play_pause_btn.configure(text="Play ►")
            progress_slider.set(-1)

        vid_player = TkinterVideo(master=self.display_frame, scaled=True, keep_aspect=True, consistant_frame_rate=True,
                                  bg="black", width=50, height=10)
        vid_player.set_resampling_method(1)
        vid_player.grid(row=2, column=0, columnspan=2, padx=20, pady=20)
        vid_player.bind("<<Duration>>", update_duration)
        vid_player.bind("<<SecondChanged>>", update_scale)
        vid_player.bind("<<Ended>>", video_ended)

        progress_slider = customtkinter.CTkSlider(master=self.display_frame, from_=-1, to=1, number_of_steps=1,
                                                  command=seek)
        progress_slider.set(-1)
        progress_slider.grid(row=3, column=0, padx=10, pady=10)

        play_pause_btn = customtkinter.CTkButton(master=self.display_frame, text="Play ►", command=play_pause)
        play_pause_btn.grid(row=4, column=0, padx=5, pady=5)

        button_1 = customtkinter.CTkButton(master=self.display_frame, text="Open Video", corner_radius=8,
                                           command=lambda: open_video(video_file))
        button_1.grid(row=5, column=0, padx=10, pady=5)

    def video_button_event(self):
        global filename, flag
        if flag != 0:
            for widget in self.display_frame.winfo_children():
                widget.destroy()
        flag = 2
        self.preview_label = customtkinter.CTkLabel(self.display_frame, text="Preview",
                                                    font=customtkinter.CTkFont(size=20, weight="bold"))
        self.preview_label.grid(row=0, column=0, padx=250)
        print("File works")
        filename = filedialog.askopenfilename(title="Select Video file")

        if filename:
            self.preview_video(filename)

    def preview_image(self, path):
        button_image = customtkinter.CTkImage(Image.open(path), size=(400, 350))
        self.image_show = customtkinter.CTkButton(self.display_frame, text="", bg_color="transparent",
                                                  image=button_image, width=400,
                                                  height=350).grid(row=1, column=0, rowspan=2, padx=0, pady=0)

    def show_camera(self):

        flag, frame = cap.read()
        frame = cv2.resize(frame, (400, 350))
        frame = cv2.flip(frame, 1)
        if flag is None:
            print("Major error!")
        elif flag:
            global last_frame
            last_frame = frame.copy()

        pic = cv2.cvtColor(last_frame,
                           cv2.COLOR_BGR2RGB)
        img = Image.fromarray(pic)
        imgtk = ImageTk.PhotoImage(image=img)
        self.display_label.imgtk = imgtk
        self.display_label.configure(image=imgtk)
        self.display_label.after(10, self.show_camera)

    def camera_button_event(self):
        global flag
        if flag != 0:
            for widget in self.display_frame.winfo_children():
                widget.destroy()
        self.preview_label = customtkinter.CTkLabel(self.display_frame, text="Preview",
                                                    font=customtkinter.CTkFont(size=20, weight="bold"))
        self.preview_label.grid(row=0, column=0, padx=250)
        self.display_label = customtkinter.CTkLabel(self.display_frame, text="")
        self.display_label.grid(row=1, column=0, padx=10)
        flag = 3
        print("Camera works")

        self.show_camera()

    def solve_button_event(self):
        global flag
        print("solve works")
        if flag == 3:
            cap.release()
            self.open_cam()
        if flag == 2:
            self.det_vid()
        if flag == 1:
            self.det_img()

    def pdf_button_event(self):
        print("pdf works")
        self.gen_report()

    def enumeration_button_event(self):
        print("enumeration works")
        self.enumeration_plot()

    def accuracy_button_event(self):
        print("accuracy works")
        self.accuracy_plot()

    def det_img(self):
        global current_status, crowd_number, max_accuracy, crowd_status, filename
        image_path = filename
        if image_path == "":
            current_status = "ERROR no image file detected"
            self.text_area.delete(index1=0.0, index2=100.100)
            self.text_area.insert(index=0.0,
                                  text=f"{current_status}\n\n{crowd_number}\n\n{max_accuracy}\n\n{crowd_status}")
            return
        current_status = "People Counted"
        self.text_area.delete(index1=0.0, index2=100.100)
        self.text_area.insert(index=0.0,
                              text=f"{current_status}\n\n{crowd_number}\n\n{max_accuracy}\n\n{crowd_status}")
        # time.sleep(1)
        self.detectByPathImage(image_path)

    def detectByPathImage(self, path):
        global max_count, framex, county, maxim, avg_acc_list, max_avg_acc_list, max_acc, max_avg_acc
        global current_status, crowd_number, max_accuracy, crowd_status

        max_count = 0
        framex = []
        county = []
        maxim = []
        avg_acc_list = []
        max_avg_acc_list = []
        max_acc = 0
        max_avg_acc = 0

        odapi = DetectorAPI()
        threshold = 0.7

        image = cv2.imread(path)
        img = cv2.resize(image, (image.shape[1], image.shape[0]))
        boxes, scores, classes, num = odapi.processFrame(img)
        person = 0
        acc = 0
        for i in range(len(boxes)):

            if classes[i] == 1 and scores[i] > threshold:
                box = boxes[i]
                person += 1
                cv2.rectangle(img, (box[1], box[0]), (box[3], box[2]), (255, 0, 0), 2)  # cv2.FILLED #BGR
                cv2.putText(img, f'P{person, round(scores[i], 2)}', (box[1] - 30, box[0] - 8),
                            cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)  # (75,0,130),
                acc += scores[i]
                if scores[i] > max_acc:
                    max_acc = scores[i]

        if person > max_count:
            max_count = person
        if person >= 1:
            if (acc / person) > max_avg_acc:
                max_avg_acc1 = (acc / person)

        for i in range(20):
            framex.append(i)
            county.append(max_count)
            maxim.append(max_count)
            avg_acc_list.append(max_avg_acc1)
            max_avg_acc_list.append(max_avg_acc1)

        if max_count > 25:
            crowd_status = "crowded"
        else:
            crowd_status = "not crowded"

        self.text_area.delete(index1=0.0, index2=100.100)
        self.text_area.insert(index=0.0,
                              text=f"{current_status}\n\nPeople Counted = {max_count}\n\nMaxiumum Confidence = {max_acc}\n\nRegion = {crowd_status}")

        cv2.imshow("Human Detection from Image", img)

    def det_vid(self):
        global filename

        video_path = filename
        args = argsParser()
        writer = None
        if args['output'] is not None:
            writer = cv2.VideoWriter(args['output'], cv2.VideoWriter_fourcc(*'MJPG'), 10, (600, 600))

        self.detectByPathVideo(video_path, writer)

    def detectByPathVideo(self, path, writer):
        global max_count, framex, county, maxim, avg_acc_list, max_avg_acc_list, max_acc, max_avg_acc
        global current_status, crowd_number, max_accuracy, crowd_status

        max_count = 0
        framex = []
        county = []
        maxim = []
        avg_acc_list = []
        max_avg_acc_list = []
        max_acc = 0
        max_avg_acc = 0

        video = cv2.VideoCapture(path)
        odapi = DetectorAPI()
        threshold = 0.76

        check, frame = video.read()
        if not check:
            print('Video Not Found. Please Enter a Valid Path (Full path of Video Should be Provided).')
            return

        x2 = 0
        while video.isOpened():
            # check is True if reading was successful
            check, frame = video.read()
            if check:
                img = cv2.resize(frame, (800, 500))
                boxes, scores, classes, num = odapi.processFrame(img)
                person = 0
                acc = 0
                for i in range(len(boxes)):
                    # print(boxes)
                    # print(scores)
                    # print(classes)
                    # print(num)
                    # print()
                    if classes[i] == 1 and scores[i] > threshold:
                        box = boxes[i]
                        person += 1
                        cv2.rectangle(img, (box[1], box[0]), (box[3], box[2]), (255, 0, 0), 2)  # cv2.FILLED
                        cv2.putText(img, f'P{person, round(scores[i], 2)}', (box[1] - 30, box[0] - 8),
                                    cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)  # (75,0,130),
                        acc += scores[i]
                        if scores[i] > max_acc:
                            max_acc = scores[i]

                if person > max_count:
                    max_count = person
                county.append(person)
                x2 += 1
                framex.append(x2)
                if person >= 1:
                    avg_acc_list.append(acc / person)
                    if (acc / person) > max_avg_acc:
                        max_avg_acc = (acc / person)
                else:
                    avg_acc_list.append(acc)

                if writer is not None:
                    writer.write(img)

                cv2.imshow("Human Detection from Video", img)
                key = cv2.waitKey(1)
                if key & 0xFF == ord('q'):
                    break
            else:
                break
        video.release()
        for i in range(len(framex)):
            maxim.append(max_count)
            max_avg_acc_list.append(max_avg_acc)
        if max_count > 25:
            crowd_status = "crowded"
        else:
            crowd_status = "not crowded"

        self.text_area.delete(index1=0.0, index2=100.100)
        self.text_area.insert(index=0.0,
                              text=f"{current_status}\n\nPeople Counted = {max_count}\n\nMaxiumum Confidence = {max_acc}\n\nRegion = {crowd_status}")

    def prev_vid(self):

        global filename2
        cap = cv2.VideoCapture(filename2)
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                img = cv2.resize(frame, (800, 500))
                cv2.imshow('Selected Video Preview', img)
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
            else:
                break
        cap.release()
        cv2.destroyAllWindows()

    def open_cam(self):
        args = argsParser()
        global current_status, crowd_number, max_accuracy, crowd_status

        current_status = "detection Complete"
        self.text_area.delete(index1=0.0, index2=100.100)
        self.text_area.insert(index=0.0, text=f"{current_status}\n\n{crowd_number}\n\n{max_accuracy}\n\n{crowd_status}")
        writer = None
        if args['output'] is not None:
            writer = cv2.VideoWriter(args['output'], cv2.VideoWriter_fourcc(*'MJPG'), 10, (600, 600))
        if True:
            self.detectByCamera(writer)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                writer.release()
                cv2.destroyAllWindows()

        # function defined to detect from camera

    def detectByCamera(self, writer):
        global max_count, framex, county, maxim, avg_acc_list, max_avg_acc_list, max_acc, max_avg_acc
        global current_status, crowd_number, max_accuracy, crowd_status

        max_count = 0
        framex = []
        county = []
        maxim = []
        avg_acc_list = []
        max_avg_acc_list = []
        max_acc = 0
        max_avg_acc = 0

        video = cv2.VideoCapture(0)
        odapi = DetectorAPI()
        threshold = 0.95

        x3 = 0
        while True:
            check, frame = video.read()
            img = cv2.resize(frame, (800, 600))
            img = cv2.flip(img, 1)
            boxes, scores, classes, num = odapi.processFrame(img)
            person = 0
            acc = 0
            for i in range(len(boxes)):

                if classes[i] == 1 and scores[i] > threshold:
                    box = boxes[i]
                    person += 1
                    cv2.rectangle(img, (box[1], box[0]), (box[3], box[2]), (255, 0, 0), 2)  # cv2.FILLED
                    cv2.putText(img, f'P{person, round(scores[i], 2)}', (box[1] - 30, box[0] - 8),
                                cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)  # (75,0,130),
                    acc += scores[i]
                    if scores[i] > max_acc:
                        max_acc = scores[i]

            if person > max_count:
                max_count = person

            if writer is not None:
                writer.write(img)
            cv2.imshow("Human Detection from Camera", img)
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break

            county.append(person)
            x3 += 1
            framex.append(x3)
            if person >= 1:
                avg_acc_list.append(acc / person)
                if (acc / person) > max_avg_acc:
                    max_avg_acc = (acc / person)
            else:
                avg_acc_list.append(acc)

        video.release()

        for i in range(len(framex)):
            maxim.append(max_count)
            max_avg_acc_list.append(max_avg_acc)

        if max_count > 25:
            crowd_status = "crowded"
        else:
            crowd_status = "not crowded"

        self.text_area.delete(index1=0.0, index2=100.100)
        self.text_area.insert(index=0.0,
                              text=f"{current_status}\n\nPeople Counted = {max_count}\n\nMaximum Confidence = {max_acc}"
                                   f"\n\nRegion = {crowd_status}")

    def enumeration_plot(self):
        plt.figure(facecolor='orange', )
        ax = plt.axes()
        ax.set_facecolor("yellow")
        plt.plot(framex, county, label="Human Count", color="green", marker='o', markerfacecolor='blue')
        plt.plot(framex, maxim, label="Max. Human Count", linestyle='dashed', color='fuchsia')
        plt.xlabel('Time (milliseconds)')
        plt.ylabel('Human Count')
        plt.legend()
        plt.title("Enumeration Plot")
        plt.show()

    def accuracy_plot(self):
        plt.figure(facecolor='orange', )
        ax = plt.axes()
        ax.set_facecolor("yellow")
        plt.plot(framex, avg_acc_list, label="Avg. Accuracy", color="green", marker='o',
                 markerfacecolor='blue')
        plt.plot(framex, max_avg_acc_list, label="Max. Avg. Accuracy", linestyle='dashed', color='fuchsia')
        plt.xlabel('Time (milliseconds)')
        plt.ylabel('Avg. Accuracy')
        plt.title('Avg. Accuracy Plot')
        plt.legend()
        plt.show()

    def gen_report(self):
        pdf = FPDF(orientation='P', unit='mm', format='A4')
        pdf.add_page()
        pdf.set_font("Arial", "", 20)
        pdf.set_text_color(128, 0, 0)
        pdf.image('Images/Crowd_Report.png', x=0, y=0, w=210, h=297)

        pdf.text(125, 150, str(max_count))
        pdf.text(105, 163, str(max_acc))
        pdf.text(125, 175, str(max_avg_acc))
        if max_count > 25:
            pdf.text(26, 220, "Max. Human Detected is greater than MAX LIMIT.")
            pdf.text(70, 235, "Region is Crowded.")
        else:
            pdf.text(26, 220, "Max. Human Detected is in range of MAX LIMIT.")
            pdf.text(65, 235, "Region is not Crowded.")

        pdf.output('Crowd_Report.pdf')
        self.text_area.delete(index1=0.0, index2=100.100)
        self.text_area.insert(index=0.0,
                              text=f"PDF Generated as Crowd_report.pdf\n\nPeople Counted = {max_count}\n\nMaximum Confidence = {max_acc}"
                                   f"\n\nRegion = {crowd_status}")


if __name__ == "__main__":
    app = App()
    app.mainloop()
