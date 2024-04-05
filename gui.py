import sys
import cv2
from PyQt5.QtCore import QEvent, Qt, pyqtSignal, QThread, QObject, QTimer
from PyQt5.QtGui import QCloseEvent, QImage, QMouseEvent, QPixmap
import PyQt5.QtWidgets as qt
import numpy as np
import selectinwindow as selectinwindow
import os
from scipy import ndimage

class VideoCaptureWorker(QObject):
    frame_ready = pyqtSignal(object)
    change_source = pyqtSignal(int)
    frame=[]
    video_source = 0
    def __init__(self, video_source):
        super().__init__()
        self.capturing = True
        self.cap = cv2.VideoCapture(video_source)
        self.video_source=video_source
    def start_capture(self):
        if not self.capturing:
            #self.cap = cv2.VideoCapture(self.video_source)
            self.capturing=True
        while self.capturing:
            ret, frame = self.cap.read()
            if ret:
                self.frame_ready.emit(frame)
                self.frame=frame
    
        black_frame = np.zeros_like(self.frame)
        self.frame_ready.emit(black_frame)

    def stop_capture(self):
        self.capturing = False
        
    def change_video_source(self, video_source):
        self.cap.release()
        self.video_source = video_source
        self.cap = cv2.VideoCapture(self.video_source)
        self.capturing=True

class CameraApp(qt.QWidget):
    captured=pyqtSignal(str)
    tab_changed = pyqtSignal(int)  # Signal emitted when the tab is changed
    start = pyqtSignal()
    def __init__(self):
        super().__init__()
        self.video_source = 0
        self.worker = VideoCaptureWorker(self.video_source)
        self.thread = QThread()

        self.video_label = qt.QLabel(self)
        #self.video_label.setAlignment(Qt.AlignCenter)

        # Move worker to thread
        self.worker.moveToThread(self.thread)

        # Connect signals and slots
        self.thread.start()
        self.worker.frame_ready.connect(self.update_frame)
        self.tab_changed.connect(self.handle_tab_change)
        self.start.connect(self.worker.start_capture)

        self.frame=[]
        

    def stop_live_feed(self):
        if self.thread.isRunning():
            self.worker.stop_capture()
              
    def change_video_source(self, video_source):
        self.worker.stop_capture()
        self.worker.change_video_source(video_source)
        self.video_source = video_source
    def capture_frame(self,folder):
        cv2.imwrite(folder, self.frame)
        self.captured.emit("Image captured")

    def update_frame(self, frame):
        if frame.shape[0]!=0:
            height, width, channel = frame.shape
            bytes_per_line = 3 * width

            # Create QImage with correct format and dimensions
            image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_BGR888)
            
            # Scale pixmap to fit QLabel
            pixmap = QPixmap.fromImage(image).scaled(self.video_label.size(), Qt.KeepAspectRatio)

            # Set the scaled pixmap to QLabel
            self.video_label.setPixmap(pixmap)
            self.frame=frame
            #cv2.imshow("frame", frame)
        

    def handle_tab_change(self, index):
        if index == 1 :
            self.worker.stop_capture()
            black_frame = np.zeros_like(self.frame)
            self.worker.frame_ready.emit(black_frame)
            
  
    def cleanup(self):
        self.thread.quit()
        self.thread.wait()

class AnalysisApp(qt.QWidget):
    tab_changed = pyqtSignal(int)  # Signal emitted when the tab is changed
    issue = pyqtSignal(str)
    def __init__(self):
        super().__init__()
        self.scale_factor = 1
        self.a_final_layout = qt.QHBoxLayout(self)
        self.a_left_layout = qt.QVBoxLayout(self)

        self.start_stop_button = qt.QPushButton("Analyze", self)


        self.right_angle_label = qt.QLabel("Right Angle", self)
        self.left_angle_label = qt.QLabel("Left Angle", self)
        self.right_angle_display = qt.QLabel("", self)
        self.left_angle_display = qt.QLabel("", self)

        # self.std_label = qt.QLabel("Standard Deviation", self)
        # self.std_display = qt.QLabel("", self) 

        self.mean_label = qt.QLabel("Mean", self)
        self.mean_display = qt.QLabel("", self)
        
        self.correction_label = qt.QLabel("Correction", self)
        self.correction_display = qt.QLineEdit("", self)

        self.image_label = qt.QLabel("Image to analyze path")
        self.image_entry = qt.QLineEdit()
        self.capture_button = qt.QPushButton("Capture Image", self)

        layout = qt.QVBoxLayout()

        # self.canny1_label = qt.QLabel("Canny1")
        # self.canny1_entry = qt.QLineEdit()
        # self.canny1_entry.setText("50")
        # layout.addWidget(self.canny1_label)
        # layout.addWidget(self.canny1_entry)

        # self.canny2_label = qt.QLabel("Canny2")
        # self.canny2_entry = qt.QLineEdit()
        # self.canny2_entry.setText("150")
        # layout.addWidget(self.canny2_label)
        # layout.addWidget(self.canny2_entry)

        self.blue_label = qt.QLabel("Blue Threshold")
        self.blue_entry = qt.QLineEdit()
        self.blue_entry.setText("150")
        layout.addWidget(self.blue_label)
        layout.addWidget(self.blue_entry)

        self.blur_size_label = qt.QLabel("Blur Size")
        self.blur_size_entry = qt.QLineEdit()
        self.blur_size_entry.setText("7")
        layout.addWidget(self.blur_size_label)
        layout.addWidget(self.blur_size_entry)

        self.aperture_size_label = qt.QLabel("Aperture Size")
        self.aperture_size_entry = qt.QLineEdit()
        self.aperture_size_entry.setText("3")
        layout.addWidget(self.aperture_size_label)
        layout.addWidget(self.aperture_size_entry)

        self.L2gradient = qt.QCheckBox("L2gradient")
        self.L2gradient.setChecked(True)
        layout.addWidget(self.L2gradient)

        self.update_param_button = qt.QPushButton("Update Parameters", self)
        layout.addWidget(self.update_param_button)
        self.update_param_button.clicked.connect(self.cv_param_changed)

        self.a_left_layout.addLayout(layout)
        self.a_left_layout.addWidget(self.right_angle_label)
        self.a_left_layout.addWidget(self.right_angle_display)
        self.a_left_layout.addWidget(self.left_angle_label)
        self.a_left_layout.addWidget(self.left_angle_display)
        # self.a_left_layout.addWidget(self.std_label)
        # self.a_left_layout.addWidget(self.std_display)
        self.a_left_layout.addWidget(self.mean_label)
        self.a_left_layout.addWidget(self.mean_display)
        self.a_left_layout.addWidget(self.correction_label)
        self.a_left_layout.addWidget(self.correction_display)
        self.a_left_layout.addWidget(self.start_stop_button)
        self.a_left_layout.addWidget(self.image_label)
        self.a_left_layout.addWidget(self.image_entry)
        self.a_left_layout.addWidget(self.capture_button)
        
        self.a_right_layout = qt.QLabel(self)
        sizePolicy = qt.QSizePolicy(qt.QSizePolicy.Expanding, qt.QSizePolicy.Expanding)
        self.a_right_layout.setSizePolicy(sizePolicy)
        self.a_final_layout.addLayout(self.a_left_layout)
        self.a_final_layout.addWidget(self.a_right_layout,stretch=70)
        self.layout=self.a_final_layout
        
        self.start_stop_button.clicked.connect(self.start_or_stop)
        self.capture_button.clicked.connect(self.capture_image)
       

        self.angle_thread = QThread()
        self.wName="OpenCV Window"
        self.tab_changed.connect(self.handle_tab_change)
        self.rectI=None
        
    def update_frame(self, frame):
        f = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # OpenCV uses BGR, PyQt5 uses RGB

       
        out = cv2.resize(f, (0, 0), fx=self.scale_factor, fy=self.scale_factor)
       
        h, w, ch = out.shape
        bytes_per_line = ch * w
        qt_image = QImage(out.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)

        # Update the layout with the pixmap
        self.a_right_layout.setPixmap(pixmap)

        # Update the dimensions of the selection rectangle
        self.rectI.keepWithin.w = int(self.a_right_layout.width() / self.scale_factor)
        self.rectI.keepWithin.h = int(self.a_right_layout.height() / self.scale_factor)

        self.a_right_layout.setAlignment(Qt.AlignTop | Qt.AlignLeft)

    def start_analysis(self):
        print("start analysis")
        image_path=self.image_entry.text()
        image_path=image_path.replace('"', '')
        if image_path=="":
            self.issue.emit("Please enter a valid image path")
            return
        self.image=cv2.imread(image_path)
        self.scale_factor = min(self.a_right_layout.width() / self.image.shape[1], self.a_right_layout.height() / self.image.shape[0])
        self.rectI=selectinwindow_v4.DragRectangle(self.image, self.wName, self.image.shape[0], self.image.shape[1])
    
        self.rectI.new_angle.connect(self.update_angles)
        self.rectI.new_frame.connect(self.update_frame)
        self.rectI.sig.connect(self.print_event)
        self.correction_display.textChanged.connect(self.correction_changed)
        self.correction_display.setText("0")
        self.event = self.event
        self.update_frame(self.rectI.image)
    
        self.start_stop_button.setText("Stop")
        self.rectI.update_param()

    def print_event(self,str):
        print(str)
    def event(self, event):
        if event.type() in (QEvent.MouseButtonPress, QEvent.MouseButtonRelease, QEvent.MouseMove):
            x,y=event.x()-self.a_right_layout.geometry().x(),event.y()-self.a_right_layout.geometry().y()
            x = int(x/self.scale_factor)
            y = int(y/self.scale_factor)
            if self.rectI is not None:
                selectinwindow_v4.dragrect(event,x,y, self.rectI)
            return super().event(event)
        return super().event(event)
    def stop_analysis(self):
        self.angle_thread.quit()
        try:
            cv2.destroyWindow(self.wName)
        except:
            pass
    def start_or_stop(self):
        if self.start_stop_button.text() == "Analyze":
            self.start_analysis()
            
        else:
            self.stop_analysis()
            self.start_stop_button.setText("Analyze")

    def correction_changed(self):
        if self.correction_display.text() not in ["","-"]:
            new_im=ndimage.rotate(self.image,float(self.correction_display.text()))
            width, height,_ = new_im.shape   # Get dimensions
            left = int((width - self.rectI.keepWithin.w)/2)
            top = int((height - self.rectI.keepWithin.h)/2)
            right = int((width + self.rectI.keepWithin.w)/2)
            bottom = int((height + self.rectI.keepWithin.h)/2)
            # Crop the center of the image
            new_im = new_im[left:right,top:bottom] 
            self.rectI.image=new_im

    def cv_param_changed(self):
        if self.rectI is not None:
            # self.rectI.canny1=int(self.canny1_entry.text())
            # self.rectI.canny2=int(self.canny2_entry.text())
            self.rectI.blur_size=int(self.blur_size_entry.text())
            self.rectI.apertureSize=int(self.aperture_size_entry.text())
            self.rectI.L2gradient=self.L2gradient.isChecked()
            self.rectI.blue_thresh=int(self.blue_entry.text())
            self.rectI.update_param()
            
    def handle_tab_change(self, index):
        if index == 0 :
            self.stop_analysis()
            self.start_stop_button.setText("Analyze")
              
    def update_angles(self,angles):
        a_l,a_r=angles
        a_l_formatted = "{:.2f}".format(a_l) if a_l is not None else None
        a_r_formatted = "{:.2f}".format(a_r) if a_r is not None else None
        self.right_angle_display.setText(str(a_r_formatted))
        self.left_angle_display.setText(str(a_l_formatted))
        std=np.std(np.array(angles))
        #std_formatted = "{:.2f}".format(std) if a_r is not None else None
        mean=np.mean(np.array(angles))
        mean_formatted = "{:.2f}".format(mean) if a_r is not None else None
        self.mean_display.setText(str(mean_formatted))
        if std < 1:
            self.mean_display.setStyleSheet("QLabel { border: 2px solid green; }")
        else:
            self.mean_display.setStyleSheet("QLabel { border: 2px solid red; }")

    def capture_image(self):
        if self.image_entry.text() == "":
            self.issue.emit("Please enter a file path")
            return
        im_name=self.image_entry.text()
        base_path, extension = os.path.splitext(im_name)
        string_to_add = self.mean_display.text()
        new_file_path = f"{base_path}_{string_to_add}{extension}"
        self.issue.emit("Saving Image at " + new_file_path)
        cv2.imwrite(new_file_path, self.rectI.tmp)
        

    def cleanup(self):
        self.stop_analysis()
        self.angle_thread.quit()

        


class MainApp(qt.QWidget):
    def __init__(self):
        super().__init__()

       
        self.resize(1200, 600)
        self.setMouseTracking(True)
        
        self.final_layout = qt.QHBoxLayout()
        self.left_layout = qt.QVBoxLayout()
        self.tabs = qt.QTabWidget(self)

        self.camera_app = CameraApp()  # Create an instance of CameraApp
        self.right_layout = self.camera_app.video_label
        self.analysis_tab = AnalysisApp()

        self.start_stop_tab = qt.QWidget()
        self.start_stop_tab.layout = qt.QVBoxLayout()
        self.camera_source_label = qt.QLabel("Camera Source")
        self.camera_source_entry = qt.QComboBox()

        

        for i in range(10):
            cap = cv2.VideoCapture(i)
            try:
                if cap.isOpened():
                    self.camera_source_entry.addItem(f"Camera {i}")
                    cap.release()
            except:
                pass

        self.folder_label = qt.QLabel("Folder")
        self.folder_entry = qt.QLineEdit()
        self.start_stop_button = qt.QPushButton("Start Live Feed", self)
        self.capture_button = qt.QPushButton("Capture Image", self)

        self.start_stop_tab.layout.addWidget(self.camera_source_label)
        self.start_stop_tab.layout.addWidget(self.camera_source_entry)
        self.start_stop_tab.layout.addWidget(self.folder_label)
        self.start_stop_tab.layout.addWidget(self.folder_entry)
        self.start_stop_tab.layout.addWidget(self.start_stop_button)
        self.start_stop_tab.layout.addWidget(self.capture_button)
        self.start_stop_tab.setLayout(self.start_stop_tab.layout)

        self.start_stop_button.clicked.connect(self.onstartstopButtonClicked)
        self.capture_button.clicked.connect(lambda : self.camera_app.capture_frame(self.folder_entry.text()))
        
        self.camera_app.captured.connect(self.show_popup)
        self.analysis_tab.issue.connect(self.show_popup)

        self.camera_source_entry.currentIndexChanged.connect(self.camera_app.change_video_source)
        self.tabs.addTab(self.start_stop_tab, "Start/Stop")
        self.tabs.addTab(self.analysis_tab, "Analysis")

        self.left_layout.addWidget(self.tabs)
        sizePolicy = qt.QSizePolicy(qt.QSizePolicy.Preferred, qt.QSizePolicy.Expanding)

        self.right_layout.setSizePolicy(sizePolicy)  # Set size policy for CameraApp

        self.final_layout.addLayout(self.left_layout)
        self.final_layout.addWidget(self.right_layout, stretch=70)  # Use camera_app instead of right_layout

        self.setLayout(self.final_layout)

        self.tabs.currentChanged.connect(self.handle_tab_change)

    def show_popup(self,text):
        msg = qt.QMessageBox()
        msg.setText(text)
        msg.setIcon(qt.QMessageBox.Question)
        msg.setStandardButtons(qt.QMessageBox.Ok|qt.QMessageBox.Cancel)
        msg.exec()

    def handle_tab_change(self, index):
        # Emit the tab_changed signal when the tab changes
        if index==1:
            self.right_layout.hide()
        else:
            self.right_layout.show()
        self.camera_app.tab_changed.emit(index)
        self.analysis_tab.tab_changed.emit(index)
        self.onstartstopButtonClicked(True)

    def onstartstopButtonClicked(self,signal):
        if signal:
            self.start_stop_button.setText('Start Live Feed')
        elif self.start_stop_button.text() == "Start Live Feed":
            self.start_stop_button.setText('Stop Live Feed')
            self.camera_app.start.emit()
        else:
            self.start_stop_button.setText('Start Live Feed')
            self.camera_app.stop_live_feed()
    
    def closeEvent(self,event):
        print("Closing main")
        self.analysis_tab.cleanup() 
        self.camera_app.cleanup()
        event.accept()

    

if __name__ == '__main__':
    app = qt.QApplication(sys.argv)
    main_app = MainApp()
    main_app.show()
    sys.exit(app.exec_())
