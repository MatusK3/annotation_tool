import cv2
import numpy as np
from typing import List
import os
import threading

from skimage import filters, morphology, util, segmentation
from scipy import ndimage as ndi

from PyQt5 import QtWidgets as qtw
from PyQt5 import QtGui as qtg
from PyQt5 import QtCore as qtc

import sys

from scan import Scan, CHANNELS
from utils import create_pixmap_from_image
from widgets.fixedScalePixmapLabel import FixedScalePixmapLabe

from segment_anything import sam_model_registry, SamPredictor

from dataset import Dataset

SIDE_BAR_WIDTH = 250


class CustomDialog(qtw.QDialog):
    form_submitted = qtc.pyqtSignal(dict)

    def __init__(self):
        
        super().__init__()

        self.setWindowTitle("Watershed dialog")
        self.layout = qtw.QFormLayout()



        self.gradient_dropdown = qtw.QComboBox()
        # self.channel_dropdown.setStatusTip("choose channel to display")
        self.gradient_dropdown.addItems(["Sobel", "Canny"])
        self.gradient_dropdown.setCurrentText("Sobel") 
        self.layout.addRow("Gradient type:", self.gradient_dropdown)

        self.gradient_channel_dropdown = qtw.QComboBox()
        # self.channel_dropdown.setStatusTip("choose channel to display")
        self.gradient_channel_dropdown.addItems(["TEXTURE", "NORMALS", "DEPTH_MAP", "ALL"])
        self.gradient_channel_dropdown.setCurrentText("ALL") 
        self.layout.addRow("Gradient channel:", self.gradient_channel_dropdown)


        self.markers_channel_dropdown = qtw.QComboBox()
        # self.channel_dropdown.setStatusTip("choose channel to display")
        self.markers_channel_dropdown.addItems(["TEXTURE", "NORMALS", "DEPTH_MAP"])
        self.markers_channel_dropdown.setCurrentText("TEXTURE") 
        self.layout.addRow("Markers channel:", self.markers_channel_dropdown)


        self.slider_label = qtw.QLabel("0.5")
        self.slider = qtw.QSlider(qtc.Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(100)
        self.slider.setValue(50)
        self.slider.setTickInterval(1)
        self.slider.valueChanged.connect(self.update_slider_label)

        self.layout.addRow("otsu treshold percentage", self.slider)
        self.layout.addWidget(self.slider_label)




        self.buttonBox = qtw.QDialogButtonBox(qtw.QDialogButtonBox.Ok | qtw.QDialogButtonBox.Cancel)
        self.buttonBox.accepted.connect(self.submit_form)
        self.buttonBox.rejected.connect(self.reject)

        self.layout.addWidget(self.buttonBox)


        self.setLayout(self.layout)

    def update_slider_label(self, value):
        self.slider_label.setText(f"{value / 100:.2f}")

    def submit_form(self):     
        form_data = {
            "gradient" : self.gradient_dropdown.currentText(),
            "gradient_channel" : self.gradient_channel_dropdown.currentText(),
            "markers_channel" : self.markers_channel_dropdown.currentText(),
            "percentage" : self.slider.value() / 100
        }

        self.form_submitted.emit(form_data)


class WorkerThread(threading.Thread):
    def __init__(self):
        super().__init__()

        self.runnable = None
        self.arguments = None
        self.result = None

    def run(self):
        self.result = self.runnable(*self.arguments)




class Main_Window(qtw.QMainWindow):
    def __init__(self, widow_stack : qtw.QStackedWidget):
        super().__init__()

        self.widow_stack = widow_stack

        self.set_up_UI()

        self.dataset = Dataset()

        self.show_masks = True
        self.displayed_scan_index = None
        self.displayed_channel = CHANNELS.TEXTURE

        self.setup_SAM()

        self.temp_mask = None
        self.inputs = []
        self.inputs_labels = []
    
    def set_up_UI(self):
        self.centralLayput = qtw.QHBoxLayout()

        self.create_side_bar()
        self.create_image_frame()

        widget = qtw.QWidget()
        widget.setLayout(self.centralLayput)
        self.setCentralWidget(widget)

        self.set_up_menu()
        self.set_up_tool_bar()
        self.set_up_status_bar()

    def create_side_bar(self):
        # side bar
        self.side_bar_widget = qtw.QWidget()
        self.side_bar_widget.setFixedWidth(SIDE_BAR_WIDTH)

        self.side_bar = qtw.QVBoxLayout()
        self.side_bar.setContentsMargins(0,0,0,0)
        self.side_bar_widget.setLayout(self.side_bar)

        self.show_all_masks_button = qtw.QPushButton("show all")
        self.hide_all_masks_button = qtw.QPushButton("hide all")
        self.show_all_masks_button.clicked.connect(self.show_segmentation)
        self.hide_all_masks_button.clicked.connect(self.hide_segmentation)
        self.side_bar.addWidget(self.show_all_masks_button)
        self.side_bar.addWidget(self.hide_all_masks_button)

        # list 
        self.list_widget = qtw.QListWidget()
        self.list_widget.itemClicked.connect(self.clicked_mask)

        scroll_bar = qtw.QScrollBar()

        self.list_widget.setVerticalScrollBar(scroll_bar)

        self.side_bar.addWidget(self.list_widget)

        self.centralLayput.addWidget(self.side_bar_widget)   

    def create_image_frame(self):
        self.image_frame = FixedScalePixmapLabe()
        self.image_frame.setAlignment(qtc.Qt.AlignCenter)
        self.image_frame.setScaledContents(True)
        self.image_frame.installEventFilter(self)
        self.centralLayput.addWidget(self.image_frame)

    def set_up_menu(self):
        self.menu = self.menuBar()
        self.file_menu = self.menu.addMenu("&File")

        self.load_folder_action = qtw.QAction("&Load folder")
        self.load_folder_action.setShortcut("Ctrl+O")
        self.load_folder_action.setStatusTip('Open Foler')
        self.load_folder_action.triggered.connect(self.load_folder)
        self.file_menu.addAction(self.load_folder_action)

        self.save_scan_segmentation_action = qtw.QAction("&Save scans")
        self.save_scan_segmentation_action.setShortcut("Ctrl+S")
        self.save_scan_segmentation_action.setStatusTip('Save scans')
        self.save_scan_segmentation_action.triggered.connect(self.save_scans)
        self.file_menu.addAction(self.save_scan_segmentation_action)

        self.Segmentation_menu = self.menu.addMenu("&Segmentation")


        self.SAM_automatic_segm_action = qtw.QAction("&SAM automatic")
        self.SAM_automatic_segm_action.triggered.connect(self.Start_SAM_automatic)
        self.Segmentation_menu.addAction(self.SAM_automatic_segm_action)

        self.Watershed_segm_action = qtw.QAction("&Watershed")
        self.Watershed_segm_action.triggered.connect(self.Start_Watershed)
        self.Segmentation_menu.addAction(self.Watershed_segm_action)

        self.Compact_watershed_segm_action = qtw.QAction("&Compact watershed")
        self.Compact_watershed_segm_action.triggered.connect(self.Start_Compact_watershed)
        self.Segmentation_menu.addAction(self.Compact_watershed_segm_action)

        self.Felzenszwalb_segm_action = qtw.QAction("&Felzenszwalb")
        self.Felzenszwalb_segm_action.triggered.connect(self.Start_Felzenszwalb)
        self.Segmentation_menu.addAction(self.Felzenszwalb_segm_action)

        self.Quickshift_segm_action = qtw.QAction("&Quickshift")
        self.Quickshift_segm_action.triggered.connect(self.Start_Quickshift)
        self.Segmentation_menu.addAction(self.Quickshift_segm_action)

        self.SLIC_segm_action = qtw.QAction("&SLIC")
        self.SLIC_segm_action.triggered.connect(self.Start_SLIC)
        self.Segmentation_menu.addAction(self.SLIC_segm_action)

        # self.SAM_semiautomatic_window_action = qtw.QAction("&SAM semiuatomatic")
        # self.SAM_semiautomatic_window_action.setShortcut("Ctrl+1")
        # self.SAM_semiautomatic_window_action.setStatusTip('Open SAM semiuatomatic segmentation window')
        # self.SAM_semiautomatic_window_action.triggered.connect(self.Start_SAM_semiautomatic)
        # self.Segmentation_menu.addAction(self.SAM_semiautomatic_window_action)

    def set_up_tool_bar(self):
        self.toolbar = qtw.QToolBar("Toolbar")
        self.addToolBar(self.toolbar)

        # SCAN NUMBER SELECTION
        self.scan_number_spin_box_label = qtw.QLabel(f"Scan: /{0}")
        self.toolbar.addWidget(self.scan_number_spin_box_label)

        self.scan_number_spin_box = qtw.QSpinBox()
        self.scan_number_spin_box.setRange(0, 0) 
        self.scan_number_spin_box.valueChanged.connect(self.changed_displayed_scan_number)
        # self.channel_dropdown.currentTextChanged.connect( 
        #     lambda channel_name : self.display_image(CHANNELS[channel_name], self.displayed_scan_index)
        # )
        self.toolbar.addWidget(self.scan_number_spin_box)

        # CHANNEL SELECTION
        self.channel_dropdown_label = qtw.QLabel("CHANNEL: ")
        self.toolbar.addWidget(self.channel_dropdown_label)

        self.channel_dropdown = qtw.QComboBox()
        # self.channel_dropdown.setStatusTip("choose channel to display")
        self.channel_dropdown.addItems(CHANNELS._member_names_)
        self.channel_dropdown.setCurrentText(CHANNELS.TEXTURE.name) 
        self.channel_dropdown.currentTextChanged.connect( 
            lambda channel_name : self.changed_channel(CHANNELS[channel_name])
        )
        self.toolbar.addWidget(self.channel_dropdown)
        # self.reset_inputs_button = QPushButton("reset inputs")
        # self.reset_inputs_button.clicked.connect(self.reset_inputs)
        # toolbar.addWidget(self.reset_inputs_button)

        # self.predicate_with_SAM_button = QPushButton("predicate with SAM")
        # self.predicate_with_SAM_button.clicked.connect(self.predicate_with_SAM)
        # toolbar.addWidget(self.predicate_with_SAM_button)


        # SAM
        self.reset_inputs_button = qtw.QPushButton("reset inputs")
        self.reset_inputs_button.clicked.connect(self.reset_inputs)
        self.toolbar.addWidget(self.reset_inputs_button)

        self.label_text_field = qtw.QLineEdit()
        self.label_text_field.setFixedWidth(200)
        self.toolbar.addWidget(self.label_text_field)

        self.confirm_mask_button = qtw.QPushButton("Confirm mask")
        self.confirm_mask_button.clicked.connect(self.confirm_mask)
        self.toolbar.addWidget(self.confirm_mask_button)


    def set_up_status_bar(self):
        self.coordinates_label = qtw.QLabel("Coordinates: ")
        self.statusBar().addWidget(self.coordinates_label)

    # def Start_SAM_semiautomatic(self):
    #     if self.dataset.get_number_of_scans == 0:
    #         return



    def Start_SAM_automatic(self):
        ...

    def Start_Watershed(self):
        self.dialog = CustomDialog()
        self.dialog.form_submitted.connect(self.add_new_segmentation)
        self.dialog.show()
        
    def Start_Compact_watershed(self):
        ...
        
    def Start_Felzenszwalb(self):
        ...
        
    def Start_Quickshift(self):
        ...
        
    def Start_SLIC(self):
        ...
        

    @qtc.pyqtSlot(dict)
    def add_new_segmentation(self, dialog_data):
        print(dialog_data)

        if self.dataset.get_number_of_scans() == 0:
            return

        scan = self.dataset.get_scan(self.displayed_scan_index)

        if dialog_data["gradient_channel"] == "ALL":
            gradient = np.zeros(scan.get_channel(CHANNELS.TEXTURE).shape[:2], dtype=np.float32)
            for channel in (CHANNELS.TEXTURE, CHANNELS.NORMALS, CHANNELS.DEPTH_MAP):
                img = scan.get_channel(channel)[:,:,0]
                img = util.img_as_ubyte(img)
                denoised = filters.rank.median(img, morphology.disk(2))
                if dialog_data["gradient"] == "Canny":
                     gradient += cv2.Canny(denoised, 100, 200)
                else:
                    gradient += filters.sobel(denoised)
            gradient = cv2.normalize(gradient, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
        else:
            img = scan.get_channel(CHANNELS[dialog_data["gradient_channel"]])[:,:,0]
            img = util.img_as_ubyte(img)
            denoised = filters.rank.median(img, morphology.disk(2))
            if dialog_data["gradient"] == "Canny":
                gradient = cv2.Canny(denoised, 100, 200)
            else:
                gradient = filters.sobel(denoised)

        img = scan.get_channel(CHANNELS[dialog_data["markers_channel"]])[:,:,0]
        img = util.img_as_ubyte(img)
        denoised = filters.rank.median(img, morphology.disk(2))
        thresh_global = filters.threshold_otsu(img)
        markers = filters.rank.gradient(denoised, morphology.disk(5)) < (thresh_global * dialog_data["percentage"])
        markers = ndi.label(markers)[0]
 
        labels = segmentation.watershed(gradient, markers)

        masks = []
        for i in range(1, np.max(labels)):
            mask = labels == i
            if np.any(mask):
                masks.append(mask)


        scan.delete_masks()
        scan.add_masks(masks, ["" for _ in range(len(masks))])


        # scan = self.dataset.get_scan(self.displayed_scan_index)
        # scan.delete_masks()
        self.show_content()



    def reset_inputs(self):
        self.inputs = []
        self.inputs_labels = []
        self.temp_mask = None

        self.show_content()
        
    def confirm_mask(self):
        self.inputs = []
        self.inputs_labels = []

        self.dataset.get_scan(self.displayed_scan_index).add_mask(self.temp_mask, self.label_text_field.text())

        self.temp_mask = None
        self.show_content()

    def set_predicators(self):
        scan = self.dataset.get_scan(self.displayed_scan_index)
        self.sam_predicator.set_image(scan.get_channel(self.displayed_channel))
        self.sam_predicate()


    

    def setup_SAM(self):
        sam_checkpoint = "trained_models/sam_vit_h_4b8939.pth"
        model_type = "vit_h"
        device = "cuda"

        self.sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        self.sam.to(device=device)


        self.sam_predicator = SamPredictor(self.sam)




    def check_loader(self, timer, loader):
        print("check")
        if not loader.is_alive():
            timer.stop()
            
            result = loader.result

            if result:
                print("loading finished")
                self.scan_number_spin_box.setRange(0, self.dataset.get_number_of_scans() - 1) 
                self.scan_number_spin_box_label.setText(f"Scan: /{self.dataset.get_number_of_scans() - 1}") 

                self.scan_number_spin_box.setValue(0)
                self.changed_displayed_scan_number()


    def load_folder(self): 
        folder = qtw.QFileDialog.getExistingDirectory(
            self, 
            'Sellect a directory',
            "C:/MATFYZ/bakalarka/datasets/SAM_data",
            qtw.QFileDialog.ShowDirsOnly
        )

        
        loader = WorkerThread()
        loader.runnable = self.dataset.load
        loader.arguments = [folder]

        timer = qtc.QTimer()
        timer.timeout.connect(lambda: self.check_loader(timer, loader))
        
        loader.start()
        timer.start(500)

        print("loading stated")

       
    def save_scans(self):
        if self.dataset.get_number_of_scans() == 0:
            return

        save_folder = qtw.QFileDialog.getExistingDirectory(
            self, 
            'Sellect a directory',
            self.loaded_folder,
            qtw.QFileDialog.ShowDirsOnly
        )

        self.dataset.save(save_folder)

    def changed_displayed_scan_number(self):
        self.displayed_scan_index = self.scan_number_spin_box.value() 

        self.set_predicators()

        self.show_content()

    

    def changed_channel(self, channel):
        self.displayed_channel = channel

        self.set_predicators()

        self.display_image()

    def show_content(self):
        self.display_image()
        self.display_mask_items()
        
    def display_mask_items(self):
        self.list_widget.clear()
        scan : Scan = self.dataset.get_scan(self.displayed_scan_index)
        for index in range(len(scan.masks)):
            mask, label = scan.masks[index]

            mask_image = np.zeros(scan.get_shape(), dtype=np.uint8)
            mask_image[mask] = scan.masks_colors[index]
            mask_pixmap = create_pixmap_from_image(mask_image)

            item = qtw.QListWidgetItem()
            item.setData(qtc.Qt.UserRole, index)
            item_widget = qtw.QWidget()

            # item_widget.setFixedWidth(int(SIDE_BAR_WIDTH * 0.8))
            item_widget.setFixedWidth(SIDE_BAR_WIDTH - 30)
            item_widget.setFixedHeight(70)

            item_layout = qtw.QHBoxLayout()
            item_layout.setContentsMargins(0,0,0,0)
            item_widget.setLayout(item_layout)

            item_index = qtw.QLabel(f"{index:03d}")
            item_layout.addWidget(item_index)

            item_label = FixedScalePixmapLabe()
            item_label.setAlignment(qtc.Qt.AlignCenter)
            item_label.setScaledContents(True)
            item_label.setPixmap(mask_pixmap)
            item_layout.addWidget(item_label)

            # label line edit
            text_field = qtw.QLineEdit(label)
            text_field.textChanged.connect(lambda _, x=index, tf=text_field: self.changed_label(x, tf))
            item_layout.addWidget(text_field)
            

            # button as icon
            item_del_button = qtw.QPushButton()
            pixmapi = getattr(qtw.QStyle, "SP_DialogDiscardButton")
            icon = self.style().standardIcon(pixmapi)
            item_del_button.setIcon(icon)
            item_del_button.clicked.connect(lambda _, x=index: self.dellete_mask(x))
            item_layout.addWidget(item_del_button)
            
            # item.setSizeHint(item_widget.sizeHint())
            item.setSizeHint(item_widget.size())
            self.list_widget.addItem(item)
            self.list_widget.setItemWidget(item, item_widget)
    
    def show_segmentation(self):
        self.show_masks = True
        self.display_image()

    def hide_segmentation(self):
        self.show_masks = False
        self.display_image()

    def changed_label(self, index, text_field):
        self.dataset.get_scan(self.displayed_scan_index).change_mask_label(index, text_field.text())

    def dellete_mask(self, index):
        self.dataset.get_scan(self.displayed_scan_index).dellete_mask(index)

        self.show_content()

    def clicked_mask(self, item):
        # index = item.data(qtc.Qt.UserRole)

        # if index in self.show_selected_masks:
        #     self.show_masks.remove(index)
        # else:
        #     self.show_masks.add(index)

        # self.display_image()
        ...

    def display_image(self):
        if self.dataset.get_number_of_scans() == 0:
            return 

        scan = self.dataset.get_scan(self.displayed_scan_index)
        if self.show_masks:
            image = scan.get_masked_channel(self.displayed_channel)
        else:
            image = scan.get_channel(self.displayed_channel)


        if(self.temp_mask is not None):
            color_index = len(scan.masks)
            image[self.temp_mask] = scan.masks_colors[color_index]

        color = (255, 0, 0)
        radius = 10
        for index in range(len(self.inputs)):
            cv2.circle(image, self.inputs[index], radius, color, -1 if self.inputs_labels[index] else 2)


    
        pixmap = create_pixmap_from_image(image)
        self.image_frame.setPixmap(pixmap)

        pixmap = create_pixmap_from_image(image)
        self.image_frame.setPixmap(pixmap)
        # QtGui.QGuiApplication.processEvents()

    def sam_predicate(self):
        print(self.inputs)
        if(len(self.inputs) == 0):
            return
        
        print("prdicting")
        masks, scores, logits = self.sam_predicator.predict(
            point_coords = np.array(self.inputs),
            point_labels = np.array(self.inputs_labels),
            multimask_output = False,
        )

        self.temp_mask = masks[0]

        print(self.temp_mask.shape)


    def eventFilter(self, source, event):
        # if the source is our QLabel, it has a valid pixmap, and the event is
        # a left click, proceed in trying to get the event position
        if (source == self.image_frame and source.pixmap() and not source.pixmap().isNull() and
            event.type() == qtc.QEvent.MouseButtonPress and
            (
                event.buttons() == qtc.Qt.LeftButton or 
                event.buttons() == qtc.Qt.RightButton
            )
        ):
            self.handle_mouse_click(self.getClickedPosition(event.pos()), event.buttons() == qtc.Qt.LeftButton)
                
        return super().eventFilter(source, event)
    
    def handle_mouse_click(self, pos, is_left_button):
        if pos is None:
            self.coordinates_label.setText(f"Coordinates:")
            return
        self.coordinates_label.setText(f"Coordinates: ({pos.x()}, {pos.y()})")

        self.inputs.append([pos.x(), pos.y()])
        self.inputs_labels.append(1 if is_left_button else 0)

        self.sam_predicate()

        self.display_image()

        # self.input_points.append([pos.x(), pos.y()])
        # self.input_labels.append(1 if is_left_button else 0)

        # self.display_image(self.displayed_channel)

    # https://stackoverflow.com/questions/59611751/pyqt5-image-coordinates
    def getClickedPosition(self, pos):
        # consider the widget contents margins
        # contentsRect = QtCore.QRectF(self.image_frame.contentsRect())
        contentsRect = self.image_frame.get_pixmap_ret()
        if pos not in contentsRect:
            # outside widget margins, ignore!
            return

        # adjust the position to the contents margins
        pos -= contentsRect.topLeft()

        pixmapRect = self.image_frame.pixmap().rect()
        if self.image_frame.hasScaledContents():
            x = pos.x() * pixmapRect.width() / contentsRect.width()
            y = pos.y() * pixmapRect.height() / contentsRect.height()
            pos = qtc.QPoint(int(x), int(y))

        else:
            align = self.image_frame.alignment()
            # for historical reasons, QRect (which is based on integer values),
            # returns right() as (left+width-1) and bottom as (top+height-1),
            # and so their opposite functions set/moveRight and set/moveBottom
            # take that into consideration; using a QRectF can prevent that; see:
            # https://doc.qt.io/qt-5/qrect.html#right
            # https://doc.qt.io/qt-5/qrect.html#bottom
            pixmapRect = qtc.QRectF(pixmapRect)

            # the pixmap is not left aligned, align it correctly
            if align & qtc.Qt.AlignRight:
                pixmapRect.moveRight(contentsRect.x() + contentsRect.width())
            elif align & qtc.Qt.AlignHCenter:
                pixmapRect.moveLeft(contentsRect.center().x() - pixmapRect.width() / 2)
            # the pixmap is not top aligned (note that the default for QLabel is
            # Qt.AlignVCenter, the vertical center)
            if align & qtc.Qt.AlignBottom:
                pixmapRect.moveBottom(contentsRect.y() + contentsRect.height())
            elif align & qtc.Qt.AlignVCenter:
                pixmapRect.moveTop(contentsRect.center().y() - pixmapRect.height() / 2)

            if not pos in pixmapRect:
                # outside image margins, ignore!
                return
            # translate coordinates to the image position and convert it back to
            # a QPoint, which is integer based
            pos = (pos - pixmapRect.topLeft()).toPoint()

        return pos


if __name__ == "__main__":
    app = qtw.QApplication(sys.argv)

    windows_stack = qtw.QStackedWidget()

    windows_stack.resize(1280, 720)
    windows_stack.setWindowTitle("Annotation tool")

    mainWindow = Main_Window(windows_stack)

    windows_stack.addWidget(mainWindow)

    windows_stack.setCurrentWidget(mainWindow)
    windows_stack.show()

    # Execute the application's main event loop
    sys.exit(app.exec_())

