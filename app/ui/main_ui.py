from typing import Dict
from pathlib import Path
import os
from functools import partial
import copy

from PySide6 import QtWidgets, QtGui
from PySide6 import QtCore
import torch

from app.ui.core.main_window import Ui_MainWindow
import app.ui.widgets.actions.common_actions as common_widget_actions
from app.ui.widgets.actions import card_actions
from app.ui.widgets.actions import layout_actions
from app.ui.widgets.actions import video_control_actions
from app.ui.widgets.actions import filter_actions
from app.ui.widgets.actions import save_load_actions
from app.ui.widgets.actions import list_view_actions
from app.ui.widgets.actions import graphics_view_actions

from app.processors.video_processor import VideoProcessor
from app.processors.models_processor import ModelsProcessor
from app.ui.widgets import widget_components
from app.ui.widgets.event_filters import GraphicsViewEventFilter, VideoSeekSliderEventFilter, videoSeekSliderLineEditEventFilter, ListWidgetEventFilter
from app.ui.widgets import ui_workers
from app.ui.widgets.common_layout_data import COMMON_LAYOUT_DATA
from app.ui.widgets.swapper_layout_data import SWAPPER_LAYOUT_DATA
from app.ui.widgets.settings_layout_data import SETTINGS_LAYOUT_DATA
from app.ui.widgets.face_editor_layout_data import FACE_EDITOR_LAYOUT_DATA
from app.helpers.miscellaneous import DFM_MODELS_DATA, ParametersDict
from app.processors.models_data import models_dir as global_models_dir # For UNet model discovery
from app.helpers.typing_helper import FacesParametersTypes, ParametersTypes, ControlTypes, MarkerTypes

ParametersWidgetTypes = Dict[str, widget_components.ToggleButton|widget_components.SelectionBox|widget_components.ParameterDecimalSlider|widget_components.ParameterSlider|widget_components.ParameterText]

class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    placeholder_update_signal = QtCore.Signal(QtWidgets.QListWidget, bool)
    gpu_memory_update_signal = QtCore.Signal(int, int)
    model_loading_signal = QtCore.Signal()
    model_loaded_signal = QtCore.Signal()
    display_messagebox_signal = QtCore.Signal(str, str, QtWidgets.QWidget)
    def initialize_variables(self):
        self.video_loader_worker: ui_workers.TargetMediaLoaderWorker|bool = False
        self.input_faces_loader_worker: ui_workers.InputFacesLoaderWorker|bool = False
        self.target_videos_filter_worker = ui_workers.FilterWorker(main_window=self, search_text='', filter_list='target_videos')
        self.input_faces_filter_worker = ui_workers.FilterWorker(main_window=self, search_text='', filter_list='input_faces')
        self.merged_embeddings_filter_worker = ui_workers.FilterWorker(main_window=self, search_text='', filter_list='merged_embeddings')
        self.video_processor = VideoProcessor(self)
        self.models_processor = ModelsProcessor(self)
        self.target_videos: Dict[int, widget_components.TargetMediaCardButton] = {} #Contains button objects of target videos (Set as list instead of single video to support batch processing in future)
        self.target_faces: Dict[int, widget_components.TargetFaceCardButton] = {} #Contains button objects of target faces
        self.input_faces: Dict[int, widget_components.InputFaceCardButton] = {} #Contains button objects of source faces (images)
        self.merged_embeddings: Dict[int, widget_components.EmbeddingCardButton] = {}
        self.cur_selected_target_face_button: widget_components.TargetFaceCardButton = False
        self.selected_video_button: widget_components.TargetMediaCardButton = False
        self.selected_target_face_id = False
        # '''
            # self.parameters dict have the following structure:
            # {
                # face_id (int): 
                # {
                    # parameter_name: parameter_value,
                    # ------
                # }
                # -----
            # }
        # '''
        self.parameters: FacesParametersTypes = {} 

        self.default_parameters: ParametersTypes = {}
        self.copied_parameters: ParametersTypes = {}
        self.current_widget_parameters: ParametersTypes = {}

        self.markers: MarkerTypes = {} #Video Markers (Contains parameters for each face)
        self.parameters_list = {}
        self.control: ControlTypes = {}
        self.parameter_widgets: ParametersWidgetTypes = {}

        # UNet related
        self.previous_kv_file_selection = "" 
        self.current_kv_tensors_map: Dict[str, torch.Tensor] #| None = None
        self.fixed_unet_model_name = "RefLDM_UNET_EXTERNAL_KV"

        self.loaded_embedding_filename: str = ''
        
        self.last_target_media_folder_path = ''
        self.last_input_media_folder_path = ''

        self.is_full_screen = False
        self.dfm_models_data = DFM_MODELS_DATA
        # This flag is used to make sure new loaded media is properly fit into the graphics frame on the first load

        # Determine project root and actual models directory path
        # main_ui.py is in app/ui/, so project root is 3 levels up.
        self.project_root_path = Path(__file__).resolve().parent.parent.parent
        self.actual_models_dir_path = self.project_root_path / global_models_dir
        self.loading_new_media = False

        self.gpu_memory_update_signal.connect(partial(common_widget_actions.set_gpu_memory_progressbar_value, self))
        self.placeholder_update_signal.connect(partial(common_widget_actions.update_placeholder_visibility, self))
        self.model_loading_signal.connect(partial(common_widget_actions.show_model_loading_dialog, self))
        self.model_loaded_signal.connect(partial(common_widget_actions.hide_model_loading_dialog, self))
        self.display_messagebox_signal.connect(partial(common_widget_actions.create_and_show_messagebox, self))

    def initialize_widgets(self):
        # Initialize QListWidget for target media
        self.targetVideosList.setFlow(QtWidgets.QListWidget.LeftToRight)
        self.targetVideosList.setWrapping(True)
        self.targetVideosList.setResizeMode(QtWidgets.QListWidget.Adjust)

        # Initialize QListWidget for face images
        self.inputFacesList.setFlow(QtWidgets.QListWidget.LeftToRight)
        self.inputFacesList.setWrapping(True)
        self.inputFacesList.setResizeMode(QtWidgets.QListWidget.Adjust)

        # Set up Menu Actions
        layout_actions.set_up_menu_actions(self)

        # Set up placeholder texts in ListWidgets (Target Videos and Input Faces)
        list_view_actions.set_up_list_widget_placeholder(self, self.targetVideosList)
        list_view_actions.set_up_list_widget_placeholder(self, self.inputFacesList)

        # Set up click to select and drop action on ListWidgets
        self.targetVideosList.setAcceptDrops(True)
        self.targetVideosList.viewport().setAcceptDrops(False)
        self.inputFacesList.setAcceptDrops(True)
        self.inputFacesList.viewport().setAcceptDrops(False)
        list_widget_event_filter = ListWidgetEventFilter(self, self)
        self.targetVideosList.installEventFilter(list_widget_event_filter)
        self.targetVideosList.viewport().installEventFilter(list_widget_event_filter)
        self.inputFacesList.installEventFilter(list_widget_event_filter)
        self.inputFacesList.viewport().installEventFilter(list_widget_event_filter)

        # Set up folder open buttons for Target and Input
        self.buttonTargetVideosPath.clicked.connect(partial(list_view_actions.select_target_medias, self, 'folder'))
        self.buttonInputFacesPath.clicked.connect(partial(list_view_actions.select_input_face_images, self, 'folder'))

        # Initialize graphics frame to view frames
        self.scene = QtWidgets.QGraphicsScene()
        self.graphicsViewFrame.setScene(self.scene)
        # Event filter to start playing when clicking on frame
        graphics_event_filter = GraphicsViewEventFilter(self, self.graphicsViewFrame,)
        self.graphicsViewFrame.installEventFilter(graphics_event_filter)

        video_control_actions.enable_zoom_and_pan(self.graphicsViewFrame)

        video_slider_event_filter = VideoSeekSliderEventFilter(self, self.videoSeekSlider)
        self.videoSeekSlider.installEventFilter(video_slider_event_filter)
        self.videoSeekSlider.valueChanged.connect(partial(video_control_actions.on_change_video_seek_slider, self))
        self.videoSeekSlider.sliderPressed.connect(partial(video_control_actions.on_slider_pressed, self))
        self.videoSeekSlider.sliderReleased.connect(partial(video_control_actions.on_slider_released, self))
        video_control_actions.set_up_video_seek_slider(self)
        self.frameAdvanceButton.clicked.connect(partial(video_control_actions.advance_video_slider_by_n_frames, self))
        self.frameRewindButton.clicked.connect(partial(video_control_actions.rewind_video_slider_by_n_frames, self))

        self.addMarkerButton.clicked.connect(partial(video_control_actions.add_video_slider_marker, self))
        self.removeMarkerButton.clicked.connect(partial(video_control_actions.remove_video_slider_marker, self))
        self.nextMarkerButton.clicked.connect(partial(video_control_actions.move_slider_to_next_nearest_marker, self))
        self.previousMarkerButton.clicked.connect(partial(video_control_actions.move_slider_to_previous_nearest_marker, self))

        self.viewFullScreenButton.clicked.connect(partial(video_control_actions.view_fullscreen, self))
        # Set up videoSeekLineEdit and add the event filter to handle changes
        video_control_actions.set_up_video_seek_line_edit(self)
        video_seek_line_edit_event_filter = videoSeekSliderLineEditEventFilter(self, self.videoSeekLineEdit)
        self.videoSeekLineEdit.installEventFilter(video_seek_line_edit_event_filter)

        # Connect the Play/Stop button to the play_video method
        self.buttonMediaPlay.toggled.connect(partial(video_control_actions.play_video, self))
        self.buttonMediaRecord.toggled.connect(partial(video_control_actions.record_video, self))
        # self.buttonMediaStop.clicked.connect(partial(self.video_processor.stop_processing))
        self.findTargetFacesButton.clicked.connect(partial(card_actions.find_target_faces, self))
        self.clearTargetFacesButton.clicked.connect(partial(card_actions.clear_target_faces, self))
        self.targetVideosSearchBox.textChanged.connect(partial(filter_actions.filter_target_videos, self))
        self.filterImagesCheckBox.clicked.connect(partial(filter_actions.filter_target_videos, self))
        self.filterVideosCheckBox.clicked.connect(partial(filter_actions.filter_target_videos, self))
#        self.filterWebcamsCheckBox.clicked.connect(partial(filter_actions.filter_target_videos, self))
#        self.filterWebcamsCheckBox.clicked.connect(partial(list_view_actions.load_target_webcams, self))

        self.inputFacesSearchBox.textChanged.connect(partial(filter_actions.filter_input_faces, self))
        self.inputEmbeddingsSearchBox.textChanged.connect(partial(filter_actions.filter_merged_embeddings, self))
        self.openEmbeddingButton.clicked.connect(partial(save_load_actions.open_embeddings_from_file, self))
        self.saveEmbeddingButton.clicked.connect(partial(save_load_actions.save_embeddings_to_file, self))
        self.saveEmbeddingAsButton.clicked.connect(partial(save_load_actions.save_embeddings_to_file, self, True))

        self.swapfacesButton.clicked.connect(partial(video_control_actions.process_swap_faces, self))
        self.editFacesButton.clicked.connect(partial(video_control_actions.process_edit_faces, self))

        self.saveImageButton.clicked.connect(partial(video_control_actions.save_current_frame_to_file, self))
        self.clearMemoryButton.clicked.connect(partial(common_widget_actions.clear_gpu_memory, self))

        self.parametersPanelCheckBox.toggled.connect(partial(layout_actions.show_hide_parameters_panel, self))
        self.facesPanelCheckBox.toggled.connect(partial(layout_actions.show_hide_faces_panel, self))
        self.mediaPanelCheckBox.toggled.connect(partial(layout_actions.show_hide_input_target_media_panel, self))

        self.faceMaskCheckBox.clicked.connect(partial(video_control_actions.process_compare_checkboxes, self))
        self.faceCompareCheckBox.clicked.connect(partial(video_control_actions.process_compare_checkboxes, self))

        # Split COMMON_LAYOUT_DATA processing: 'UNet Denoiser' as control, others as parameter
        common_parameters_layout_data = {}
        common_controls_layout_data = {}

        for group_name, widgets_in_group in COMMON_LAYOUT_DATA.items():
            # UNet Denoiser group now contains mostly controls
            if group_name == 'ReF-LDM Denoiser':
                common_controls_layout_data[group_name] = widgets_in_group
            else: # Other groups like 'Face Restorer' are parameters
                common_parameters_layout_data[group_name] = widgets_in_group

        if common_parameters_layout_data:
            layout_actions.add_widgets_to_tab_layout(self, LAYOUT_DATA=common_parameters_layout_data, layoutWidget=self.commonWidgetsLayout, data_type='parameter')
        if common_controls_layout_data:
            layout_actions.add_widgets_to_tab_layout(self, LAYOUT_DATA=common_controls_layout_data, layoutWidget=self.commonWidgetsLayout, data_type='control')
        
        layout_actions.add_widgets_to_tab_layout(self, LAYOUT_DATA=SWAPPER_LAYOUT_DATA, layoutWidget=self.swapWidgetsLayout, data_type='parameter')
        layout_actions.add_widgets_to_tab_layout(self, LAYOUT_DATA=SETTINGS_LAYOUT_DATA, layoutWidget=self.settingsWidgetsLayout, data_type='control')
        layout_actions.add_widgets_to_tab_layout(self, LAYOUT_DATA=FACE_EDITOR_LAYOUT_DATA, layoutWidget=self.faceEditorWidgetsLayout, data_type='parameter')

        # Set up output folder select button (It is inside the settings tab Widget)
        self.outputFolderButton.clicked.connect(partial(list_view_actions.select_output_media_folder, self))
        common_widget_actions.create_control(self, 'OutputMediaFolder', '')
        
        # Initialize current_widget_parameters with default values
        self.current_widget_parameters = ParametersDict(copy.deepcopy(self.default_parameters), self.default_parameters)

        # Populate Reference K/V Tensors dropdown (AFTER connecting the signal)
        self._populate_reference_kv_tensors()
        
        # Initialize the button states
        video_control_actions.reset_media_buttons(self)

        # Set GPU Memory Progressbar
        font = self.vramProgressBar.font()
        font.setBold(True)
        self.vramProgressBar.setFont(font)
        common_widget_actions.update_gpu_memory_progressbar(self)
        # Set face_swap_tab as the default focused tab
        self.tabWidget.setCurrentIndex(0)
        # widget_actions.add_groupbox_and_widgets_from_layout_map(self)

        # Connect Denoiser Mode SelectionBox signals to update visibility
        denoiser_mode_before_combo = self.parameter_widgets.get('DenoiserModeSelectionBefore')
        if denoiser_mode_before_combo:
            # Pass the new text (current_mode_text) from the signal to the handler
            denoiser_mode_before_combo.currentTextChanged.connect(
                lambda text, ps="Before": self.update_denoiser_controls_visibility_for_pass(ps, text)
            )
            # Initial call using the value from self.control, which should be the default
            initial_mode_before = self.control.get('DenoiserModeSelectionBefore', "Single Step (Fast)")
            self.update_denoiser_controls_visibility_for_pass("Before", initial_mode_before)

        denoiser_mode_after_first_combo = self.parameter_widgets.get('DenoiserModeSelectionAfterFirst')
        if denoiser_mode_after_first_combo:
            denoiser_mode_after_first_combo.currentTextChanged.connect(
                lambda text, ps="AfterFirst": self.update_denoiser_controls_visibility_for_pass(ps, text)
            )
            initial_mode_after_first = self.control.get('DenoiserModeSelectionAfterFirst', "Single Step (Fast)")
            self.update_denoiser_controls_visibility_for_pass("AfterFirst", initial_mode_after_first)

        denoiser_mode_after_combo = self.parameter_widgets.get('DenoiserModeSelectionAfter')
        if denoiser_mode_after_combo:
            denoiser_mode_after_combo.currentTextChanged.connect(
                lambda text, ps="After": self.update_denoiser_controls_visibility_for_pass(ps, text)
            )
            initial_mode_after = self.control.get('DenoiserModeSelectionAfter', "Single Step (Fast)")
            self.update_denoiser_controls_visibility_for_pass("After", initial_mode_after)

    def update_denoiser_controls_visibility_for_pass(self, pass_suffix: str, current_mode_text: str):
        """
        Updates visibility of denoiser controls for a specific pass (Before, AfterFirst, After)
        based on the provided current_mode_text.
        """
        # current_mode = self.control.get(mode_selection_control_name, "Single Step (Fast)") # Old way
        current_mode = current_mode_text # Use the passed text directly

        # Define widget names based on the pass_suffix
        single_step_slider_name = f'DenoiserSingleStepTimestepSlider{pass_suffix}'
        ddim_steps_slider_name = f'DenoiserDDIMStepsSlider{pass_suffix}'
        cfg_scale_slider_name = f'DenoiserCFGScaleDecimalSlider{pass_suffix}'

        # Get widget instances from self.parameter_widgets
        single_step_widget = self.parameter_widgets.get(single_step_slider_name)
        ddim_steps_widget = self.parameter_widgets.get(ddim_steps_slider_name)
        cfg_scale_widget = self.parameter_widgets.get(cfg_scale_slider_name)

        # Helper to set visibility for a widget and its associated label and reset button
        def set_widget_visibility(widget_instance, is_visible):
            if widget_instance:
                widget_instance.setVisible(is_visible)
                if hasattr(widget_instance, 'label_widget') and widget_instance.label_widget:
                    widget_instance.label_widget.setVisible(is_visible)
                if hasattr(widget_instance, 'reset_default_button') and widget_instance.reset_default_button:
                    widget_instance.reset_default_button.setVisible(is_visible)
                if hasattr(widget_instance, 'line_edit') and widget_instance.line_edit:
                    widget_instance.line_edit.setVisible(is_visible)

        # Set visibility for Single Step controls
        is_single_step_mode = (current_mode == "Single Step (Fast)")
        set_widget_visibility(single_step_widget, is_single_step_mode)

        # Set visibility for Full Restore (DDIM) controls
        is_full_restore_mode = (current_mode == "Full Restore (DDIM)")
        set_widget_visibility(ddim_steps_widget, is_full_restore_mode)
        set_widget_visibility(cfg_scale_widget, is_full_restore_mode)

    def _populate_reference_kv_tensors(self):
        kv_tensor_files = []
        # Define the directory for K/V tensor files
        # global_models_dir points to 'model_assets'
        kv_tensors_dir = os.path.join(global_models_dir, "reference_kv_data")

        if os.path.exists(kv_tensors_dir):
            for f_name in os.listdir(kv_tensors_dir):
                kv_tensor_files.append(f_name)
        
        kv_tensor_files.sort() # Sort alphabetically for consistent order

        # Assuming the widget control name is 'ReferenceKVTensorsSelection'
        kv_tensor_widget = self.parameter_widgets.get("ReferenceKVTensorsSelection")
        if kv_tensor_widget and isinstance(kv_tensor_widget, widget_components.SelectionBox):
            current_selection_in_control = self.control.get("ReferenceKVTensorsSelection")
            kv_tensor_widget.clear()

            if kv_tensor_files:
                kv_tensor_widget.addItems(kv_tensor_files)
                
                if not current_selection_in_control or current_selection_in_control not in kv_tensor_files:
                    new_selection = kv_tensor_files[0]
                    self.control["ReferenceKVTensorsSelection"] = new_selection
                    kv_tensor_widget.setCurrentText(new_selection)
                else:
                    kv_tensor_widget.setCurrentText(current_selection_in_control)
            else:
                kv_tensor_widget.addItem("No K/V Tensors found")
                self.control["ReferenceKVTensorsSelection"] = "" # No file selected
                kv_tensor_widget.setCurrentText("No K/V Tensors found")
    
    def handle_reference_kv_file_change(self, new_kv_file_name: str): 

        with self.models_processor.model_lock: # Protect access to shared K/V map attributes
            # Always try to unload/load
            self.current_kv_tensors_map = None 
            
            self.control['ReferenceKVTensorsSelection'] = new_kv_file_name 
            self.previous_kv_file_selection = new_kv_file_name # Update this under lock
            
            if new_kv_file_name and new_kv_file_name != "No K/V tensor files found":
                # Use the robustly calculated path
                kv_file_path = self.actual_models_dir_path / "reference_kv_data" / new_kv_file_name
                if kv_file_path.exists():
                    try:
                        self.model_loading_signal.emit() 
                        kv_payload = torch.load(kv_file_path, map_location='cpu', weights_only=False) 
                        self.current_kv_tensors_map = kv_payload.get("kv_map")
                        if self.current_kv_tensors_map:
                            print(f"Successfully loaded K/V map from {new_kv_file_name} for {len(self.current_kv_tensors_map)} layers.")
                        else:
                            print(f"Warning: 'kv_map' not found in {new_kv_file_name}.")
                            self.current_kv_tensors_map = None
                        self.model_loaded_signal.emit() 
                    except Exception as e:
                        print(f"Error loading K/V tensor file {kv_file_path}: {e}")
                        self.current_kv_tensors_map = None
                    self.model_loaded_signal.emit() 
                else:
                    print(f"K/V tensor file not found: {kv_file_path}")
                    self.current_kv_tensors_map = None
            else:
                self.current_kv_tensors_map = None

        # Frame refresh logic (outside the lock)
        denoiser_enabled_before = self.control.get('DenoiserUNetEnableBeforeRestorersToggle', False)
        denoiser_enabled_after_first = self.control.get('DenoiserAfterFirstRestorerToggle', False)
        denoiser_enabled_after = self.control.get('DenoiserAfterRestorersToggle', False)

        # Refresh frame if any denoiser is active and K/V selection might have changed its state
        if (denoiser_enabled_before or denoiser_enabled_after_first or denoiser_enabled_after):
            # Trigger refresh if a K/V file was selected/deselected,
            # as this affects whether kv_tensor_map_for_this_run will be None or populated.
            if new_kv_file_name: # True if a file is selected or "No K/V..." is chosen (i.e., selection changed)
                common_widget_actions.refresh_frame(self)
            
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setupUi(self)
        self.initialize_variables()
        self.initialize_widgets()
        self.load_last_workspace()

    def resizeEvent(self, event: QtGui.QResizeEvent):
        # print("Called resizeEvent()")
        super().resizeEvent(event)
        # Call the method to fit the image to the view whenever the window resizes
        if self.scene.items():
            pixmap_item = self.scene.items()[0]
            # Set the scene rectangle to the bounding rectangle of the pixmap
            scene_rect = pixmap_item.boundingRect()
            self.graphicsViewFrame.setSceneRect(scene_rect)
            graphics_view_actions.fit_image_to_view(self, pixmap_item, scene_rect )

    def keyPressEvent(self, event):
        match event.key():
            case QtCore.Qt.Key_F11:
                video_control_actions.view_fullscreen(self)
            case QtCore.Qt.Key_V:
                video_control_actions.advance_video_slider_by_n_frames(self, n=1)
            case QtCore.Qt.Key_C:
                video_control_actions.rewind_video_slider_by_n_frames(self, n=1)
            case QtCore.Qt.Key_D:
                video_control_actions.advance_video_slider_by_n_frames(self, n=30)
            case QtCore.Qt.Key_A:
                video_control_actions.rewind_video_slider_by_n_frames(self, n=30)
            case QtCore.Qt.Key_Z:
                self.videoSeekSlider.setValue(0)
            case QtCore.Qt.Key_Space:
                self.buttonMediaPlay.click()
            case QtCore.Qt.Key_R:
                self.buttonMediaRecord.click()
            case QtCore.Qt.Key_F:
                if event.modifiers() & QtCore.Qt.KeyboardModifier.AltModifier:
                    video_control_actions.remove_video_slider_marker(self)
                else:
                    video_control_actions.add_video_slider_marker(self)
            case QtCore.Qt.Key_W:
                video_control_actions.move_slider_to_nearest_marker(self, 'next')
            case QtCore.Qt.Key_Q:
                video_control_actions.move_slider_to_nearest_marker(self, 'previous')
            case QtCore.Qt.Key_S:
                self.swapfacesButton.click()

    def closeEvent(self, event):
        print("MainWindow: closeEvent called.")

        self.video_processor.stop_processing()
        list_view_actions.clear_stop_loading_input_media(self)
        list_view_actions.clear_stop_loading_target_media(self)

        save_load_actions.save_current_workspace(self, 'last_workspace.json')
        # Optionally handle the event if needed
        event.accept()

    def load_last_workspace(self):
        # Show the load workspace dialog if the file exists
        if Path('last_workspace.json').is_file():
            load_dialog = widget_components.LoadLastWorkspaceDialog(self)
            load_dialog.exec_()
            self._populate_reference_kv_tensors()

    def save_last_workspace(self):
        pass