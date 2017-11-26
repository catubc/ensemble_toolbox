import numpy as np
import os

from PyQt4 import QtGui, QtCore
from PyQt4.QtCore import *
from PyQt4.QtGui import *

from utils import plot_contours, correct_ROIs, nb_view_patches

class NeuronEditor(QtGui.QWidget):
    def __init__(self, parent):
        super(NeuronEditor, self).__init__(parent)
        
        #self.setGeometry(50, 50, 600, 400)
        self.parent=parent
        self.parent.resize(800, 600)
        self.parent.setWindowTitle("NeuronEditor")
        self.setWindowIcon(QtGui.QIcon('pythonlogo.png'))
                
        #self.parent.root_dir = '/media/cat/12TB/in_vivo/tim/cat/'
        self.root_dir = '/mnt/244f644c-15b8-46be-8f8a-50b5b2d8c6e1/in_vivo/rafa/alejandro/G2M5/20170511/000/'
        #self.parent.root_dir = '/media/cat/All.Data.3TB/in_vivo/tim/cat/2016_05_27_gcamp/tsf_files/'
        #self.parent.root_dir = '/media/cat/All.Data.3TB/in_vivo/nick/ptc21/tr5c/'
        
        #***************************** Cat DATA *************************************
        ##ptc21 tr5c 
        self.file_name = '/mnt/244f644c-15b8-46be-8f8a-50b5b2d8c6e1/in_vivo/rafa/alejandro/G2M5/20170511/000/G2M5_C1V1_GCaMP6s_20170511_000.npz'
        #self.lfp_event_file = '/media/cat/8TB/in_vivo/nick/lfp_clustering/ptc21/tr5c/synch_sort/61-tr5c-blankscreen_alltrack_lfp_50compressed.ptcs'
        #self.lfp_tsf_file = '/media/cat/8TB/in_vivo/nick/lfp_clustering/ptc21/tr5c/synch_sort/61-tr5c-blankscreen_alltrack_lfp.tsf'

        row_index = 0

        layout = QtGui.QGridLayout()

        #Select file_name
        self.button_select_recording = QPushButton('Select File')
        self.button_select_recording.setMaximumWidth(200)
        self.button_select_recording.clicked.connect(self.slct_recording)
        layout.addWidget(self.button_select_recording, row_index, 0)
        
        self.selected_recording  = os.path.split(self.file_name)[1]
        self.select_recording_lbl = QLabel(self.selected_recording, self)
        layout.addWidget(self.select_recording_lbl, row_index,1); row_index+=1

        #Correct ROIs
        self.button_correct_ROIs = QPushButton('Review/Correct ROIs')
        self.button_correct_ROIs.setMaximumWidth(200)
        self.button_correct_ROIs.clicked.connect(self.correctROIs)
        layout.addWidget(self.button_correct_ROIs, row_index, 0); row_index+=1
        
        #
        self.button_plotcontours = QPushButton('View ROIs (original)')
        self.button_plotcontours.setMaximumWidth(200)
        self.button_plotcontours.clicked.connect(self.plotContours)
        layout.addWidget(self.button_plotcontours, row_index, 0); row_index+=1
        
        
        #View traces
        self.button_viewpatches = QPushButton('ROIs/Traces (Bokeh)')
        self.button_viewpatches.setMaximumWidth(200)
        self.button_viewpatches.clicked.connect(self.viewPatches)
        layout.addWidget(self.button_viewpatches, row_index, 0); row_index+=1
        
                
        self.setLayout(layout)
        

        #**** PLOT RESULTS IN BOKEH FROM demo_OnACID_mesoscope output ********
        
        print "... loading default file ..."
        data_in = np.load(self.file_name)

        self.Yr = data_in['Yr']
        print self.Yr.shape

        self.YrA = data_in['YrA']
        print self.YrA.shape

        A = data_in['A']        #Convert array from sparse to dense
        self.A = A[()].toarray()
        print self.A.shape


        self.C = data_in['C']
        self.b = data_in['b']
        self.f = data_in['f']
        self.Cn = data_in['Cn']

        #np.save(file_name[:-4]+"_yarray", Yr)
        

    def slct_recording(self):
        #self.selected_recording =  QtGui.QFileDialog.getOpenFileName(self, 'Load File', self.selected_recording)
        self.lfp_tsf_file =  QtGui.QFileDialog.getOpenFileName(self, ".npz", self.root_dir,"*.npz *.npy")

        path_name, file_name = os.path.split(self.lfp_tsf_file)
        self.select_recording_lbl.setText(file_name)


    def tif_to_npy(self):
        #***** CONVERT TIF to NPY
        pass
        images = tiff.imread(file_name)
        np.save(file_name[:-4], images)

        
    def correctROIs(self):
        #from utils import correct_ROIs
        correct_ROIs(self.file_name, self.A, self.Cn, thr=0.95)


    def plotContours(self):
        plot_contours(self.A, self.Cn, thr=0.9)

        
    def viewPatches(self):
        nb_view_patches(self.file_name, self.Yr, self.A, self.C, self.b, self.f, 250, 250, YrA = self.YrA, thr = 0.8, image_neurons=self.Cn, denoised_color='red')


