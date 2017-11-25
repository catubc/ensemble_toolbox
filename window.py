import glob, os, sys

from PyQt4 import QtGui, QtCore
from PyQt4.QtCore import *
from PyQt4.QtGui import *

from neuron_editor import NeuronEditor
#from rat_tools import RatTools

class Window(QtGui.QMainWindow):

    def __init__(self):
        super(Window, self).__init__()
        self.setGeometry(50, 50, 500, 300)
        #self.setWindowTitle("OpenNeuron")
        self.setWindowIcon(QtGui.QIcon('pythonlogo.png'))


        #Set widget to show up with viewbox
        toolMenu = QtGui.QMenuBar()
        toolMenu.setNativeMenuBar(False) # <--Sets the menu with the widget; otherwise shows up as global (i.e. at top desktop screen)
        self.setMenuBar(toolMenu)

        #***** TEXT PARAMETERS FIELDS ******
        self.animal_name_text='' 
        self.rec_name_text=''

        #Load default experiment
        #self.animal.rec_length = self.animal.tsf.n_vd_samples/float(self.animal.tsf.SampleFrequency)

        #Menu Item Lists
        self.make_menu()

        #LOAD CENTRAL WIDGET
        self.central_widget = QtGui.QStackedWidget()
        self.setCentralWidget(self.central_widget)
        
        #SET DEFAULT WIDGET TO PROCESS
        #self.load_widget = Load(self)
        #self.central_widget.addWidget(self.load_widget)
        
        self.show()

    def make_menu(self):
        

        #PROCESSING MENUS
        cnmfCaiman = QtGui.QAction("&CNMF - CaIman", self)
        cnmfCaiman.setStatusTip('CNMF - CaImAn')
        cnmfCaiman.triggered.connect(self.cnmf_caiman)
        

        #REVIEW MENUS
        reviewNeurons = QtGui.QAction("&Review ROIs", self)
        reviewNeurons.setStatusTip('Review ROIs')
        reviewNeurons.triggered.connect(self.review_neurons)
        
        
        #ENSEMBLE MENUS
        viewEnsembles = QtGui.QAction("&Review ROIs", self)
        viewEnsembles.setStatusTip('Review ROIs')
        viewEnsembles.triggered.connect(self.view_ensembles)
        
        exitApplication = QtGui.QAction("&Exit Application", self)
        exitApplication.setStatusTip('Exit')
        exitApplication.triggered.connect(self.close_application)
        
        #MAKE TOP BAR MENUO
        mainMenu = self.menuBar()
        
        fileMenu = mainMenu.addMenu('&File')
        fileMenu.addAction(exitApplication)

        fileMenu = mainMenu.addMenu('Process')
        fileMenu.addAction(cnmfCaiman)

        fileMenu = mainMenu.addMenu('Review')
        fileMenu.addAction(reviewNeurons)

        fileMenu = mainMenu.addMenu('Ensembles')
        fileMenu.addAction(viewEnsembles)


    #********************************************************************************
    #***************************** LOAD FILE MENUS **********************************
    #********************************************************************************
    def cnmf_caiman(self):
        
        #print "... clicked on cnmf_caiman...."

        print "... running cnmf_caiman..."
        return

        self.load_widget.load_mouse(self)   #Pass main widget to subwidgets as it contains needed parameters.

        self.exp_type = 'mouse'

        #RESTART Process widget with updated info; SEEMS THERE IS A BETTER WAY TO DO THIS
        self.load_widget = Neuron(self)
        self.central_widget.addWidget(self.load_widget)
        self.central_widget.setCurrentWidget(self.load_widget)


    def review_neurons(self):
        
        print "... clicked on view_patches...."
        
        #self.load_widget.load_mouse_lever(self)   #Pass main widget to subwidgets as it contains needed parameters.

        #self.exp_type = 'mouse_lever'

        #RESTART Process widget with updated info; SEEMS THERE IS A BETTER WAY TO DO THIS
        self.load_widget = NeuronEditor(self)
        self.central_widget.addWidget(self.load_widget)
        self.central_widget.setCurrentWidget(self.load_widget)
    
    def view_ensembles(self):
        
        print "... clicked on view_patches.... NOT IMPLEMENTED"
        
        #RESTART Process widget with updated info; SEEMS THERE IS A BETTER WAY TO DO THIS
        #self.load_widget = NeuronEditor(self)
        #self.central_widget.addWidget(self.load_widget)
        #self.central_widget.setCurrentWidget(self.load_widget)

                
    def close_application(self):
        print("...clean exit, good job! ")
       
        sys.exit()


    #************************************************************************
    #************************* EXP TOOLS MENUS ******************************
    #************************************************************************
    def ophys_tools(self):
        ophys_widget = EventTriggeredImaging(self)
        self.central_widget.addWidget(ophys_widget)  
        self.central_widget.setCurrentWidget(ophys_widget)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
