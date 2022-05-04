# PyQt5 Video player
#!/usr/bin/env python

from PyQt5.QtCore import QDir, Qt, QUrl
from PyQt5.QtMultimedia import QMediaContent, QMediaPlayer, QAudioProbe
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtWidgets import (QApplication, QFileDialog, QHBoxLayout, QLabel, QPushButton, QSizePolicy, QSlider, QStyle, QVBoxLayout, QWidget,QLineEdit)
from PyQt5.QtWidgets import QMainWindow,QWidget, QPushButton, QAction, QApplication
from PyQt5.QtGui import QIcon, QIntValidator
from matplotlib.pyplot import Figure
from matplotlib.backends.qt_compat import QtCore, QtWidgets, is_pyqt5
if is_pyqt5():
    from matplotlib.backends.backend_qt5agg import (
        FigureCanvas, NavigationToolbar2QT as NavigationToolbar)
else:
    from matplotlib.backends.backend_qt4agg import (
        FigureCanvas, NavigationToolbar2QT as NavigationToolbar)
import sys
import os
from pydub import AudioSegment
import numpy as np
from scipy import signal

class VideoWindow(QMainWindow):

    def __init__(self, parent=None):
        super(VideoWindow, self).__init__(parent)
        
        self.videoPosition = 0
        self.audioPosition = 0
        self.audioPositionStart = 0
        
        self.videoFileName = ''
        self.audioFileName = ''
        
        self.playersTimeDifference = 0
        
        self.setWindowTitle("InZnak Explorer v. 1.1.1 FINAL SOLUTION FOR INZNAK") 
        
        # https://stackoverflow.com/questions/25989610/pyqt5-keyboard-shortcuts-w-qaction

        self.mediaPlayer = QMediaPlayer(None, QMediaPlayer.VideoSurface)
        self.audioPlayer = QMediaPlayer()
        self.onDemandAudioReader = None

        videoWidget = QVideoWidget()

        self.playButton = QPushButton(self)
        self.playButton.setEnabled(False)
        self.playButton.setShortcut('Space')
        self.playButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.playButton.clicked.connect(self.play)

        self.moviePositionSlider = QSlider(Qt.Horizontal)
        self.moviePositionSlider.setRange(0, 0)
        self.moviePositionSlider.sliderMoved.connect(self.setPosition)

        self.errorLabel = QLabel()
        self.errorLabel.setSizePolicy(QSizePolicy.Preferred,
                QSizePolicy.Maximum)

        # Create new action
        openMovieAction = QAction(QIcon('open.png'), '&Open Movie', self)
        openMovieAction.setShortcut('Ctrl+O')
        openMovieAction.setStatusTip('Open movie')
        openMovieAction.triggered.connect(self.openMovieFile)

        # Create new action
        openAudioAction = QAction(QIcon('open.png'), '&Open Audio', self)
        openAudioAction.setShortcut('Ctrl+A')
        openAudioAction.setStatusTip('Open audio')
        openAudioAction.triggered.connect(self.openAudioFile)

        # Create exit action
        exitAction = QAction(QIcon('exit.png'), '&Exit', self)        
        exitAction.setShortcut('Ctrl+Q')
        exitAction.setStatusTip('Exit application')
        exitAction.triggered.connect(self.exitCall)

        # Create menu bar and add action
        menuBar = self.menuBar()
        fileMenu = menuBar.addMenu('&File')
        #fileMenu.addAction(newAction)
        fileMenu.addAction(openMovieAction)
        fileMenu.addAction(openAudioAction)
        fileMenu.addAction(exitAction)

        # Create a widget for window contents
        wid = QWidget(self)
        self.setCentralWidget(wid)

        # Create layouts to place inside widget
        controlLayout = QHBoxLayout()
        controlLayout.setContentsMargins(0, 0, 0, 0)
        controlLayout.addWidget(self.playButton)
        controlLayout.addWidget(self.moviePositionSlider)
        
        
        #-----------------------------------------------------
        shiftLayout = QHBoxLayout()
        self.videoShiftTextbox = QLineEdit(self)
        self.videoShiftTextbox.setValidator(QIntValidator())
        self.videoShiftTextbox.setText('0')
        self.timestampWriteAction = QAction()
        self.timestampWriteAction.triggered.connect(self.writeTimestamps)
        self.timestampWriteAction.setShortcut('Return')
        self.addAction(self.timestampWriteAction)
        #-----------------------------------------------------
        self.videoShiftLabel = QLabel('Video shift factor: ')
        self.videoShiftButtonB = QPushButton()
        self.videoShiftButtonB.setText('<<<\n(Ctrl+Left)')
        self.videoShiftButtonB.setShortcut('Ctrl+Left')
        self.videoShiftButtonB.clicked.connect(self.setVideoShiftBackward)
        self.videoShiftButtonF = QPushButton()
        self.videoShiftButtonF.setText('>>>\n(Ctrl+Right)')
        self.videoShiftButtonF.clicked.connect(self.setVideoShiftForward)
        self.videoShiftButtonF.setShortcut('Ctrl+Right')
        shiftLayout.addWidget(self.videoShiftLabel)
        shiftLayout.addWidget(self.videoShiftTextbox)
        shiftLayout.addWidget(self.videoShiftButtonB)
        shiftLayout.addWidget(self.videoShiftButtonF)
        #-----------------------------------------------------
        self.audioShiftLabel = QLabel('Audio shift factor: ')
        self.audioShiftTextbox = QLineEdit(self)
        self.audioShiftTextbox.setValidator(QIntValidator())
        self.audioShiftTextbox.setText('0')
        self.audioShiftButtonB = QPushButton()
        self.audioShiftButtonB.setText('<<<\n(Alt+Left)')
        self.audioShiftButtonB.setShortcut('Alt+Left')
        # self.audioShiftButtonB.setShortcut('Left')
        self.audioShiftButtonB.clicked.connect(self.setAudioShiftBackward)
        self.audioShiftButtonF = QPushButton()
        self.audioShiftButtonF.setText('>>>\n(Alt+Right)')
        self.audioShiftButtonF.setShortcut('Alt+Right')
        # self.audioShiftButtonF.setShortcut('Right')
        self.audioShiftButtonF.clicked.connect(self.setAudioShiftForward)
        shiftLayout.addWidget(self.audioShiftLabel)
        shiftLayout.addWidget(self.audioShiftTextbox)
        shiftLayout.addWidget(self.audioShiftButtonB)
        shiftLayout.addWidget(self.audioShiftButtonF)
        
        
        globalShiftPanel = QHBoxLayout()
        #-----------------------------------------------------
        self.globalShiftLabel = QLabel('Global shift factor: ')
        self.globalShiftTextbox = QLineEdit(self)
        self.globalShiftTextbox.setValidator(QIntValidator())
        self.globalShiftTextbox.setText('200')
        self.globalShiftButtonB = QPushButton()
        self.globalShiftButtonB.setText('<<<\n(Left)')
        self.globalShiftButtonB.setShortcut('Left')
        self.globalShiftButtonB.clicked.connect(self.globalBackward)
        self.globalShiftButtonF = QPushButton()
        self.globalShiftButtonF.setText('>>>\n(Right)')
        self.globalShiftButtonF.setShortcut('Right')
        self.globalShiftButtonF.clicked.connect(self.globalForward)
        globalShiftPanel.addWidget(self.globalShiftLabel)
        globalShiftPanel.addWidget(self.globalShiftTextbox)
        globalShiftPanel.addWidget(self.globalShiftButtonB)
        globalShiftPanel.addWidget(self.globalShiftButtonF)
        #-----------------------------------------------------
        self.globalIncrementLabel = QLabel('Increment factor: ')
        self.globalIncrementTextbox = QLineEdit(self)
        self.globalIncrementTextbox.setValidator(QIntValidator())
        self.globalIncrementTextbox.setText('100')
        self.globalIncrementButtonB = QPushButton()
        self.globalIncrementButtonB.setText('v\n(Down)')
        self.globalIncrementButtonB.setShortcut('Down')
        self.globalIncrementButtonB.clicked.connect(self.decreaseIncrement)
        self.globalIncrementButtonF = QPushButton()
        self.globalIncrementButtonF.setText('^\n(Up)')
        self.globalIncrementButtonF.setShortcut('Up')
        self.globalIncrementButtonF.clicked.connect(self.increaseIncrement)
        globalShiftPanel.addWidget(self.globalIncrementLabel)
        globalShiftPanel.addWidget(self.globalIncrementTextbox)
        globalShiftPanel.addWidget(self.globalIncrementButtonB)
        globalShiftPanel.addWidget(self.globalIncrementButtonF)
        
        self.classMotorcyclePresent = False
        self.classCarPresent        = False
        self.classVanPresent        = False
        self.classTruckPresent      = False
        self.classBusPresent        = False
        utilityPanel = QHBoxLayout()
        self.resetButton = QPushButton()
        self.resetButton.setText('reset position\n(Ctrl+R)')
        self.resetButton.clicked.connect(self.resetPosition)
        self.resetButton.setShortcut('Ctrl+R')
        utilityPanel.addWidget(self.resetButton)
        utilityPanel.addWidget(QLabel('Classes: '))
        self.classMotorcycleButton = QPushButton()
        self.classCarButton = QPushButton()
        self.classVanButton = QPushButton()
        self.classTruckButton = QPushButton()
        self.classBusButton = QPushButton()
        self.classMotorcycleButton.setText('Motorcycle\n(Z)')
        self.classCarButton.setText('Car\n(X)')
        self.classVanButton.setText('Van\n(C)')
        self.classTruckButton.setText('Truck\n(V)')
        self.classBusButton.setText('Bus\n(B)')
        self.classMotorcycleButton.clicked.connect(self.toggleMotorcycle)
        self.classCarButton.clicked.connect(self.toggleCar)
        self.classVanButton.clicked.connect(self.toggleVan)
        self.classTruckButton.clicked.connect(self.toggleTruck)
        self.classBusButton.clicked.connect(self.toggleBus)
        self.classMotorcycleButton.setShortcut('Z')
        self.classCarButton.setShortcut('X')
        self.classVanButton.setShortcut('C')
        self.classTruckButton.setShortcut('V')
        self.classBusButton.setShortcut('B')
        utilityPanel.addWidget(self.classMotorcycleButton)
        utilityPanel.addWidget(self.classCarButton)
        utilityPanel.addWidget(self.classVanButton)
        utilityPanel.addWidget(self.classTruckButton)
        utilityPanel.addWidget(self.classBusButton)
        
        self.positionDisplayLabel = QLabel('--position--')
        videoWidget.setContentsMargins(0,0,0,0)
        self.positionDisplayLabel.setContentsMargins(0,0,0,0)
        
        self.audioFigure = Figure(figsize=(5, 3))
        self.plot_canvas = FigureCanvas(self.audioFigure)
        self.audioPlotAx = self.audioFigure.add_subplot(111)
        # probe.record()
        
        layout = QVBoxLayout()
        layout.addWidget(videoWidget,35)
        layout.addWidget(self.plot_canvas,25)
        layout.addWidget(self.positionDisplayLabel)
        layout.addLayout(controlLayout)
        layout.addWidget(self.errorLabel)
        layout.addLayout(shiftLayout)
        layout.addLayout(globalShiftPanel)
        layout.addLayout(utilityPanel)

        # Set widget to contain window contents
        wid.setLayout(layout)

        self.mediaPlayer.setVideoOutput(videoWidget)
        self.mediaPlayer.stateChanged.connect(self.mediaStateChanged)
        
        self.mediaPlayer.setNotifyInterval(100)
        self.audioPlayer.setNotifyInterval(100)
        self.mediaPlayer.positionChanged.connect(self.audioPreviewUpdate)
        self.mediaPlayer.positionChanged.connect(self.positionChanged)
        self.mediaPlayer.positionChanged.connect(self.setVideoPosLabel)
        self.audioPlayer.positionChanged.connect(self.setAudioPosLabel)
        
        self.mediaPlayer.durationChanged.connect(self.durationChanged)
        self.mediaPlayer.error.connect(self.handleError)
        
        self.playButton.setFocus()
    
    
    def toggleMotorcycle(self):
        if self.classMotorcyclePresent:
            self.classMotorcycleButton.setStyleSheet("background-color: none")
            self.classMotorcyclePresent = False
        else:
            self.classMotorcycleButton.setStyleSheet("background-color: lime")
            self.classMotorcyclePresent = True
            self.audioPositionStart = self.audioPosition
    def toggleCar(self):
        if self.classCarPresent:
            self.classCarButton.setStyleSheet("background-color: none")
            self.classCarPresent = False
        else:
            self.classCarButton.setStyleSheet("background-color: lime")
            self.classCarPresent = True
            self.audioPositionStart = self.audioPosition
    def toggleVan(self):
        if self.classVanPresent:
            self.classVanButton.setStyleSheet("background-color: none")
            self.classVanPresent = False
        else:
            self.classVanButton.setStyleSheet("background-color: lime")
            self.classVanPresent = True
            self.audioPositionStart = self.audioPosition
    def toggleTruck(self):
        if self.classTruckPresent:
            self.classTruckButton.setStyleSheet("background-color: none")
            self.classTruckPresent = False
        else:
            self.classTruckButton.setStyleSheet("background-color: lime")
            self.classTruckPresent = True
            self.audioPositionStart = self.audioPosition
    def toggleBus(self):
        if self.classBusPresent:
            self.classBusButton.setStyleSheet("background-color: none")
            self.classBusPresent = False
        else:
            self.classBusButton.setStyleSheet("background-color: lime")
            self.classBusPresent = True
            self.audioPositionStart = self.audioPosition
    
    def increaseIncrement(self):
        increment_value = self.globalIncrementTextbox.text()
        if increment_value != '':
            self.globalShiftTextbox.setText(str(int(self.globalShiftTextbox.text()) + int(increment_value)))
        
    def decreaseIncrement(self):
        increment_value = self.globalIncrementTextbox.text()
        if increment_value != '':
            self.globalShiftTextbox.setText(str(int(self.globalShiftTextbox.text()) - int(increment_value)))
        
    def setMovieFile(self, fileName):
        if fileName != '':
            self.mediaPlayer.setMedia(
                    QMediaContent(QUrl.fromLocalFile(fileName)))
            self.playButton.setEnabled(True)
            self.mediaPlayer.setVolume(0)
            self.videoFileName = fileName
    
    def setAudioFile(self, fileName):
        if fileName != '':
            self.audioPlayer.setMedia(
                    QMediaContent(QUrl.fromLocalFile(fileName)))
            self.playButton.setEnabled(True)
            self.audioFileName = fileName
            self.onDemandAudioReader = AudioSegment.from_file(self.audioFileName, format="wav")
    
    def audioPreviewUpdate(self):
        self.audioPlotAx.cla()
        self.audioPlotAx.set_ylim([-1.1,1.1])
        time_start = self.audioPosition
        time_stop  = self.audioPosition+5000
        signal_for_visualization = self.onDemandAudioReader[time_start:time_stop].get_array_of_samples()
        signal_for_visualization = np.array(signal_for_visualization)/500.
        signal_for_visualization = np.abs(signal_for_visualization)
        signal_for_visualization = signal.fftconvolve(signal_for_visualization, np.ones(2048)/2048)
        signal_for_visualization = signal_for_visualization[2048:-2048:100]
        
        t_vec = np.linspace(time_start,time_stop,len(signal_for_visualization))
        self.audioPlotAx.fill_between(t_vec,-signal_for_visualization,signal_for_visualization)
        self.audioPlotAx.grid()
        
        self.plot_canvas.draw()

    def openMovieFile(self):
        fileName, _ = QFileDialog.getOpenFileName(self, "Open Movie",
                os.path.dirname(os.path.realpath(__file__)))
        self.setMovieFile(fileName)
            
    def openAudioFile(self):
        fileName, _ = QFileDialog.getOpenFileName(self, "Open Audio",
                os.path.dirname(os.path.realpath(__file__)))
        self.setAudioFile(fileName)

    def exitCall(self):
        sys.exit(app.exec_())

    def play(self):
        if self.mediaPlayer.state() == QMediaPlayer.PlayingState:
            self.mediaPlayer.pause()
            self.audioPlayer.pause()
        else:
            self.mediaPlayer.play()
            self.audioPlayer.play()

    def mediaStateChanged(self, state):
        if self.mediaPlayer.state() == QMediaPlayer.PlayingState:
            self.playButton.setIcon(
                    self.style().standardIcon(QStyle.SP_MediaPause))
        else:
            self.playButton.setIcon(
                    self.style().standardIcon(QStyle.SP_MediaPlay))

    def positionChanged(self, position):
        self.moviePositionSlider.setValue(position)
    
    def createAVPosLabelText(self):
        return 'video position: %i, audio position: %i [ms]\n(press space to start/stop stream, left/right arrow to nawigate both video&audio, ENTER to write a cut info)'%(self.videoPosition,self.audioPosition)
    def updatePosText(self):
        self.positionDisplayLabel.setText(self.createAVPosLabelText())
        
    def setVideoPosLabel(self, position):
        self.videoPosition = position
        self.updatePosText()
        video_distance = int(self.videoShiftTextbox.text()) - int(self.audioShiftTextbox.text())
        if video_distance <= 0:
            self.audioPlayer.setPosition(position-video_distance)
    def setAudioPosLabel(self, position):
        self.audioPosition = position
        self.updatePosText()
        audio_distance = int(self.audioShiftTextbox.text()) - int(self.videoShiftTextbox.text())
        if audio_distance < 0:
            self.mediaPlayer.setPosition(position-audio_distance)

    def durationChanged(self, duration):
        self.moviePositionSlider.setRange(0, duration)

    def setPosition(self, position):
        self.mediaPlayer.setPosition(position)
        self.audioPlayer.setPosition(position)
        
    def resetPosition(self):
        self.setPosition(0)

    def handleError(self):
        self.playButton.setEnabled(False)
        self.errorLabel.setText("Error: " + self.mediaPlayer.errorString())

    def setVideoShiftForward(self):
        shift_value = int(self.videoShiftTextbox.text())
        increment_value = int(self.globalIncrementTextbox.text())
        if increment_value != '':
            self.videoShiftTextbox.setText(str(shift_value+increment_value))
        self.updatePosText()
        self.updatePosText()
    def setVideoShiftBackward(self):
        shift_value = int(self.videoShiftTextbox.text())
        increment_value = int(self.globalIncrementTextbox.text())
        if increment_value != '':
            self.videoShiftTextbox.setText(str(shift_value-increment_value))
        self.updatePosText()
    
    def setAudioShiftForward(self):        
        shift_value = int(self.audioShiftTextbox.text())
        increment_value = int(self.globalIncrementTextbox.text())
        if increment_value != '':
            self.audioShiftTextbox.setText(str(shift_value+increment_value))
        self.updatePosText()
        self.updatePosText()
    def setAudioShiftBackward(self):        
        shift_value = int(self.audioShiftTextbox.text())
        increment_value = int(self.globalIncrementTextbox.text())
        if increment_value != '':
            self.audioShiftTextbox.setText(str(shift_value-increment_value))
        self.updatePosText()
    
    def globalForward(self):
        increment_value = int(self.globalShiftTextbox.text())
        self.mediaPlayer.setPosition(self.videoPosition+increment_value)
        self.audioPlayer.setPosition(self.audioPosition+increment_value)
        self.setVideoPosLabel(self.videoPosition)
        self.setAudioPosLabel(self.audioPosition)
                
    def globalBackward(self):
        increment_value = int(self.globalShiftTextbox.text())
        self.mediaPlayer.setPosition(self.videoPosition-increment_value)
        self.audioPlayer.setPosition(self.audioPosition-increment_value)
        self.setVideoPosLabel(self.videoPosition)
        self.setAudioPosLabel(self.audioPosition)
        
    def writeTimestamps(self):
        if QApplication.focusWidget().__class__.__name__ == 'QLineEdit':
            self.playButton.setFocus()
            return
        
        if not os.path.isfile('log.csv'):
            with open('log.csv', 'a') as f:
                f.write('video_file_name,audio_file_name,video_position,audio_position_start,audio_position_end,motorcycle_present,car_present,van_present,truck_present,bus_present')
        with open('log.csv', 'a') as f:
            f.write('\n%s,%s,%i,%i,%i,%i,%i,%i,%i,%i'%(self.videoFileName, self.audioFileName, self.videoPosition, self.audioPositionStart, self.audioPosition, self.classMotorcyclePresent,self.classCarPresent,self.classVanPresent,self.classTruckPresent,self.classBusPresent))

if __name__ == '__main__':
    app = QApplication(sys.argv)
    player = VideoWindow()
    player.resize(640, 720)
    player.setAudioFile('G:/Moje dane/ProjektBadawczy/INZNAK Viewer/data/audio_data_20190524T120000.wav')
    player.setMovieFile('G:/Moje dane/ProjektBadawczy/INZNAK Viewer/data/rec_0524140000.avi')
    player.show()
    sys.exit(app.exec_())

        