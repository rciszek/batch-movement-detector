from PyQt5.QtWidgets import QMainWindow, QAction, qApp, QApplication,QFileDialog,QHBoxLayout,QLabel,QWidget,QVBoxLayout,QSlider,QGridLayout,QLineEdit,QGroupBox,QFormLayout, QSpacerItem,QSizePolicy,QPushButton,QDialog,QDialogButtonBox,QProgressBar 
from PyQt5.QtGui import QIcon,QPixmap,QImage,QFont,QDoubleValidator,QIntValidator 
from PyQt5 import QtGui 
from PyQt5 import QtCore
from PyQt5.QtCore import QThread, pyqtSignal, Qt,QLocale
from VideoReader import VideoReader
from VideoAnalyzer import VideoAnalyzer, MovementTracker
import cv2
import numpy as np
import logging,re,glob,time,sys,locale
import logging.config
from deployment import resource_path,loadStyleSheet
from time import sleep


class MotionDetector(QMainWindow):
    
    def __init__(self):
        super().__init__()
        
        self.initUI()
        
        
    def initUI(self):
        
        self.setWindowTitle('MovementDetector')
      
        openFileAct = QAction('Open', self)        
        openFileAct.setShortcut('Ctrl+O')
        openFileAct.setStatusTip('Open a single file')
        openFileAct.triggered.connect(self.showFileDialog)     
        
        batchProcessAct = QAction('Batch process', self)        
        batchProcessAct.setShortcut('Ctrl+B')
        batchProcessAct.setStatusTip('Process multiple files')
        batchProcessAct.triggered.connect(self.showBatchDialog)     
        
        exitAct = QAction('&Exit', self)        
        exitAct.setShortcut('Ctrl+Q')
        exitAct.setStatusTip('Exit application')
        exitAct.triggered.connect(qApp.quit)
        
        aboutAct = QAction('&About', self)        
        aboutAct.setStatusTip('About MovementDetector')
        aboutAct.triggered.connect(self.showAboutDialog)        

        self.statusBar()

        menubar = self.menuBar()
        fileMenu = menubar.addMenu('&File')
        fileMenu.addAction(openFileAct)      
        fileMenu.addAction(batchProcessAct)           
        fileMenu.addAction(exitAct)     
        
        aboutMenu = menubar.addMenu('&Help')
        aboutMenu.addAction(aboutAct)           
        self.showVideoPlaybackView()
 

  
    def showVideoPlaybackView(self, maximize=True):
        self.videoWidget = VideoWidget()
        self.videoWidget.playbackStarted.connect(lambda : self.showMaximized() )

        self.setCentralWidget(self.videoWidget)
        if maximize:
            self.showMaximized()     
      
    def showFileDialog(self):

        fname = QFileDialog.getOpenFileName(self, 'Open file', '/home')

        if fname[0]:
            self.videoWidget.displayVideo(fname[0])   

            
    def showAboutDialog(self):
        self.videoWidget.videoThread.pause()
        dialog = AboutDialog()
        dialog.exec_()    
            
    def showBatchDialog(self):
        dialog = BatchDialog()
        return_code = dialog.exec_()
        if return_code == QDialog.Accepted:
            parameters, source_path, target_path = dialog.getParameters()                           
            self.performBatchAnalysis(VideoAnalyzer(**parameters),source_path,target_path)
                
    def performBatchAnalysis(self,videoAnalyser,source_folder,target_folder):
        progressWidget = ProgressWidget()
      
        self.closeVideoPlayback()
        progressWidget.cancelButton.clicked.connect(self.cancelBatch)
        self.setCentralWidget(progressWidget)

        self.batchAnalyzer = BatchAnalyzer(videoAnalyser,source_folder,target_folder)
        self.batchAnalyzer.progressed.connect(progressWidget.updateProgress)
        self.batchAnalyzer.start()
        
    def cancelBatch(self):
        self.batchAnalyzer.stopBatch()
        self.showVideoPlaybackView(maximize=False)
                       
    def closeEvent(self, evnt):
        self.videoWidget.videoThread.terminate()
        super().closeEvent(evnt)
        
        
    def closeVideoPlayback(self):
        if self.videoWidget.videoThread.videoReader != None:
            self.videoWidget.videoThread.videoReader.close()
            while self.videoWidget.videoThread.videoReader.isClosed() != True:
                sleep(0.1)    
        self.videoWidget.videoThread.terminate()  
            
class BatchAnalyzer(QThread):  
     
    progressed = pyqtSignal(int,int)   
    
    def __init__(self, videoAnalyser, source, target):
        QThread.__init__(self, parent=None)        
        self.videoAnalyzer = videoAnalyser
        self.source = source
        self.target = target
        self.cancel = False
        
    def processFile(self, file_path):
        file_path = file_path.replace('\\','/')
        videoReader = VideoReader(file_path)     
        movementTracker = MovementTracker()

        captured = re.search('(?<=/)([A-Za-z0-9\-_]+)(?=.avi)', file_path, re.IGNORECASE)
        
        if captured is None:
            return
        
        file_name = captured.group(0)
        while True:
            if self.cancel == True:
                break
            
            frame = videoReader.nextFrame()  

            if frame is None:
                np.savetxt(self.target  + "/" + file_name+".csv",movementTracker.getEvents(), delimiter=",", fmt='%.2f')
                break;       
       
            foreground_mask, movement = self.videoAnalyzer.detectMovement(frame)          
            movementTracker.update(movement, videoReader.currentPositionInSeconds())
        videoReader.close()
        
    def stopBatch(self):
        self.cancel = True 

    def run(self):  
        video_files = glob.glob(self.source+'/**/*.avi', recursive=True)
        for i in range(0,len(video_files)):
            if self.cancel == False:
                self.processFile(video_files[i])    
            self.progressed.emit(i+1,len(video_files))
            
      
        
        
       
class BatchDialog(QDialog):
    def __init__(self,parent=None):
        
        self.videoAnalyzer = VideoAnalyzer()
        self.parameters = {}
        
        QDialog.__init__(self,parent)
                
        self.setupUi(self)               
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.setWindowFlags(self.windowFlags() & (~QtCore.Qt.WindowContextHelpButtonHint))
     
        
    def setupUi(self, Dialog):
        self.setWindowTitle("Batch detection")
        self.resize(308, 170)
        self.layout = QVBoxLayout(Dialog)
        self.detectionSettingsWidget = DetectionSettingsWidget(self.videoAnalyzer)
        self.detectionSettingsWidget.setParent(Dialog)
        self.detectionSettingsWidget.updatedSignal.connect(self.updateProceed)
        self.layout.addWidget(self.detectionSettingsWidget)
        self.sourceHBox = QHBoxLayout(Dialog) 
        self.sourceLabel = QLabel('Source:',parent=Dialog)        
        self.sourceEditline = QLineEdit(Dialog)
        self.sourceEditline.setReadOnly(True)
        self.sourcePushButton = QPushButton("Select",parent=Dialog)   
        self.sourceEditline.textChanged.connect(self.updateProceed)
        
        self.sourcePushButton.clicked.connect( lambda d: self.showFolderDialog(self.sourceEditline))
        
        self.sourceHBox.addWidget(self.sourceLabel)        
        self.sourceHBox.addWidget(self.sourceEditline)
        self.sourceHBox.addWidget(self.sourcePushButton)
        self.layout.addLayout(self.sourceHBox) 

       
        self.targetHBox = QHBoxLayout(Dialog) 
        self.targetLabel = QLabel('Target:',parent=Dialog)          
        self.targetEditline = QLineEdit(Dialog)
        self.targetEditline.setReadOnly(True)
        self.targetEditline.textChanged.connect(self.updateProceed)
        self.targetPushButton = QPushButton("Select",parent=Dialog)      
        
        self.targetPushButton.clicked.connect( lambda d: self.showFolderDialog(self.targetEditline))
        self.targetHBox.addWidget(self.targetLabel)
        self.targetHBox.addWidget(self.targetEditline)
        self.targetHBox.addWidget(self.targetPushButton)        
        self.layout.addLayout(self.targetHBox)    

        
        self.buttonBox = QDialogButtonBox(Dialog)
        self.buttonBox.setGeometry(QtCore.QRect(0, 250, 310, 32))
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QDialogButtonBox.Cancel|QDialogButtonBox.Ok)
        self.buttonBox.button(QDialogButtonBox.Cancel).clicked.connect(Dialog.reject)
        self.buttonBox.button(QDialogButtonBox.Ok).clicked.connect(Dialog.accept)    
        self.buttonBox.button(QDialogButtonBox.Ok).setText("Detect")
        self.buttonBox.button(QDialogButtonBox.Ok).setEnabled(False)
        self.layout.addWidget(self.buttonBox)
        
    def getParameters(self):
        return self.parameters, self.source_path, self.target_path        
      
        
    def showFolderDialog(self, targetedLineEdit):
        dname = str(QFileDialog.getExistingDirectory(self, "Select Directory"))

        if len(dname) > 0:
            targetedLineEdit.setText(dname)
        
    def updateProceed(self):

        self.parameters = self.videoAnalyzer.getParameters()
        
        if self.detectionSettingsWidget.allSettingsValid() and ( self.targetEditline.text() is not '') and ( self.sourceEditline.text() is not ''):
            self.buttonBox.button(QDialogButtonBox.Ok).setEnabled(True)
            self.source_path = self.sourceEditline.text()
            self.target_path = self.targetEditline.text()  
        else:
            self.buttonBox.button(QDialogButtonBox.Ok).setEnabled(False)
        
class AboutDialog(QDialog):
    def __init__(self,parent=None):
        
        QDialog.__init__(self,parent)
              
        self.setupUi(self)               
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.setWindowFlags(self.windowFlags() & (~QtCore.Qt.WindowContextHelpButtonHint))
     
        
    def setupUi(self, Dialog):
        self.setWindowTitle("About")
        self.resize(100, 120)
        
        self.layout = QHBoxLayout(Dialog)
        self.vbox = QVBoxLayout(Dialog)
        self.vbox.setAlignment(Qt.AlignHCenter)
        self.layout.addLayout(self.vbox)
        
        self.nameLabel = QLabel("MovementDetector")
        self.infoHbox = QHBoxLayout(Dialog)
        self.infoLabel = QLabel("Version: 0.3 Support: ciszek@uef.fi")  
        self.infoLabel.setMaximumWidth(100)
        self.infoLabel.setWordWrap(True)         
        
        self.vbox.addWidget(self.nameLabel)
        self.vbox.addWidget(self.infoLabel)
        self.vbox.addStretch(1)
        self.buttonHBox = QHBoxLayout()
        
        self.buttonBox = QDialogButtonBox(Dialog)


        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QDialogButtonBox.Ok)
        self.buttonBox.button(QDialogButtonBox.Ok).clicked.connect(Dialog.accept)   
        self.buttonHBox.addStretch(1)
        self.buttonHBox.addWidget(self.buttonBox)
        self.buttonHBox.addStretch(1)       
        self.vbox.addLayout(self.buttonHBox)


        
class ProgressWidget(QWidget):


    progress_in_progress_style = loadStyleSheet(resource_path('stylesheets/progress_in_progress.css'))  
    progress_completed_style = loadStyleSheet(resource_path('stylesheets/progress_completed.css'))
  
    
    def __init__(self,):
        super().__init__()
        self.vbox = QVBoxLayout() 

        self.outerHbox = QHBoxLayout()  
        
        self.outerHbox.setContentsMargins(30, 0, 30, 0)  
     
        self.outerHbox.addLayout(self.vbox)
       
                    
        self.progressBar = QProgressBar(self)  
        self.progressBar.setStyleSheet(ProgressWidget.progress_in_progress_style) 
        self.progressBar.setValue(0)
        
        self.vbox.addStretch(1)   
        self.vbox.setSpacing(10)
        self.vbox.setAlignment(Qt.AlignHCenter)
        self.vbox.addWidget(self.progressBar)
        
        self.buttonHBox = QHBoxLayout()
        
        self.cancelButton = QPushButton("Cancel")
        self.cancelButton.setFixedWidth(50)  
        self.buttonHBox.addStretch(1)           
        self.buttonHBox.addWidget(self.cancelButton)
        self.buttonHBox.addStretch(1)           
        self.vbox.addLayout(self.buttonHBox)
        self.vbox.addStretch(1)          

        self.setLayout(self.outerHbox )
        
        
        
        
        
    def updateProgress(self, currentProgress, maximum):
        self.progressBar.setMaximum(maximum)  
        self.progressBar.setValue(currentProgress)  
        
        if currentProgress == maximum:
            self.doCompleted()
        
        
    def doCompleted(self):
        self.progressBar.setStyleSheet(ProgressWidget.progress_completed_style)   
        self.cancelButton.setText("Ok")

     
        
class VideoWidget(QWidget):
    
    playbackStarted = pyqtSignal()
    
    def __init__(self):
        super().__init__()
        self.videoAnalyzer = VideoAnalyzer()        
        self.videoThread = VideoThread(self);
        self.__layout()
        self.toggleVideoControls(False)

    def toggleVideoControls(self, state):
        for child in self.findChildren(QWidget):
            child.setEnabled(state)
        

    def __layout(self):
        
        self.original = QLabel(self)
        self.processed = QLabel(self)  
        
        self.originalPixmap = QPixmap(50,50)
        self.originalPixmap.fill(QtCore.Qt.transparent)
        self.original.setPixmap(self.originalPixmap)
        self.processed.setPixmap(self.originalPixmap)  
        
        self.currentTimeLabel = QLabel("00:00:00")
        self.currentTimeLabel.setFixedWidth(45)
        self.totalTimeLabel = QLabel("00:00:00")
        self.totalTimeLabel.setFixedWidth(45)        
        self.timeSlider = QSlider(Qt.Horizontal, self)
        self.timeSlider.setFocusPolicy(Qt.NoFocus)
        self.timeSlider.setGeometry(30, 40, 100, 30)
  
        self.timeSlider.sliderMoved[int].connect(self.videoThread.changePosition)
        self.detectionSettingsBox = QGroupBox("Settings")        
        
        self.vbox = QVBoxLayout(self)   
        self.hbox = QHBoxLayout(self)   
        self.hbox.setAlignment(Qt.AlignCenter)       
        self.hbox.addWidget(self.processed)
        self.hbox.addWidget(self.original)
                 
        self.vbox.addStretch(1)
        self.vbox.addLayout(self.hbox ) 
        
        self.timePositionLayout = QHBoxLayout(self)  
        self.timePositionLayout.addWidget(self.currentTimeLabel)
        self.timePositionLayout.addWidget(self.timeSlider) 
        self.timePositionLayout.addWidget(self.totalTimeLabel)        
        self.vbox.addLayout(self.timePositionLayout)      

        self.detectionSettingsWidget = DetectionSettingsWidget(self.videoAnalyzer)
        
        bottomHbox = QHBoxLayout(self)
        bottomHbox.addStretch(1)            
        bottomHbox.addWidget(self.detectionSettingsWidget)
        bottomHbox.addStretch(1)  
        
        self.stopButton = QPushButton()
        stopIcon = QIcon()
        stopIcon.addPixmap(QPixmap(resource_path('icons/Stop.png')))
        self.stopButton.setIcon(stopIcon)
        self.stopButton.setFixedWidth(50)
        self.stopButton.setFixedHeight(40)   
        self.stopButton.clicked.connect( lambda: self.videoThread.pause())
        bottomHbox.addWidget(self.stopButton)

        self.playButton = QPushButton()
        self.playButton.setFixedWidth(50)
        self.playButton.setFixedHeight(40)
        self.playButton.clicked.connect( lambda: self.videoThread.play())
        
        playIcon = QIcon()
        playIcon.addPixmap(QPixmap(resource_path('icons/Play.png')))
        self.playButton.setIcon(playIcon)
        bottomHbox.addWidget(self.playButton)
      
        bottomHbox.addStretch(1)      

        spacer = QSpacerItem(320, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)
        bottomHbox.addItem(spacer)
        bottomHbox.addStretch(1)            
        self.vbox.addLayout(bottomHbox)      
        self.setLayout(self.vbox )  
        
        
    def displayVideo(self,file_path):

        self.toggleVideoControls(True)
        
        videoReader = VideoReader(file_path)
        self.totalTimeLabel.setText(time.strftime('%H:%M:%S', time.gmtime(videoReader.lengthInSeconds())))
        self.timeSlider.setTickInterval(1)
        self.timeSlider.setRange(0,videoReader.duration)

        self.videoThread.changeOriginalPixmap.connect(lambda p: self.original.setPixmap(p))
        self.videoThread.changeProcessedPixmap.connect(lambda p: self.processed.setPixmap(p))        
        self.videoThread.changecurrentTime.connect(lambda t: self.__updateSlider(t))
        self.videoThread.videoReader = videoReader
        self.videoThread.videoAnalyzer = self.videoAnalyzer
        self.videoThread.videoAnalyzer.updateParameters()
        if  self.videoThread.isRunning() == False:
            self.videoThread.start()
        self.playbackStarted.emit()
        
    def __updateSlider(self, t):
        self.currentTimeLabel.setText(time.strftime('%H:%M:%S', time.gmtime(t)))
        self.timeSlider.setValue(t)
        
        
class DetectionSettingsWidget(QWidget):
    
    __movementThresholdValidator = QDoubleValidator(0.000001, 1.0, 6)
    __historyValidator = QIntValidator(1, 1000)    
    __kernerValidator = QIntValidator(1, 20)   
    __mixtureValidator = QIntValidator(1, 10)  
    __backgroundRatioValidator = QDoubleValidator(0.000001, 1,6) 
    __complexityThresholdValidator = QDoubleValidator(0.000001, 1.0, 6)    
        
    updatedSignal = pyqtSignal()    
    
    def __init__(self, videoAnalyzer):
        super().__init__()

        self.videoAnalyzer = videoAnalyzer
        self.__layout()

    
    def __layout(self):
        self.setFixedWidth(300)
        hbox = QHBoxLayout()
        hbox.setSpacing(5)
        leftFormLayout = QFormLayout()    
        
        self.movementThresholdLineEdit = QLineEdit("0,001", parent=self)
        self.movementThresholdLineEdit.setMaximumWidth(40)
        self.movementThresholdLineEdit.setValidator(self.__movementThresholdValidator)
        leftFormLayout.addRow(QLabel("Threshold:",parent=self),self.movementThresholdLineEdit)     
        
        self.kernelSizeLineEdit = QLineEdit("5",parent=self)
        self.kernelSizeLineEdit.setMaximumWidth(40)   
        self.kernelSizeLineEdit.setValidator(self.__kernerValidator)        
        leftFormLayout.addRow(QLabel("Kernel:",parent=self),self.kernelSizeLineEdit)           
        
        middleFormLayout = QFormLayout() 
        self.historyLineEdit = QLineEdit("100",parent=self)
        self.historyLineEdit.setMaximumWidth(40)
        self.historyLineEdit.setValidator(self.__historyValidator)           
        middleFormLayout.addRow(QLabel("History:",parent=self),self.historyLineEdit)      
        
        self.complexityReductionThresholdLineEdit = QLineEdit("0,05",parent=self)
        self.complexityReductionThresholdLineEdit.setMaximumWidth(40)
        self.complexityReductionThresholdLineEdit.setValidator(self.__complexityThresholdValidator)           
        middleFormLayout.addRow(QLabel("CRT:",parent=self),self.complexityReductionThresholdLineEdit)            
        
        rightFormLayout = QFormLayout()   
         
        self.mixtureLineEdit = QLineEdit("5",parent=self)
        self.mixtureLineEdit.setMaximumWidth(40)
        self.mixtureLineEdit.setValidator(self.__mixtureValidator)              
        rightFormLayout.addRow(QLabel("Mixtures:",parent=self),self.mixtureLineEdit) 
        
        self.backgroundRatioLineEdit = QLineEdit("0,8",parent=self)
        self.backgroundRatioLineEdit.setMaximumWidth(40)
        self.backgroundRatioLineEdit.setValidator(self.__backgroundRatioValidator)           
        rightFormLayout.addRow(QLabel("Bg ratio:",parent=self),self.backgroundRatioLineEdit)         
   
        hbox.addLayout(leftFormLayout)

        hbox.addLayout(middleFormLayout)   
       
        hbox.addLayout(rightFormLayout)
        
        self.movementThresholdLineEdit.textChanged.connect( lambda t: self.updateSettings(self.movementThresholdLineEdit,self.videoAnalyzer,'movement_threshold','updateParameters') )
        self.kernelSizeLineEdit.textChanged.connect( lambda t: self.updateSettings(self.kernelSizeLineEdit,self.videoAnalyzer,'open_kernel_size','updateParameters')  )        
        self.historyLineEdit.textChanged.connect( lambda t: self.updateSettings(self.historyLineEdit,self.videoAnalyzer,'history','updateParameters') )
        self.complexityReductionThresholdLineEdit.textChanged.connect( lambda t: self.updateSettings(self.complexityReductionThresholdLineEdit,self.videoAnalyzer,'complexity_reduction_threshold','updateParameters')  )           
        self.mixtureLineEdit.textChanged.connect( lambda t: self.updateSettings(self.mixtureLineEdit,self.videoAnalyzer,'mixtures','updateParameters')  ) 
        self.backgroundRatioLineEdit.textChanged.connect( lambda t: self.updateSettings(self.backgroundRatioLineEdit,self.videoAnalyzer,'background_ratio','updateParameters')  )
        
        self.setLayout(hbox)
        
    def allSettingsValid(self):
        for setting in self.findChildren(QLineEdit):
            if setting.validator().validate(setting.text(), 0)[0] != QtGui.QValidator.Acceptable:
                return False        
        return True
        
    def updateSettings(self, source, target, setting, update_function_name=None):
        
        validator = source.validator() 
        color = '#f6989d'
      
        if validator.validate(source.text(), 0)[0] == QtGui.QValidator.Acceptable:
            color = 'white'
            setattr( target, setting, float( source.text().replace(',', '.')))     

        update_function = getattr(target, update_function_name)
        source.setStyleSheet('QLineEdit { background-color: %s }'%color)
        update_function()
        self.updatedSignal.emit()
        
    
class VideoThread(QThread):
    changeOriginalPixmap = pyqtSignal(QPixmap)
    changeProcessedPixmap = pyqtSignal(QPixmap) 
    changecurrentTime = pyqtSignal([float])   
    
    def __init__(self, parent=None ):
        QThread.__init__(self, parent=parent)
        self.jump_to_position = -1
        self.videoReader = None
        self.videoAnalyzer = None
        self.paused = False

    def pause(self):
        self.paused = True 
        
    def play(self):
        self.paused = False         

    def run(self):

        self.Paused = False
        
        movementTracker = MovementTracker()        
        
        while True:
            
            begin_time = int(round(time.time() * 1000))
            
            if self.paused:
                continue
            
            if self.jump_to_position >= 0:
                self.videoReader.setPositionInSeconds(self.jump_to_position)
                self.jump_to_position = -1
            
            frame = self.videoReader.nextFrame()    
            
            if frame is None or frame is []:
                continue;
            
            height, width, channel, bytesPerLine = self.videoReader.getFrameData(frame)


            foreground_mask, movement = self.videoAnalyzer.detectMovement(frame)
            
            movementTracker.update(movement, self.videoReader.currentPositionInSeconds())
            
            gs_foreground_mask = cv2.cvtColor(foreground_mask,cv2.COLOR_GRAY2RGB)
            
            if gs_foreground_mask is None:
                continue
            
            height, width, channel, bytesPerLine = self.videoReader.getFrameData(gs_foreground_mask)

            processedFrameImage = QImage(gs_foreground_mask.data, width, height, bytesPerLine, QImage.Format_RGB888)         
            processedFramePixmap = QPixmap.fromImage(processedFrameImage)  

            im2,contours,hierarchy = cv2.findContours(foreground_mask, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) > 0 and movement:
                for cnt in contours:
                    x,y,w,h = cv2.boundingRect(cnt)
                    cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
                
            originalFrameImage = QImage(frame.data, width, height, bytesPerLine, QImage.Format_RGB888)
            originalFramePixmap = QPixmap.fromImage(originalFrameImage)      
            
            self.changeOriginalPixmap.emit(originalFramePixmap)  
            self.changeProcessedPixmap.emit(processedFramePixmap)  
            self.changecurrentTime.emit(round(self.videoReader.currentPositionInSeconds()))
            
            delta_time = ( int(round(time.time() * 1000)) - begin_time ) / 1000.0

            wait_time = np.max([(1.0/self.videoReader.fps)-delta_time,0])
            if wait_time > 0:
                sleep(wait_time)
            
    def changePosition(self,position):
        self.jump_to_position = position

    def terminate(self):
        if self.videoReader is not None:
            self.videoReader.close()
        super().terminate()
        
        
        
if __name__ == '__main__':
    #Set up logging

    #logging.config.fileConfig('logging.conf')
    #logging.getLogger('eventLog').info("Testing")
    app=0  

    locale.setlocale(locale.LC_ALL, '') 
    app = QApplication(sys.argv)
    ex = MotionDetector()
    app.exec_()
    #sys.exit(app.exec_())