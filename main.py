import os
import sys
import numpy as np
import pandas as pd

import classify
import segment
import compress

from PyQt5 import QtCore
from PyQt5 import QtGui
from PyQt5.QtWidgets import * 
from tkinter import filedialog, messagebox

from keras.models import load_model

import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#Main Window
class App(QWidget):
    def __init__(self):
        super().__init__()
        self.classTrainPath = 'Satellite Images for Classify'
        self.classTestPath = ''
        self.classTestImage = None
        
        if os.path.isfile('model/classification_mdl.h5'):
            self.classMdl = load_model('model/classification_mdl.h5')
        else:
            self.classMdl = None
        
        self.segmentTrainPath = 'Satellite Images for Segment'
        self.segmentTestPath = ''
        self.segemntTestImage = None
        
        if os.path.isfile('model/segment/checkpoint'):            
            self.segmentMdl = segment.init_model()
            self.segmentMdl.load_weights('model/segment/segment')
        else:
            self.segmentMdl = None
            
        self.sortPath = ''
        self.sortedImages = None
        self.sortedAreas = None
        self.sortImage = None
        self.sortIdx = None
        self.sortMin = None
        self.sortMax = None
        
        self.compressTestPath = ''
        self.compressTestImage = None

        self.title = 'Satellite Image Process'
        self.left = 300
        self.top = 300
        self.width = 900
        self.height = 600
   
        self.setFixedSize(1200, 600)
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())
        self.createWidget()
        self.initParameters()

    #Create control panel
    def createWidget(self):

        # main pannel
        self.paraPannel = QGridLayout()
        self.paraPannel.setSpacing(20)

        # 1. Classification
        ## train data path for classify
        self.classTrainEdit = QLineEdit('')
        self.classTrainEdit.setEnabled(False)
        
        ## test image path for classify test
        self.classTestEdit = QLineEdit('')
        self.classTestEdit.setEnabled(False)
          
        ## button to select test image for classify
        self.classSelectBtn = QPushButton('...')
        self.classSelectBtn.clicked.connect(self.classSelect)
         
        self.classTestBox = QHBoxLayout()
        self.classTestBox.addWidget(self.classTestEdit)
        self.classTestBox.addWidget(self.classSelectBtn)
        
        ## button to train classification model
        self.classTrainBtn = QPushButton('Train')
        self.classTrainBtn.clicked.connect(self.classTrain)
        
        ## button to classify test image
        self.classTestBtn = QPushButton('Classify')
        self.classTestBtn.clicked.connect(self.classTest)
        
        self.classBtnBox = QHBoxLayout()
        self.classBtnBox.addWidget(self.classTrainBtn)
        self.classBtnBox.addWidget(self.classTestBtn)
        
        ## Classify Group
        self.classGroup = QGroupBox("Satellite Image Classification")
        self.classBox = QVBoxLayout()
        self.classGroup.setLayout(self.classBox)

        self.paraPannel.addWidget(self.classGroup)
        self.classBox.addWidget(self.classTrainEdit)
        self.classBox.addLayout(self.classTestBox)
        self.classBox.addLayout(self.classBtnBox)

        # 2. Water Segmentation
        ## train data path for segment
        self.segmentTrainEdit = QLineEdit('')
        self.segmentTrainEdit.setEnabled(False)
        
        ## test image path for segment test
        self.segmentTestEdit = QLineEdit('')
        self.segmentTestEdit.setEnabled(False)
          
        ## button to select test image for segment
        self.segmentSelectBtn = QPushButton('...')
        self.segmentSelectBtn.clicked.connect(self.segmentSelect)
        
        self.segmentTestBox = QHBoxLayout()
        self.segmentTestBox.addWidget(self.segmentTestEdit)
        self.segmentTestBox.addWidget(self.segmentSelectBtn)
        
        ## button to train segment model
        self.segmentTrainBtn = QPushButton('Train')
        self.segmentTrainBtn.clicked.connect(self.segmentTrain)
        
        ## button to segment test image
        self.segmentTestBtn = QPushButton('Segment')
        self.segmentTestBtn.clicked.connect(self.segmentTest)
        
        self.segmentBtnBox = QHBoxLayout()
        self.segmentBtnBox.addWidget(self.segmentTrainBtn)
        self.segmentBtnBox.addWidget(self.segmentTestBtn)
        
        ## Segment Group
        self.segmentGroup = QGroupBox("Water Satellite Image Segment")
        self.segmentBox = QVBoxLayout()
        self.segmentGroup.setLayout(self.segmentBox)

        self.paraPannel.addWidget(self.segmentGroup)
        self.segmentBox.addWidget(self.segmentTrainEdit)
        self.segmentBox.addLayout(self.segmentTestBox)
        self.segmentBox.addLayout(self.segmentBtnBox)
        
        # 3. Sort Satellite Image
        ## sort image data path
        self.sortEdit = QLineEdit('')
        self.sortEdit.setEnabled(False)
        # self.segmentTrainEdit.setFixedWidth(160)
        
        ## button to sort images
        self.sortBtn = QPushButton('Sort')
        self.sortBtn.clicked.connect(self.imageSort)
        
        ## previous button to sort image
        self.sortPrevBtn = QPushButton('<<')
        self.sortPrevBtn.clicked.connect(self.sortPrev)
        
        ## next button to sort image
        self.sortNextBtn = QPushButton('>>')
        self.sortNextBtn.clicked.connect(self.sortNext)
        
        self.sortBtnBox = QHBoxLayout()
        self.sortBtnBox.addWidget(self.sortBtn)
        self.sortBtnBox.addWidget(self.sortPrevBtn)
        self.sortBtnBox.addWidget(self.sortNextBtn)
        
        ## sort Group
        self.sortGroup = QGroupBox("Sort Satellite Image")
        self.sortBox = QVBoxLayout()
        self.sortGroup.setLayout(self.sortBox)

        self.paraPannel.addWidget(self.sortGroup)
        self.sortBox.addWidget(self.sortEdit)
        self.sortBox.addLayout(self.sortBtnBox)
        
        
        # 4. Satellite Image Compress
        ## test image path for segment test
        self.compressTestEdit = QLineEdit('')
        self.compressTestEdit.setEnabled(False)
          
        ## button to select test image for segment
        self.compressSelectBtn = QPushButton('...')
        self.compressSelectBtn.clicked.connect(self.compressSelect)
        
        self.compressTestBox = QHBoxLayout()
        self.compressTestBox.addWidget(self.compressTestEdit)
        self.compressTestBox.addWidget(self.compressSelectBtn)
        
        ## button to sort images
        self.compressBtn = QPushButton('Compress')
        self.compressBtn.clicked.connect(self.compressImage)

        self.compressResultEdit = QTextEdit('')
        self.compressResultEdit.setEnabled(False)
        
        ## sort Group
        self.compressGroup = QGroupBox("Compress Satellite Image")
        self.compressBox = QVBoxLayout()
        self.compressGroup.setLayout(self.compressBox)

        self.paraPannel.addWidget(self.compressGroup)
        self.compressBox.addLayout(self.compressTestBox)
        self.compressBox.addWidget(self.compressBtn)
        self.compressBox.addWidget(self.compressResultEdit)
        

        ## Figure Pannel
        self.testPannel = QGridLayout()
        self.processLabel = QLabel('This is process label!!!')
        self.processLabel.setAlignment(QtCore.Qt.AlignHCenter)

        self.imageLabel = QLabel('')
        self.imageLabel.setPixmap(QtGui.QPixmap("data/satellite.jpg"))
        
        self.fileLabel = QLabel('')
        self.fileLabel.setAlignment(QtCore.Qt.AlignHCenter)
        
        self.testPannel.addWidget(self.processLabel)
        self.testPannel.addWidget(self.imageLabel)
        self.testPannel.addWidget(self.fileLabel)

        self.layout = QGridLayout()
        self.layout.setSpacing(10)
        self.layout.addLayout(self.paraPannel, 1, 0, 1, 1)
        self.layout.addLayout(self.testPannel, 1, 1, 1,3)
        self.setLayout(self.layout)

    def initParameters(self):
        self.classTrainEdit.setText('data/Classify')
        self.segmentTrainEdit.setText('data/Segment')
        self.sortEdit.setText('test/Sort')

    def classSelect(self):
        # Use filedialog to select input file
        filetypes = [("JPEG files", "*.jpg")]
        path = filedialog.askopenfilename(title='Open a file',
                                        initialdir='test/Classify/',
                                        filetypes=filetypes)

        # Update label with selected file path
        if path: 
            self.classTestEdit.setText(path)
            self.fileLabel.setText(path)
            self.classTestPath = path
            self.classTestImage = QtGui.QPixmap(path)
            self.classTestImage = self.classTestImage.scaled(900, 540)
            self.imageLabel.setPixmap(self.classTestImage)
        else:
            messagebox.showerror("Error", 'Please select image file correctly!')

    def segmentSelect(self):
        # Use filedialog to select input file
        filetypes = [("JPEG files", "*.jpg")]
        path = filedialog.askopenfilename(title='Open a file',
                                        initialdir='test/Segment/',
                                        filetypes=filetypes)

        # Update label with selected file path
        if path: 
            self.segmentTestEdit.setText(path)
            self.fileLabel.setText(path)
            self.segmentTestPath = path
            self.segmentTestImage = QtGui.QPixmap(path)
            self.segmentTestImage = self.segmentTestImage.scaled(900, 540)
            self.imageLabel.setPixmap(self.segmentTestImage)
            self.processLabel.setText('')
        else:
            messagebox.showerror("Error", 'Please select image file correctly!')

    def compressSelect(self):
        # Use filedialog to select input file
        filetypes = [("JPEG files", "*.jpg")]
        path = filedialog.askopenfilename(title='Open a file',
                                        initialdir='test/Compress/Original',
                                        filetypes=filetypes)

        # Update label with selected file path
        if path: 
            self.compressTestEdit.setText(path)
            self.fileLabel.setText(path)
            self.compressTestPath = path
            self.compressTestImage = QtGui.QPixmap(path)
            self.compressTestImage = self.compressTestImage.scaled(900, 540)
            self.imageLabel.setPixmap(self.compressTestImage)
        else:
            messagebox.showerror("Error", 'Please select image file correctly!')

    # train classification model
    def classTrain(self):
        res=messagebox.askquestion('Model', 'Do you really want to rebuild the classification model?')
        if res == 'yes' :
            classify.build_model()
            messagebox.showinfo('Success', 'Classification model trained successfuly!')
        else :
            messagebox.showinfo('Return', 'Use pre-trained model!')       
        
    # classify test image
    def classTest(self):
        path = self.classTestEdit.text()
        if path == '': 
            messagebox.showerror("Error", 'There is not seletecd image file!')
        else:
            self.fileLabel.setText(path)
            self.classTestPath = path
            self.classTestImage = QtGui.QPixmap(path)
            self.classTestImage = self.classTestImage.scaled(900, 540)
            self.imageLabel.setPixmap(self.classTestImage)
        
        if self.classMdl == None:
            messagebox.showerror("Error", 'There is not pre-trained classification model!') 
            
        if path != '' and self.classMdl != None:
            pred_label = classify.predict(self.classMdl, path)
            real_label = path.split('/')[-1].split(' (')[0]
            result = 'Real: {}, Predict:{}'.format(real_label, pred_label)
            
            self.processLabel.setText("<font color='blue'>{}</font>".format(result))

        
    def segmentTrain(self):
        res=messagebox.askquestion('Model', 'Do you really want to rebuild the segment model?')
        if res == 'yes' :
            segment.build_model()
            messagebox.showinfo('Success', 'Segment model trained successfuly!')
        else :
            messagebox.showinfo('Return', 'Use pre-trained model!') 
        
    def segmentTest(self):
        path = self.segmentTestEdit.text()
        if path == '': 
            messagebox.showerror("Error", 'There is not seletecd image file!')
        else:
            self.fileLabel.setText(path)
            self.segmentTestPath = path
            self.segmentTestImage = QtGui.QPixmap(path)
            self.imageLabel.setPixmap(self.segmentTestImage)
        
        if self.classMdl == None:
            messagebox.showerror("Error", 'There is not pre-trained segment model!') 
            
        if path != '' and self.segmentMdl != None:
            bw_mask, pair_img = segment.predict(self.segmentMdl, path)
            w, h = bw_mask.shape
            area_rate = np.round(np.sum(bw_mask)/(w*h)*100, 2)
            
            height, width, channel = pair_img.shape
            bytesPerLine = 3 * width
            self.segemntTestImage = QtGui.QPixmap(QtGui.QImage(pair_img, width, height, bytesPerLine, QtGui.QImage.Format_RGB888))
            self.segemntTestImage = self.segemntTestImage.scaled(900, 540)
            self.imageLabel.setPixmap(self.segemntTestImage)

            result = 'Water area rate: {}%'.format(area_rate)            
            self.processLabel.setText("<font color='blue'>{}</font>".format(result))

        
    def drawSortImage(self):
        self.sortImage = QtGui.QPixmap(self.sortedImages[self.sortIdx])
        self.sortImage = self.sortImage.scaled(900, 540)
        self.imageLabel.setPixmap(self.sortImage)
        
        result = 'Rank: {} - Detected water area: {}%'.format(self.sortIdx, self.sortedAreas[self.sortIdx])
        self.processLabel.setText("<font color='blue'>{}</font>".format(result))
        self.fileLabel.setText(self.sortedImages[self.sortIdx + 1])
        
    def imageSort(self):
        self.sortPath = self.sortEdit.text()        
        if self.sortedImages == None:
            if self.segmentMdl != None:
                self.sortedAreas, self.sortedImages = segment.sortImage(self.segmentMdl, self.sortPath)
                self.sortIdx = 0
                self.sortMin = 0
                self.sortMax = len(self.sortedImages)
                
                self.drawSortImage()
            else:            
                messagebox.showerror("Error", 'There is not pre-trained segment model!') 
        else:
            res=messagebox.askquestion('Sort', 'Sorted already! Do you want to resort?')
            if res == 'yes':
                self.sortedAreas, self.sortedImages = segment.sortImage(self.segmentMdl, self.sortPath)
                self.sortIdx = 0
                self.sortMin = 0
                self.sortMax = len(self.sortedImages)
                
                self.drawSortImage()
            else:
                messagebox.showinfo('Return', 'Use pre-sorted images!')      
        
    def sortPrev(self):
        print(self.sortIdx)
        if self.segmentMdl != None:
            if self.sortIdx != None:
                self.sortIdx = self.sortIdx - 1
                if self.sortIdx == -1:
                    self.sortIdx = self.sortMax - 2
                self.drawSortImage()
            else:
                messagebox.showerror("Error", 'Images does not sorted!') 
        else:            
            messagebox.showerror("Error", 'There is not pre-trained segment model!') 
        
    def sortNext(self):
        if self.segmentMdl != None:
            if self.sortIdx != None:
                self.sortIdx = self.sortIdx + 1
                if self.sortIdx == self.sortMax - 1:
                    self.sortIdx = 0
                self.drawSortImage()
            else:
                messagebox.showerror("Error", 'Images does not sorted!') 
        else:            
            messagebox.showerror("Error", 'There is not pre-trained segment model!') 
        
    def compressImage(self):
        path = self.compressTestEdit.text()
        if path == '': 
            messagebox.showerror("Error", 'There is not seletecd image file!')
        else:
            self.fileLabel.setText(path)
            self.compressTestPath = path
            compress_process, pair_img = compress.compress_img(self.compressTestPath)
            
            height, width, channel = pair_img.shape
            bytesPerLine = 3 * width
            self.compressTestImage = QtGui.QPixmap(QtGui.QImage(pair_img, width, height, bytesPerLine, QtGui.QImage.Format_RGB888))
            self.compressTestImage = self.compressTestImage.scaled(900, 540)
            self.imageLabel.setPixmap(self.compressTestImage)

            self.compressResultEdit.setText('')
            for i in range(5):
                self.compressResultEdit.append(compress_process[i])

            self.processLabel.setText("<font color='blue'>{}</font>".format(compress_process[5]))
        
def run():
    app = QApplication(sys.argv)
    ex = App()
    ex.show()
    sys.exit(app.exec_())
if __name__ == '__main__':
    run()
