# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Attendance_Main_Gui.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1024, 586)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(MainWindow.sizePolicy().hasHeightForWidth())
        MainWindow.setSizePolicy(sizePolicy)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.centralwidget)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.imgLabel = QtWidgets.QLabel(self.centralwidget)
        self.imgLabel.setFrameShape(QtWidgets.QFrame.Box)
        self.imgLabel.setText("")
        self.imgLabel.setObjectName("imgLabel")
        self.horizontalLayout_2.addWidget(self.imgLabel)
        self.groupBox_5 = QtWidgets.QGroupBox(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.groupBox_5.sizePolicy().hasHeightForWidth())
        self.groupBox_5.setSizePolicy(sizePolicy)
        self.groupBox_5.setTitle("")
        self.groupBox_5.setObjectName("groupBox_5")
        self.gridLayout_4 = QtWidgets.QGridLayout(self.groupBox_5)
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.gridLayout_3 = QtWidgets.QGridLayout()
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.groupBox_3 = QtWidgets.QGroupBox(self.groupBox_5)
        self.groupBox_3.setObjectName("groupBox_3")
        self.gridLayout = QtWidgets.QGridLayout(self.groupBox_3)
        self.gridLayout.setObjectName("gridLayout")
        self.LiveManualAttendanceButton = QtWidgets.QPushButton(self.groupBox_3)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.LiveManualAttendanceButton.sizePolicy().hasHeightForWidth())
        self.LiveManualAttendanceButton.setSizePolicy(sizePolicy)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("images/attendance.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.LiveManualAttendanceButton.setIcon(icon)
        self.LiveManualAttendanceButton.setObjectName("LiveManualAttendanceButton")
        self.gridLayout.addWidget(self.LiveManualAttendanceButton, 1, 1, 1, 1)
        self.TakePhotoButton = QtWidgets.QPushButton(self.groupBox_3)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.TakePhotoButton.sizePolicy().hasHeightForWidth())
        self.TakePhotoButton.setSizePolicy(sizePolicy)
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap("images/picture.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.TakePhotoButton.setIcon(icon1)
        self.TakePhotoButton.setObjectName("TakePhotoButton")
        self.gridLayout.addWidget(self.TakePhotoButton, 0, 1, 1, 1)
        self.OpenCameraButton = QtWidgets.QPushButton(self.groupBox_3)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.OpenCameraButton.sizePolicy().hasHeightForWidth())
        self.OpenCameraButton.setSizePolicy(sizePolicy)
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap("images/photo-camera.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.OpenCameraButton.setIcon(icon2)
        self.OpenCameraButton.setCheckable(False)
        self.OpenCameraButton.setObjectName("OpenCameraButton")
        self.gridLayout.addWidget(self.OpenCameraButton, 0, 0, 1, 1)
        self.StopCameraButton = QtWidgets.QPushButton(self.groupBox_3)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.StopCameraButton.sizePolicy().hasHeightForWidth())
        self.StopCameraButton.setSizePolicy(sizePolicy)
        icon3 = QtGui.QIcon()
        icon3.addPixmap(QtGui.QPixmap("images/close camera.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.StopCameraButton.setIcon(icon3)
        self.StopCameraButton.setObjectName("StopCameraButton")
        self.gridLayout.addWidget(self.StopCameraButton, 1, 0, 1, 1)
        self.Monitor_checkBox = QtWidgets.QCheckBox(self.groupBox_3)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.Monitor_checkBox.sizePolicy().hasHeightForWidth())
        self.Monitor_checkBox.setSizePolicy(sizePolicy)
        self.Monitor_checkBox.setObjectName("Monitor_checkBox")
        self.gridLayout.addWidget(self.Monitor_checkBox, 2, 0, 1, 1)
        self.ShowResultsButton = QtWidgets.QPushButton(self.groupBox_3)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.ShowResultsButton.sizePolicy().hasHeightForWidth())
        self.ShowResultsButton.setSizePolicy(sizePolicy)
        self.ShowResultsButton.setObjectName("ShowResultsButton")
        self.gridLayout.addWidget(self.ShowResultsButton, 2, 1, 1, 1)
        self.gridLayout_3.addWidget(self.groupBox_3, 2, 0, 1, 2)
        self.groupBox_6 = QtWidgets.QGroupBox(self.groupBox_5)
        self.groupBox_6.setObjectName("groupBox_6")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.groupBox_6)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.timeEdit = QtWidgets.QTimeEdit(self.groupBox_6)
        self.timeEdit.setTime(QtCore.QTime(10, 0, 0))
        self.timeEdit.setObjectName("timeEdit")
        self.verticalLayout_2.addWidget(self.timeEdit)
        self.StartAutoAttendanceButton = QtWidgets.QPushButton(self.groupBox_6)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.StartAutoAttendanceButton.sizePolicy().hasHeightForWidth())
        self.StartAutoAttendanceButton.setSizePolicy(sizePolicy)
        self.StartAutoAttendanceButton.setObjectName("StartAutoAttendanceButton")
        self.verticalLayout_2.addWidget(self.StartAutoAttendanceButton)
        self.gridLayout_3.addWidget(self.groupBox_6, 1, 0, 1, 1)
        self.groupBox_2 = QtWidgets.QGroupBox(self.groupBox_5)
        self.groupBox_2.setObjectName("groupBox_2")
        self.gridLayout_7 = QtWidgets.QGridLayout(self.groupBox_2)
        self.gridLayout_7.setObjectName("gridLayout_7")
        self.BrowseButton = QtWidgets.QPushButton(self.groupBox_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.BrowseButton.sizePolicy().hasHeightForWidth())
        self.BrowseButton.setSizePolicy(sizePolicy)
        icon4 = QtGui.QIcon()
        icon4.addPixmap(QtGui.QPixmap("images/image files.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.BrowseButton.setIcon(icon4)
        self.BrowseButton.setObjectName("BrowseButton")
        self.gridLayout_7.addWidget(self.BrowseButton, 0, 0, 1, 1)
        self.UpdateImageAttendanceButton = QtWidgets.QPushButton(self.groupBox_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.UpdateImageAttendanceButton.sizePolicy().hasHeightForWidth())
        self.UpdateImageAttendanceButton.setSizePolicy(sizePolicy)
        self.UpdateImageAttendanceButton.setIcon(icon)
        self.UpdateImageAttendanceButton.setObjectName("UpdateImageAttendanceButton")
        self.gridLayout_7.addWidget(self.UpdateImageAttendanceButton, 1, 0, 1, 1)
        self.gridLayout_3.addWidget(self.groupBox_2, 1, 1, 1, 1)
        self.groupBox_4 = QtWidgets.QGroupBox(self.groupBox_5)
        self.groupBox_4.setObjectName("groupBox_4")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.groupBox_4)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.GenerateCSVButton = QtWidgets.QPushButton(self.groupBox_4)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.GenerateCSVButton.sizePolicy().hasHeightForWidth())
        self.GenerateCSVButton.setSizePolicy(sizePolicy)
        icon5 = QtGui.QIcon()
        icon5.addPixmap(QtGui.QPixmap("images/csv file.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.GenerateCSVButton.setIcon(icon5)
        self.GenerateCSVButton.setIconSize(QtCore.QSize(26, 16))
        self.GenerateCSVButton.setObjectName("GenerateCSVButton")
        self.gridLayout_2.addWidget(self.GenerateCSVButton, 0, 3, 1, 1)
        self.studentNameComboBox = QtWidgets.QComboBox(self.groupBox_4)
        self.studentNameComboBox.setObjectName("studentNameComboBox")
        self.gridLayout_2.addWidget(self.studentNameComboBox, 1, 0, 1, 2)
        self.percentageAttendanceButton = QtWidgets.QPushButton(self.groupBox_4)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.percentageAttendanceButton.sizePolicy().hasHeightForWidth())
        self.percentageAttendanceButton.setSizePolicy(sizePolicy)
        icon6 = QtGui.QIcon()
        icon6.addPixmap(QtGui.QPixmap("images/percentage.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.percentageAttendanceButton.setIcon(icon6)
        self.percentageAttendanceButton.setIconSize(QtCore.QSize(20, 20))
        self.percentageAttendanceButton.setObjectName("percentageAttendanceButton")
        self.gridLayout_2.addWidget(self.percentageAttendanceButton, 1, 2, 1, 2)
        self.monthComboBox = QtWidgets.QComboBox(self.groupBox_4)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.monthComboBox.sizePolicy().hasHeightForWidth())
        self.monthComboBox.setSizePolicy(sizePolicy)
        self.monthComboBox.setObjectName("monthComboBox")
        self.gridLayout_2.addWidget(self.monthComboBox, 2, 0, 1, 2)
        self.percentage_attendance_label = QtWidgets.QLabel(self.groupBox_4)
        font = QtGui.QFont()
        font.setPointSize(8)
        self.percentage_attendance_label.setFont(font)
        self.percentage_attendance_label.setFrameShape(QtWidgets.QFrame.Box)
        self.percentage_attendance_label.setText("")
        self.percentage_attendance_label.setObjectName("percentage_attendance_label")
        self.gridLayout_2.addWidget(self.percentage_attendance_label, 2, 2, 1, 2)
        self.refreshComboBoxButton = QtWidgets.QPushButton(self.groupBox_4)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.refreshComboBoxButton.sizePolicy().hasHeightForWidth())
        self.refreshComboBoxButton.setSizePolicy(sizePolicy)
        self.refreshComboBoxButton.setText("")
        icon7 = QtGui.QIcon()
        icon7.addPixmap(QtGui.QPixmap("images/refresh.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.refreshComboBoxButton.setIcon(icon7)
        self.refreshComboBoxButton.setIconSize(QtCore.QSize(25, 25))
        self.refreshComboBoxButton.setObjectName("refreshComboBoxButton")
        self.gridLayout_2.addWidget(self.refreshComboBoxButton, 0, 2, 1, 1)
        self.ShowRecordButton = QtWidgets.QPushButton(self.groupBox_4)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.ShowRecordButton.sizePolicy().hasHeightForWidth())
        self.ShowRecordButton.setSizePolicy(sizePolicy)
        icon8 = QtGui.QIcon()
        icon8.addPixmap(QtGui.QPixmap("images/database.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.ShowRecordButton.setIcon(icon8)
        self.ShowRecordButton.setObjectName("ShowRecordButton")
        self.gridLayout_2.addWidget(self.ShowRecordButton, 0, 0, 1, 2)
        self.gridLayout_3.addWidget(self.groupBox_4, 3, 0, 1, 2)
        self.verticalLayout_4 = QtWidgets.QVBoxLayout()
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.groupBox_7 = QtWidgets.QGroupBox(self.groupBox_5)
        self.groupBox_7.setObjectName("groupBox_7")
        self.gridLayout_5 = QtWidgets.QGridLayout(self.groupBox_7)
        self.gridLayout_5.setObjectName("gridLayout_5")
        self.addPersonButton = QtWidgets.QPushButton(self.groupBox_7)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.addPersonButton.sizePolicy().hasHeightForWidth())
        self.addPersonButton.setSizePolicy(sizePolicy)
        icon9 = QtGui.QIcon()
        icon9.addPixmap(QtGui.QPixmap("images/persons.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.addPersonButton.setIcon(icon9)
        self.addPersonButton.setIconSize(QtCore.QSize(25, 25))
        self.addPersonButton.setObjectName("addPersonButton")
        self.gridLayout_5.addWidget(self.addPersonButton, 0, 0, 1, 1)
        self.verticalLayout_4.addWidget(self.groupBox_7)
        self.gridLayout_3.addLayout(self.verticalLayout_4, 0, 0, 1, 1)
        self.ExitButton = QtWidgets.QPushButton(self.groupBox_5)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.ExitButton.sizePolicy().hasHeightForWidth())
        self.ExitButton.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(14)
        font.setBold(True)
        font.setItalic(False)
        font.setWeight(75)
        self.ExitButton.setFont(font)
        icon10 = QtGui.QIcon()
        icon10.addPixmap(QtGui.QPixmap("exit.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.ExitButton.setIcon(icon10)
        self.ExitButton.setObjectName("ExitButton")
        self.gridLayout_3.addWidget(self.ExitButton, 5, 0, 1, 2)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label_2 = QtWidgets.QLabel(self.groupBox_5)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_2.sizePolicy().hasHeightForWidth())
        self.label_2.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.horizontalLayout.addWidget(self.label_2)
        self.gridLayout_3.addLayout(self.horizontalLayout, 4, 0, 1, 1)
        self.groupBox = QtWidgets.QGroupBox(self.groupBox_5)
        self.groupBox.setObjectName("groupBox")
        self.gridLayout_6 = QtWidgets.QGridLayout(self.groupBox)
        self.gridLayout_6.setObjectName("gridLayout_6")
        self.TrainButton = QtWidgets.QPushButton(self.groupBox)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.TrainButton.sizePolicy().hasHeightForWidth())
        self.TrainButton.setSizePolicy(sizePolicy)
        icon11 = QtGui.QIcon()
        icon11.addPixmap(QtGui.QPixmap("images/train.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.TrainButton.setIcon(icon11)
        self.TrainButton.setObjectName("TrainButton")
        self.gridLayout_6.addWidget(self.TrainButton, 0, 0, 1, 1)
        self.AccuracyButton = QtWidgets.QPushButton(self.groupBox)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.AccuracyButton.sizePolicy().hasHeightForWidth())
        self.AccuracyButton.setSizePolicy(sizePolicy)
        icon12 = QtGui.QIcon()
        icon12.addPixmap(QtGui.QPixmap("images/accuracy.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.AccuracyButton.setIcon(icon12)
        self.AccuracyButton.setObjectName("AccuracyButton")
        self.gridLayout_6.addWidget(self.AccuracyButton, 1, 0, 1, 1)
        self.gridLayout_3.addWidget(self.groupBox, 0, 1, 1, 1)
        self.Accuracy_label = QtWidgets.QLabel(self.groupBox_5)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.Accuracy_label.sizePolicy().hasHeightForWidth())
        self.Accuracy_label.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(14)
        self.Accuracy_label.setFont(font)
        self.Accuracy_label.setFrameShape(QtWidgets.QFrame.Box)
        self.Accuracy_label.setText("")
        self.Accuracy_label.setObjectName("Accuracy_label")
        self.gridLayout_3.addWidget(self.Accuracy_label, 4, 1, 1, 1)
        self.gridLayout_4.addLayout(self.gridLayout_3, 0, 0, 1, 1)
        self.horizontalLayout_2.addWidget(self.groupBox_5)
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.groupBox_3.setTitle(_translate("MainWindow", "Manual Recognition"))
        self.LiveManualAttendanceButton.setText(_translate("MainWindow", "Attendance"))
        self.TakePhotoButton.setText(_translate("MainWindow", "Take Photo"))
        self.OpenCameraButton.setText(_translate("MainWindow", "Open Camera"))
        self.StopCameraButton.setText(_translate("MainWindow", "Stop Camera"))
        self.Monitor_checkBox.setText(_translate("MainWindow", "Monitor Attentiveness"))
        self.ShowResultsButton.setText(_translate("MainWindow", "Display Results"))
        self.groupBox_6.setTitle(_translate("MainWindow", "Auto Attandance"))
        self.StartAutoAttendanceButton.setToolTip(_translate("MainWindow", "Start Automatic Attendance"))
        self.StartAutoAttendanceButton.setText(_translate("MainWindow", "Start Auto"))
        self.groupBox_2.setTitle(_translate("MainWindow", "Image Recognition"))
        self.BrowseButton.setToolTip(_translate("MainWindow", "Browse Local Image for Recognition"))
        self.BrowseButton.setText(_translate("MainWindow", "Browse Image"))
        self.UpdateImageAttendanceButton.setToolTip(_translate("MainWindow", "Mark Attendance in MySQL Database"))
        self.UpdateImageAttendanceButton.setText(_translate("MainWindow", "Attendance"))
        self.groupBox_4.setTitle(_translate("MainWindow", "Attendance Record"))
        self.GenerateCSVButton.setToolTip(_translate("MainWindow", "Make CSV file for further Inquiries"))
        self.GenerateCSVButton.setText(_translate("MainWindow", "Make CSV File"))
        self.studentNameComboBox.setToolTip(_translate("MainWindow", "Select Students"))
        self.percentageAttendanceButton.setToolTip(_translate("MainWindow", "Get Percentile Attendance of Selected Student"))
        self.percentageAttendanceButton.setText(_translate("MainWindow", "Attendance"))
        self.monthComboBox.setToolTip(_translate("MainWindow", "Select Month"))
        self.refreshComboBoxButton.setToolTip(_translate("MainWindow", "Refresh Combo Boxes"))
        self.ShowRecordButton.setToolTip(_translate("MainWindow", "Show Complete Year Data"))
        self.ShowRecordButton.setText(_translate("MainWindow", "Show All Records"))
        self.groupBox_7.setTitle(_translate("MainWindow", "Dataset Creation"))
        self.addPersonButton.setToolTip(_translate("MainWindow", "Add New person to Dataset"))
        self.addPersonButton.setText(_translate("MainWindow", "Add Person"))
        self.ExitButton.setToolTip(_translate("MainWindow", "EXIT Program"))
        self.ExitButton.setText(_translate("MainWindow", "Exit"))
        self.label_2.setText(_translate("MainWindow", "Accuracy (%) :"))
        self.groupBox.setTitle(_translate("MainWindow", "Training Process"))
        self.TrainButton.setToolTip(_translate("MainWindow", "Train the LBPFaceRecogniser"))
        self.TrainButton.setText(_translate("MainWindow", "Training"))
        self.AccuracyButton.setToolTip(_translate("MainWindow", "Calculate the accuracy of Dataset"))
        self.AccuracyButton.setText(_translate("MainWindow", "Accuracy"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
