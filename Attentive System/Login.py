from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Login_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(1024, 768)
        Dialog.setStyleSheet("*{\n"
"        font-family:century gothic;\n"
"        font-size:24px;\n"
"    \n"
"}\n"
"\n"
"QDialog{\n"
"    background-image: url(:/images/images/background.jpg);\n"
"    background-repeat:no-repeat;\n"
"\n"
"}\n"
"\n"
"QLabel{\n"
"    color:white;\n"
"\n"
"}\n"
"\n"
"QFrame{\n"
"    background:#333;\n"
"    border-radius:15px;\n"
"    \n"
"}\n"
"\n"
"QLineEdit{\n"
"    background:transparent;\n"
"    border: none;\n"
"    color:white;\n"
"    border-bottom:1px solid #717072;\n"
"}\n"
"\n"
"QToolButton{\n"
"    background:red;\n"
"    border-radius:60px;\n"
"}")
        self.frame = QtWidgets.QFrame(Dialog)
        self.frame.setGeometry(QtCore.QRect(300, 220, 431, 421))
        self.frame.setStyleSheet("")
        self.frame.setObjectName("frame")
        self.username_lineEdit = QtWidgets.QLineEdit(self.frame)
        self.username_lineEdit.setGeometry(QtCore.QRect(60, 130, 321, 33))
        self.username_lineEdit.setObjectName("username_lineEdit")
        self.password_lineEdit = QtWidgets.QLineEdit(self.frame)
        self.password_lineEdit.setGeometry(QtCore.QRect(60, 200, 321, 33))
        self.password_lineEdit.setEchoMode(QtWidgets.QLineEdit.Password)
        self.password_lineEdit.setObjectName("password_lineEdit")
        self.label = QtWidgets.QLabel(self.frame)
        self.label.setGeometry(QtCore.QRect(160, 60, 131, 31))
        self.label.setObjectName("label")
        self.buttonBox = QtWidgets.QDialogButtonBox(self.frame)
        self.buttonBox.setGeometry(QtCore.QRect(60, 310, 321, 81))
        self.buttonBox.setOrientation(QtCore.Qt.Vertical)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.toolButton = QtWidgets.QToolButton(Dialog)
        self.toolButton.setGeometry(QtCore.QRect(460, 140, 121, 121))
        self.toolButton.setText("")
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/images/images/user.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.toolButton.setIcon(icon)
        self.toolButton.setIconSize(QtCore.QSize(128, 128))
        self.toolButton.setObjectName("toolButton")
        self.label_2 = QtWidgets.QLabel(Dialog)
        self.label_2.setGeometry(QtCore.QRect(60, 20, 891, 51))
        self.label_2.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.label_2.setStyleSheet("font: 75 30pt \"Consolas\";\n"
"background:transparent;")
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_2.setObjectName("label_2")

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Admin Login Dialog"))
        self.username_lineEdit.setPlaceholderText(_translate("Dialog", "UserName"))
        self.password_lineEdit.setPlaceholderText(_translate("Dialog", "Password"))
        self.label.setText(_translate("Dialog", "Login Here"))
        self.label_2.setText(_translate("Dialog", "FACE RECOGNITION BASED ATTENDANCE SYSTEM"))

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Dialog = QtWidgets.QDialog()
    ui = Ui_Login_Dialog()
    ui.setupUi(Dialog)
    Dialog.show()
    sys.exit(app.exec_())

