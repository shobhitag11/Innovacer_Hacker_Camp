# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'DBRecords.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_DBRecords(object):
    def setupUi(self, DBRecords):
        DBRecords.setObjectName("DBRecords")
        DBRecords.resize(750, 460)
        DBRecords.setMaximumSize(QtCore.QSize(750, 460))
        self.tableWidget = QtWidgets.QTableWidget(DBRecords)
        self.tableWidget.setGeometry(QtCore.QRect(10, 10, 731, 441))
        self.tableWidget.setRowCount(356)
        self.tableWidget.setColumnCount(100)
        self.tableWidget.setObjectName("tableWidget")
        self.tableWidget.horizontalHeader().setVisible(True)

        self.retranslateUi(DBRecords)
        QtCore.QMetaObject.connectSlotsByName(DBRecords)

    def retranslateUi(self, DBRecords):
        _translate = QtCore.QCoreApplication.translate
        DBRecords.setWindowTitle(_translate("DBRecords", "Dialog"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    DBRecords = QtWidgets.QDialog()
    ui = Ui_DBRecords()
    ui.setupUi(DBRecords)
    DBRecords.show()
    sys.exit(app.exec_())

