import copy
import csv
import datetime
import os
import pickle
import sys
from pathlib import Path

import pandas as pd
import win32com.client as wincl
import dlib
from imutils import face_utils

import MySQLdb as mdb
import cv2
import numpy as np
from PIL import Image
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import pyqtSlot, QTimer, QAbstractTableModel
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import label_binarize

from PyQt5.uic import loadUi
from AddPersonDialog import Ui_Dialog
from DBRecords import Ui_DBRecords
from Login import Ui_Login_Dialog
from results import Ui_Dialog_results

class AttendanceSystemML(QMainWindow):
    def __init__(self):
        super(AttendanceSystemML, self).__init__()

        loadUi('Attendance_Main_Gui.ui', self)

        self.addPersonButton.clicked.connect(self.addNewPersonClicked)
        self.TrainButton.clicked.connect(self.trainingClicked)
        self.AccuracyButton.clicked.connect(self.calculateAccuracyClicked)
        self.BrowseButton.clicked.connect(self.browseClicked)
        self.UpdateImageAttendanceButton.clicked.connect(self.detectAttendanceClicked)
        self.OpenCameraButton.clicked.connect(self.openCameraClicked)
        self.TakePhotoButton.clicked.connect(self.takePhotoClicked)
        self.LiveManualAttendanceButton.clicked.connect(self.detectAttendanceClicked)
        self.ShowResultsButton.clicked.connect(self.Display_Results)
        self.StopCameraButton.clicked.connect(self.stopCameraClicked)
        self.StartAutoAttendanceButton.clicked.connect(self.startAutoAttendanceClicked)
        self.ShowRecordButton.clicked.connect(self.showdbRecords)
        self.refreshComboBoxButton.clicked.connect(self.refreshComboBoxesClicked)
        self.GenerateCSVButton.clicked.connect(self.generateCSVfile)
        self.percentageAttendanceButton.clicked.connect(self.percentileAttendanceClicked)
        self.ExitButton.clicked.connect(self.exitClicked)

        # grab the indexes of the facial landmarks for the left and
        # right eye, respectively

        self.predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

        (self.lStart, self.lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (self.rStart, self.rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
        (self.mStart, self.mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

        self.faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()

        my_file = Path("./trainingmodel.yml")

        if my_file.is_file():
            self.recognizer.read('./trainingmodel.yml')
            print('Training Data Loaded !\n')
        else:
            print('No training File exists')

        names_file = Path("./trainlabelnames.pkl")

        if names_file.is_file():
            with open('trainlabelnames.pkl', 'rb') as f:
                self.names = pickle.load(f)
                self.studentNameComboBox.addItems(self.names)
        else:
            print('No Names File exists')

        months = []

        for i in range(1, 13):
            months.append(datetime.date(2008, i, 1).strftime('%B'))

        self.monthComboBox.addItems(months)

        self.font = cv2.FONT_HERSHEY_PLAIN
        self.global_features = []
        self.global_labels = []
        # train_test_split size
        self.test_size = 0.10
        # seed for reproducing same results
        self.seed = 9

        self.EYE_AR_THRESH = 0.3
        self.EYE_AR_CONSEC_FRAMES = 20
        self.COUNTER = 0
        self.YAWN_COUNTER = 0
        self.FACE_COUNTER = 0
        self.ALARM_ON = False
        self.start_detection_Flag = False
        self.frequency = 2500
        self.duration = 500
        self.speak = wincl.Dispatch("SAPI.SpVoice")
        self.systemStatusFlag = False
        self.global_features = []
        self.global_labels = []

        self.person_1_eye_close_count = 0
        self.person_2_eye_close_count = 0
        self.person_3_eye_close_count = 0
        self.person_4_eye_close_count = 0
        self.person_5_eye_close_count = 0

        self.person_1_yawn_count = 0
        self.person_2_yawn_count = 0
        self.person_3_yawn_count = 0
        self.person_4_yawn_count = 0
        self.person_5_yawn_count = 0


    @pyqtSlot()
    def addNewPersonClicked(self):
        self.openDialogWindow()

    def openDialogWindow(self):
        Dialog = QtWidgets.QDialog()
        ui = Ui_Dialog()
        ui.setupUi(Dialog)
        Dialog.show()
        response = Dialog.exec_()

        if response == QtWidgets.QDialog.Accepted:
            self.Dialogtext = ui.lineEdit.text()
            # print( self.Dialogtext)
            if self.Dialogtext:
                print(self.Dialogtext)
                self.CreateNewPersonData(self.Dialogtext)
            else:
                print("No variable set")
        else:
            print("No variable set")

    def CreateNewPersonData(self, personName):
        path = 'dataset'
        folderPath = [os.path.join(path, f) for f in os.listdir(path)]
        numfolders = len(folderPath)
        newfoldername = str(numfolders + 1) + '.' + personName
        finalpath = os.path.join(path, newfoldername)
        print(newfoldername, finalpath)
        os.mkdir(finalpath)

        self.capture = cv2.VideoCapture(0)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)

        for i in range(100):
            ret, self.image = self.capture.read()
            self.image = cv2.flip(self.image, 1)
            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            faces = self.faceCascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                self.DisplayImage(self.image, 1)
                imagename = os.path.join(finalpath, 'capture_' + str(i) + '.jpg ')
                # cv2.imwrite(imagename, gray[y:y + h, x:x + w])
                cv2.imwrite(imagename, self.image)

        self.capture.release()
        del self.capture
        cv2.destroyAllWindows()

    def manipulateRollRecords(self):
        names_file = Path("./trainlabelnames.pkl")

        if names_file.is_file():
            with open('trainlabelnames.pkl', 'rb') as f:
                names = pickle.load(f)

                sql_delete_roll_record = "DELETE FROM `student_roll_record` WHERE 1"
                self.openMySQLdb()  # open database connection
                cursor = self.db.cursor()

                try:
                    cursor.execute(sql_delete_roll_record)
                    self.db.commit()

                    for indx, student_name in enumerate(names, 1):
                        #    print(indx, student_name)
                        db_alias = 'student' + '_' + str(indx)
                        # print(db_alias, names[indx - 1])
                        # INSERT INTO `student_roll_record`(`roll_number`, `student_db_alias`, `student_name`) VALUES ('1','student_1','Salmaan')
                        sql_insert_new_student_roll_record = "INSERT INTO `student_roll_record`(`roll_number`, `student_db_alias`, `student_name`) " \
                                                             "VALUES (" "'" + str(indx) + "'" "," + "'" + db_alias + "'" "," + "'" + names[
                                                                 indx - 1] + "'" ")"
                        cursor.execute(sql_insert_new_student_roll_record)
                        self.db.commit()
                except:
                    self.db.rollback()  # Rollback in case there is any error
                self.closeMYSQLdb()  # close database connection
        else:
            print('No Names File exists')

    def OpenDBRecordWindow(self):
        self.window = QtWidgets.QDialog()
        self.ui = Ui_DBRecords()
        self.ui.setupUi(self.window)
        self.window.show()

        self.openMySQLdb()

        cur = self.db.cursor()

        cur.execute("SHOW COLUMNS FROM student_attendance_table")
        result_temp = cur.fetchall()

        my_list = []

        for row_num_temp, row_data_temp in enumerate(result_temp):
            my_list.append(row_data_temp[0])

        self.ui.tableWidget.setHorizontalHeaderLabels(my_list)

        cur.execute("SELECT * FROM student_attendance_table")
        result = cur.fetchall()
        self.ui.tableWidget.setRowCount(0)

        for row_num, row_data in enumerate(result):
            self.ui.tableWidget.insertRow(row_num)
            for col_num, data in enumerate(row_data):
                self.ui.tableWidget.setItem(row_num, col_num, QtWidgets.QTableWidgetItem(str(data)))

        self.closeMYSQLdb()

    @pyqtSlot()
    def trainingClicked(self):

        faceSamples, ids = self.get_images_and_labels(plotting=1)
        self.global_features = faceSamples
        self.global_labels = np.array(ids)
        self.recognizer.train(faceSamples, np.array(ids))
        self.recognizer.write('trainingmodel.yml')
        self.manipulateRollRecords()

    def get_images_and_labels(self, plotting):
        path = 'dataset'
        faceSamples = []
        ids = []
        names = []
        imagePath = [os.path.join(path, f) for f in os.listdir(path)]

        for imagePath in imagePath:
            tempid = os.path.split(imagePath)
            person_id = tempid[1]
            retval = person_id.split('.')
            names.append(retval[1])
            # print(person_id)

            filelist = os.listdir(imagePath)
            # print(filelist)

            for imgfilename in filelist:
                imgfilenamewithpath = os.path.join(imagePath, imgfilename)
                print(imgfilenamewithpath)

                PIL_img = Image.open(imgfilenamewithpath).convert('L')
                img_numpy = np.array(PIL_img, 'uint8')

                faces = self.faceCascade.detectMultiScale(img_numpy)

                for (x, y, w, h) in faces:
                    self.image = cv2.imread(imgfilenamewithpath)

                    if plotting == 1:
                        cv2.rectangle(self.image, (x - 20, y - 20), (x + w + 20, y + h + 20), (0, 255, 0), 4)
                        self.DisplayImage(self.image, 1)

                    faceSamples.append(img_numpy[y:y + h, x:x + w])
                    # ids.append(int(person_id))
                    ids.append(int(retval[0]))

        with open('trainlabelnames.pkl', 'wb') as f:
            pickle.dump(names, f)

        return faceSamples, ids

    @pyqtSlot()
    def calculateAccuracyClicked(self):

        global_features, global_labels = self.get_images_and_labels(plotting=0)

        num_rows = global_features.__len__()
        y_true = copy.deepcopy(global_labels)
        y_pred = []
        confidence = []

        for i in range(num_rows):
            sample = global_features[i]
            new_training_feature = copy.deepcopy(global_features)
            new_training_label = copy.deepcopy(global_labels)
            del new_training_feature[i]
            del new_training_label[i]
            self.recognizer.train(new_training_feature, np.array(new_training_label))
            y, conf = self.recognizer.predict(np.array(sample))

            y_pred.append(y)
            confidence.append(conf)

        conf_mat = confusion_matrix(y_true, y_pred)
        acc = accuracy_score(y_true, y_pred)

        print("CONFUSION MATRIX\n")
        print(conf_mat)

        print("Accuracy: ", acc * 100, "%")
        self.Accuracy_label.setText(str(acc * 100))

    @pyqtSlot()
    def browseClicked(self):
        fname, filter = QFileDialog.getOpenFileName(self, 'Open File', '.\\', "Image Files (*.*)")
        if fname:
            self.LoadImageFunction(fname)
        else:
            print("Invalid Image")

    def LoadImageFunction(self, fname):
        self.image = cv2.imread(fname)
        self.DisplayImage(self.image, 1)

    def DisplayImage(self, img, window=1):
        qformat = QImage.Format_Indexed8
        if len(img.shape) == 3:
            if (img.shape[2]) == 4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888

        outImg = QImage(img, img.shape[1], img.shape[0], img.strides[0], qformat)

        outImg = outImg.rgbSwapped()

        if window == 1:
            self.imgLabel.setPixmap(QPixmap.fromImage(outImg))
            self.imgLabel.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)
            self.imgLabel.setScaledContents(False)

    @pyqtSlot()
    def takePhotoClicked(self):
        ret, self.image = self.capture.read()
        self.image = cv2.flip(self.image, 1)
        self.stopCameraClicked()
        self.DisplayImage(self.image, 1)

    @pyqtSlot()
    def openCameraClicked(self):
        self.capture = cv2.VideoCapture(0)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(0.1)

    def update_frame(self):
        ret, self.image = self.capture.read()
        self.image = cv2.flip(self.image, 1)

        if self.Monitor_checkBox.isChecked():
            self.MonitorAttentivenessFunction(self.image)
        else:
            self.DisplayImage(self.image, 1)

    @pyqtSlot()
    def detectAttendanceClicked(self):
        self.DetectFacesFunction(self.image)

    def DetectFacesFunction(self, img):

        # cv2.resize(image, (width, height))

        img = cv2.resize(img, (640, 480))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.faceCascade.detectMultiScale(gray, 1.2, 5, minSize=(90, 90))
        persons_list = []

        if len(faces) is not 0:

            for (x, y, w, h) in faces:

                # Recognize the face belongs to which ID
                person_id, confidence = self.recognizer.predict(gray[y:y + h, x:x + w])
                persons_list.append(person_id)
                print("Person ID: ", str(person_id), " Confidence: ", str(confidence))

                with open('trainlabelnames.pkl', 'rb') as f:
                    names = pickle.load(f)

                if 20 < confidence < 50:
                    if person_id == 0:
                        person = "Unknown"
                        print('Unknown Person !')
                    else:
                        person = names[person_id - 1]
                        print('Person Recognised is', person)
                        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
                        cv2.putText(img, str(person), (x, y - 10), self.font, 2, (0, 255, 0), 2)
                        self.DisplayImage(img, 1)

                # If recognition confidence is above threshold, then it is Unknown

                else:
                    person = "Unknown"
                    print('Unknown Person !')
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    cv2.putText(img, str(person), (x, y - 10), self.font, 2, (0, 255, 0), 2)
                    self.DisplayImage(img, 1)

            print("Persons list: ", persons_list)
            self.updateDatabase(persons_list)
        else:
            print("No Face Detected !")

    def updateDatabase(self, persons_list):
        print("Database Open And update students attendance record\n")

        ''' *************************************************************************************************

                        D A T B A S E       C R E A T I O N

            *************************************************************************************************
        '''

        ''' *************************************************************************************************
            the index 0 is for unknown person, so we don't need to check for the unknown person
            we only need to check the rest of the students
            
            thus, the classes will be [1, 2, 3, 4]       
            arr = label_binarize(persons_list, classes=[0, 1, 2, 3]) 
            *************************************************************************************************
        '''

        # Check the existing Database for names and number of persons

        with open('trainlabelnames.pkl', 'rb') as f:
            names = pickle.load(f)

        num_persons = names.__len__()

        '''     THE FOLLOWING IS THE CONSTRAINT OF label_binarize() FUNCTION
                https://stackoverflow.com/questions/31947140/sklearn-labelbinarizer-returns-vector-when-there-are-2-classes        
        '''

        if num_persons <= 2:
            self.showMessagebox("Atleast 3 students are required to be enrolled in database. Add another student.")
        else:

            ''' *************************************************************************************************
                Find number of students existing in database using
                SELECT COUNT(*) FROM student_roll_record 
                *************************************************************************************************
            '''

            sql_find_students = "SELECT COUNT(*) FROM student_roll_record"

            self.openMySQLdb()  # open database connection

            cursor = self.db.cursor()
            try:
                cursor.execute(sql_find_students)  # Execute the SQL command
                results = cursor.fetchall()

                if len(results) is not 0:
                    self.db.commit()  # Commit your changes in the database
                    for row in results:
                        student_count = row[0]
                        # print(row[0])
                else:
                    print("No student List Found")

            except:
                self.db.rollback()  # Rollback in case there is any error

            self.closeMYSQLdb()  # close database connection

            ''' *************************************************************************************************
                Updating Student ROLL NUMBER/NAME Database with new records !
                *************************************************************************************************
            '''

            if student_count == num_persons:
                print("All students Enrolled")
            elif student_count < num_persons:
                print("Database has less number of students enrolled than in training dataset")
                print("Updating Student ROLL NUMBER/NAME Database with new records !")

                sql_delete_roll_record = "DELETE FROM `student_roll_record` WHERE 1"

                self.openMySQLdb()  # open database connection

                cursor = self.db.cursor()

                try:
                    cursor.execute(sql_delete_roll_record)
                    self.db.commit()

                    for indx, student_name in enumerate(names, 1):
                        #    print(indx, student_name)
                        db_alias = 'student' + '_' + str(indx)
                        # print(db_alias, names[indx - 1])

                        # INSERT INTO `student_roll_record`(`roll_number`, `student_db_alias`, `student_name`) VALUES ('1','student_1','Salmaan')

                        sql_insert_new_student_roll_record = "INSERT INTO `student_roll_record`(`roll_number`, `student_db_alias`, `student_name`) " \
                                                             "VALUES (" "'" + str(indx) + "'" "," + "'" + db_alias + "'" "," + "'" + names[
                                                                 indx - 1] + "'" ")"

                        cursor.execute(sql_insert_new_student_roll_record)
                        self.db.commit()

                except:
                    self.db.rollback()  # Rollback in case there is any error

                self.closeMYSQLdb()  # close database connection

            else:
                print("Check Database !")

            ''' *************************************************************************************************
                Check if attendance record has all the students aliases using
                SELECT COUNT(*) FROM information_schema.columns WHERE TABLE_NAME='student_attendance_table'
                SHOW columns FROM `student_attendance_table` 
                *************************************************************************************************
            '''

            sql_find_columns_names = "SHOW columns FROM `student_attendance_table` "

            self.openMySQLdb()  # open database connection

            cursor = self.db.cursor()
            try:
                cursor.execute(sql_find_columns_names)  # Execute the SQL command
                results = cursor.fetchall()

                num_columns = results.__len__()

                if num_columns == 8:  # there are 8 columns in default database record
                    print("No Student enrolled in database !")
                    print("Adding students in train dataset to MySQL database.")

                    try:
                        for indx, student_name in enumerate(names, 1):
                            #    print(indx, student_name)
                            db_alias = 'student' + '_' + str(indx)
                            # print(db_alias, names[indx - 1])

                            sql_add_new_student_column = "ALTER TABLE `student_attendance_table` ADD `" + db_alias + "` CHAR(1) NULL DEFAULT 'A'"

                            # sql_insert_new_student_roll_record = "INSERT INTO `student_roll_record`(`roll_number`, `student_db_alias`, `student_name`) " \
                            #                                      "VALUES (" "'" + str(indx) + "'" "," + "'" + db_alias + "'" "," + "'" + names[indx - 1] + "'" ")"

                            cursor.execute(sql_add_new_student_column)
                            self.db.commit()

                    except Exception:
                        self.db.rollback()  # Rollback in case there is any error

                # if database has less number of persons than in train dataset, then add the remaining students to database
                elif num_columns - 8 < num_persons:

                    num_persons_already_in_db = num_columns - 8
                    # num_of_persons_to_be_added = num_persons - num_persons_already_in_db

                    try:
                        for i in range(num_persons_already_in_db, num_persons):
                            index = 1 + i
                            db_alias_to_be_added = 'student' + '_' + str(index)

                            # ALTER TABLE `student_attendance_table` ADD `student_3` CHAR(1) NULL DEFAULT 'A' AFTER `student_2`;

                            sql_add_new_student_column = "ALTER TABLE `student_attendance_table` ADD `" + db_alias_to_be_added + "` CHAR(1) NULL DEFAULT 'A'"
                            cursor.execute(sql_add_new_student_column)
                            self.db.commit()

                    except Exception:
                        self.db.rollback()  # Rollback in case there is any error

            except Exception:
                self.db.rollback()  # Rollback in case there is any error

            self.closeMYSQLdb()  # close database connection

            ''' *************************************************************************************************
                    
                    D A T B A S E       U P D A T I O N
            
                *************************************************************************************************
            '''

            labels = [x + 1 for x in range(num_persons)]

            arr = label_binarize(persons_list, classes=labels)

            # arr = label_binarize(persons_list, classes=[1, 2, 3, 4])
            arr_t = sum(arr)
            # arr_t_ss = str(arr_t[0]) + "," + str(arr_t[1]) + "," + str(arr_t[2]) + "," + str(arr_t[3])

            # sql_fetch_current_date_attendance = "SELECT * FROM `student_attendance_table` WHERE `db_date` = CURRENT_DATE"

            self.openMySQLdb()  # open database connection
            cursor = self.db.cursor()

            try:
                for x in range(num_persons):

                    if arr_t[x] == 1:
                        status = 'P'
                    else:
                        status = 'A'

                    db_alias_to_be_updated = 'student' + '_' + str(x + 1)
                    # sql_update_attendance = "INSERT INTO `student_attendance_table`(`" + db_alias_to_be_updated + "`) VALUES (" + status + ")"
                    sql_update_attendance = "UPDATE `student_attendance_table` SET `" + db_alias_to_be_updated + "`= " "'" + status + "'" "WHERE `db_date` = CURRENT_DATE"
                    cursor.execute(sql_update_attendance)
                    self.db.commit()

            except:
                self.db.rollback()  # Rollback in case there is any error
            self.closeMYSQLdb()  # close database connection

    def showMessagebox(self, text):
        mb = QMessageBox()
        mb.setIcon(QMessageBox.Warning)
        # mb.setText("Atleast 3 students are required to be enrolled in database. Add another student.")
        mb.setText(text)
        mb.setWindowTitle("Warning")
        mb.setStandardButtons(QMessageBox.Ok)
        mb.exec_()

    def MonitorAttentivenessFunction(self, img):

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

        if len(faces) is not 0:
            self.FACE_COUNTER = 0

            with open('trainlabelnames.pkl', 'rb') as f:
                names = pickle.load(f)

            for (x, y, w, h) in faces:

                # Recognize the face belongs to which ID
                person_id, confidence = self.recognizer.predict(gray[y:y + h, x:x + w])
                print("Person ID: ", str(person_id), " Confidence: ", str(confidence))

                if 20 < confidence < 50:
                    if person_id == 0:
                        person = "Unknown"
                        print('Unknown Person !')
                    else:
                        person = names[person_id - 1]
                        print('Person Recognised is', person)
                else:
                    person = "Unknown"
                    print('Unknown Person !')

                ##################################################################################

                rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
                shape = self.predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)

                leftEye = shape[self.lStart:self.lEnd]
                rightEye = shape[self.rStart:self.rEnd]
                leftEAR = self.eye_aspect_ratio(leftEye)
                rightEAR = self.eye_aspect_ratio(rightEye)

                mouthshape = shape[self.mStart:self.mEnd]
                mouthOpenDistance = self.euclidean_dist(mouthshape[18], mouthshape[14])
                
                print("mouthOpenDistance: ", mouthOpenDistance)

                ear = (leftEAR + rightEAR) / 2.0

                leftEyeHull = cv2.convexHull(leftEye)
                rightEyeHull = cv2.convexHull(rightEye)
                cv2.drawContours(img, [leftEyeHull], -1, (0, 255, 0), 1)
                cv2.drawContours(img, [rightEyeHull], -1, (0, 255, 0), 1)

                if ear < self.EYE_AR_THRESH:
                    self.COUNTER += 1
                    # print('Counter:',self.COUNTER)

                    # if the eyes were closed for a sufficient number of
                    # frames, then sound the alarm
                    if self.COUNTER >= self.EYE_AR_CONSEC_FRAMES:
                        # if the alarm is not on, turn it on
                        if not self.ALARM_ON:
                            self.ALARM_ON = True
                            # print("ALARM_ON")
                            # winsound.Beep(self.frequency, self.duration)

                        if person_id == 1:
                            self.person_1_eye_close_count += 1
                        elif person_id == 2:
                            self.person_2_eye_close_count += 1
                        elif person_id == 3:
                            self.person_3_eye_close_count += 1
                        elif person_id == 4:
                            self.person_4_eye_close_count += 1
                        elif person_id == 5:
                            self.person_5_eye_close_count += 1

                        # draw an alarm on the frame
                        start_point = (x, y)
                        end_point = (x + w, y + h)
                        color = (0, 0, 255)
                        thickness = 1
                        img = cv2.rectangle(img, start_point, end_point, color, thickness)
                        cv2.putText(img, str(person), (x, y - 10), self.font, 2, (0, 255, 0), 2)
                        cv2.putText(img, "InAttentive !", (int(x), int(y - 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                        print('Eye Close Counts: ', [self.person_1_eye_close_count, self.person_2_eye_close_count, self.person_3_eye_close_count,
                                                     self.person_4_eye_close_count, self.person_5_eye_close_count])

                        self.DisplayImage(img, 1)

                    # otherwise, the eye aspect ratio is not below the blink
                    # threshold, so reset the counter and alarm
                else:
                    self.COUNTER = 0
                    self.ALARM_ON = False

                if mouthOpenDistance > 5:
                    self.YAWN_COUNTER += 1

                    if self.YAWN_COUNTER >= 1:
                        print('Yawning !')
                        # self.speak.Speak("feeling sleepy")

                        if not self.ALARM_ON:
                            self.ALARM_ON = True
                            # print("ALARM_ON")
                            # winsound.Beep(self.frequency, self.duration)

                            if person_id == 1:
                                self.person_1_yawn_count += 1
                            elif person_id == 2:
                                self.person_2_yawn_count += 1
                            elif person_id == 3:
                                self.person_3_yawn_count += 1
                            elif person_id == 4:
                                self.person_4_yawn_count += 1
                            elif person_id == 5:
                                self.person_5_yawn_count += 1

                            # draw an alarm on the frame
                            start_point = (x, y)
                            end_point = (x + w, y + h)
                            color = (0, 0, 255)
                            thickness = 1
                            img = cv2.rectangle(img, start_point, end_point, color, thickness)
                            cv2.putText(img, "YAWNING !", (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                            print('Yawn Counts: ',
                                  [self.person_1_yawn_count, self.person_2_yawn_count, self.person_3_yawn_count, self.person_4_yawn_count, self.person_5_yawn_count])

                            self.DisplayImage(img, 1)

                else:
                    self.YAWN_COUNTER = 0
                    self.ALARM_ON = False

                # draw the computed eye aspect ratio on the frame to help
                # with debugging and setting the correct eye aspect ratio
                # thresholds and frame counters
                cv2.putText(img, "EAR: {:.3f}".format(ear), (500, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                self.DisplayImage(img, 1)

            else:
                self.FACE_COUNTER += 1

                if self.FACE_COUNTER >= 15:
                    print('Not AWAKE !')
                    self.speak.Speak("Wake UP")

                    cv2.putText(img, "Not AWAKE !", (100, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                    self.DisplayImage(img, 1)

    @pyqtSlot()
    def Display_Results(self):
        self.Process_Attentiveness()
        self.Prepare_Results()

    def Process_Attentiveness(self):
        # #####################################################
        #               PROCESS     ATTENTIVENESS
        # #####################################################

        eye_close_counts = [self.person_1_eye_close_count, self.person_2_eye_close_count, self.person_3_eye_close_count,
                            self.person_4_eye_close_count, self.person_5_eye_close_count]
        yawn_counts = [self.person_1_yawn_count, self.person_2_yawn_count, self.person_3_yawn_count, self.person_4_yawn_count,
                       self.person_5_yawn_count]

        max_eye_close_counts = max(eye_close_counts)
        max_yawn_counts = max(yawn_counts)

        eye_close_count_all = []
        eye_close_percents = []
        yawn_counts_count_all = []
        yawn_counts_percents = []

        if max_eye_close_counts is not 0:  # prevent division by zero
            for i in range(len(eye_close_counts)):
                temp = eye_close_counts[i] / max_eye_close_counts
                eye_close_count_all.append(temp)
                temp_per = (1 - temp) * 100
                eye_close_percents.append(temp_per)

        if max_yawn_counts is not 0:  # prevent division by zero
            for j in range(len(yawn_counts)):
                temp = yawn_counts[j] / max_yawn_counts
                yawn_counts_count_all.append(temp)
                temp_per = (1 - temp) * 100
                yawn_counts_percents.append(temp_per)

        print('eye_close_percents', eye_close_percents)
        print('yawn_counts_percents', yawn_counts_percents)

        with open('Eye_Close_and_Yawn_Percentages.pkl', 'wb') as f:
            pickle.dump([eye_close_counts, eye_close_percents, yawn_counts, yawn_counts_percents], f)

    def Prepare_Results(self):

        with open('trainlabelnames.pkl', 'rb') as f:
            names = pickle.load(f)

        with open('Eye_Close_and_Yawn_Percentages.pkl', 'rb') as f:
            eye_close_count_all, eye_close_percents, yawn_counts_count_all, yawn_counts_percents = pickle.load(f)

        df = pd.DataFrame(list(zip(names, eye_close_count_all, eye_close_percents, yawn_counts_count_all, yawn_counts_percents)),
                          columns=['Names', 'Eye Blink Count', 'Attentiveness Percentage', 'Yawn Count', 'Non Fatigue Percentage'])

        Dialog = QtWidgets.QDialog()
        ui = Ui_Dialog_results()
        ui.setupUi(Dialog)
        model = pandasModel(df)
        ui.tableView.setModel(model)
        Dialog.show()
        response = Dialog.exec_()

        if response == QtWidgets.QDialog.Accepted:
            print(df)
        else:
            print("No variable set")

    def eye_aspect_ratio(self, eye):
        # compute the euclidean distances between the two sets of
        # vertical eye landmarks (x, y)-coordinates

        A = self.euclidean_dist(eye[1], eye[5])
        B = self.euclidean_dist(eye[2], eye[4])

        # compute the euclidean distance between the horizontal
        # eye landmark (x, y)-coordinates
        C = self.euclidean_dist(eye[0], eye[3])

        # compute the eye aspect ratio
        ear = (A + B) / (2.0 * C)

        # return the eye aspect ratio
        return ear

    def euclidean_dist(self, ptA, ptB):
        # compute and return the euclidean distance between the two
        # points
        return np.linalg.norm(ptA - ptB)

    @pyqtSlot()
    def startAutoAttendanceClicked(self):
        now = datetime.datetime.now()
        current_hour = now.strftime("%H")
        current_minute = now.strftime("%M")
        current_seconds = now.strftime("%S")
        current_time = now.strftime("%H:%M:%S")

        set_time_temp = self.timeEdit.time()
        set_time = set_time_temp.toString()

        while current_time != set_time:
            now = datetime.datetime.now()
            current_time = now.strftime("%H:%M:%S")
            print(current_time, end='\r', flush=True)

        self.openCameraClicked()
        self.takePhotoClicked()
        self.detectAttendanceClicked()

    def openMySQLdb(self):
        try:
            self.db = mdb.connect('localhost', 'root', '', 'attendance_system')
            # QMessageBox.about(self, 'Connection', 'Successfully Connected to DB')
            print("Successfully Connected to DB")
        except mdb.Error as err:
            # QMessageBox.about(self, 'Connection', 'Not Connected to DB')
            print("Not Connected to DB. Error: ", err)
            sys.exit(1)

    def closeMYSQLdb(self):
        if self.db:
            self.db.close()
        else:
            print("No earlier Database connection available.")

    @pyqtSlot()
    def showdbRecords(self):
        self.OpenDBRecordWindow()

    @pyqtSlot()
    def refreshComboBoxesClicked(self):

        self.studentNameComboBox.clear()
        self.monthComboBox.clear()

        names_file = Path("./trainlabelnames.pkl")

        if names_file.is_file():
            with open('trainlabelnames.pkl', 'rb') as f:
                self.names = pickle.load(f)
                self.studentNameComboBox.addItems(self.names)
        else:
            print('No Names File exists')

        months = []

        for i in range(1, 13):  # fill months in combobox   http://strftime.org/
            months.append(datetime.date(2008, i, 1).strftime('%B'))

        self.monthComboBox.addItems(months)

    @pyqtSlot()
    def generateCSVfile(self):

        self.openMySQLdb()
        cur = self.db.cursor()
        cur.execute("SHOW columns FROM student_attendance_table")

        myheaderlist = []
        header_names = cur.fetchall()
        # header_names[0][0] = db_date
        # header_names[1][0] = year
        # header_names[2][0] = month
        # header_names[3][0] = day
        # header_names[4][0] = quarter
        # header_names[5][0] = week
        # and so on ...

        for i in range(len(header_names)):
            myheaderlist.append(header_names[i][0])

        cur.execute("SELECT * FROM student_attendance_table")

        result_temp = cur.fetchall()

        with open('outfile.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerow(myheaderlist)

            for row in result_temp:
                writer.writerow(row)

            f.close()
        self.closeMYSQLdb()

    @pyqtSlot()
    def stopCameraClicked(self):
        self.timer.stop()
        self.capture.release()

    @pyqtSlot()
    def percentileAttendanceClicked(self):
        selected_student = str(self.studentNameComboBox.currentText())
        selected_month = str(self.monthComboBox.currentText())

        # print(selected_student, selected_month)

        # get student roll record from `student_roll_record` table and attendance data from `student_attendance_table`
        # SELECT * FROM `student_roll_record` WHERE `student_name` = 'Salmaan'

        sql_student_roll = "SELECT * FROM `student_roll_record` WHERE `student_name` = '" + selected_student + "' "

        self.openMySQLdb()

        cur = self.db.cursor()
        cur.execute(sql_student_roll)
        self.db.commit()

        result_temp = cur.fetchall()
        student_db_alias = result_temp[0][1]
        student_roll_num = result_temp[0][0]

        # SELECT `student_1` FROM `student_attendance_table` WHERE `month_name` = 'July'

        sql_student_record_monthly = "SELECT `" + student_db_alias + "` FROM `student_attendance_table` WHERE `month_name` = '" + selected_month + "' "

        cur = self.db.cursor()
        cur.execute(sql_student_record_monthly)
        self.db.commit()

        result_temp_month = cur.fetchall()
        # print(result_temp_month)

        attendance_score = []

        for j in range(result_temp_month.__len__()):
            status = result_temp_month[j][0]

            if status == 'P':
                attendance_score.append(1)
            else:
                attendance_score.append(0)

        percentage = (sum(attendance_score) / result_temp_month.__len__()) * 100
        print("Student Name: ", selected_student, ", % Attendance for Month of: ", selected_month, "is: ", round(percentage, 2))
        self.percentage_attendance_label.setText(str(round(percentage, 2)))
        self.closeMYSQLdb()

    @pyqtSlot()
    def exitClicked(self):
        QApplication.instance().quit()


class pandasModel(QAbstractTableModel):

    def __init__(self, data):
        QAbstractTableModel.__init__(self)
        self._data = data

    def rowCount(self, parent=None):
        return self._data.shape[0]

    def columnCount(self, parent=None):
        return self._data.shape[1]

    def data(self, index, role=QtCore.Qt.DisplayRole):
        if index.isValid():
            if role == QtCore.Qt.DisplayRole:
                return str(self._data.iloc[index.row(), index.column()])
            return None

    def headerData(self, col, orientation, role):
        if orientation == QtCore.Qt.Horizontal and role == QtCore.Qt.DisplayRole:
            return self._data.columns[col]
        return None


def Display_Login_Dialog():
    Dialog = QtWidgets.QDialog()
    login_ui = Ui_Login_Dialog()
    login_ui.setupUi(Dialog)
    Dialog.show()
    response = Dialog.exec_()

    if response == QtWidgets.QDialog.Accepted:
        username = login_ui.username_lineEdit.text()
        password = login_ui.password_lineEdit.text()
        if username == 'admin' and password == 'admin':
            print("The username: %s and password: %s" % (username, password))
            return 1
        else:
            print("NO VALID USERNAME ENTERED !")
            return 0


if __name__ == '__main__':
    app = QApplication(sys.argv)
    retval = Display_Login_Dialog()

    if retval == 1:
        window = AttendanceSystemML()
        window.setWindowTitle('Attentiveness System')
        window.show()
        sys.exit(app.exec_())
    else:
        print('Incorrect Admin Credentials !\nPlease try with correct username and password !')
        sys.exit(app.exec_())

