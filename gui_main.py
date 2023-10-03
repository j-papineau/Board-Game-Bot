from PyQt6.QtWidgets import *
from PyQt6.QtGui import QFont, QFontDatabase
import sys
from Main_Tab import *
from Second_Tab import *

class Window(QMainWindow):

    def __init__(self):
        super().__init__()

        # title
        self.setWindowTitle("Joel's Python Experiments")

        self.Width = 1920
        self.height = int(0.618 * self.Width)
        self.resize(self.Width, self.height)

        # global vars

        self.title_font = QFont("Arial", pointSize=24, weight=QFont.Weight.Bold)
        self.normal_font = QFont("Arial")
        self.bold_font = QFont("Arial", weight=QFont.Weight.Bold)

        # add side buttons

        self.btn_1 = QPushButton('Main', self)
        self.btn_2 = QPushButton('Serial Output \n(Listening: 0x033FA2 - 0x034100)', self)
        self.btn_3 = QPushButton('Arduino Connection', self)
        
        self.btn_2.setEnabled(False)
        self.btn_3.setEnabled(False)
        
        self.btn_1.clicked.connect(self.btn_1_handler)
        self.btn_2.clicked.connect(self.btn_2_handler)
        self.btn_3.clicked.connect(self.btn_3_handler)

        self.btn_1.setFont(self.bold_font)

        self.tab1 = self.ui1()
        self.tab2 = self.ui2()
        self.tab3 = self.ui3()

        self.initUI()

    def initUI(self):
        left_layout = QVBoxLayout()
        self.sidebar_title = QLabel("Tabs")
        self.sidebar_title.setFont(self.title_font)
        self.sidebar_title.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        
        left_layout.addWidget(self.sidebar_title)
        left_layout.addWidget(self.btn_1)
        left_layout.addWidget(self.btn_2)
        left_layout.addWidget(self.btn_3)
        left_layout.addStretch(5)
        left_layout.setSpacing(20)
        left_widget = QWidget()
        left_widget.setLayout(left_layout)

        self.right_widget = QTabWidget()
        self.right_widget.tabBar().setObjectName("mainTab")

        self.right_widget.addTab(self.tab1, '')
        self.right_widget.addTab(self.tab2, '')
        self.right_widget.addTab(self.tab3, '')

        self.right_widget.setCurrentIndex(0)
        self.right_widget.setStyleSheet('''QTabBar::tab{width: 0; \
            height: 0; margin: 0; padding: 0; border: none;}''')
        
        main_layout = QHBoxLayout()
        main_layout.addWidget(left_widget)
        main_layout.addWidget(self.right_widget)
        main_layout.setStretch(0, 40)
        main_layout.setStretch(1, 200)
        main_widget = QWidget()
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)


        # button handlers

    def btn_1_handler(self):
        self.right_widget.setCurrentIndex(0)
        self.main_tab.start_feed()
        self.btn_1.setFont(self.bold_font)
        self.btn_2.setFont(self.normal_font)
        self.btn_3.setFont(self.normal_font)
          
    def btn_2_handler(self):
        self.right_widget.setCurrentIndex(1)
        self.main_tab.cancel_feed()
        self.btn_2.setFont(self.bold_font)
        self.btn_1.setFont(self.normal_font)
        self.btn_3.setFont(self.normal_font)

    def btn_3_handler(self):
        self.right_widget.setCurrentIndex(2)
        self.main_tab.cancel_feed()
        self.btn_3.setFont(self.bold_font)
        self.btn_1.setFont(self.normal_font)
        self.btn_2.setFont(self.normal_font)

    
        # tab ui's

    def ui1(self):
        main_layout = QVBoxLayout()
        self.main_tab = Main_Tab()
        main_layout.addWidget(self.main_tab)
        main = QWidget()
        main.setLayout(main_layout)
        return main
        
    def ui2(self):
        main_layout = QVBoxLayout()
        self.second_tab = Second_Tab()
        main_layout.addWidget(self.second_tab)
        main = QWidget()
        main.setLayout(main_layout)
        return main

    def ui3(self):
        main_layout = QVBoxLayout()
        main_layout.addWidget(QLabel('Something awesome will go here (eventually ((probably not)))'))
        main_layout.addStretch(5)
        main = QWidget()
        main.setLayout(main_layout)
        return main


if __name__ == '__main__':
    app = QApplication(sys.argv)
    # oxygen, breeze, qtcurve, windows, fusion
    app.setStyle("Fusion")
    ex = Window()
    ex.show()

    sys.exit(app.exec())