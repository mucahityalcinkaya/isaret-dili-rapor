import sys, cv2, numpy as np, torch, datetime
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QVBoxLayout, 
                             QHBoxLayout, QWidget, QTextEdit, QListWidget, QPushButton)
from PyQt5.QtCore import QTimer, Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QFont

from kameradetecdeneme2 import MediaPipeExtractor, SignLanguagePredictor
from demoo import get_best_translation # Akıllı fonksiyonu çektik

class LLMWorker(QThread):
    finished = pyqtSignal(str, str)
    def __init__(self, words, history):
        super().__init__(); self.words = words; self.history = history

    def run(self):
        # Arka planda tüm varyasyonlar denenir
        sentence, summary = get_best_translation(self.words, self.history)
        self.finished.emit(sentence, summary)

class SignApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Medikal İşaret Dili Tercümanı v3.0")
        self.setMinimumSize(1280, 850)
        self.setStyleSheet("background-color: #1e1e1e; color: white;")

        self.predictor = SignLanguagePredictor("../best_model_balanced.pth", "../class_to_idx.json")
        self.extractor = MediaPipeExtractor()
        
        self.cap = cv2.VideoCapture(0)
        self.recording = False
        self.frames_buffer = []
        self.detected_glosses = []
        self.history = []
        self.conversation_log = []
        self.missing_ctr = 0
        self.last_confirmed_text = "" 

        self.init_ui()
        self.timer = QTimer()
        self.timer.timeout.connect(self.process_frame)
        self.timer.start(30)

    def init_ui(self):
        central_widget = QWidget(); self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # SOL PANEL
        left_layout = QVBoxLayout()
        self.video_label = QLabel()
        self.video_label.setFixedSize(850, 640)
        self.video_label.setStyleSheet("border: 3px solid #34495e; border-radius: 10px; background-color: black;")
        left_layout.addWidget(self.video_label)
        self.status_label = QLabel("● HAZIR"); self.status_label.setStyleSheet("color: #2ecc71; font-weight: bold; font-size: 16px;")
        left_layout.addWidget(self.status_label)
        main_layout.addLayout(left_layout, stretch=2)

        # SAĞ PANEL
        right_panel = QVBoxLayout()
        self.btn_delete = QPushButton("Sonuncuyu Sil")
        self.btn_delete.setStyleSheet("background-color: #e74c3c; height: 35px;")
        self.btn_delete.clicked.connect(self.delete_last_gloss)
        right_panel.addWidget(self.btn_delete)

        self.gloss_list = QListWidget(); right_panel.addWidget(self.gloss_list)

        self.btn_confirm = QPushButton("Cümleyi Onayla ve Yeniyi Başlat")
        self.btn_confirm.setStyleSheet("background-color: #3498db; height: 50px; font-weight: bold;")
        self.btn_confirm.clicked.connect(self.confirm_sentence)
        right_panel.addWidget(self.btn_confirm)

        self.ll_output = QTextEdit(); self.ll_output.setReadOnly(True)
        self.ll_output.setStyleSheet("background-color: #2c3e50; font-size: 14px; border: 1px solid #555;")
        right_panel.addWidget(self.ll_output)

        self.btn_save = QPushButton("Görüşmeyi Kaydet (.txt)")
        self.btn_save.clicked.connect(self.save_conversation)
        right_panel.addWidget(self.btn_save)
        main_layout.addLayout(right_panel, stretch=1)

    def draw_skeleton(self, frame, kdict):
        for hand in [kdict['left_hand'], kdict['right_hand']]:
            if np.any(hand):
                for pt in hand: cv2.circle(frame, (int(pt[0]), int(pt[1])), 4, (0, 255, 255), -1)
        if np.any(kdict['pose']):
            for pt in kdict['pose']:
                if pt[2] > 0.5: cv2.circle(frame, (int(pt[0]), int(pt[1])), 5, (255, 0, 0), -1)

    def process_frame(self):
        ret, frame = self.cap.read()
        if not ret: return
        frame = cv2.flip(frame, 1)
        kdict, W, H, lp, rp = self.extractor.extract_keypoints(frame)
        self.draw_skeleton(frame, kdict)

        if lp or rp:
            self.recording = True; self.missing_ctr = 0
            self.frames_buffer.append(self.extractor.normalize_keypoints(kdict, W, H))
            self.status_label.setText("● KAYDEDİLİYOR...")
            self.status_label.setStyleSheet("color: #e74c3c;")
        elif self.recording:
            self.missing_ctr += 1
            if self.missing_ctr > 12: self.process_gesture()

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = QImage(rgb.data, W, H, 3 * W, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(img).scaled(850, 640, Qt.KeepAspectRatio))

    def process_gesture(self):
        self.recording = False
        if len(self.frames_buffer) > 10:
            res = self.predictor.predict_skeleton(self.predictor.resample_to_64(self.frames_buffer), top_k=1)
            gloss = res[0]['class']
            self.detected_glosses.append(gloss); self.gloss_list.addItem(f">> {gloss}")
            self.run_llm()
        self.frames_buffer = []; self.missing_ctr = 0

    def delete_last_gloss(self):
        if self.detected_glosses:
            self.detected_glosses.pop(); self.gloss_list.takeItem(self.gloss_list.count()-1)
            if self.detected_glosses: self.run_llm()
            else: self.ll_output.setText(self.last_confirmed_text)

    def confirm_sentence(self):
        if not self.detected_glosses: return
        self.history = [] 
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        self.last_confirmed_text = self.ll_output.toPlainText() + f"\n✅ [{ts}] ONAYLANDI\n" + "-"*30 + "\n"
        self.ll_output.setText(self.last_confirmed_text)
        self.detected_glosses = []; self.gloss_list.clear()
        self.status_label.setText("● YENİ BAĞLAM BAŞLATILDI"); self.status_label.setStyleSheet("color: #3498db;")

    def run_llm(self):
        self.llm_thread = LLMWorker(self.detected_glosses, self.history)
        self.llm_thread.finished.connect(lambda s, sum: self.ll_output.setText(self.last_confirmed_text + f"Hasta: {s}\nÖzet: {sum}"))
        self.llm_thread.start()

    def save_conversation(self):
        filename = f"rapor_{datetime.datetime.now().strftime('%H%M%S')}.txt"
        with open(filename, "w", encoding="utf-8") as f: f.write(self.ll_output.toPlainText())
        self.ll_output.append(f"\n[SİSTEM]: Kaydedildi -> {filename}")

if __name__ == "__main__":
    app = QApplication(sys.argv); window = SignApp(); window.show(); sys.exit(app.exec_())