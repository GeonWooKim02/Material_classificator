"""
Material Analysis System - 일괄 분석 (Keras 모델 버전)
필요 라이브러리: pip install customtkinter joblib pandas openpyxl scikit-learn tensorflow

필요 파일 (프로그램/EXE와 같은 폴더에 두세요. 파일만 교체하면 최신 모델이 바로 적용됩니다)
  - model.keras
  - scaler.pkl
  - label_encoder.pkl
"""

import os
import sys
import threading

import numpy as np
import pandas as pd
import joblib
import sklearn  # PyInstaller가 자동으로 감지하도록 import 유지
import customtkinter as ctk
from tkinter import filedialog

N_FEATURES = 500  # 학습 때와 동일: 엑셀의 앞 500개 열이 입력값

MODEL_FILE = "model.keras"
SCALER_FILE = "scaler.pkl"
ENCODER_FILE = "label_encoder.pkl"


# ---------------------------------------------------------
# 모델 파일 위치 찾기
#  1순위: EXE(또는 .py)와 같은 폴더  -> 파일만 바꿔 끼우면 갱신됨
#  2순위: PyInstaller 내부(--add-data로 포함한 경우)
#  3순위: 현재 작업 폴더
# ---------------------------------------------------------
def find_file(name):
    if getattr(sys, "frozen", False):
        app_dir = os.path.dirname(sys.executable)
    else:
        app_dir = os.path.dirname(os.path.abspath(__file__))

    candidates = [os.path.join(app_dir, name)]
    if hasattr(sys, "_MEIPASS"):
        candidates.append(os.path.join(sys._MEIPASS, name))
    candidates.append(os.path.join(os.path.abspath("."), name))

    for path in candidates:
        if os.path.exists(path):
            return path
    return None


# ---------------------------------------------------------
# 외형 설정
# ---------------------------------------------------------
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

ACCENT = "#3B82F6"
ACCENT_HOVER = "#2563EB"
BG_CARD = "#1E1E2E"
BG_RESULT = "#15151f"
SEPARATOR = "-" * 48

# 확신도 색상 기준
COLOR_GREEN = "#22C55E"    # 90% 이상
COLOR_YELLOW = "#FACC15"   # 70% 이상 ~ 90% 미만
COLOR_ORANGE = "#FB923C"   # 70% 미만
COLOR_RED = "#EF4444"      # 오류


def confidence_tag(conf):
    if conf >= 90:
        return "green"
    if conf >= 70:
        return "yellow"
    return "orange"


class WaferAnalyzerApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Material Analysis System")
        self.geometry("640x720")
        self.minsize(620, 600)

        self.card = ctk.CTkFrame(
            self, corner_radius=20, fg_color=BG_CARD,
            border_width=1, border_color="#33334d"
        )
        self.card.pack(padx=24, pady=24, fill="both", expand=True)

        # ---------- 제목 (정중앙) ----------
        ctk.CTkLabel(
            self.card, text="Material Analysis System",
            font=ctk.CTkFont(size=24, weight="bold")
        ).pack(pady=(28, 4))

        # ---------- 색상 범례 (카드 좌측 상단에 고정) ----------
        legend = ctk.CTkFrame(self.card, fg_color="transparent")
        legend.place(x=18, y=14)

        items = [
            (COLOR_GREEN, "확신도 90% 이상"),
            (COLOR_YELLOW, "70~90%"),
            (COLOR_ORANGE, "70% 미만"),
            (COLOR_RED, "오류"),
        ]
        for i, (color, text) in enumerate(items):
            cell = ctk.CTkFrame(legend, fg_color="transparent")
            cell.grid(row=i, column=0, sticky="w")
            ctk.CTkLabel(cell, text="●", text_color=color, height=14,
                         font=ctk.CTkFont(size=11)).pack(side="left")
            ctk.CTkLabel(cell, text=text, text_color="#9CA3AF", height=14,
                         font=ctk.CTkFont(size=10)).pack(side="left", padx=(2, 0))

        ctk.CTkLabel(
            self.card,
            text="엑셀 파일을 선택하면 모든 데이터를 자동 분석합니다.",
            font=ctk.CTkFont(size=13),
            text_color="#9CA3AF"
        ).pack(pady=(0, 18))

        self.analyze_btn = ctk.CTkButton(
            self.card,
            text="파일 선택 및 일괄 분석 시작",
            font=ctk.CTkFont(size=14, weight="bold"),
            height=46,
            corner_radius=14,
            fg_color=ACCENT,
            hover_color=ACCENT_HOVER,
            command=self.start_analysis
        )
        self.analyze_btn.pack(fill="x", padx=30, pady=(0, 10))

        self.progress = ctk.CTkProgressBar(self.card, mode="indeterminate")

        self.result_box = ctk.CTkTextbox(
            self.card,
            font=ctk.CTkFont(family="Consolas", size=13),
            fg_color=BG_RESULT,
            text_color="#E5E7EB",
            corner_radius=12,
            border_width=1,
            border_color="#33334d",
            wrap="none",
        )
        self.result_box.pack(fill="both", expand=True, padx=30, pady=(10, 28))
        self.result_box.tag_config("green", foreground=COLOR_GREEN)
        self.result_box.tag_config("yellow", foreground=COLOR_YELLOW)
        self.result_box.tag_config("orange", foreground=COLOR_ORANGE)
        self.result_box.tag_config("red", foreground=COLOR_RED)
        self.result_box.insert("end", "분석 결과가 여기에 표시됩니다.")
        self.result_box.configure(state="disabled")

        legend.lift()  # 범례를 다른 위젯 위로 올려서 가려지지 않게 함

    # -----------------------------------------------------
    # UI 헬퍼 (스레드에서 호출해도 메인 스레드에서 실행되도록 처리)
    # -----------------------------------------------------
    def _write(self, content, clear=False, tag=None):
        # content: 문자열 또는 [(텍스트, 색상태그), ...] 목록
        segments = [(content, tag)] if isinstance(content, str) else content

        def _do():
            self.result_box.configure(state="normal")
            if clear:
                self.result_box.delete("1.0", "end")
            for text, t in segments:
                if t:
                    self.result_box.insert("end", text, t)
                else:
                    self.result_box.insert("end", text)
            self.result_box.see("end")
            self.result_box.configure(state="disabled")
        self.after(0, _do)

    def _set_busy(self, busy):
        def _do():
            if busy:
                self.analyze_btn.configure(state="disabled", text="분석 중...")
                self.progress.pack(fill="x", padx=30, pady=(0, 10), before=self.result_box)
                self.progress.start()
            else:
                self.progress.stop()
                self.progress.pack_forget()
                self.analyze_btn.configure(state="normal", text="파일 선택 및 일괄 분석 시작")
        self.after(0, _do)

    # -----------------------------------------------------
    # 분석 로직
    # -----------------------------------------------------
    def start_analysis(self):
        missing = [n for n in (MODEL_FILE, SCALER_FILE, ENCODER_FILE) if find_file(n) is None]
        if missing:
            self._write(
                "❌ 필요한 파일을 찾을 수 없습니다:\n  " + "\n  ".join(missing) +
                "\n\n프로그램(EXE)과 같은 폴더에 있는지 확인하세요.",
                clear=True, tag="red",
            )
            return

        file_path = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx")])
        if not file_path:
            return

        threading.Thread(target=self.analyze, args=(file_path,), daemon=True).start()

    def analyze(self, file_path):
        self._set_busy(True)
        self._write(f"분석 대상: {os.path.basename(file_path)}\n{SEPARATOR}\n"
                    "모델 불러오는 중...\n", clear=True)
        try:
            # tensorflow는 무거우므로 분석 시작 시점에 불러옴 (창이 빨리 뜸)
            import tensorflow as tf

            # 분석할 때마다 새로 읽으므로 파일을 교체하면 바로 최신 모델이 적용됨
            model = tf.keras.models.load_model(find_file(MODEL_FILE), compile=False)
            scaler = joblib.load(find_file(SCALER_FILE))
            encoder = joblib.load(find_file(ENCODER_FILE))

            df = pd.read_excel(file_path, engine="openpyxl")
            df = df.dropna(how="all")

            if df.shape[1] < N_FEATURES:
                raise ValueError(
                    f"열이 {df.shape[1]}개뿐입니다. 앞 {N_FEATURES}개 열에 "
                    f"측정값이 있어야 합니다."
                )

            # 학습 때와 동일한 전처리: 앞 500열 -> 음수 0 처리 -> log10 -> 표준화
            raw = df.iloc[:, :N_FEATURES].apply(pd.to_numeric, errors="coerce")
            excel_rows = (raw.index + 2).tolist()  # 1행은 헤더 -> 데이터는 2행부터

            valid_mask = ~raw.isna().any(axis=1).values
            lines = [(f"분석 대상: {os.path.basename(file_path)}\n{SEPARATOR}\n", None)]

            preds = {}
            if valid_mask.any():
                X = raw.values[valid_mask].astype(np.float32)
                L = np.log10(np.maximum(np.atleast_2d(X).astype(float), 1e-30))
                X = scaler.transform(L - L.mean(axis=1, keepdims=True))
                probs = model.predict(X, verbose=0)
                idx = probs.argmax(axis=1)
                labels = encoder.inverse_transform(idx)
                conf = probs.max(axis=1) * 100
                valid_rows = [r for r, ok in zip(excel_rows, valid_mask) if ok]
                for r, lab, c in zip(valid_rows, labels, conf):
                    preds[r] = (str(lab), float(c))

            counts = {}
            for r in excel_rows:
                if r in preds:
                    lab, c = preds[r]
                    counts[lab] = counts.get(lab, 0) + 1
                    lines.append((
                        f"● [{r}번 행] 예측: {lab} (확신도: {c:.2f}%)\n",
                        confidence_tag(c),
                    ))
                else:
                    lines.append((
                        f"● [{r}번 행] 오류: 비어 있거나 숫자가 아닌 값이 있습니다\n",
                        "red",
                    ))

            summary = ", ".join(f"{k} {v}건" for k, v in sorted(counts.items()))
            lines.append((
                f"{SEPARATOR}\n총 {len(preds)}건 분석 완료"
                + (f"  ({summary})" if summary else "") + "\n",
                None,
            ))

            self._write(lines, clear=True)

        except Exception as e:
            self._write(f"⚠️ 오류가 발생했습니다\n{e}\n", clear=True, tag="red")
        finally:
            self._set_busy(False)


if __name__ == "__main__":
    app = WaferAnalyzerApp()
    app.mainloop()