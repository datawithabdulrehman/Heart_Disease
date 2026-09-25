"""
Heart Disease Risk Assessment System — by ABXREHMAN
==================================================
A single-page desktop GUI (CustomTkinter) with full icon integration,
Heart branding, and a high-contrast Black, White, Cyan & Red theme.

Required files (in the same directory):
    - knn_heart_model.pkl    (trained classifier)
    - heart_scaler.pkl       (fitted feature scaler)
    - heart_columns.pkl      (expected one-hot-encoded column order)

Run:
    pip install customtkinter pandas joblib scikit-learn
    python heart_disease_app.py
"""

import os
import webbrowser
import customtkinter as ctk
import pandas as pd
import joblib
from tkinter import messagebox

# ---------------------------------------------------------------------------
# Global Appearance & Color Palette (Black, White, Cyan, Red)
# ---------------------------------------------------------------------------
ctk.set_appearance_mode("dark")

MODEL_FILE = "knn_heart_model.pkl"
SCALER_FILE = "heart_scaler.pkl"
COLUMNS_FILE = "heart_columns.pkl"

# --- BLACK, WHITE, CYAN & RED PALETTE ---
COLOR_BG = "#0A0A0A"             # Deep Black
COLOR_CARD = "#141416"           # Dark Charcoal Card
COLOR_ACCENT = "#00E5FF"         # Electric Cyan
COLOR_ACCENT_HOVER = "#00B8D4"   # Dark Cyan Hover
COLOR_HIGH_RISK = "#FF1744"      # Crimson Red
COLOR_LOW_RISK = "#00E676"       # Mint Green
COLOR_TEXT_MAIN = "#FFFFFF"      # Pure White
COLOR_MUTED_TEXT = "#8E8E93"     # Muted Slate Gray


class HeartDiseaseApp(ctk.CTk):
    """Main application window with full icon support."""

    def __init__(self):
        super().__init__()

        self.title("Heart Disease Risk Assessment System — ABXREHMAN")
        self.geometry("900x840")
        self.minsize(820, 740)
        self.configure(fg_color=COLOR_BG)

        # Model artifacts
        self.model = None
        self.scaler = None
        self.expected_columns = None
        self.models_loaded = False
        self.load_error = None

        self._load_artifacts()

        # Build Single Page UI
        self._build_header()
        self._build_main_form()
        self._build_action_area()
        self._build_result_card()
        self._build_footer()

        if not self.models_loaded:
            self.after(300, self._show_missing_files_warning)

    # ------------------------------------------------------------------
    # Artifact Loading
    # ------------------------------------------------------------------
    def _load_artifacts(self):
        missing = [f for f in (MODEL_FILE, SCALER_FILE, COLUMNS_FILE) if not os.path.exists(f)]
        if missing:
            self.load_error = (
                "The following required file(s) were not found:\n\n"
                + "\n".join(f"  \u2022 {name}" for name in missing)
                + "\n\nPlace them in the same folder as this script, then restart."
            )
            self.models_loaded = False
            return

        try:
            self.model = joblib.load(MODEL_FILE)
            self.scaler = joblib.load(SCALER_FILE)
            self.expected_columns = joblib.load(COLUMNS_FILE)
            self.models_loaded = True
        except Exception as exc:  # noqa: BLE001
            self.models_loaded = False
            self.load_error = f"An error occurred while loading model files:\n\n{exc}"

    def _show_missing_files_warning(self):
        messagebox.showwarning("Model Files Not Found", self.load_error)

    # ------------------------------------------------------------------
    # Header
    # ------------------------------------------------------------------
    def _build_header(self):
        header = ctk.CTkFrame(self, corner_radius=0, fg_color=COLOR_CARD, height=90)
        header.pack(fill="x", side="top")
        header.pack_propagate(False)

        ctk.CTkLabel(
            header,
            text="🫀  Heart Disease Risk Assessment System",
            font=ctk.CTkFont(size=24, weight="bold"),
            text_color=COLOR_TEXT_MAIN,
        ).pack(pady=(14, 0))

        ctk.CTkLabel(
            header,
            text="❤️ Machine Learning Diagnostics  •  Single-Page Input Form",
            font=ctk.CTkFont(size=12),
            text_color=COLOR_ACCENT,
        ).pack(pady=(2, 0))

        if not self.models_loaded:
            ctk.CTkLabel(
                header,
                text="⚠️ Model files missing — predictions disabled",
                font=ctk.CTkFont(size=11, weight="bold"),
                text_color=COLOR_HIGH_RISK,
            ).pack(pady=(2, 0))

    # ------------------------------------------------------------------
    # Consolidated Single-Page Form with Icons
    # ------------------------------------------------------------------
    def _build_main_form(self):
        self.form_container = ctk.CTkScrollableFrame(
            self, fg_color="transparent", corner_radius=0
        )
        self.form_container.pack(fill="both", expand=True, padx=20, pady=(10, 5))

        # --- Section 1: Demographics ---
        self._build_section_header("👤  1. Patient Demographics")
        demo_frame = ctk.CTkFrame(self.form_container, fg_color=COLOR_CARD, corner_radius=10)
        demo_frame.pack(fill="x", pady=(0, 15), ipady=5)
        demo_frame.grid_columnconfigure((0, 1, 2), weight=1)

        self.age_slider = self._make_slider_field(
            demo_frame, row=0, col=0, label_text="🎂 Age (years)", frm=18, to=100, default=40, steps=82
        )
        self.sex_combo = self._make_combo_field(
            demo_frame, row=0, col=1, label_text="⚧ Sex", values=["M", "F"]
        )
        self.chest_pain_combo = self._make_combo_field(
            demo_frame, row=0, col=2, label_text="💔 Chest Pain Type",
            values=["ATA", "NAP", "TA", "ASY"],
        )

        # --- Section 2: Clinical Measurements ---
        self._build_section_header("🩺  2. Clinical Measurements")
        clinical_frame = ctk.CTkFrame(self.form_container, fg_color=COLOR_CARD, corner_radius=10)
        clinical_frame.pack(fill="x", pady=(0, 15), ipady=5)
        clinical_frame.grid_columnconfigure((0, 1), weight=1)

        self.resting_bp_entry = self._make_entry_field(
            clinical_frame, row=0, col=0, label_text="🩸 Resting Blood Pressure (mm Hg)",
            default=120, placeholder="80 - 200",
        )
        self.cholesterol_entry = self._make_entry_field(
            clinical_frame, row=0, col=1, label_text="🧪 Cholesterol (mg/dL)",
            default=200, placeholder="100 - 600",
        )
        self.fasting_bs_combo = self._make_combo_field(
            clinical_frame, row=1, col=0, label_text="🍬 Fasting Blood Sugar > 120 mg/dL",
            values=["0", "1"],
        )
        self.max_hr_slider = self._make_slider_field(
            clinical_frame, row=1, col=1, label_text="⚡ Max Heart Rate", frm=60, to=220, default=150, steps=160
        )

        # --- Section 3: Cardiac Test Results ---
        self._build_section_header("📈  3. Cardiac Test Results")
        cardiac_frame = ctk.CTkFrame(self.form_container, fg_color=COLOR_CARD, corner_radius=10)
        cardiac_frame.pack(fill="x", pady=(0, 10), ipady=5)
        cardiac_frame.grid_columnconfigure((0, 1), weight=1)

        self.resting_ecg_combo = self._make_combo_field(
            cardiac_frame, row=0, col=0, label_text="📊 Resting ECG",
            values=["Normal", "ST", "LVH"],
        )
        self.exercise_angina_combo = self._make_combo_field(
            cardiac_frame, row=0, col=1, label_text="🏃 Exercise-Induced Angina", values=["Y", "N"]
        )
        self.oldpeak_slider = self._make_slider_field(
            cardiac_frame, row=1, col=0, label_text="📉 Oldpeak (ST Depression)",
            frm=0.0, to=6.0, default=1.0, is_float=True, steps=60,
        )
        self.st_slope_combo = self._make_combo_field(
            cardiac_frame, row=1, col=1, label_text="📐 ST Slope", values=["Up", "Flat", "Down"]
        )

    def _build_section_header(self, text):
        lbl = ctk.CTkLabel(
            self.form_container,
            text=text,
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=COLOR_ACCENT,
            anchor="w"
        )
        lbl.pack(fill="x", pady=(5, 5))

    # -- Generic Input Field Components ------------------------------------
    def _make_slider_field(self, parent, row, col, label_text, frm, to, default,
                            is_float=False, steps=None, suffix=""):
        frame = ctk.CTkFrame(parent, fg_color="transparent")
        frame.grid(row=row, column=col, padx=15, pady=10, sticky="nsew")

        top = ctk.CTkFrame(frame, fg_color="transparent")
        top.pack(fill="x")
        ctk.CTkLabel(
            top, text=label_text, font=ctk.CTkFont(size=13, weight="bold"),
            text_color=COLOR_TEXT_MAIN, anchor="w"
        ).pack(side="left")

        fmt = f"{{:.1f}}{suffix}" if is_float else f"{{:d}}{suffix}"
        value_var = ctk.StringVar(
            value=fmt.format(default) if is_float else fmt.format(int(default))
        )
        ctk.CTkLabel(
            top, textvariable=value_var, font=ctk.CTkFont(size=13, weight="bold"),
            text_color=COLOR_ACCENT,
        ).pack(side="right")

        def on_change(v):
            value_var.set(fmt.format(float(v)) if is_float else fmt.format(int(round(float(v)))))

        slider = ctk.CTkSlider(
            frame, from_=frm, to=to, number_of_steps=steps, command=on_change,
            button_color=COLOR_ACCENT, button_hover_color=COLOR_ACCENT_HOVER,
            progress_color=COLOR_ACCENT
        )
        slider.set(default)
        slider.pack(fill="x", pady=(6, 0))

        lo_hi = ctk.CTkFrame(frame, fg_color="transparent")
        lo_hi.pack(fill="x")
        ctk.CTkLabel(lo_hi, text=str(frm), font=ctk.CTkFont(size=10),
                     text_color=COLOR_MUTED_TEXT).pack(side="left")
        ctk.CTkLabel(lo_hi, text=str(to), font=ctk.CTkFont(size=10),
                     text_color=COLOR_MUTED_TEXT).pack(side="right")

        return slider

    def _make_combo_field(self, parent, row, col, label_text, values, default=None):
        frame = ctk.CTkFrame(parent, fg_color="transparent")
        frame.grid(row=row, column=col, padx=15, pady=10, sticky="nsew")

        ctk.CTkLabel(
            frame, text=label_text, font=ctk.CTkFont(size=13, weight="bold"),
            text_color=COLOR_TEXT_MAIN, anchor="w"
        ).pack(fill="x")

        combo = ctk.CTkComboBox(
            frame, values=values, state="readonly",
            button_color=COLOR_ACCENT, button_hover_color=COLOR_ACCENT_HOVER,
            border_color=COLOR_ACCENT, dropdown_fg_color=COLOR_CARD
        )
        combo.set(default if default else values[0])
        combo.pack(fill="x", pady=(6, 0))
        return combo

    def _make_entry_field(self, parent, row, col, label_text, default, placeholder=""):
        frame = ctk.CTkFrame(parent, fg_color="transparent")
        frame.grid(row=row, column=col, padx=15, pady=10, sticky="nsew")

        ctk.CTkLabel(
            frame, text=label_text, font=ctk.CTkFont(size=13, weight="bold"),
            text_color=COLOR_TEXT_MAIN, anchor="w"
        ).pack(fill="x")

        entry = ctk.CTkEntry(
            frame, placeholder_text=placeholder,
            border_color=COLOR_ACCENT, fg_color="#1F1F23"
        )
        entry.insert(0, str(default))
        entry.pack(fill="x", pady=(6, 0))
        return entry

    # ------------------------------------------------------------------
    # Action Area / Predict Button
    # ------------------------------------------------------------------
    def _build_action_area(self):
        action_frame = ctk.CTkFrame(self, fg_color="transparent")
        action_frame.pack(fill="x", padx=20, pady=(2, 6))

        self.predict_btn = ctk.CTkButton(
            action_frame,
            text="🫀  Predict Heart Risk",
            font=ctk.CTkFont(size=16, weight="bold"),
            height=46,
            corner_radius=10,
            fg_color=COLOR_ACCENT,
            hover_color=COLOR_ACCENT_HOVER,
            text_color="#000000",
            command=self.predict,
        )
        self.predict_btn.pack(fill="x")

        if not self.models_loaded:
            self.predict_btn.configure(
                state="disabled",
                text="⚠️ Predict Disabled — Model Files Missing",
                fg_color="#3A3A3C", text_color="#A1A1AA"
            )

    # ------------------------------------------------------------------
    # Result Card
    # ------------------------------------------------------------------
    def _build_result_card(self):
        self.result_card = ctk.CTkFrame(self, corner_radius=10, fg_color=COLOR_CARD, height=75)
        self.result_card.pack(fill="x", padx=20, pady=(0, 8))
        self.result_card.pack_propagate(False)

        self.result_label = ctk.CTkLabel(
            self.result_card,
            text='📋 Fill in patient details above and click "Predict Heart Risk".',
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=COLOR_TEXT_MAIN,
        )
        self.result_label.pack(expand=True, pady=8)

    # ------------------------------------------------------------------
    # Footer Area (ABXREHMAN Branding & Links with Icons)
    # ------------------------------------------------------------------
    def _build_footer(self):
        footer_frame = ctk.CTkFrame(self, fg_color="transparent")
        footer_frame.pack(fill="x", side="bottom", pady=(0, 12))

        # Disclaimer
        ctk.CTkLabel(
            footer_frame,
            text="⚠️ Disclaimer: For educational purposes only. Not a substitute for professional medical advice.",
            font=ctk.CTkFont(size=10),
            text_color=COLOR_MUTED_TEXT,
        ).pack(pady=(0, 4))

        # Built by
        ctk.CTkLabel(
            footer_frame,
            text="❤️ Built by ABXREHMAN",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=COLOR_TEXT_MAIN,
        ).pack()

        # Links
        links_frame = ctk.CTkFrame(footer_frame, fg_color="transparent")
        links_frame.pack(pady=(2, 0))

        def create_hyperlink(parent, text, url):
            link = ctk.CTkLabel(
                parent, 
                text=text, 
                font=ctk.CTkFont(size=11, underline=True), 
                text_color=COLOR_ACCENT,
                cursor="hand2"
            )
            link.pack(side="left", padx=10)
            link.bind("<Button-1>", lambda e: webbrowser.open(url))

        create_hyperlink(links_frame, "💼 LinkedIn", "https://www.linkedin.com/in/datawithabdulrehman")
        create_hyperlink(links_frame, "📊 Kaggle", "https://www.kaggle.com/datawithabxrehman")
        create_hyperlink(links_frame, "💻 GitHub", "https://github.com/datawithabdulrehman")

    # ------------------------------------------------------------------
    # Input Gathering & Prediction Logic
    # ------------------------------------------------------------------
    def _gather_input(self) -> dict:
        age = int(round(self.age_slider.get()))
        sex = self.sex_combo.get()
        chest_pain = self.chest_pain_combo.get()
        fasting_bs = int(self.fasting_bs_combo.get())
        max_hr = int(round(self.max_hr_slider.get()))
        resting_ecg = self.resting_ecg_combo.get()
        exercise_angina = self.exercise_angina_combo.get()
        oldpeak = round(float(self.oldpeak_slider.get()), 1)
        st_slope = self.st_slope_combo.get()

        resting_bp_raw = self.resting_bp_entry.get().strip()
        cholesterol_raw = self.cholesterol_entry.get().strip()

        if not resting_bp_raw or not cholesterol_raw:
            raise ValueError("Resting Blood Pressure and Cholesterol cannot be empty.")

        try:
            resting_bp = int(resting_bp_raw)
            cholesterol = int(cholesterol_raw)
        except ValueError as exc:
            raise ValueError("Resting Blood Pressure and Cholesterol must be whole numbers.") from exc

        if not (80 <= resting_bp <= 200):
            raise ValueError("Resting Blood Pressure should be between 80 and 200 mm Hg.")
        if not (100 <= cholesterol <= 600):
            raise ValueError("Cholesterol should be between 100 and 600 mg/dL.")

        raw_input = {
            "Age": age,
            "RestingBP": resting_bp,
            "Cholesterol": cholesterol,
            "FastingBS": fasting_bs,
            "MaxHR": max_hr,
            "Oldpeak": oldpeak,
            f"Sex_{sex}": 1,
            f"ChestPainType_{chest_pain}": 1,
            f"RestingECG_{resting_ecg}": 1,
            f"ExerciseAngina_{exercise_angina}": 1,
            f"ST_Slope_{st_slope}": 1,
        }
        return raw_input

    def predict(self):
        if not self.models_loaded:
            messagebox.showerror("Model Not Loaded", self.load_error or "Model files are missing.")
            return

        try:
            raw_input = self._gather_input()
        except ValueError as exc:
            messagebox.showerror("Invalid Input", str(exc))
            return

        try:
            input_df = pd.DataFrame([raw_input])

            # Fill missing one-hot columns with 0
            for col in self.expected_columns:
                if col not in input_df.columns:
                    input_df[col] = 0

            # Reorder columns to match training order
            input_df = input_df[self.expected_columns]

            scaled_input = self.scaler.transform(input_df)
            prediction = self.model.predict(scaled_input)[0]

            self._show_result(prediction)

        except Exception as exc:  # noqa: BLE001
            messagebox.showerror(
                "Prediction Error", f"Something went wrong while predicting:\n\n{exc}"
            )

    def _show_result(self, prediction):
        if prediction == 1:
            self.result_card.configure(fg_color=COLOR_HIGH_RISK)
            self.result_label.configure(
                text="⚠️  HIGH RISK of Heart Disease",
                text_color="#FFFFFF"
            )
        else:
            self.result_card.configure(fg_color=COLOR_LOW_RISK)
            self.result_label.configure(
                text="✅  LOW RISK of Heart Disease",
                text_color="#000000"
            )


if __name__ == "__main__":
    app = HeartDiseaseApp()
    app.mainloop()