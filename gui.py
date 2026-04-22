import tkinter as tk
from tkinter import messagebox
import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn as nn
import torch.nn.functional as F

MODEL_PATH = "mnist_cnn.pt"
IMAGE_SIZE = 28
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

class DigitRecognizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Распознавание рукописных цифр (MNIST, PyTorch)")

        self.model = CNN().to(DEVICE)
        try:
            self.model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
            self.model.eval()
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось загрузить модель:\n{e}")
            self.root.destroy()
            return

        self.canvas_size = 280
        self.bg_color = "black"
        self.fg_color = "white"
        self.pen_width = 15

        self.canvas = tk.Canvas(
            self.root,
            width=self.canvas_size,
            height=self.canvas_size,
            bg=self.bg_color
        )
        self.canvas.grid(row=0, column=0, columnspan=4, padx=10, pady=10)

        self.image = Image.new("L", (self.canvas_size, self.canvas_size), color=0)
        self.draw = ImageDraw.Draw(self.image)

        self.btn_predict = tk.Button(self.root, text="Распознать", command=self.predict)
        self.btn_predict.grid(row=1, column=0, padx=5, pady=5, sticky="ew")

        self.btn_clear = tk.Button(self.root, text="Очистить", command=self.clear_canvas)
        self.btn_clear.grid(row=1, column=1, padx=5, pady=5, sticky="ew")

        self.lbl_result = tk.Label(self.root, text="Нарисуйте цифру (0–9)", font=("Arial", 14))
        self.lbl_result.grid(row=1, column=2, columnspan=2, padx=5, pady=5, sticky="w")

        self.canvas.bind("<Button-1>", self.start_draw)
        self.canvas.bind("<B1-Motion>", self.draw_lines)

        self.last_x, self.last_y = None, None

    def start_draw(self, event):
        self.last_x, self.last_y = event.x, event.y

    def draw_lines(self, event):
        x, y = event.x, event.y
        if self.last_x is not None and self.last_y is not None:
            self.canvas.create_line(
                self.last_x, self.last_y, x, y,
                width=self.pen_width,
                fill=self.fg_color,
                capstyle=tk.ROUND,
                smooth=True
            )
            self.draw.line(
                [self.last_x, self.last_y, x, y],
                fill=255,
                width=self.pen_width
            )
        self.last_x, self.last_y = x, y

    def clear_canvas(self):
        self.canvas.delete("all")
        self.draw.rectangle([0, 0, self.canvas_size, self.canvas_size], fill=0)
        self.lbl_result.config(text="Нарисуйте цифру (0–9)")
        self.last_x, self.last_y = None, None

    def preprocess_image(self):
        img = self.image.resize((IMAGE_SIZE, IMAGE_SIZE))
        img_array = np.array(img).astype("float32") / 255.0

        # нормализация как при обучении: (x - 0.5) / 0.5
        img_array = (img_array - 0.5) / 0.5

        img_tensor = torch.from_numpy(img_array).unsqueeze(0).unsqueeze(0)  # (1,1,28,28)
        img_tensor = img_tensor.to(DEVICE)
        return img_tensor

    def predict(self):
        img_tensor = self.preprocess_image()
        with torch.no_grad():
            outputs = self.model(img_tensor)
            probs = F.softmax(outputs, dim=1).cpu().numpy()[0]

        predicted_digit = int(np.argmax(probs))
        prob = float(np.max(probs)) * 100.0

        self.lbl_result.config(
            text=f"Предсказание: {predicted_digit} (вероятность {prob:.1f}%)"
        )

        probs_text = "\n".join([f"{i}: {p*100:.1f}%" for i, p in enumerate(probs)])
        print("Вероятности классов:\n" + probs_text)

def main():
    root = tk.Tk()
    app = DigitRecognizerApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()