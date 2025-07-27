import tkinter as tk
from tkinter import filedialog, messagebox
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import torch.nn.functional as F
import matplotlib as mpl
import numpy as np

mpl.rcParams['font.family'] = 'Malgun Gothic'
mpl.rcParams['axes.unicode_minus'] = False

model_path = None
folder_path = None
resize_size = 224

def browse_model_file():
    global model_path
    model_path = filedialog.askopenfilename(filetypes=[("Model Files", "*.pt;*.pth")])
    if model_path:
        model_label.config(text=f"모델 경로: {model_path}")
        messagebox.showinfo("알림", f"모델 경로가 설정되었습니다: {model_path}")
    else:
        messagebox.showinfo("알림", "모델 파일을 선택하지 않았습니다.")

def browse_image_folder():
    global folder_path
    folder_path = filedialog.askdirectory()
    if folder_path:
        folder_label.config(text=f"이미지 폴더 경로: {folder_path}")
        messagebox.showinfo("알림", f"이미지 폴더 경로가 설정되었습니다: {folder_path}")
    else:
        messagebox.showinfo("알림", "이미지 폴더를 선택하지 않았습니다.")

def load_images_from_folder(folder_path, resize_size):
    transform = transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.ToTensor(),
    ])
    images = []
    image_paths = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith((".jpg", ".png", ".jpeg")):
            image_path = os.path.join(folder_path, filename)
            image = Image.open(image_path).convert("RGB")
            images.append(transform(image).unsqueeze(0))
            image_paths.append(image_path)
    return images, image_paths

def run_model_with_condition(images, condition_vector, model, image_paths):
    try:
        reconstructed_images = []
        with torch.no_grad():
            for img in images:
                _, reconstructed_image, _ = model(img, condition_vector)
                reconstructed_images.append(reconstructed_image)

        condition_class = f"V16{condition_var.get()}"
        visualize_images(images, reconstructed_images, image_paths, condition_class)
    except Exception as e:
        messagebox.showerror("에러", f"오류가 발생했습니다: {str(e)}")

def set_condition_vector():
    condition_idx = condition_var.get()
    condition_vector = F.one_hot(
        torch.tensor([condition_idx]), num_classes=8
    ).float()
    condition_vector = condition_vector.to(torch.device("cpu"))
    return condition_vector

def classify_comment():
    global resize_size

    if model_path is None:
        messagebox.showinfo("알림", "모델 파일을 선택하세요.")
        return
    if folder_path is None:
        messagebox.showinfo("알림", "이미지 폴더를 선택하세요.")
        return

    try:
        model = torch.load(rf'{model_path}', map_location=torch.device('cpu'))
        model.eval()

        try:
            resize_size = int(resize_entry.get())
        except ValueError:
            messagebox.showerror("에러", "올바른 이미지 크기를 입력하세요.")
            return

        images, image_paths = load_images_from_folder(folder_path, resize_size)

        if not images:
            messagebox.showinfo("알림", "폴더에 유효한 이미지가 없습니다.")
            return

        condition_window = tk.Toplevel(window)
        condition_window.title("클래스 선택")
        condition_var.set(0)

        tk.Label(condition_window, text="클래스 선택:").pack(pady=10)
        for i in range(8):
            tk.Radiobutton(
                condition_window, text=f"V16{i}", variable=condition_var, value=i
            ).pack(anchor="w")

        def on_condition_set():
            condition_vector = set_condition_vector()
            run_model_with_condition(images, condition_vector, model, image_paths)

        tk.Button(condition_window, text="클래스 설정", command=on_condition_set).pack(pady=10)
    except Exception as e:
        messagebox.showerror("에러", f"오류가 발생했습니다: {str(e)}")

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

def visualize_images(originals, reconstructed, image_paths, condition_class):
    scroll_window = tk.Toplevel(window)
    scroll_window.title("이미지 재구성 결과")
    
    canvas = tk.Canvas(scroll_window)
    scrollbar = tk.Scrollbar(scroll_window, orient="vertical", command=canvas.yview)
    scrollable_frame = tk.Frame(canvas)

    scrollable_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )
    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)

    def _on_mouse_wheel(event):
        canvas.yview_scroll(-1 * (event.delta // 120), "units")

    canvas.bind_all("<MouseWheel>", _on_mouse_wheel)

    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    fig, axes = plt.subplots(len(originals), 2, figsize=(12, len(originals) * 3))
    if len(originals) == 1: 
        axes = [axes]

    fig.subplots_adjust(left=0.2, right=0.8)

    for i, (original, reconstructed_image) in enumerate(zip(originals, reconstructed)):
        original = original.squeeze(0).permute(1, 2, 0).numpy()
        reconstructed_image = reconstructed_image.squeeze(0).permute(1, 2, 0).numpy()
        
        reconstruction_error = np.mean((original - reconstructed_image) ** 2)

        axes[i, 0].imshow(original)
        axes[i, 0].set_title(f"원본 이미지 ({condition_class})", fontsize=12)
        axes[i, 0].axis("off")

        axes[i, 1].imshow(reconstructed_image)
        axes[i, 1].set_title(f"reconstruction error: {reconstruction_error:.4f}", fontsize=12)
        axes[i, 1].axis("off")

    plt.tight_layout()

    canvas_agg = FigureCanvasTkAgg(fig, scrollable_frame)
    canvas_widget = canvas_agg.get_tk_widget()
    canvas_widget.pack(side="top", fill="both", expand=True)

    def on_close():
        plt.close(fig)
        scroll_window.destroy()

    scroll_window.protocol("WM_DELETE_WINDOW", on_close)

window = tk.Tk()
window.title("이미지 재구성")

browse_model_button = tk.Button(window, text="모델 찾기", command=browse_model_file, font=("Helvetica", 12))
browse_model_button.grid(row=0, column=0, padx=10, pady=5)

model_label = tk.Label(window, text="", font=("Helvetica", 10), anchor="e")
model_label.grid(row=0, column=1, padx=10, pady=5)

browse_folder_button = tk.Button(window, text="폴더 찾기", command=browse_image_folder, font=("Helvetica", 12))
browse_folder_button.grid(row=1, column=0, padx=10, pady=5)

folder_label = tk.Label(window, text="", font=("Helvetica", 10), anchor="e")
folder_label.grid(row=1, column=1, padx=10, pady=5)

resize_label = tk.Label(window, text="이미지 크기:", font=("Helvetica", 12))
resize_label.grid(row=2, column=0, padx=10, pady=5)

resize_entry = tk.Entry(window, font=("Helvetica", 12))
resize_entry.insert(0, "224")
resize_entry.grid(row=2, column=1, padx=10, pady=5)

classify_button = tk.Button(window, text="재구성하기", command=classify_comment, font=("Helvetica", 12))
classify_button.grid(row=3, column=0, columnspan=2, pady=10)

condition_var = tk.IntVar(value=0)

window.mainloop()
