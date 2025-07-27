import tkinter as tk
from tkinter import filedialog, messagebox
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn.functional as F
import matplotlib as mpl

mpl.rcParams['font.family'] = 'Malgun Gothic'
mpl.rcParams['axes.unicode_minus'] = False

model_path = None
image_path = None

def browse_model_file():
    global model_path
    model_path = filedialog.askopenfilename(filetypes=[("Model Files", "*.pt;*.pth")])
    if model_path:
        model_label.config(text=f"모델 경로: {model_path}")
        messagebox.showinfo("알림", f"모델 경로가 설정되었습니다: {model_path}")
    else:
        messagebox.showinfo("알림", "모델 파일을 선택하지 않았습니다.")

def browse_image_file():
    global image_path
    image_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.png;*.jpeg")])
    if image_path:
        image_label.config(text=f"이미지 경로: {image_path}")
        messagebox.showinfo("알림", f"이미지 경로가 설정되었습니다: {image_path}")
    else:
        messagebox.showinfo("알림", "이미지를 선택하지 않았습니다.")

def load_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)

def run_model_with_condition(input_image, condition_vector, model):
    try:
        with torch.no_grad():
            z, reconstructed_image, z_hat = model(input_image, condition_vector)

        visualize_images(input_image, reconstructed_image)
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
    if model_path is None:
        messagebox.showinfo("알림", "모델 파일을 선택하세요.")
        return
    if image_path is None:
        messagebox.showinfo("알림", "이미지 파일을 선택하세요.")
        return

    try:
        model = torch.load(rf'{model_path}', map_location=torch.device('cpu'))
        model.eval()

        input_image = load_image(image_path)

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
            run_model_with_condition(input_image, condition_vector, model)

        tk.Button(condition_window, text="클래스 설정", command=on_condition_set).pack(pady=10)
    except Exception as e:
        messagebox.showerror("에러", f"오류가 발생했습니다: {str(e)}")

def visualize_images(original, reconstructed):
    """원본 이미지와 재구성된 이미지를 시각화"""
    original = original.squeeze(0).permute(1, 2, 0).numpy()
    reconstructed = reconstructed.squeeze(0).permute(1, 2, 0).numpy()

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(original)
    axes[0].set_title("원본 이미지")
    axes[0].axis("off")

    axes[1].imshow(reconstructed)
    axes[1].set_title("재구성된 이미지")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()

window = tk.Tk()
window.title("이미지 재구성")

browse_model_button = tk.Button(window, text="모델 찾기", command=browse_model_file, font=("Helvetica", 12))
browse_model_button.grid(row=0, column=0, padx=10, pady=5)

model_label = tk.Label(window, text="", font=("Helvetica", 10), anchor="e")
model_label.grid(row=0, column=1, padx=10, pady=5)

browse_image_button = tk.Button(window, text="이미지 찾기", command=browse_image_file, font=("Helvetica", 12))
browse_image_button.grid(row=1, column=0, padx=10, pady=5)

image_label = tk.Label(window, text="", font=("Helvetica", 10), anchor="e")
image_label.grid(row=1, column=1, padx=10, pady=5)

classify_button = tk.Button(window, text="재구성하기", command=classify_comment, font=("Helvetica", 12))
classify_button.grid(row=2, column=0, columnspan=2, pady=10)

condition_var = tk.IntVar(value=0)

window.mainloop()
