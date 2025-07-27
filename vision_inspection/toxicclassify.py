import tkinter as tk
from tkinter import filedialog, messagebox
import torch
from transformers import AutoTokenizer

PATH = None

def browse_file():
    global PATH
    PATH = filedialog.askopenfilename()
    if PATH:
        path_label.config(text=f"파일 경로: {PATH}")
        messagebox.showinfo("알림", f"파일 경로가 설정되었습니다: {PATH}")
    else:
        messagebox.showinfo("알림", "파일을 선택하지 않았습니다.")

def classify_comment():
    if PATH is None:
        messagebox.showinfo("알림", "모델 파일을 선택하세요.")
        return
    
    model = torch.load(rf'{PATH}')
    tokenizer = AutoTokenizer.from_pretrained("beomi/KcELECTRA-base")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_text = text_input.get().strip()
    if input_text:
        sequences = tokenizer(input_text, padding=True, truncation=True, return_tensors="pt")
        input_ids = sequences['input_ids'].to(device)
        attention_mask = sequences['attention_mask'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1)

        if preds[0].item() == 0:
            result_text.config(text="정상", fg="green", font=("Helvetica", 14, "bold"))
        else:
            result_text.config(text="악플", fg="red", font=("Helvetica", 14, "bold"))
    else:
        messagebox.showinfo("알림", "입력이 비어 있습니다.")

window = tk.Tk()
window.title("악플 분류기")

browse_button = tk.Button(window, text="파일 찾기", command=browse_file, font=("Helvetica", 12))
browse_button.grid(row=0, column=0, padx=10, pady=5)

path_label = tk.Label(window, text="", font=("Helvetica", 10), anchor="e")
path_label.grid(row=0, column=1, padx=10, pady=5)

input_label = tk.Label(window, text="텍스트를 입력하세요:", font=("Helvetica", 12))
input_label.grid(row=1, column=0, padx=10, pady=5)

text_input = tk.Entry(window, width=50, font=("Helvetica", 12))
text_input.grid(row=1, column=1, padx=10, pady=5, sticky="ew")

classify_button = tk.Button(window, text="분류하기", command=classify_comment, font=("Helvetica", 12))
classify_button.grid(row=1, column=2, padx=10, pady=5)

result_frame = tk.Frame(window, relief=tk.GROOVE, borderwidth=5)
result_frame.grid(row=2, column=1, columnspan=2, padx=10, pady=5, sticky='w')

result_text = tk.Label(result_frame, text="", font=("Helvetica", 14), justify="center")
result_text.pack()

result_label = tk.Label(window, text="결과 :", font=("Helvetica", 12))
result_label.grid(row=2, column=0, padx=10, pady=5, sticky="e")

window.mainloop()
