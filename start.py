import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import normalize
from models import ISNetDIS
from skimage import io

def apply_mask_to_image(image, mask):
    image = image.convert("RGBA")
    mask = mask.convert("L")
    bg_removed = Image.new("RGBA", image.size)
    for x in range(image.width):
        for y in range(image.height):
            r, g, b, a = image.getpixel((x, y))
            mask_pixel = mask.getpixel((x, y))
            bg_removed.putpixel((x, y), (r, g, b, mask_pixel))
    return bg_removed

def remove_background(image_path):
    model_path = "saved_model/isnet-general-use.pth"
    image = Image.open(image_path).convert("RGB")
    input_size = [1024, 1024]
    net = ISNetDIS()
    if torch.cuda.is_available():
        net.load_state_dict(torch.load(model_path))
        net = net.cuda()
    else:
        net.load_state_dict(torch.load(model_path, map_location="cpu"))
    net.eval()
    with torch.no_grad():
        im = io.imread(image_path)
        pil_image = Image.fromarray(im)
        if len(im.shape) < 3:
            im = im[:, :, np.newaxis]
        im_shp = im.shape[0:2]
        im_tensor = torch.tensor(im, dtype=torch.float32).permute(2, 0, 1)
        im_tensor = F.interpolate(torch.unsqueeze(im_tensor, 0), input_size, mode="bilinear").type(torch.uint8)
        image = torch.divide(im_tensor, 255.0)
        image = normalize(image, [0.5, 0.5, 0.5], [1.0, 1.0, 1.0])
        if torch.cuda.is_available():
            image = image.cuda()
        result = net(image)
        result = torch.squeeze(F.interpolate(result[0][0], im_shp, mode='bilinear'), 0)
        ma = torch.max(result)
        mi = torch.min(result)
        result = (result - mi) / (ma - mi)
        mask = (result * 255).permute(1, 2, 0).cpu().data.numpy().astype(np.uint8)
        mask_pil_image = Image.fromarray(mask.squeeze(), mode='L')
        bg_removed_image = apply_mask_to_image(pil_image, mask_pil_image)
    return bg_removed_image

def load_image():
    global processed_image
    file_path = filedialog.askopenfilename()
    display_size = (450, 300)
    if file_path:
        try:
            image = Image.open(file_path)
            image.thumbnail(display_size)
            photo = ImageTk.PhotoImage(image)
            image_label.config(image=photo)
            image_label.image = photo
            processed_image = remove_background(file_path)
            processed_image.thumbnail(display_size)
            processed_photo = ImageTk.PhotoImage(processed_image)
            processed_image_label.config(image=processed_photo)
            processed_image_label.image = processed_photo
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")

def save_image():
    if processed_image:
        file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png"), ("All files", "*.*")])
        if file_path:
            processed_image.save(file_path)
            messagebox.showinfo("Save Image", "Image saved successfully!")
    else:
        messagebox.showwarning("Save Image", "No processed image to save.")

app = tk.Tk()
app.title("Background Removal Tool")
app.geometry("1000x400")

app.grid_columnconfigure(0, weight=1)
app.grid_columnconfigure(1, weight=1)
app.grid_rowconfigure(0, weight=1)
app.grid_rowconfigure(1, weight=0)

frame = tk.Frame(app)
frame.grid(row=0, column=0, columnspan=2, sticky="nsew", padx=10, pady=10)

frame.grid_rowconfigure(0, weight=1)
frame.grid_columnconfigure(0, weight=1)
frame.grid_columnconfigure(1, weight=1)

image_label = tk.Label(frame, width=450, height=300, bg="black")
image_label.grid(row=0, column=0, padx=10)

processed_image_label = tk.Label(frame, width=450, height=300, bg="black")
processed_image_label.grid(row=0, column=1, padx=10)

button_frame = tk.Frame(app)
button_frame.grid(row=1, column=0, columnspan=2, pady=10)

load_button = tk.Button(button_frame, text="Load Image", command=load_image)
load_button.grid(row=0, column=0, padx=10, pady=5)

save_button = tk.Button(button_frame, text="Save Image", command=save_image)
save_button.grid(row=0, column=1, padx=10, pady=5)

exit_button = tk.Button(button_frame, text="Exit", command=app.quit)
exit_button.grid(row=0, column=2, padx=10, pady=5)

app.mainloop()