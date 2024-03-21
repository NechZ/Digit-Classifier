import tkinter
import Classifier
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageOps
from matplotlib import pyplot as plt
import numpy as np


class GUI:
    def __init__(self):
        self.root = tkinter.Tk()
        self.root.title("Digit Classifier")
        self.canvas = tkinter.Canvas(self.root, width=280, height=280, bg="white")
        self.canvas.grid(row=0, column=0, columnspan=2, padx=10, pady=10)
        self.prediction = tkinter.Label(self.root, text="Prediction: None")
        self.prediction.grid(row=1, column=0, columnspan=2)
        self.classify = tkinter.Button(self.root, text="Classify", command=self.classify)
        self.classify.grid(row=2, column=0, columnspan=2, pady=10)

    def main(self):
        self.canvas.bind("<Button-1>", self.draw)
        self.canvas.bind("<B1-Motion>", self.draw)
        self.root.mainloop()

    def draw(self, event):
        x = event.x // 10
        y = event.y // 10
        self.canvas.create_rectangle(x*10, y*10, (x+2) * 10, (y+2) * 10, fill="black", outline="black")
        pass

    def classify(self):
        # Get Image
        self.canvas.postscript(file="image.eps")
        img = Image.open("image.eps")
        img.save("image.png", "png")
        image = Image.open("image.png").convert("L")
        image = ImageOps.invert(image)
        image = image.resize((28, 28))
        image = transforms.ToTensor()(image)
        image = torch.flatten(image, start_dim=1)
        # Run Model
        output = model(image)
        confidence, prediction = torch.max(output, 1)
        self.canvas.delete("all")
        confidence = np.round(confidence.item() * 100, 2)
        self.prediction.config(text=f"Prediction: {prediction.item()} with {confidence.item()}% confidence")
        # Display Image
        plt.imshow(image.view(28, 28).detach().numpy(), cmap="gray")
        plt.show()

if __name__ == "__main__":
    model = Classifier.DigitClassifier()
    model.load_state_dict(torch.load("Digit_classifier.pth"))
    GUI().main()
