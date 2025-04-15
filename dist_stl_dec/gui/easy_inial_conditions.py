import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Circle
import tkinter as tk
import numpy as np

class InteractivePlot:
    def __init__(self, root):
        self.root = root
        self.root.title("Interactive Plot with Circular Patches")
        
        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.root)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=1)
        self.canvas.draw()

        # Set limits and add grid
        self.ax.set_xlim(-10, 10)
        self.ax.set_ylim(-10, 10)
        self.ax.grid(True)

        self.cid = self.canvas.mpl_connect('button_press_event', self.on_click)
        self.patches = []
        self.texts = []  # List to store text labels
        self.centers = {}
        self.counter = 0

        self.remove_button = tk.Button(master=self.root, text="Remove Last Patch", command=self.remove_patch)
        self.remove_button.pack(side=tk.BOTTOM)
        self.save_button = tk.Button(master=self.root, text="Save Centers", command=self.save_centers)
        self.save_button.pack(side=tk.BOTTOM)
        self.load_button = tk.Button(master=self.root, text="Load Centers", command=self.load_centers)
        self.load_button.pack(side=tk.BOTTOM)

    def on_click(self, event):
        if event.inaxes != self.ax:
            return
        # Add a circle patch at the mouse click location
        circle = Circle((event.xdata, event.ydata), 0.1, color='blue', alpha=0.5)
        self.ax.add_patch(circle)
        self.patches.append(circle)

        # Add text label next to the circle
        text = self.ax.text(event.xdata, event.ydata, str(self.counter), color='red', fontsize=12, ha='center', va='center')
        self.texts.append(text)

        self.centers[self.counter] = np.array([event.xdata, event.ydata])
        self.counter += 1
        self.canvas.draw()

    def remove_patch(self):
        if self.patches:
            # Remove the last added patch and text label
            patch = self.patches.pop()
            patch.remove()

            text = self.texts.pop()
            text.remove()

            self.counter -= 1
            del self.centers[self.counter]
            self.canvas.draw()

    def save_centers(self):
        if self.centers:
            # Save centers to a file
            centers_array = np.array(list(self.centers.values()))
            np.save('centers.npy', centers_array)
            print("Centers saved to centers.npy")

    def load_centers(self):
        try:
            # Load centers from a file
            centers_array = np.load('centers.npy')
            self.centers = {i: center for i, center in enumerate(centers_array)}
            self.counter = len(self.centers)
            self.redraw_patches()
            print("Centers loaded from centers.npy")
        except FileNotFoundError:
            print("No centers.npy file found")

    def redraw_patches(self):
        self.ax.clear()
        # Reset limits and grid
        self.ax.set_xlim(-10, 10)
        self.ax.set_ylim(-10, 10)
        self.ax.grid(True)
        self.patches.clear()
        self.texts.clear()  # Clear text labels
        for i, center in self.centers.items():
            circle = Circle(center, 0.1, color='blue', alpha=0.5)
            self.ax.add_patch(circle)
            self.patches.append(circle)

            # Add text label next to the circle
            text = self.ax.text(center[0], center[1], str(i), color='red', fontsize=12, ha='center', va='center')
            self.texts.append(text)
            
        self.canvas.draw()

if __name__ == "__main__":
    root = tk.Tk()
    app = InteractivePlot(root)
    root.mainloop()