# import tkinter as tk
# from tkinter import ttk, filedialog
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# import numpy as np
# from preprocessing import preprocess_images
# from model_training import load_model

# class PneumoScan3D:
#     def __init__(self, root):
#         self.root = root
#         self.root.title("Veridis Scan 3D - Diagnostic Assisté")

#         self.model = load_model('/Users/dalm1/Desktop/reroll/Progra/par20/results/best_traditional_model.pkl')

#         self.main_frame = ttk.Frame(self.root)
#         self.main_frame.pack(fill=tk.BOTH, expand=True)

#         self.create_controls()
#         self.create_3d_canvas()
#         self.create_result_panel()

#     def create_controls(self):
#         control_frame = ttk.Frame(self.main_frame)
#         control_frame.pack(fill=tk.X, padx=10, pady=10)

#         self.btn_load = ttk.Button(control_frame, text="📁 Charger Radiographie",
#                                  command=self.load_image)
#         self.btn_load.pack(side=tk.LEFT)

#     def create_3d_canvas(self):
#         # 3D Visualization canvas
#         self.fig = plt.figure(figsize=(8, 6))
#         self.ax = self.fig.add_subplot(111, projection='3d')

#         # Initial empty visualization
#         self.create_accuracy_cone(0.5)  # Default 50% confidence

#         self.canvas = FigureCanvasTkAgg(self.fig, master=self.main_frame)
#         self.canvas.draw()
#         self.canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

#     def create_result_panel(self):
#         result_frame = ttk.Frame(self.main_frame)
#         result_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=10)

#         self.lbl_prediction = ttk.Label(result_frame, text="Prédiction ", font=('Helvetica', 14))
#         self.lbl_prediction.pack(pady=20)

#         self.lbl_confidence = ttk.Label(result_frame, text="Accuratie ", font=('Helvetica', 12))
#         self.lbl_confidence.pack(pady=10)

#     def load_image(self):
#         file_path = filedialog.askopenfilename()
#         if file_path:
#             self.analyze_image(file_path)

#     def analyze_image(self, image_path):
#         processed_img = preprocess_images([image_path])
#         probability = self.model.predict_proba(processed_img)[0][1]

#         prediction = "PROB DESEASE" if probability > 0.5 else "NORMAL"
#         self.lbl_prediction.config(text=f"Prédiction {prediction}")
#         self.lbl_confidence.config(text=f"Accuratie {probability*100:.2f}%")

#         self.update_3d_visualization(probability)

#     def create_accuracy_cone(self, height):
#         # Create rotating 3D cone
#         theta = np.linspace(0, 2*np.pi, 100)
#         z = np.linspace(0, height, 100)
#         theta, z = np.meshgrid(theta, z)
#         x = z * np.cos(theta)
#         y = z * np.sin(theta)

#         self.ax.clear()
#         self.ax.plot_surface(x, y, z, cmap='viridis', alpha=0.8)
#         self.ax.set_title('Niveau de Confiance 3D', fontsize=14)
#         self.ax.set_zlim(0, 1)

#     def update_3d_visualization(self, accuracy):
#         self.create_accuracy_cone(accuracy)
#         self.canvas.draw()

# if __name__ == "__main__":
#     root = tk.Tk()
#     root.geometry("1200x800")
#     app = PneumoScan3D(root)
#     root.mainloop()
