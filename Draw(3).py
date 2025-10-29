import json
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
try:
    from scipy.signal import savgol_filter
except ImportError:
    savgol_filter = None

class SilentChartApp:
    def __init__(self, root):
        self.root = root
        self.root.title("静默数据可视化工具")
        self.file_path = ""
        self.raw_data = None

        # 初始化所有Tkinter变量
        self.smooth_enabled = tk.BooleanVar(value=False)
        self.window_length = tk.IntVar(value=5)
        self.polyorder = tk.IntVar(value=2)

        # 初始化界面组件
        self.create_widgets()
        self.init_chart()  # 仅保留一次初始化

    def create_widgets(self):
        # 创建控制面板
        control_frame = ttk.Frame(self.root, padding=10)
        control_frame.pack(side=tk.TOP, fill=tk.X)

        # 文件选择组件
        ttk.Button(
            control_frame,
            text="打开JSON文件",
            command=self.open_file
        ).pack(side=tk.LEFT, padx=5)

        # 坐标轴选择组件
        ttk.Label(control_frame, text="X轴字段:").pack(side=tk.LEFT, padx=5)
        self.x_axis_combobox = ttk.Combobox(control_frame, width=20)
        self.x_axis_combobox.pack(side=tk.LEFT, padx=5)

        ttk.Label(control_frame, text="Y轴字段:").pack(side=tk.LEFT, padx=5)
        self.y_axis_combobox = ttk.Combobox(control_frame, width=20)
        self.y_axis_combobox.pack(side=tk.LEFT, padx=5)

        # 操作按钮
        ttk.Button(
            control_frame,
            text="刷新图表",
            command=self.update_chart
        ).pack(side=tk.LEFT, padx=5)

        control_right = ttk.Frame(control_frame)
        control_right.pack(side=tk.RIGHT)
        
        # 保存按钮
        ttk.Button(
            control_right,
            text="保存图表",
            command=self.save_chart
        ).pack(side=tk.LEFT, padx=5)
        
        # 平滑功能控件
        ttk.Checkbutton(
            control_right,
            text="平滑曲线",
            variable=self.smooth_enabled,
            command=self.toggle_smooth_options
        ).pack(side=tk.LEFT, padx=5)
        
        self.smooth_frame = ttk.Frame(control_right)
        
        ttk.Label(self.smooth_frame, text="窗口:").pack(side=tk.LEFT)
        ttk.Spinbox(
            self.smooth_frame,
            from_=3,
            to=21,
            increment=2,
            textvariable=self.window_length,
            width=5
        ).pack(side=tk.LEFT)
        
        ttk.Label(self.smooth_frame, text="阶数:").pack(side=tk.LEFT)
        ttk.Spinbox(
            self.smooth_frame,
            from_=2,
            to=5,
            textvariable=self.polyorder,
            width=3
        ).pack(side=tk.LEFT)

    def toggle_smooth_options(self):
        """控制平滑参数控件的显示"""
        if self.smooth_enabled.get():
            self.smooth_frame.pack(side=tk.LEFT, padx=5)
        else:
            self.smooth_frame.pack_forget()

    def apply_smoothing(self, y_data):
        """应用Savitzky-Golay平滑"""
        if not savgol_filter:
            messagebox.showerror("错误", "请先安装scipy库：pip install scipy")
            return y_data
            
        window = self.window_length.get()
        polyorder = self.polyorder.get()
        
        # 确保窗口为奇数且小于数据长度
        window = min(len(y_data)//2*2-1, window)  # 最大可用奇数
        window = max(3, window)
        if window % 2 == 0:
            window += 1
            
        try:
            return savgol_filter(y_data, window, polyorder)
        except Exception as e:
            messagebox.showerror("平滑错误", str(e))
            return y_data

    def save_chart(self):
        """保存图表到文件"""
        if not hasattr(self, 'figure') or len(self.figure.axes) == 0:
            messagebox.showwarning("警告", "请先生成图表")
            return
            
        filetypes = [
            ('PNG 图片', '*.png'),
            ('JPEG 图片', '*.jpg'),
            ('PDF 文档', '*.pdf'),
            ('SVG 矢量图', '*.svg')
        ]
        
        path = filedialog.asksaveasfilename(
            filetypes=filetypes,
            defaultextension=".png"
        )
        
        if path:
            try:
                self.figure.savefig(path, dpi=300, bbox_inches='tight')
                messagebox.showinfo("成功", f"图表已保存至：{path}")
            except Exception as e:
                messagebox.showerror("保存失败", str(e))

    def init_chart(self):
        # 初始化图表
        self.figure = Figure(figsize=(10, 6))
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.root)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def open_file(self):
        self.file_path = filedialog.askopenfilename(
            filetypes=[("JSON文件", "*.json")]
        )
        if self.file_path and self.load_data():
            keys = ['num'] + list(self.raw_data.keys())
            self.x_axis_combobox['values'] = keys
            self.y_axis_combobox['values'] = list(self.raw_data.keys())
            self.x_axis_combobox.set('num')
            if self.raw_data:
                self.y_axis_combobox.set(next(iter(self.raw_data.keys()), ))

    def load_data(self):
        try:
            with open(self.file_path, 'r') as f:
                data = json.load(f)
                
            # 强制转换为字典格式
            if not isinstance(data, dict):
                data = {"data": [data]}
                
            # 转换所有值为列表
            self.raw_data = {}
            for k, v in data.items():
                if not isinstance(v, list):
                    self.raw_data[k] = [v]
                else:
                    self.raw_data[k] = v
            return True
        except:
            return False

    def validate_entries(self):
        x_key = self.x_axis_combobox.get().strip()
        y_key = self.y_axis_combobox.get().strip()
        return (
            x_key and y_key and
            (x_key == 'num' or x_key in self.raw_data) and
            y_key in self.raw_data
        )

    def process_data(self, x_key, y_key):
        try:
            # 获取Y轴数据并过滤无效值
            raw_y = self.raw_data[y_key]
            y_data = []
            for y in raw_y:
                try:
                    y_data.append(float(y))
                except (ValueError, TypeError):
                    continue

            # 生成X轴数据
            if x_key == 'num':
                x_data = list(range(len(y_data)))
            else:
                x_data = []
                for x in self.raw_data[x_key]:
                    try:
                        x_data.append(float(x))
                    except (ValueError, TypeError):
                        continue

            # 自动对齐数据长度
            min_length = min(len(x_data), len(y_data))
            return x_data[:min_length], y_data[:min_length]
        except:
            return None

    def update_chart(self):
        if not self.raw_data or not self.validate_entries():
            return
            
        x_key = self.x_axis_combobox.get().strip()
        y_key = self.y_axis_combobox.get().strip()
        processed = self.process_data(x_key, y_key)
        if processed and all(len(d) > 0 for d in processed):
            self.draw_chart(*processed, x_key, y_key)

    def draw_chart(self, x_data, y_data, x_label, y_label):
        """更新后的绘图方法"""
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        # 应用平滑
        if self.smooth_enabled.get():
            y_data = self.apply_smoothing(y_data)
            line_label = f"{y_label} (平滑后)"
        else:
            line_label = y_label
            
        # 绘制主曲线
        ax.plot(
            x_data, y_data,
            linestyle='-',
            color='steelblue',
            linewidth=2,
            label=line_label
        )
        
        # 添加辅助元素
        ax.set_title(f"{y_label} value change curve", fontsize=14, pad=15)
        ax.legend(loc='upper right')
        
        # 增强样式配置
        ax.set_xlabel(x_label, fontsize=12, labelpad=10)
        ax.set_ylabel(y_label, fontsize=12, labelpad=10)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        self.figure.tight_layout()
        self.canvas.draw()


if __name__ == "__main__":
    root = tk.Tk()
    app = SilentChartApp(root)
    root.mainloop()