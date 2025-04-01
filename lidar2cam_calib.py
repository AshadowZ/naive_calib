import sys
import numpy as np
import cv2
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QPushButton, QDoubleSpinBox, 
                             QGroupBox, QSizePolicy, QMessageBox, QSpacerItem,
                             QLineEdit, QSpinBox, QFileDialog)
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QImage, QPixmap
from sensor_msgs import point_cloud2
from cv_bridge import CvBridge
from scipy.spatial.transform import Rotation
import rosbag


class LidarProjectionApp(QMainWindow):
    def __init__(self, bag_file=None, camera_model=None, intrinsics=None):
        super().__init__()
        self.bag_file = bag_file
        self.camera_model = camera_model
        self.intrinsics = intrinsics
        self.pc_msgs = []
        self.img_msgs = []
        self.current_frame = 0
        self.points_lidar = None
        try:
            self.load_bag_data()
            self.initUI()
            self.initData()
        except Exception as e:
            QMessageBox.critical(self, "Initialization Error", f"Failed to initialize: {str(e)}")
            sys.exit(1)

    def load_bag_data(self):
        if not self.bag_file:
            raise ValueError("No bag file provided")
            
        bag = rosbag.Bag(self.bag_file, 'r')
        try:
            # 存储所有点云和图像消息
            for topic, msg, t in bag.read_messages():
                if msg._type == 'sensor_msgs/PointCloud2':
                    self.pc_msgs.append(msg)
                elif msg._type == 'sensor_msgs/CompressedImage' or msg._type == 'sensor_msgs/Image':
                    self.img_msgs.append(msg)
                    
            if not self.pc_msgs:
                raise ValueError("No point cloud messages found in the bag")
            if not self.img_msgs:
                raise ValueError("No image messages found in the bag")
                
            # 确保时间同步，这里简化处理，假设顺序匹配
            self.min_frames = min(len(self.pc_msgs), len(self.img_msgs))
        finally:
            bag.close()

    def initUI(self):
        self.setWindowTitle('Lidar to Camera Projection - Multi Frame')
        self.setGeometry(100, 100, 1200, 800)
        
        # 主窗口容器
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        
        # 使用网格布局实现更好的自适应
        main_layout = QHBoxLayout(main_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        
        # 左侧控制面板
        control_panel = QGroupBox("Controls")
        control_panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        control_layout = QVBoxLayout()
        control_layout.setContentsMargins(5, 5, 5, 5)

        # 帧导航控制
        frame_group = QGroupBox("Frame Navigation")
        frame_layout = QHBoxLayout()
        
        self.frame_label = QLabel("Frame:")
        self.frame_input = QSpinBox()
        self.frame_input.setRange(0, self.min_frames-1)
        self.frame_input.setValue(0)
        self.frame_input.valueChanged.connect(self.change_frame)
        
        self.total_frames_label = QLabel(f"/ {self.min_frames-1}")
        
        prev_btn = QPushButton("Previous")
        prev_btn.clicked.connect(self.prev_frame)
        next_btn = QPushButton("Next")
        next_btn.clicked.connect(self.next_frame)
        
        frame_layout.addWidget(self.frame_label)
        frame_layout.addWidget(self.frame_input)
        frame_layout.addWidget(self.total_frames_label)
        frame_layout.addWidget(prev_btn)
        frame_layout.addWidget(next_btn)
        frame_group.setLayout(frame_layout)
        control_layout.addWidget(frame_group)

        # 步长控制
        step_group = QGroupBox("Step Control")
        step_layout = QHBoxLayout()
        
        trans_step_label = QLabel("Translation Step (m):")
        self.trans_step = QDoubleSpinBox()
        self.trans_step.setRange(0.0001, 1.0)
        self.trans_step.setValue(0.01)
        self.trans_step.setDecimals(4)
        
        rot_step_label = QLabel("Rotation Step (deg):")
        self.rot_step = QDoubleSpinBox()
        self.rot_step.setRange(0.01, 10.0)
        self.rot_step.setValue(1.0)
        self.rot_step.setDecimals(2)
        
        step_layout.addWidget(trans_step_label)
        step_layout.addWidget(self.trans_step)
        step_layout.addWidget(rot_step_label)
        step_layout.addWidget(self.rot_step)
        step_group.setLayout(step_layout)
        control_layout.addWidget(step_group)

        # 平移参数输入
        trans_group = QGroupBox("Translation (m)")
        trans_layout = QVBoxLayout()
        
        self.x_input = self.create_param_control("X :     ", -10.0, 10.0, 0.045, 6, is_rotation=False)
        self.y_input = self.create_param_control("Y :     ", -10.0, 10.0, -0.035, 6, is_rotation=False)
        self.z_input = self.create_param_control("Z :     ", -10.0, 10.0, -0.048, 6, is_rotation=False)
        
        trans_layout.addWidget(self.x_input)
        trans_layout.addWidget(self.y_input)
        trans_layout.addWidget(self.z_input)
        trans_group.setLayout(trans_layout)
        
        # 旋转参数输入
        rot_group = QGroupBox("Rotation (degrees)")
        rot_layout = QVBoxLayout()
        
        self.roll_input = self.create_param_control("Roll :     ", -180, 180, 0, 4, is_rotation=True)
        self.pitch_input = self.create_param_control("Pitch :     ", -180, 180, 0, 4, is_rotation=True)
        self.yaw_input = self.create_param_control("Yaw :     ", -180, 180, 0, 4, is_rotation=True)
        
        rot_layout.addWidget(self.roll_input)
        rot_layout.addWidget(self.pitch_input)
        rot_layout.addWidget(self.yaw_input)

        # 添加四元数显示
        self.quat_label = QLabel("Quaternion: [0.000, 0.000, 0.000, 1.000]")
        self.quat_label.setAlignment(Qt.AlignCenter)
        rot_layout.addWidget(self.quat_label)

        rot_group.setLayout(rot_layout)
        
        # 添加保存按钮
        save_group = QGroupBox("Save Extrinsic")
        save_layout = QHBoxLayout()
        save_btn = QPushButton("Save Extrinsic Parameters")
        save_btn.clicked.connect(self.save_extrinsic)
        save_layout.addWidget(save_btn)
        save_group.setLayout(save_layout)
        
        control_layout.addWidget(trans_group)
        control_layout.addWidget(rot_group)
        control_layout.addWidget(save_group)
        control_layout.addStretch()
        control_panel.setLayout(control_layout)
        
        # 右侧图像显示
        image_container = QWidget()
        image_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        image_layout = QVBoxLayout(image_container)
        image_layout.setContentsMargins(0, 0, 0, 0)
        
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.image_label.setMinimumSize(640, 480)
        
        image_layout.addWidget(self.image_label)
        
        # 主布局添加组件
        main_layout.addWidget(control_panel, stretch=1)
        main_layout.addWidget(image_container, stretch=3)


    def save_extrinsic(self):
        try:
            # 获取当前外参参数
            dx = self.x_input.findChild(QDoubleSpinBox).value()
            dy = self.y_input.findChild(QDoubleSpinBox).value()
            dz = self.z_input.findChild(QDoubleSpinBox).value()
            roll = np.deg2rad(self.roll_input.findChild(QDoubleSpinBox).value())
            pitch = np.deg2rad(self.pitch_input.findChild(QDoubleSpinBox).value())
            yaw = np.deg2rad(self.yaw_input.findChild(QDoubleSpinBox).value())
            
            # 创建旋转矩阵
            rot = Rotation.from_euler('xyz', [roll, pitch, yaw])
            rotation_matrix = rot.as_matrix()
            quat = rot.as_quat()  # 返回 [qx, qy, qz, qw]
            
            # 弹出文件保存对话框
            options = QFileDialog.Options()
            file_path, _ = QFileDialog.getSaveFileName(self, 
                                                      "Save Extrinsic Parameters", 
                                                      "extrinsic_params.txt", 
                                                      "Text Files (*.txt)", 
                                                      options=options)
            if file_path:
                with open(file_path, 'w') as f:
                    # 写入平移参数
                    f.write(f"Translation (m):\n")
                    f.write(f"x: {dx}\n")
                    f.write(f"y: {dy}\n")
                    f.write(f"z: {dz}\n\n")
                    
                    # 写入旋转矩阵
                    f.write("Rotation Matrix:\n")
                    for row in rotation_matrix:
                        f.write(f"{row[0]:.6f}, {row[1]:.6f}, {row[2]:.6f}\n")
                    f.write("\n")
                    
                    # 写入四元数
                    f.write("Quaternion:\n")
                    f.write(f"qx: {quat[0]}\n")
                    f.write(f"qy: {quat[1]}\n")
                    f.write(f"qz: {quat[2]}\n")
                    f.write(f"qw: {quat[3]}\n")
                
                QMessageBox.information(self, "Success", "Extrinsic parameters saved successfully!")
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save extrinsic parameters: {str(e)}")

    def create_param_control(self, label, min_val, max_val, default_val, decimals, is_rotation):
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)
        
        value_label = QLabel(label)
        value_label.setMinimumWidth(30)
        value_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        
        spinbox = QDoubleSpinBox()
        spinbox.setRange(min_val, max_val)
        spinbox.setValue(default_val)
        spinbox.setDecimals(decimals)
        spinbox.valueChanged.connect(self.update_projection)
        spinbox.setMinimumWidth(100)
        
        btn_container = QWidget()
        btn_layout = QHBoxLayout(btn_container)
        btn_layout.setContentsMargins(0, 0, 0, 0)
        btn_layout.setSpacing(2)
        
        btn_decrease = QPushButton("-")
        btn_decrease.setFixedWidth(80)
        btn_decrease.clicked.connect(lambda: self.adjust_param(label, -1, is_rotation))
        
        btn_increase = QPushButton("+")
        btn_increase.setFixedWidth(80)
        btn_increase.clicked.connect(lambda: self.adjust_param(label, 1, is_rotation))
        
        btn_layout.addWidget(btn_decrease)
        btn_layout.addWidget(btn_increase)
        
        layout.addWidget(value_label)
        layout.addWidget(spinbox)
        layout.addWidget(btn_container)
        
        return widget

    def initData(self):
        if not self.pc_msgs or not self.img_msgs:
            raise ValueError("No point cloud or image messages loaded")
        
        # 确保在加载数据前所有必要的UI控件都已创建
        if hasattr(self, 'frame_input'):
            self.load_frame_data(0)
        else:
            # 如果UI还没完全初始化，稍后会在initUI完成后自动调用
            pass

    def load_frame_data(self, frame_idx):
        if frame_idx < 0 or frame_idx >= self.min_frames:
            return
            
        self.current_frame = frame_idx
        if hasattr(self, 'frame_input'):
            self.frame_input.setValue(frame_idx)
        
        # 加载当前帧的点云和图像
        self.pc_msg = self.pc_msgs[frame_idx]
        self.img_msg = self.img_msgs[frame_idx]
        
        # 确保点云数据被正确加载
        self.points_lidar = np.array(list(point_cloud2.read_points(
            self.pc_msg, field_names=("x", "y", "z"), skip_nans=True)))
        
        bridge = CvBridge()
        try:
            self.img = bridge.compressed_imgmsg_to_cv2(self.img_msg, "bgr8")
        except AttributeError:
            self.img = bridge.imgmsg_to_cv2(self.img_msg, "bgr8")
        
        # 检查内参是否已设置
        if not hasattr(self, 'intrinsics') or self.intrinsics is None:
            QMessageBox.critical(self, "Error", "Camera intrinsics not set!")
            return
        
        # 只有在UI控件已创建且数据加载完成时才更新投影
        if hasattr(self, 'image_label') and self.points_lidar is not None:
            self.update_projection()

    def change_frame(self, frame_idx):
        if frame_idx != self.current_frame:
            self.load_frame_data(frame_idx)

    def prev_frame(self):
        if self.current_frame > 0:
            self.load_frame_data(self.current_frame - 1)

    def next_frame(self):
        if self.current_frame < self.min_frames - 1:
            self.load_frame_data(self.current_frame + 1)

    def adjust_param(self, label, direction, is_rotation):
        step = self.rot_step.value() if is_rotation else self.trans_step.value()
        
        spinbox = None
        if label == "X :     ":
            spinbox = self.x_input.findChild(QDoubleSpinBox)
        elif label == "Y :     ":
            spinbox = self.y_input.findChild(QDoubleSpinBox)
        elif label == "Z :     ":
            spinbox = self.z_input.findChild(QDoubleSpinBox)
        elif label == "Roll :     ":
            spinbox = self.roll_input.findChild(QDoubleSpinBox)
        elif label == "Pitch :     ":
            spinbox = self.pitch_input.findChild(QDoubleSpinBox)
        elif label == "Yaw :     ":
            spinbox = self.yaw_input.findChild(QDoubleSpinBox)
        
        if spinbox:
            if is_rotation:
                # 对于旋转参数，我们使用增量式更新
                # 获取当前旋转矩阵
                current_roll = np.deg2rad(self.roll_input.findChild(QDoubleSpinBox).value())
                current_pitch = np.deg2rad(self.pitch_input.findChild(QDoubleSpinBox).value())
                current_yaw = np.deg2rad(self.yaw_input.findChild(QDoubleSpinBox).value())
                current_rot = Rotation.from_euler('xyz', [current_roll, current_pitch, current_yaw])
                
                # 创建增量旋转
                if label == "Roll :     ":
                    delta_rot = Rotation.from_euler('x', direction * np.deg2rad(step))
                elif label == "Pitch :     ":
                    delta_rot = Rotation.from_euler('y', direction * np.deg2rad(step))
                elif label == "Yaw :     ":
                    delta_rot = Rotation.from_euler('z', direction * np.deg2rad(step))
                
                # 应用增量旋转
                new_rot = delta_rot * current_rot
                new_euler = new_rot.as_euler('xyz', degrees=True)
                
                # 更新UI显示
                self.roll_input.findChild(QDoubleSpinBox).setValue(new_euler[0])
                self.pitch_input.findChild(QDoubleSpinBox).setValue(new_euler[1])
                self.yaw_input.findChild(QDoubleSpinBox).setValue(new_euler[2])
            else:
                # 平移参数保持原样
                new_value = spinbox.value() + (direction * step)
                spinbox.setValue(new_value)

    def resizeEvent(self, event):
        self.update_projection()
        super().resizeEvent(event)

    def ds_project(self, points_cam, intrinsics):
        fx = intrinsics["fx"]
        fy = intrinsics["fy"]
        cx = intrinsics["cx"]
        cy = intrinsics["cy"]
        xi = intrinsics["xi"]
        alpha = intrinsics["alpha"]
        
        uv = []
        valid_indices = []
        for i, p in enumerate(points_cam):
            x, y, z = p
            if z <= 0:
                continue
            
            d1 = np.sqrt(x**2 + y**2 + z**2)
            d2 = np.sqrt(x**2 + y**2 + (xi*d1 + z)**2)
            k = alpha * d2 + (1 - alpha)*(xi*d1 + z)
            
            if k <= 0:
                continue
            
            u = (x / k) * fx + cx
            v = (y / k) * fy + cy
            uv.append((u, v))
            valid_indices.append(i)
        
        return np.array(uv), valid_indices

    def update_projection(self):
        try:
            # 添加检查确保points_lidar已初始化
            if not hasattr(self, 'points_lidar') or self.points_lidar is None:
                return
                
            # 获取用户输入的参数
            dx = self.x_input.findChild(QDoubleSpinBox).value()
            dy = self.y_input.findChild(QDoubleSpinBox).value()
            dz = self.z_input.findChild(QDoubleSpinBox).value()
            roll = np.deg2rad(self.roll_input.findChild(QDoubleSpinBox).value())
            pitch = np.deg2rad(self.pitch_input.findChild(QDoubleSpinBox).value())
            yaw = np.deg2rad(self.yaw_input.findChild(QDoubleSpinBox).value())
            
            # 创建旋转矩阵
            rot = Rotation.from_euler('xyz', [roll, pitch, yaw])
            quat = rot.as_quat()
            self.quat_label.setText(f"Quaternion: [{quat[0]:.3f}, {quat[1]:.3f}, {quat[2]:.3f}, {quat[3]:.3f}]")
            rot = rot.as_matrix()

            # 创建变换矩阵
            T = np.eye(4)
            T[:3, :3] = rot
            T[:3, 3] = [dx, dy, dz]
            
            # 坐标变换
            points_hom = np.hstack([self.points_lidar, np.ones((len(self.points_lidar), 1))])
            points_cam = (T @ points_hom.T).T[:, :3]
            
            # 投影到图像
            uv, valid_indices = self.ds_project(points_cam, self.intrinsics)
            
            # 创建图像副本
            img_copy = self.img.copy()
            
            if len(valid_indices) > 0:
                uv = uv.astype(int)
                points_valid = self.points_lidar[valid_indices]
                distances = np.linalg.norm(points_valid, axis=1)
                
                min_dist = 0.0
                max_dist = 10.0
                normalized_dist = np.clip((distances - min_dist) / (max_dist - min_dist), 0.0, 1.0)
                
                colors = cv2.applyColorMap(
                    (normalized_dist * 255).astype(np.uint8),
                    cv2.COLORMAP_JET
                ).squeeze()
                
                for (u, v), color in zip(uv, colors):
                    if 0 <= u < img_copy.shape[1] and 0 <= v < img_copy.shape[0]:
                        cv2.circle(img_copy, (u, v), 2, color.tolist(), -1)
                
                img_copy = self.add_right_color_scale(img_copy, min_dist, max_dist)
            
            # 显示图像
            height, width = img_copy.shape[:2]
            bytes_per_line = 3 * width
            q_img = QImage(img_copy.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
            
            pixmap = QPixmap.fromImage(q_img)
            scaled_pixmap = pixmap.scaled(
                self.image_label.size(), 
                Qt.KeepAspectRatio, 
                Qt.SmoothTransformation
            )
            self.image_label.setPixmap(scaled_pixmap)
            
        except Exception as e:
            print(f"Error in projection: {str(e)}")

    def add_right_color_scale(self, img, min_val, max_val):
        height, width = img.shape[:2]
        scale_width = 50
        scale_height = height - 100
        margin = 40
        text_offset = 15
        
        text_width = 100
        new_width = width + scale_width + margin + text_width
        new_img = np.zeros((height, new_width, 3), dtype=np.uint8)
        new_img[:, :width] = img
        
        for i in range(scale_height):
            value = 1.0 - (i / scale_height)
            color = cv2.applyColorMap(np.array([value * 255], dtype=np.uint8), cv2.COLORMAP_JET)[0][0]
            cv2.line(new_img, 
                    (width + margin, margin + i), 
                    (width + margin + scale_width, margin + i), 
                    color.tolist(), 5)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        font_thickness = 2
        text_color = (255, 255, 255)
        text_bg_color = (0, 0, 0, 150)

        text_x = width + margin + scale_width//2 + 20
        
        def put_text_with_bg(img, text, y_pos):
            (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, font_thickness)
            overlay = img.copy()
            cv2.rectangle(overlay,
                        (text_x - 5, y_pos - text_h - 5),
                        (text_x + text_w + 5, y_pos + 5),
                        text_bg_color, -1)
            cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
            cv2.putText(img, text, (text_x, y_pos), 
                    font, font_scale, text_color, 
                    font_thickness, cv2.LINE_AA)

        put_text_with_bg(new_img, f"{max_val:.1f}", margin + 35)
        put_text_with_bg(new_img, f"{(max_val+min_val)/2:.1f}", margin + scale_height//2 + 20)
        put_text_with_bg(new_img, f"{min_val:.1f}", margin + scale_height - 20)
        
        unit_text = "meters"
        (unit_w, unit_h), _ = cv2.getTextSize(unit_text, font, font_scale, font_thickness)
        unit_x = width + margin + scale_width//2 + 20 - unit_w//2
        overlay = new_img.copy()
        cv2.rectangle(overlay,
                    (unit_x - 10, margin + scale_height + 20),
                    (unit_x + unit_w + 10, margin + scale_height + 20 + unit_h + 15),
                    text_bg_color, -1)
        cv2.addWeighted(overlay, 0.6, new_img, 0.4, 0, new_img)
        cv2.putText(new_img, unit_text,
                (unit_x, margin + scale_height + 20 + unit_h + 5),
                font, font_scale, text_color,
                font_thickness, cv2.LINE_AA)

        return new_img


def main():
    if len(sys.argv) < 8:
        print("Usage: python script.py <rosbag_file> <camera_model> <fx> <fy> <cx> <cy> <xi> <alpha>")
        print("Example: python3 test4.py mid-cam-cali.bag ds 301.3403358317853 301.2014464145018 556.7394553845984 531.3838845784604 -0.175461293087452 0.6673152472283514")
        sys.exit(1)
    
    bag_file_name = sys.argv[1]
    camera_model = sys.argv[2]
    intrinsics = {
        "fx": float(sys.argv[3]),
        "fy": float(sys.argv[4]),
        "cx": float(sys.argv[5]),
        "cy": float(sys.argv[6]),
        "xi": float(sys.argv[7]),
        "alpha": float(sys.argv[8])
    }
    
    app = QApplication(sys.argv)
    ex = LidarProjectionApp(bag_file_name, camera_model, intrinsics)
    ex.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()