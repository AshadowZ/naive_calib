import sys
import os
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
from sensor_msgs.msg import PointCloud2, Image, CompressedImage
import bisect
import open3d as o3d


class LidarProjectionApp(QMainWindow):
    def __init__(self, base_dir=None, camera_model=None, intrinsics=None):
        super().__init__()
        # 添加预分配内存
        self.points_hom = None
        self.points_cam = None
        self.color_map = None

        self.base_dir = base_dir
        self.camera_model = camera_model
        self.intrinsics = intrinsics
        
        # 修改为使用文件路径列表
        self.img_files = []
        self.pcd_files = []
        self.pcd_poses = []
        
        self.current_frame = 0
        self.points_lidar = None
        
        try:
            if base_dir:
                self.load_file_data(base_dir)
            self.initUI()
            self.initData()
        except Exception as e:
            QMessageBox.critical(self, "Initialization Error", f"Failed to initialize: {str(e)}")
            sys.exit(1)

    def load_file_data(self, base_dir, time_tolerance=0.02):
        try:
            images_dir = os.path.join(base_dir, "images")
            pcds_dir = os.path.join(base_dir, "pcds")
            images_time_file = os.path.join(base_dir, "images_time.txt")
            pcds_time_pose_file = os.path.join(base_dir, "pcds_time_pose.txt")

            # 验证目录和文件是否存在
            if not os.path.exists(images_dir) or not os.path.exists(pcds_dir):
                raise FileNotFoundError("Images or PCDs directory not found")
            if not os.path.exists(images_time_file) or not os.path.exists(pcds_time_pose_file):
                raise FileNotFoundError("Timestamp files not found")

            # 读取图像时间戳
            img_times = []
            img_files = []
            with open(images_time_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    parts = line.split()
                    if len(parts) >= 2:
                        idx, ts = parts[0], parts[1]
                        img_path = os.path.join(images_dir, f"{idx}.png")
                        if os.path.exists(img_path):
                            img_times.append(float(ts))
                            img_files.append(img_path)
                        else:
                            print(f"Warning: Image file not found: {img_path}")

            # 读取点云时间戳 + pose
            pcd_times = []
            pcd_files = []
            pcd_poses = []
            with open(pcds_time_pose_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    parts = line.split()
                    if len(parts) >= 9:
                        idx = parts[0]
                        ts = float(parts[1])
                        tx, ty, tz = map(float, parts[2:5])
                        qw, qx, qy, qz = map(float, parts[5:9])
                        pcd_path = os.path.join(pcds_dir, f"{idx}.pcd")
                        if os.path.exists(pcd_path):
                            pcd_times.append(ts)
                            pcd_files.append(pcd_path)
                            pcd_poses.append((tx, ty, tz, qx, qy, qz, qw))
                        else:
                            print(f"Warning: Point cloud file not found: {pcd_path}")

            if not img_times or not pcd_times:
                raise ValueError("No valid image or point cloud data found")

            # 匹配图像与点云
            img_times = np.array(img_times)
            pcd_times = np.array(pcd_times)

            matched_img_files = []
            matched_pcd_files = []
            matched_poses = []

            for ts, pcd_file, pose in zip(pcd_times, pcd_files, pcd_poses):
                idx = bisect.bisect_left(img_times, ts)
                candidates = []
                if idx < len(img_times):
                    candidates.append((abs(img_times[idx] - ts), idx))
                if idx > 0:
                    candidates.append((abs(img_times[idx-1] - ts), idx-1))
                if not candidates:
                    continue
                dt, best_idx = min(candidates, key=lambda x: x[0])
                if dt <= time_tolerance:
                    matched_img_files.append(img_files[best_idx])
                    matched_pcd_files.append(pcd_file)
                    matched_poses.append(pose)

            self.min_frames = len(matched_img_files)
            if self.min_frames == 0:
                raise ValueError("No synchronized frames found")

            # 缓存数据
            self.img_files = matched_img_files
            self.pcd_files = matched_pcd_files
            self.pcd_poses = matched_poses
            print(f"Successfully matched {self.min_frames} image-pcd pairs")

        except Exception as e:
            raise ValueError(f"Failed to load file data: {str(e)}")

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
        
        # 在控制面板中添加距离范围控制（放在step_group后面）
        dist_group = QGroupBox("Distance Range (m)")
        dist_layout = QHBoxLayout()
        
        min_dist_label = QLabel("Min Dist:")
        self.min_dist_input = QDoubleSpinBox()
        self.min_dist_input.setRange(0.0, 100.0)
        self.min_dist_input.setValue(0.0)
        self.min_dist_input.setDecimals(1)
        self.min_dist_input.valueChanged.connect(self.update_projection)
        
        max_dist_label = QLabel("Max Dist:")
        self.max_dist_input = QDoubleSpinBox()
        self.max_dist_input.setRange(0.1, 100.0)
        self.max_dist_input.setValue(10.0)
        self.max_dist_input.setDecimals(1)
        self.max_dist_input.valueChanged.connect(self.update_projection)
        
        dist_layout.addWidget(min_dist_label)
        dist_layout.addWidget(self.min_dist_input)
        dist_layout.addWidget(max_dist_label)
        dist_layout.addWidget(self.max_dist_input)
        dist_group.setLayout(dist_layout)
        
        # 将距离范围控件插入到布局中（例如放在step_group后面）
        control_layout.insertWidget(2, dist_group)  # 调整插入位置为适合你的UI布局

        # 在控制面板中添加模式切换按钮
        mode_group = QGroupBox("Visualization Mode")
        mode_layout = QHBoxLayout()
        
        self.mode_btn = QPushButton("Switch to Dense Mode")
        self.mode_btn.setCheckable(True)
        self.mode_btn.clicked.connect(self.toggle_visualization_mode)

        self.num_frames_label = QLabel("Frames to Accumulate:")
        self.num_frames_input = QSpinBox()
        self.num_frames_input.setRange(1, 50)
        self.num_frames_input.setValue(5)
        self.num_frames_input.setEnabled(False)
        self.num_frames_input.valueChanged.connect(self.on_num_frames_changed)
        
        mode_layout.addWidget(self.mode_btn)
        mode_layout.addWidget(self.num_frames_label)
        mode_layout.addWidget(self.num_frames_input)
        mode_group.setLayout(mode_layout)

        # 将模式控件插入到布局中合适位置
        control_layout.insertWidget(3, mode_group)
        
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

    def on_num_frames_changed(self):
        """当累积帧数改变时重新累积点云"""
        if self.mode_btn.isChecked():
            self.accumulate_point_clouds()
            self.update_projection()

    def toggle_visualization_mode(self):
        if self.mode_btn.isChecked():
            # 切换到稠密模式
            self.mode_btn.setText("Switch to Sparse Mode")
            self.frame_input.setEnabled(False)
            self.num_frames_input.setEnabled(True)
            self.accumulate_point_clouds()
        else:
            # 切换回稀疏模式
            self.mode_btn.setText("Switch to Dense Mode")
            self.frame_input.setEnabled(True)
            self.num_frames_input.setEnabled(False)
            self.load_frame_data(self.current_frame)  # 重新加载当前单帧
        
        self.update_projection()

    def accumulate_point_clouds(self):
        """累积多帧点云数据并转换到当前帧坐标系"""
        if not hasattr(self, 'pcd_files') or not self.pcd_files:
            return
        
        num_frames = self.num_frames_input.value()
        center_idx = self.current_frame
        start_idx = max(0, center_idx - num_frames // 2)
        end_idx = min(len(self.pcd_files), start_idx + num_frames)
        
        # 获取当前帧的位姿（body to world）
        current_tx, current_ty, current_tz, current_qx, current_qy, current_qz, current_qw = self.pcd_poses[center_idx]
        T_current_to_world = np.eye(4)
        T_current_to_world[:3, :3] = Rotation.from_quat([current_qx, current_qy, current_qz, current_qw]).as_matrix()
        T_current_to_world[:3, 3] = [current_tx, current_ty, current_tz]
        
        # 计算世界到当前帧的变换
        T_world_to_current = np.eye(4)
        T_world_to_current[:3, :3] = T_current_to_world[:3, :3].T
        T_world_to_current[:3, 3] = -T_world_to_current[:3, :3] @ T_current_to_world[:3, 3]
        
        accumulated_points = []
        
        for i in range(start_idx, end_idx):
            try:
                pcd = o3d.io.read_point_cloud(self.pcd_files[i])
                points = np.asarray(pcd.points)
                
                # 获取该帧的位姿（body to world）
                tx, ty, tz, qx, qy, qz, qw = self.pcd_poses[i]
                T_frame_to_world = np.eye(4)
                T_frame_to_world[:3, :3] = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
                T_frame_to_world[:3, 3] = [tx, ty, tz]
                
                # 计算该帧到当前帧的变换: T_current = T_world_to_current @ T_frame_to_world
                T_frame_to_current = T_world_to_current @ T_frame_to_world
                
                # 转换点云到当前帧坐标系
                points_hom = np.hstack([points, np.ones((len(points), 1))])
                points_current = (T_frame_to_current @ points_hom.T).T[:, :3]
                
                accumulated_points.append(points_current)
            except Exception as e:
                print(f"Error loading frame {i}: {str(e)}")
        
        if accumulated_points:
            self.points_lidar = np.vstack(accumulated_points)
            self.points_hom = np.hstack([self.points_lidar, np.ones((len(self.points_lidar), 1))])
        else:
            self.points_lidar = None
            self.points_hom = None

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
        # 修改检查条件
        if not hasattr(self, 'img_files') or not self.img_files or not hasattr(self, 'pcd_files') or not self.pcd_files:
            raise ValueError("No image or point cloud files loaded")
        
        # 确保在加载数据前所有必要的UI控件都已创建
        if hasattr(self, 'frame_input'):
            self.load_frame_data(0)

    def load_frame_data(self, frame_idx):
        if frame_idx < 0 or frame_idx >= self.min_frames:
            return

        self.current_frame = frame_idx
        if hasattr(self, 'frame_input'):
            self.frame_input.setValue(frame_idx)

        try:
            # 读取图像
            img_path = self.img_files[frame_idx]
            self.img = cv2.imread(img_path)
            if self.img is None:
                raise ValueError(f"Failed to load image: {img_path}")

            # 读取点云
            pcd_path = self.pcd_files[frame_idx]
            pcd = o3d.io.read_point_cloud(pcd_path)
            self.points_lidar = np.asarray(pcd.points)
            if len(self.points_lidar) == 0:
                raise ValueError(f"Empty point cloud: {pcd_path}")

            # 预计算齐次坐标
            self.points_hom = np.hstack([self.points_lidar, np.ones((len(self.points_lidar), 1))])
            self.color_map = cv2.applyColorMap(np.arange(256, dtype=np.uint8), cv2.COLORMAP_JET).squeeze()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load frame data: {str(e)}")
            return

        if hasattr(self, 'image_label'):
            self.update_projection()

    def change_frame(self, frame_idx):
        if self.mode_btn.isChecked():
            QMessageBox.warning(self, "Warning", "Cannot change frame in dense mode. Switch to sparse mode first.")
            return
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
        """支持双球面(ds)和pinhole-radtan两种相机模型的投影"""
        fx = intrinsics["fx"]
        fy = intrinsics["fy"]
        cx = intrinsics["cx"]
        cy = intrinsics["cy"]
        
        uv = []
        valid_indices = []
        
        if self.camera_model == "ds":
            # 双球面模型投影
            xi = intrinsics["xi"]
            alpha = intrinsics["alpha"]
            
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
                
        elif self.camera_model == "pinhole-radtan":
            # pinhole-radtan模型投影
            k1 = intrinsics.get("k1", 0)
            k2 = intrinsics.get("k2", 0)
            p1 = intrinsics.get("p1", 0)
            p2 = intrinsics.get("p2", 0)
            
            for i, p in enumerate(points_cam):
                x, y, z = p
                if z <= 0:
                    continue
                    
                # 归一化平面坐标
                xn = x / z
                yn = y / z
                
                # 径向畸变
                r2 = xn*xn + yn*yn
                radial_dist = 1 + k1*r2 + k2*r2*r2
                
                # 切向畸变
                xy = xn * yn
                tangential_x = 2*p1*xy + p2*(r2 + 2*xn*xn)
                tangential_y = p1*(r2 + 2*yn*yn) + 2*p2*xy
                
                # 应用畸变
                xd = xn * radial_dist + tangential_x
                yd = yn * radial_dist + tangential_y
                
                # 投影到像素平面
                u = xd * fx + cx
                v = yd * fy + cy
                if 0 <= u < self.img.shape[1] and 0 <= v < self.img.shape[0]:
                    uv.append((u, v))
                    valid_indices.append(i)
                    
        else:
            raise ValueError(f"Unsupported camera model: {self.camera_model}")
            
        return np.array(uv), valid_indices

    def update_projection(self):
        try:
            # 添加检查确保必要数据已初始化
            if (not hasattr(self, 'img') or self.img is None or
                not hasattr(self, 'intrinsics') or self.intrinsics is None):
                return
                
            # 如果是稠密模式且点云未加载，则重新累积
            if self.mode_btn.isChecked() and not hasattr(self, 'points_lidar'):
                self.accumulate_point_clouds()
                
            # 获取用户输入的参数
            dx = self.x_input.findChild(QDoubleSpinBox).value()
            dy = self.y_input.findChild(QDoubleSpinBox).value()
            dz = self.z_input.findChild(QDoubleSpinBox).value()
            roll = np.deg2rad(self.roll_input.findChild(QDoubleSpinBox).value())
            pitch = np.deg2rad(self.pitch_input.findChild(QDoubleSpinBox).value())
            yaw = np.deg2rad(self.yaw_input.findChild(QDoubleSpinBox).value())
            
            # 直接计算旋转矩阵，避免创建Rotation对象
            cr, sr = np.cos(roll), np.sin(roll)
            cp, sp = np.cos(pitch), np.sin(pitch)
            cy, sy = np.cos(yaw), np.sin(yaw)
            
            rot = np.array([
                [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
                [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
                [-sp, cp*sr, cp*cr]
            ])

            # 计算四元数并更新显示
            quat = Rotation.from_matrix(rot).as_quat()
            self.quat_label.setText(f"Quaternion: [{quat[0]:.3f}, {quat[1]:.3f}, {quat[2]:.3f}, {quat[3]:.3f}]")
            
            # 直接计算变换后的坐标
            if hasattr(self, 'points_lidar') and self.points_lidar is not None:
                self.points_cam = (rot @ self.points_lidar.T).T + np.array([dx, dy, dz])
            
            # 投影到图像
            if hasattr(self, 'points_cam') and self.points_cam is not None:
                uv, valid_indices = self.ds_project(self.points_cam, self.intrinsics)
            else:
                uv, valid_indices = np.array([]), []
            
            img_copy = self.img.copy()
            
            # 检查是否有点云点被成功投影到图像上
            if not valid_indices:
                print("No lidar points can be projected onto the current image.")
            else:
                # 获取用户设置的距离范围
                min_dist = self.min_dist_input.value()
                max_dist = self.max_dist_input.value()
                
                # 确保最大距离大于最小距离
                if max_dist <= min_dist:
                    max_dist = min_dist + 0.1
                    self.max_dist_input.setValue(max_dist)
                
                if len(valid_indices) > 0 and uv.size > 0:
                    uv = uv.astype(int)
                    points_valid = self.points_lidar[valid_indices]
                    distances = np.linalg.norm(points_valid, axis=1)
                    
                    # 使用用户设置的距离范围进行归一化
                    norm_dist = np.clip((distances - min_dist) / (max_dist - min_dist), 0.0, 1.0)
                    color_indices = (norm_dist * 255).astype(np.uint8)
                    
                    # 使用向量化操作绘制点
                    mask = (uv[:,0] >= 0) & (uv[:,0] < img_copy.shape[1]) & \
                        (uv[:,1] >= 0) & (uv[:,1] < img_copy.shape[0])
                    
                    uv_valid = uv[mask]
                    colors = self.color_map[color_indices[mask]]
                    
                    # 批量绘制点，加大点的大小
                    for (u, v), color in zip(uv_valid, colors):
                        cv2.circle(img_copy, (u, v), 2, color.tolist(), -1)
            
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
        print("Usage: python script.py <base_dir> <camera_model> <fx> <fy> <cx> <cy> [model_params...]")
        print("For ds model: python script.py base_dir ds fx fy cx cy xi alpha")
        print("For pinhole-radtan model: python script.py base_dir pinhole-radtan fx fy cx cy k1 k2 p1 p2")
        sys.exit(1)

    base_dir = sys.argv[1]
    camera_model = sys.argv[2]
    
    # 基本内参
    intrinsics = {
        "fx": float(sys.argv[3]),
        "fy": float(sys.argv[4]),
        "cx": float(sys.argv[5]),
        "cy": float(sys.argv[6])
    }
    
    # 根据相机模型添加特定参数
    if camera_model == "ds":
        if len(sys.argv) < 9:
            print("DS model requires xi and alpha parameters")
            sys.exit(1)
        intrinsics.update({
            "xi": float(sys.argv[7]),
            "alpha": float(sys.argv[8])
        })
    elif camera_model == "pinhole-radtan":
        if len(sys.argv) < 11:
            print("Pinhole-radtan model requires k1, k2, p1, p2 parameters")
            sys.exit(1)
        intrinsics.update({
            "k1": float(sys.argv[7]),
            "k2": float(sys.argv[8]),
            "p1": float(sys.argv[9]),
            "p2": float(sys.argv[10])
        })
    else:
        print(f"Unsupported camera model: {camera_model}")
        sys.exit(1)

    app = QApplication(sys.argv)
    ex = LidarProjectionApp(base_dir, camera_model, intrinsics)
    ex.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()