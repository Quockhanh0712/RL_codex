# Nhóm 13 IAI-RL-codex - Dự án Orbit Wars Agent

Dự án thiết kế và huấn luyện Agent thông minh tham gia môi trường giả lập **Orbit Wars** sử dụng kết hợp giữa Học tăng cường (Reinforcement Learning - PPO/JAX) và các chiến thuật Heuristic tối ưu.

---

## 👥 Thành viên nhóm & Phân công đóng góp

| STT | Họ và tên | MSSV | Vai trò & Phân công đóng góp | Tỷ lệ đóng góp |
| :---: | :--- | :---: | :--- | :---: |
| 1 | **Trần Quốc Khánh** (Đại diện) | 23020387 | • Thiết kế & huấn luyện mô hình RL (JAX/PPO/Flax)<br>• Cấu trúc môi trường JAX và trích xuất đặc trưng<br>• Đồng quản lý mã nguồn & tích hợp hệ thống | **50%** |
| 2 | **Hoàng Ngọc Nam** | 23020403 | • Phát triển các phiên bản Heuristic (Target-Scoring)<br>• Cài đặt quỹ đạo kinematics, di chuyển và lập kế hoạch (Planner)<br>• Thiết kế các kịch bản phân tích metrics & thử nghiệm | **50%** |

*   **Thành viên đại diện nộp bài:** Trần Quốc Khánh.
*   **Kho lưu trữ mã nguồn (Repository):** [https://github.com/Quockhanh0712/RL_codex.git](https://github.com/Quockhanh0712/RL_codex.git)

---

## 📂 Cấu trúc mã nguồn thư mục

Dự án chia làm các hướng tiếp cận chính và bộ công cụ phân tích đi kèm:

1.  **`OrbitWars_JAX_RL/`**: Agent học máy sử dụng Học tăng cường (Reinforcement Learning).
    *   `train_ppo.py`: Luồng huấn luyện thuật toán **PPO (Proximal Policy Optimization)** được viết bằng **JAX** tối ưu hóa cao.
    *   `jax_env.py`: Môi trường giả lập Orbit Wars tương thích tính toán song song vector hóa của JAX.
    *   `jax_features.py`: Trích xuất đặc trưng từ hành tinh (planet features) và hạm đội (fleet features) làm đầu vào mô hình.
    *   `jax_model.py`: Mạng nơ-ron Transformer (`EntityTransformer`) sử dụng **Flax/Linen** để xử lý số lượng thực thể linh hoạt.
    *   `kaggle.py` & `main_agent.py`: Agent giao tiếp với môi trường thi đấu chính thức.
    *   `submission_final.tar.gz`: Gói nộp bài Agent RL đã huấn luyện xong.
2.  **`Version1_Target-Scoring Heuristic Agent/`**: Phiên bản Heuristic đời đầu dựa trên phương pháp tính toán điểm mục tiêu để ra quyết định gửi quân.
3.  **`Version3/`**: Phiên bản Heuristic cải tiến với độ chính xác cao về quỹ đạo vật lý:
    *   `garrison.py`: Quản lý lính đồn trú và phân bổ lực lượng.
    *   `kinematics.py` & `movement.py`: Giải thuật tính toán động học, quỹ đạo bay của tàu vũ trụ.
    *   `planner.py` & `runtime.py`: Bộ hoạch định thời gian thực đưa ra chiến thuật tấn công/phòng thủ tối ưu.
4.  **`metrics/`**: Bộ công cụ phân tích hiệu năng:
    *   `analyze_orbit_replay.py`: Phân tích file replay của trận đấu để đo lường các chỉ số thắng/thua, hiệu suất chiến đấu của Agent.
5.  **`reference/`**: Các tài liệu kỹ thuật, baseline tham khảo và kinh nghiệm phát triển Orbit Wars.

---

## 🛠️ Công nghệ sử dụng

*   **Học máy & Tối ưu:** [JAX](https://github.com/google/jax), [Flax](https://github.com/google/flax), [Optax](https://github.com/google-deepmind/optax) (giúp tối ưu hóa huấn luyện song song cực nhanh trên GPU/TPU).
*   **Ngôn ngữ lập trình:** Python 3.10+.
