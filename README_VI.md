# AI Auto Labeler for CVAT

Ứng dụng Streamlit kết nối CVAT, chạy YOLO/SegFormer, xem và duyệt annotation theo từng frame, rồi thêm kết quả đã xác nhận vào Job. Ảnh và mask của một lần inference được lưu theo frame trong `outputs/predictions/` để không giữ cả task trong RAM.

## Cài đặt và chạy

Yêu cầu Python 3.10 trở lên để chạy YOLO và ứng dụng. Trong thư mục repo:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
streamlit run app.py
```

Mở `http://localhost:8501`. Tool tự ưu tiên GPU CUDA nếu PyTorch nhận được GPU; nếu không, YOLO/SegFormer tiếp tục chạy CPU.

### Bật GPU và SAM 3 trên Windows

Máy có GPU NVIDIA nhưng PyTorch bản CPU vẫn sẽ chạy CPU. Tạo một môi trường riêng đã cài CUDA bằng lệnh:

```powershell
.\setup_gpu.ps1
```

Sau khi hoàn tất, mở app bằng môi trường mới:

```powershell
.\run_gpu.ps1
```

Script cài PyTorch CUDA 12.6, dependencies của tool và CLIP cho text/exemplar prompt. SAM 3 cần checkpoint đã được Meta cấp quyền tải. Sau khi request được chấp nhận, đăng nhập bằng `\.venv-sam3\Scripts\hf.exe auth login`; `run_gpu.ps1` sẽ tự tải checkpoint về `models/sam3.pt` khi chưa có. Có thể dùng checkpoint `sam3.1_multiplex.pt` cho inference ảnh bằng cách cấu hình `SAM3_WEIGHTS`. Tool dùng FP16 và ảnh kích thước 640 để giảm VRAM; GPU 4 GB có thể cần xử lý từng frame nếu hết bộ nhớ.

## Canvas local khi tài khoản không thấy Task

Trong sidebar, chọn **Canvas local** để làm việc mà không cần quyền xem Task CVAT. Tải ảnh lên, nhập danh sách nhãn, chọn Polygon hoặc Rectangle rồi vẽ trực tiếp trên canvas. Polygon kết thúc bằng nút **Hoàn tất polygon** hoặc nhấp đúp; **Undo** lùi một điểm đang vẽ hoặc xóa shape cuối.

Nút **Tải bộ dữ liệu COCO (.zip)** xuất ảnh cùng annotation segmentation theo định dạng COCO. Tạo/chọn Task CVAT có cùng nhãn rồi nhập ZIP này vào CVAT. Chế độ local không ghi annotation thẳng vào CVAT.

Canvas dùng Streamlit Custom Components v2; cần Streamlit 1.57 trở lên.

## Kết nối CVAT

Trong sidebar, chọn môi trường và URL:

- Dán trực tiếp link dạng `https://cvat.edufun.ai/tasks/226/jobs/982` vào ô **Liên kết Task / Job**. Tool tự nhận server, Task và Job; kết nối CVAT rồi bấm **Tải Task / Job**.

- **CVAT Local**: mặc định `http://localhost:8080`.
- **CVAT Online**: mặc định `https://cvat.edufun.ai`.
- **Custom**: nhập URL của server khác.

Chọn đăng nhập bằng username/password để tạo session CVAT, hoặc Personal Access Token (PAT). PAT được CVAT khuyến nghị cho tích hợp API. Tạo PAT trong CVAT ở phần User settings → Security, rồi nhập vào trường token. Ứng dụng giữ session/token trong bộ nhớ của phiên Streamlit hiện tại và không ghi chúng vào file. Khi session hết hạn, đăng nhập lại.

Nút **Test Connection** kiểm tra đăng nhập và thông tin máy chủ. Nhập **Organization slug** (với workspace này là `cohort-5a`) để gửi đúng organization context trong API request. Với quyền Worker, hãy dán link Job trực tiếp rồi bấm **Tải Task / Job**; công cụ đọc labels, frame range, ảnh và annotations qua API của Job, không cần đọc Task trước.

Nếu nhận lỗi 401, kiểm tra thông tin đăng nhập hoặc token. CVAT có thể cho Worker xem và làm Job được giao nhưng từ chối trang/API của Task; chọn tổ chức `cohort-5a` chỉ đổi workspace, không tự cấp vai trò hay quyền. Trong trường hợp đó, dùng link Job trực tiếp. Nếu chính `GET /api/jobs/<id>` cũng trả 403, nhờ Owner/Supervisor kiểm tra thành viên trong tổ chức và giao Job cho đúng tài khoản ở stage **Annotation**. Xem thêm [vai trò CVAT](https://docs.cvat.ai/docs/account_management/user-roles/) và [trang Jobs](https://docs.cvat.ai/docs/workspace/jobs-page/).

## Chạy inference

Chọn một bước trên thanh quy trình: **Tạo nhãn**, **Duyệt & chỉnh sửa**, **Đánh giá**, **Đồng bộ CVAT**, hoặc **Xóa nhãn frame**. Kết nối, chọn Task/Job và Canvas local vẫn ở sidebar.

Trong **Xóa nhãn frame**, chọn số frame CVAT để xem danh sách shape/tag rồi xác nhận xóa. Tool tải annotation mới nhất, chỉ xóa shape/tag độc lập đúng frame và kiểm tra lại trên server; không cần chạy inference. Nếu có track đang đi qua frame, thao tác xóa riêng frame bị khóa để tránh xóa track ở các frame khác. Để dọn nhanh cả Job, dùng **Xóa toàn bộ nhãn của Job**; thao tác này xóa shape, tag, track và interval trên mọi frame của Job, giữ nguyên ảnh/frame, đồng thời yêu cầu xác nhận riêng.

Các chế độ inference:

Trong **Cấu hình model**, ngưỡng confidence mặc định là 0.35 và có thanh **Bỏ vùng mask nhỏ hơn** (mặc định 96 pixel). Mask vạch đường cùng label được gộp để duyệt chung, mảnh nhiễu bị lọc trước QC; khi đồng bộ, polyline ngắn hơn 32 pixel bị bỏ và tool chặn trước khi gửi nếu một frame vượt 250 shape. Có thể hạ ngưỡng/diện tích/độ dài để giữ chi tiết nhỏ, hoặc tăng giới hạn sau khi kiểm tra preview.

Trong **Duyệt & chỉnh sửa**, dùng **Duyệt mask không chồng lấn · mọi ảnh** hoặc **Duyệt tất cả mask · mọi ảnh** để đánh dấu hàng loạt trên toàn bộ frame đã chạy inference. Nút này chỉ đánh dấu mask đã duyệt; qua **Đồng bộ CVAT** để xem preview và xác nhận gửi. Có thể dùng **Bỏ duyệt mọi ảnh** để bỏ chọn các mask chưa đồng bộ.

1. **Manual + AI**: giữ model theo loại segmentation của task hiện tại, sau đó người dùng kiểm tra và duyệt từng kết quả.
2. **YOLO Only**: dùng weights đã chọn để phát hiện box hoặc instance mask. Bấm **Đọc danh sách class từ weights YOLO** để xem độ phủ và ánh xạ class của model sang label CVAT; label không có trong taxonomy cần weights đã huấn luyện cho label đó.
3. **SAM3 Only**: dùng box, positive/negative points, text concept hoặc exemplar box để tạo mask trên ảnh; text prompt có thể chạy trên nhiều frame.
4. **YOLO + SAM3**: đưa box YOLO sang SAM 3 để tạo mask.
5. **SAM3 Video Track**: chọn frame bắt đầu, đặt box/points hoặc text/exemplar để SAM 3 theo dõi object qua các frame tiếp theo. Khi duyệt hết mask, đồng bộ được CVAT track giữ ID qua frame; nếu chỉ duyệt một phần, vẫn có thể gửi từng mask frame.

Confidence là điểm model, không phải độ chính xác được đảm bảo. Hãy kiểm tra kết quả và chỉnh mask trước khi đồng bộ. Cityscapes/COCO chỉ nhận diện taxonomy đã huấn luyện; trong **Manual + AI**, các label ngoài taxonomy tự được gửi qua SAM 3 text prompt. Tool có prompt riêng cho `area/alternative`, `area/direct`/`area/drivable`, crosswalk, curb và từng nhóm vạch trắng/vàng/other của BDD100K. Cần checkpoint SAM 3 được cấp quyền và CUDA. **YOLO Only** chạy thuần YOLO và chỉ gán class được weights hỗ trợ; app tự khớp tên, cho phép ánh xạ từng class YOLO sang label CVAT. Muốn nhận diện label mới cần weights YOLO đã huấn luyện với các label đó; ánh xạ chỉ đổi tên dự đoán. Muốn fallback zero-shot cho label ngoài taxonomy, chọn **Manual + AI** hoặc **YOLO + SAM3**. SAM 3 zero-shot có thể nhầm vùng drivable trực tiếp/thay thế hoặc kiểu vạch gần nhau, nên duyệt kỹ và dùng weights huấn luyện theo dữ liệu nếu cần độ chính xác ổn định. Chọn thêm thiết bị và weights YOLO. Có thể chọn một file `.pt` trong repo hoặc nhập đường dẫn riêng. Đầu ra YOLO được đối chiếu với tên labels của task CVAT. Model box-only có thể đồng bộ Bounding box; Mask chỉ được bật khi model trả mask thật.

Inference theo batch xử lý từng frame, hiện progress và ghi ảnh/mask xuống `outputs/predictions/task_<id>/job_<id>/<run-id>/`. Video tracking tạo clip tạm từ frame đã chọn tới cuối Job, lưu mask và track ID theo từng frame rồi xóa clip tạm. Khi hoàn tất, xem ảnh với zoom/pan, dùng nút Prev/Next, kiểm tra QC rồi duyệt hoặc bỏ từng kết quả. Có thể sửa mask bằng cách thêm/xóa vùng tròn tại vị trí nhấp trên ảnh; mọi chỉnh sửa sẽ lưu theo frame và cần duyệt lại. SAM 3 tinh chỉnh mask đang chọn bằng prompt mask cùng positive/negative points trước khi duyệt.

## Cấu hình SAM 3

SAM 3 không được giả lập. Cần checkpoint thật mà người dùng có quyền tải từ [Meta/Hugging Face](https://huggingface.co/facebook/sam3). Đặt file ở `models/sam3.pt` hoặc cấu hình `SAM3_WEIGHTS` trong `.env`.

Runtime SAM 3 yêu cầu Python 3.12+, PyTorch 2.7+ và CUDA 12.6+; `setup_gpu.ps1` tạo `.venv-sam3` và cài CUDA PyTorch thay vì ghi đè môi trường Python hiện tại. Tool hỗ trợ point/box prompt, text concept, exemplar box và video tracking qua API Ultralytics; mask prompt được dùng để tinh chỉnh mask ảnh đang chọn bằng positive/negative points. SAM 3 được tải theo nhu cầu, dùng FP16 ở 640 px, và chuyển giữa image/video predictor để hạn chế giữ model trùng trong VRAM. Video track có thể gửi lên CVAT dưới dạng annotation tracks với mask keyframe theo frame. Checkpoint `sam3.1_multiplex.pt` dùng cho ảnh; video tracking cần `sam3.pt`. Môi trường CPU-only tiếp tục chạy các model khác; SAM 3 cần CUDA và checkpoint hợp lệ.

## Preview và đồng bộ annotation

Luồng đồng bộ có preset **Tự nhận diện theo taxonomy**, **Đường phố: box + drivable area + lane marking**, và **Semantic segmentation: polygon**. Có thể mở **Tùy chỉnh kiểu shape cho từng label** để đổi riêng từng class; một frame có thể gửi rectangle, polygon, polyline và mask cùng lúc. Preset đường phố gán object instance thành bounding box, `area/*` thành polygon và `lane/*` thành polyline. Preset semantic segmentation chuyển 19 class thành polygon.

Polyline được tạo từ đường tâm của mask lane marking. Bounding box cần dự đoán instance riêng; mask semantic đã gộp nhiều object sẽ không được xuất thành một box chung. Kiểu shape vẫn phải tương thích với label type trong CVAT. Preview cho phép gắn `occluded` theo từng instance và `truncated` nếu label CVAT có checkbox attribute `truncated` có thể chỉnh sửa. Keypoints/Skeleton chỉ dùng khi Task có label skeleton với sublabels khớp keypoint; các điểm không khớp bị bỏ qua. Chuyển mask sang polygon có thể làm mất lỗ hoặc chi tiết nhỏ.

Trước khi gửi, kiểm tra preview và xác nhận lại sau mỗi lần đổi shape hoặc attribute. Ứng dụng dùng thao tác thêm shape qua PATCH, không thay toàn bộ annotation của Job. Nó đọc shape hiện có, lọc shape trùng và chỉ gửi phần mới. Kiểu shape không tương thích với label sẽ bị bỏ qua và báo rõ.

### Facial Landmark-50 trực tiếp trên CVAT Job

Nếu chưa có model auto-label, chạy một lần `powershell -ExecutionPolicy Bypass -File .\setup_facial_landmark.ps1` trong môi trường app.

Khi Task có đúng 7 label Facial Landmark-50, app chuyển khỏi luồng segmentation sang workspace **Facial Landmark-50 · CVAT Job**. Chọn frame, bấm **AI tự gán 50 điểm cho frame**, duyệt pre-label rồi xác nhận **Save to CVAT**. App dùng MediaPipe Face Landmarker, lưu nháp với trạng thái review pending, và chỉ ghi/cập nhật nhóm skeleton của frame đang chọn; shape/tag/track khác được giữ nguyên, sau đó app đọc lại để xác nhận.

Trước khi bật lưu, app kiểm tra label cha kiểu Skeleton, số point sublabel từng nhóm và ID được mã hóa trong tên point (ví dụ `point_14`). Nếu API có SVG, app xác minh edges; nếu thiếu hoặc SVG lỗi, app ghi log và hiện cảnh báo, rồi vẫn cho vẽ/lưu theo topology guideline. CVAT có thể không hiển thị đường nối nếu cấu hình SVG trên Task thực sự thiếu/sai. Bản nháp vẫn được lưu cục bộ.

Hiện tại workspace này hỗ trợ một khuôn mặt người lái trên mỗi frame. Model Facial Landmark chưa được tích hợp và chưa có mapping đã xác minh sang 50 Point ID; nút Save to CVAT chỉ ghi annotation người dùng đã vẽ.

## Đánh giá Ground Truth

Trong phần Ground-truth evaluation, xác nhận rằng annotation hiện có trong Job là Ground Truth đã được kiểm tra. Công cụ so sánh polygon, rectangle và mask theo class/frame, rồi báo Precision, Recall, mean matched IoU, Dice và mAP@50 cùng TP/FP/FN. Polyline, skeleton, tags và tracks không được tính như mask Ground Truth.

Các metric chỉ mô tả dữ liệu Ground Truth đã chọn. Confidence threshold không phải độ chính xác. Hãy đánh giá trên tập ảnh đại diện, đã kiểm tra nhãn và chưa dùng để điều chỉnh threshold; mục tiêu trên 70% cần được xác minh bằng các metric thực tế đó.

Để tìm ngưỡng hướng tới Precision 90%, chạy inference trên một Job Ground Truth đã kiểm tra với confidence 0%, rồi dùng nút **Đánh giá và tìm ngưỡng confidence**. Công cụ chọn ngưỡng giữ Recall cao nhất trong các ngưỡng đạt mục tiêu trên chính Job đó; sau đó đặt slider theo gợi ý, chạy lại inference và đánh giá lại. Kết quả trên Job calibrate không bảo đảm cho ảnh mới. Hãy xác nhận trên một tập holdout độc lập trước khi dùng hàng loạt.

## Kiểm tra

```powershell
python -m pytest tests -q
```

Các test unit kiểm tra mask codec CVAT, pagination, PAT header, dedupe khi append, lưu prediction theo frame và metric Ground Truth. Một số test dataset cũ sẽ skip nếu fixture `K5-L2-DAY03-NguyenThanhQuang-2A202603083-Image-Segmentation/data/manifest.json` không có trên máy.

Trạng thái ban đầu của Python mặc định: driver nhận NVIDIA GeForce RTX 3050 Ti Laptop GPU 4 GB, nhưng PyTorch là bản CPU-only. Môi trường `.venv-sam3` được cài PyTorch CUDA riêng. Chưa có checkpoint SAM 3 trong `models/`; đặt checkpoint được cấp quyền để chạy SAM 3.

## Giới hạn và troubleshooting

- Không có CVAT Local đang chạy thì `http://localhost:8080` không thể kiểm tra kết nối thật.
- CVAT Online cần quyền truy cập mạng và thông tin đăng nhập/PAT của người dùng.
- SAM 3 cần checkpoint được cấp quyền và GPU/CUDA tương thích; CPU YOLO không thay thế SAM 3.
- `outputs/predictions/` giữ các lần inference để có thể xem kết quả sau lỗi giữa chừng. Xóa các run cũ khi không cần nữa để giải phóng dung lượng.
- Các model pretrained trên Cityscapes/COCO có thể bỏ sót hoặc nhầm class trên dữ liệu khác; duyệt từng kết quả và đo với Ground Truth trước khi sử dụng hàng loạt.
