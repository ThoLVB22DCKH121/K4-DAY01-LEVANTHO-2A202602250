# Báo cáo bài thực hành Ngày 1 – Đọc nhãn từ đầu ra YOLO11

**Ngày chạy:**

**Runtime Colab:** CPU/GPU

**Python / PyTorch / Ultralytics:**

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Thay đổi so với notebook nguồn:** Không / mô tả rõ thay đổi

> ZIP do notebook tạo có tên `KX-DAY01-report.zip`. Giải nén rồi đặt trực tiếp `REPORT.md` và
> `day1_lab_outputs/` vào thư mục `report/` của repository tạo từ template. Không ghi họ tên, MSSV,
> email, số điện thoại hoặc dữ liệu cá nhân khác. Nộp link repository trên VLearn; tài khoản VLearn xác
> định người nộp.

## 1. Phân loại ảnh – prediction cấp ảnh

Nguồn evidence: `classification_predictions.json`, sample `traffic`.

- Record hạng 1 (`class_id`, `class_name`, `rank`, `score`, `taxonomy_name`):
    (`class_id`: 468, `class_name`: cab, `rank`: 1, `score`: 0.510915, `taxonomy_name`: ImageNet-1K)
- Record này mô tả toàn ảnh như thế nào?
    Đây là dự đoán duy nhất cho toàn bộ ảnh: model cho rằng nội dung ảnh gần với lớp **cab** nhất trong số 1000 lớp mà nó biết, với độ tự tin ~51%.
- Ai định nghĩa class list mà checkpoint có thể dự đoán?
    Class list do bộ dữ liệu dùng để huấn luyện model quyết định. Với `yolo11n-cls.pt`, đó là taxonomy **ImageNet-1K** (1000 lớp) — model chỉ có thể chọn nhãn trong danh sách này, không thể "tự nghĩ ra" lớp mới.
- Vì sao cần giữ cả ID, tên lớp và tên taxonomy?
    - `class_id`: giá trị số ổn định, dùng để so khớp chính xác giữa các hệ thống.
    - `class_name`: giúp con người đọc và hiểu nhãn dễ dàng.
    - `taxonomy_name`: cho biết ID/tên đó thuộc bộ nhãn nào, tránh nhầm lẫn khi hai taxonomy khác nhau dùng trùng ID hoặc trùng tên.
- Nếu ảnh có nhiều chủ thể, guideline cần quy định điều gì?
    Ảnh `traffic` là ví dụ điển hình: top-5 gồm nhiều lớp phương tiện gần nghĩa nhau (cab 51%, minibus 16%, police_van 8.6%, recreational_vehicle 5.4%, streetcar 4.8%), cho thấy ảnh có thể chứa nhiều loại xe cùng lúc nhưng classification chỉ cho phép một nhãn/ảnh. Guideline cần quy định rõ:
    - Quy tắc chọn nhãn đại diện khi có nhiều chủ thể (ví dụ: vật thể chiếm diện tích lớn nhất, ở tiêu điểm ảnh, hoặc theo mục đích gán nhãn của dự án).
    - Ngưỡng chênh lệch score tối thiểu giữa rank 1 và rank 2 để annotator được phép tự chốt nhãn mà không cần hỏi thêm.
    - Trường hợp ảnh thực sự đa chủ thể rõ rệt, nên chuyển sang tác vụ object detection hoặc instance segmentation thay vì ép về một nhãn duy nhất.
- Vì sao model score không phải ground truth?
    Score chỉ phản ánh mức độ tự tin nội tại của model dựa trên phân phối xác suất nó học được, chứ không phải một xác nhận đã qua kiểm chứng của con người. Ở đây score cao nhất chỉ đạt ~51%, nghĩa là chính model cũng không chắc chắn và hoàn toàn có thể sai. Ground truth chỉ được xác lập khi con người gán nhãn dựa trên guideline đã thống nhất .

## 2. Phát hiện vật thể – lớp và box cho từng object

Nguồn evidence: `detection_predictions.json` và `visuals/detection_predictions.png`, sample `kitchen`.

- Một record (`class_name`, `score`, `bbox_xyxy`, `bbox_width`, `bbox_height`):
    (`class_name`: person, `score`: 0.912625, `bbox_xyxy`: [385.33, 69.24, 498.92, 348.92], `bbox_width`: 113.58, `bbox_height`: 279.68)
- Diễn giải vị trí box bằng lời:
    Box là một hình chữ nhật đứng, có góc trên-trái tại (x=385.33, y=69.24) và góc dưới-phải tại (x=498.92, y=348.92). Box ôm sát người trong ảnh từ đầu xuống gần chân, nằm lệch về nửa phải khung hình.
- So sánh số prediction ở hai threshold:
    Threshold càng tăng thì số prediction giữ lại càng giảm (model chỉ trả về những box có độ tự tin cao); ngược lại threshold càng giảm thì số prediction càng tăng.
- Điều gì thay đổi đối với độ bao phủ và khối lượng reviewer cần xem?
    - Ngưỡng thấp: độ bao phủ tăng vì bắt được nhiều vật thể mờ/nhỏ hơn, nhưng false positive cũng tăng theo, khiến reviewer phải xem và loại bỏ nhiều box sai hơn.
    - Ngưỡng cao: số box ít hơn, độ chính xác trung bình cao hơn, nhưng dễ bỏ sót vật thể thật, buộc reviewer/annotator phải bù bằng cách bổ sung nhãn thủ công.
- Đề xuất một quy tắc box chặt:
    Box phải ôm sát đúng biên ngoài cùng của vật thể ở cả 4 cạnh — không để dư khoảng trống (box lỏng) và không cắt vào phần thân vật thể (box hụt).
- Với object bị che khuất/cắt mép, điều gì cần guideline hoặc escalation quyết định?
    - Có gán nhãn cho vật thể bị cắt bởi mép ảnh hay không, và nếu có, box nên vẽ tới đúng mép ảnh hay chỉ theo phần nhìn thấy được.
    - Khi hai vật thể cùng lớp chồng lấn nhiều, có tách thành 2 box riêng hay gộp lại — trường hợp này cần escalation lên reviewer/chuyên gia quyết định thay vì để annotator tự đoán.

## 3. Phân đoạn theo từng đối tượng – polygon cho mỗi instance

Nguồn evidence: `segmentation_predictions.json` và `visuals/segmentation_prediction.png`, sample `kitchen`.

- Một record (`instance_id`, `class_name`, `score`, số điểm và một phần `polygon_xy`):
    (`instance_id`: kitchen-001, `class_name`: person, `score`: 0.899318, polygon_point_count: 348, polygon_xy: [[446.0, 70.0], [445.0, 71.0], [444.0, 71.0], [443.0, 72.0], [442.0, 72.0], [441.0, 73.0], ...])
- Polygon bổ sung chi tiết gì so với box?
    Bounding box chỉ là một hình chữ nhật bao quanh vật thể, nên bên trong box vẫn có thể chứa nhiều vùng nền không thuộc vật thể. Polygon gồm hàng trăm điểm chạy dọc theo đường biên thực tế của vật thể, nên mô tả đúng hình dạng của nó chứ không chỉ khung bao ngoài.
- `instance_id` dùng để làm gì và không phải loại ID nào?
    Dùng để phân biệt từng đối tượng cụ thể trong một ảnh, kể cả khi nhiều đối tượng thuộc cùng một lớp (ví dụ ảnh `kitchen` có 2 person, 4 bowl, mỗi đối tượng vẫn có `instance_id` riêng dù chung `class_name`). Đây không phải `class_id` (không đại diện cho loại vật thể) và cũng không phải ID cố định xuyên suốt nhiều ảnh — nó chỉ có ý nghĩa trong phạm vi một ảnh/một lần dự đoán.
- Đề xuất một quy tắc biên mask:
    Đường biên mask phải bám sát ranh giới thực của vật thể ở độ phân giải pixel, không cắt góc và không lấn vào vùng nền hoặc vật thể liền kề.
- Với vùng mờ/tiếp xúc/che khuất, điều gì cần guideline hoặc escalation quyết định?
    Guideline cần quy định rõ ai có quyền quyết định đường phân chia khi các vật thể tiếp xúc hoặc chồng lấn nhau, và khi vật thể bị che khuất một phần thì vẽ mask theo phần suy đoán (amodal) hay chỉ phần nhìn thấy (visible-only). Đây là quyết định chính sách nên cần escalation để thống nhất trong toàn dự án, không để từng annotator tự xử lý khác nhau.

## 4. Vòng đời và kiểm tra chất lượng

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

| Tác vụ | Đơn vị/định dạng ground truth | Lỗi hoặc điểm mơ hồ quan sát được | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Một `class_id`/`class_name` cho toàn ảnh, theo taxonomy ImageNet-1K | Ảnh có nhiều chủ thể (traffic: cab/minibus/police_van đều hợp lý); score cao nhất chỉ ~51% cho thấy độ mơ hồ | Chọn nhãn đại diện theo guideline, không lấy trực tiếp rank 1 của model | Kiểm tra nhãn có đúng chủ thể chính không, có nhất quán với guideline chọn chủ thể khi đa vật thể không |
| Phát hiện vật thể | Một box `xyxy` (pixel) + `class_id`/`class_name` cho mỗi vật thể, theo taxonomy COCO-80 | Vật thể bị che khuất/cắt mép (person góc trái); box lỏng lẻo ở ngưỡng thấp; model bỏ sót vật thể thật | Vẽ box chặt sát biên, bổ sung vật thể model bỏ sót, xử lý occlusion theo guideline | Kiểm tra độ khít của box, box có bỏ sót vật thể không, có box thừa/sai lớp không |
| Instance segmentation | Một polygon/mask + `instance_id` + `class_id`/`class_name` cho mỗi đối tượng, theo taxonomy COCO-80 | Ranh giới mờ giữa các vật thể tiếp xúc (bowl/spoon chồng lấn); vật thể bị che khuất một phần (person bị bếp che) | Vẽ mask bám sát biên pixel, quyết định amodal hay visible-only theo guideline, gán `instance_id` riêng cho từng đối tượng | Kiểm tra biên mask có chính xác không, các instance cùng lớp có bị gộp nhầm không, xử lý occlusion có nhất quán không |

## 5. An toàn dữ liệu

- Một quy tắc bảo vệ dữ liệu:
    Chỉ sử dụng ảnh/dữ liệu nằm trong phạm vi được cấp phép của dự án (ví dụ ảnh COCO có license CC BY 2.0 kèm attribution rõ ràng như trong `IMAGE_ATTRIBUTION.md`); không thu thập, lưu trữ hay chia sẻ dữ liệu cá nhân, dữ liệu nhạy cảm, hoặc ảnh không rõ nguồn gốc/license hợp lệ.
- Nếu thấy ảnh hoặc dữ liệu không đúng phạm vi, tôi sẽ dừng và báo cho:
    Tôi sẽ dừng xử lý ngay và báo cho reviewer, giảng viên hoặc người phụ trách dữ liệu/guideline để được xác nhận trước khi tiếp tục.

## 6. Danh sách bằng chứng

- [X] `classification_predictions.json`
- [X] `detection_predictions.json`
- [X] `segmentation_predictions.json`
- [X] `IMAGE_ATTRIBUTION.md`
- [X] `visuals/classification_top5.png`
- [X] `visuals/detection_predictions.png`
- [X] `visuals/segmentation_prediction.png`
- [X] Ô validation cuối notebook báo `PASS`.
- [X] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output.
