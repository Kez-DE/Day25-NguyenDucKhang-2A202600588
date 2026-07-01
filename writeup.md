# Lab 25 - GPU FinOps: Bài viết ngắn

NimbusAI GPU Cost Optimization, baseline vs. optimized.

## 1. Baseline vs. Optimized

Chi phí GPU tổng hợp của NimbusAI giảm từ $27,133/tháng (baseline: mọi thứ chạy on-demand, không
cache, không cascade) xuống $14,554/tháng (optimized), tiết kiệm $12,579, tương đương 46%. Ở tầng
inference, đơn vị đo quan trọng hơn là `$/1M-token`, và con số này giảm mạnh hơn nhiều: từ $6.488
xuống $1.126, tức 82.6%. Sự chênh lệch giữa hai tỷ lệ (46% tổng so với 82.6% riêng inference)
không phải nghịch lý. Phần lớn tiền của NimbusAI nằm ở purchasing (8 workload training và
inference chạy hàng trăm giờ GPU mỗi tháng), còn inference chỉ là 2,400 request/ngày. Tối ưu
được 82.6% ở một phần nhỏ của hoá đơn vẫn cho ra một con số tổng khiêm tốn hơn.

## 2. Phân tích từng đòn bẩy

| Đòn bẩy | Tiết kiệm/tháng | Đóng góp |
|---|---|---|
| Purchasing (spot/reserved) | $10,112 | 80% |
| Inference (cascade/cache/batch) | $1,212 | 10% |
| Right-size util-lies | $655 | 5% |
| Kill idle GPUs | $600 | 5% |

Purchasing đóng góp nhiều nhất, đơn giản vì đó là nơi có nhiều tiền nhất để cắt. Đây cũng là bài
học chính của lab: đòn bẩy có % giảm ấn tượng nhất (cascade/cache/batch, 82.6%) không nhất thiết
là đòn bẩy đáng làm trước. Nên bắt đầu từ nơi có số tiền tuyệt đối lớn nhất, purchasing, rồi mới
đến các phần còn lại.

## 3. GPU-Util Lie

Hai GPU bị gắn cờ "lie": gpu-h100-4 (H100, util 98%, MFU chỉ 19%, MBU 21%) và gpu-a10g-1 (A10G,
util 97%, MFU 27%, MBU 30%). Điểm đáng chú ý là cả MFU lẫn MBU của hai GPU này đều thấp cùng lúc.
Nếu chỉ MFU thấp còn MBU cao, đó sẽ là dấu hiệu của workload memory-bound (như decode), hợp lý và
không đáng ngại. Nhưng ở đây cả hai chỉ số cùng thấp, nên cách đọc hợp lý hơn là: GPU được lập
lịch bận (SM được cấp phát, nvidia-smi đếm là "active") nhưng phần lớn thời gian đó tiêu tốn vào
overhead đồng bộ hoá, batch size quá nhỏ hoặc chờ collective-comm giữa các GPU trong cùng job
training, chứ không thực sự làm FLOPs hay di chuyển dữ liệu qua HBM. `nvidia-smi` không phân biệt
được "đang tính toán" với "đang chờ", nên 98% util là một con số gây hiểu lầm nếu không nhìn thêm
MFU và MBU.

Về tài chính, hai GPU này đang bị tính tiền như H100/A10G full giá trong khi chỉ khai thác khoảng
20-30% FLOPs thực tế. Right-size chúng xuống một tier thấp hơn (H100 xuống A100, A10G xuống L4)
tiết kiệm $655/tháng, không phải khoản lớn nhất trong bảng, nhưng là chi phí đang bị giấu sau một
con số trông có vẻ hiệu quả.

## 4. Phần mở rộng đã làm

### Extension 1: cải thiện recommend_tier()

Thêm interruption rate theo GPU type (H100/B200 khoảng 2-3%, A10G/L4 khoảng 12-15%, vì các dòng
flagship thường được provider giữ dự phòng nhiều hơn), và so sánh 1yr với 3yr reserved dựa trên
`kind` của job (training không nên khoá 3yr vì là dự án có thời hạn) cùng tỷ lệ ngày hoạt động
trong tháng billing (job chạy gần như liên tục cả tháng mới đáng khoá 3yr). Kết quả: savings của
M3 tăng từ 39.1% (policy cũ) lên 39.4% (policy mới). Con số thay đổi không lớn vì phân loại
spot/reserved/on-demand cho 8 job hiện có vẫn giữ nguyên, nhưng phần chi phí spot giờ được tính
đúng theo tỷ lệ gián đoạn thực tế của từng GPU, và term reserved được chọn đúng theo mức độ liên
tục của job thay vì mặc định luôn là 3yr. Bảng ma trận GPU-type × duty-cycle × interruptible
(`tier_recommendation_matrix()`) cho thấy chính sách phản ứng nhất quán trên toàn bộ không gian
quyết định, không chỉ đúng cho 8 job có sẵn trong dữ liệu.

### Extension 3: cache_is_worth_it()

Với write premium 1.25x và read discount 0.10 (theo kiểu Anthropic prompt caching), break-even
chỉ là 1.39 lần đọc lại. Dữ liệu thực tế từ `token_usage.csv` cho thấy tier `small` có trung bình
237.8 lần đọc mỗi project (gấp 171 lần break-even) và tier `large` có 62.2 lần đọc mỗi project (gấp
44.8 lần break-even). Cả hai đều vượt xa ngưỡng, nghĩa là cache trong dataset này đang thực sự
sinh lời chứ không phải một chi phí ẩn. Nếu một đội có traffic thấp hơn nhiều, dưới khoảng 1.4 lần
đọc mỗi prefix, hàm này sẽ tự động tắt discount cache cho tier đó thay vì áp dụng mù quáng.

## 5. Khuyến nghị cho NimbusAI

Nếu tôi là FinOps lead, ba việc đầu tiên sẽ làm:

1. Sửa purchasing trước tiên. Đây là 80% của khoản tiết kiệm khả dụng ($10,112 trên tổng $12,579).
   Áp dụng chính sách tier mới (GPU-aware và duration-aware) cho toàn bộ workload, không chỉ 8 job
   mẫu trong lab này.
2. Điều tra hai GPU-Util lie trước khi mua thêm phần cứng. Nếu nguyên nhân thực sự là batch size
   nhỏ hoặc đồng bộ hoá kém giữa các node trong cùng job training, sửa phần mềm sẽ rẻ hơn nhiều so
   với mua thêm GPU để bù cho hiệu năng thấp.
3. Chuyển các workload có thể di chuyển sang europe-north1. Vùng này rẻ hơn 25% và phát thải ít
   hơn 92% so với us-east-1 cho cùng một query, nên đây là một trong số ít lever mà chi phí và
   carbon đi cùng chiều thay vì đánh đổi lẫn nhau.

---

Viết dựa trên số liệu thực tế từ lần chạy `python missions/run_all.py` gần nhất (seed=25, tháng
6/2026 as-of). Trước khi nộp, đã chạy lại `verify.py` (11/11) và `pytest` (23/23, gồm 8 test tự
viết cho hai phần mở rộng) để xác nhận số liệu khớp với báo cáo.
