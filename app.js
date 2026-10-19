// ============================================================
// CONFIGURATION & MODEL METADATA (YOLOv8)
// ============================================================
const DEFAULT_INPUT_SIZE = 640;
let CONF_THRESHOLD = 0.45; // Ngưỡng tự tin phát hiện
let IOU_THRESHOLD = 0.45;  // Ngưỡng lọc trùng lặp NMS
let INFERENCE_INTERVAL = 150; // Khoảng cách giữa các frame (ms) để realtime mượt và nhẹ máy

// Cấu hình các model hỗ trợ
const MODEL_CONFIGS = {
    best_12_flowers: {
        id: "best_12_flowers",
        name: "best.onnx",
        badge: "best.onnx (12 classes)",
        headerBadge: "YOLOv8 • 12 LOÀI HOA (MỚI)",
        path: "./best.onnx",
        inputSize: 320,
        classes: [
            { id: 0, raw: "chieu_tim", vn: "Hoa chiều tím", en: "Ruellia Simplex", color: [139, 92, 246] },
            { id: 1, raw: "chuoi_duoc_do_rung", vn: "Hoa chuối đuốc đỏ rừng", en: "Heliconia Rostrata", color: [225, 29, 72] },
            { id: 2, raw: "chuoi_mo_ket", vn: "Hoa chuối mỏ két", en: "Heliconia Psittacorum", color: [249, 115, 22] },
            { id: 3, raw: "dam_but_cam", vn: "Hoa dâm bụt cam", en: "Orange Hibiscus", color: [251, 146, 60] },
            { id: 4, raw: "dam_but_vang", vn: "Hoa dâm bụt vàng", en: "Yellow Hibiscus", color: [250, 204, 21] },
            { id: 5, raw: "hong", vn: "Hoa hồng", en: "Rose", color: [244, 63, 94] },
            { id: 6, raw: "huynh_anh", vn: "Hoa huỳnh anh", en: "Allamanda Cathartica", color: [234, 179, 8] },
            { id: 7, raw: "mau_don_do", vn: "Hoa mẫu đơn đỏ (trang đỏ)", en: "Red Ixora", color: [220, 38, 38] },
            { id: 8, raw: "mau_don_trang", vn: "Hoa mẫu đơn trắng (trang trắng)", en: "White Ixora", color: [226, 232, 240] },
            { id: 9, raw: "mau_don_vang", vn: "Hoa mẫu đơn vàng (trang vàng)", en: "Yellow Ixora", color: [217, 119, 6] },
            { id: 10, raw: "mia_do", vn: "Hoa mía đỏ (gừng đỏ)", en: "Red Button Ginger", color: [190, 18, 60] },
            { id: 11, raw: "ngoc_but", vn: "Hoa ngọc bút (lài tây)", en: "Crape Jasmine", color: [20, 184, 166] }
        ]
    },
    best_12_flowers_int8: {
        id: "best_12_flowers_int8",
        name: "best_12_flowers_int8.onnx",
        badge: "best_12_flowers_int8.onnx (12 classes INT8 - 20MB)",
        headerBadge: "YOLOv8 • 12 LOÀI HOA (INT8 NÉN)",
        path: "./best_12_flowers_int8.onnx",
        classes: [
            { id: 0, raw: "chieu_tim", vn: "Hoa chiều tím", en: "Ruellia Simplex", color: [139, 92, 246] },
            { id: 1, raw: "chuoi_duoc_do_rung", vn: "Hoa chuối đuốc đỏ rừng", en: "Heliconia Rostrata", color: [225, 29, 72] },
            { id: 2, raw: "chuoi_mo_ket", vn: "Hoa chuối mỏ két", en: "Heliconia Psittacorum", color: [249, 115, 22] },
            { id: 3, raw: "dam_but_cam", vn: "Hoa dâm bụt cam", en: "Orange Hibiscus", color: [251, 146, 60] },
            { id: 4, raw: "dam_but_vang", vn: "Hoa dâm bụt vàng", en: "Yellow Hibiscus", color: [250, 204, 21] },
            { id: 5, raw: "hong", vn: "Hoa hồng", en: "Rose", color: [244, 63, 94] },
            { id: 6, raw: "huynh_anh", vn: "Hoa huỳnh anh", en: "Allamanda Cathartica", color: [234, 179, 8] },
            { id: 7, raw: "mau_don_do", vn: "Hoa mẫu đơn đỏ (trang đỏ)", en: "Red Ixora", color: [220, 38, 38] },
            { id: 8, raw: "mau_don_trang", vn: "Hoa mẫu đơn trắng (trang trắng)", en: "White Ixora", color: [226, 232, 240] },
            { id: 9, raw: "mau_don_vang", vn: "Hoa mẫu đơn vàng (trang vàng)", en: "Yellow Ixora", color: [217, 119, 6] },
            { id: 10, raw: "mia_do", vn: "Hoa mía đỏ (gừng đỏ)", en: "Red Button Ginger", color: [190, 18, 60] },
            { id: 11, raw: "ngoc_but", vn: "Hoa ngọc bút (lài tây)", en: "Crape Jasmine", color: [20, 184, 166] }
        ]
    },
    flower_10: {
        id: "flower_10",
        name: "flower.onnx",
        badge: "flower.onnx (10 classes)",
        headerBadge: "YOLOv8 • 10 LOÀI HOA",
        path: "./flower.onnx",
        classes: [
            { id: 0, raw: "Allamanda Cathartica", vn: "Hoa hoàng anh (huỳnh anh vàng)", en: "Allamanda Cathartica", color: [245, 158, 11] },
            { id: 1, raw: "Heliconia Psittacorum", vn: "Hoa chuối mỏ két (thiên điểu)", en: "Heliconia Psittacorum", color: [239, 68, 68] },
            { id: 2, raw: "Hibiscus", vn: "Hoa dâm bụt", en: "Hibiscus", color: [236, 72, 153] },
            { id: 3, raw: "Ixora Coccinea", vn: "Hoa trang đỏ (bông trang)", en: "Ixora Coccinea", color: [220, 38, 38] },
            { id: 4, raw: "Red Button Ginger", vn: "Hoa phụng hoàng thảo (gừng đỏ)", en: "Red Button Ginger", color: [168, 85, 247] },
            { id: 5, raw: "Ruellia Simplex", vn: "Hoa chiều tím", en: "Ruellia Simplex", color: [99, 102, 241] },
            { id: 6, raw: "Tabernaemontana", vn: "Hoa lài tây (ngọc bút)", en: "Tabernaemontana", color: [20, 184, 166] },
            { id: 7, raw: "Tecoma Stans", vn: "Hoa chuông vàng (huỳnh liên)", en: "Tecoma Stans", color: [234, 179, 8] },
            { id: 8, raw: "White Rose", vn: "Hoa hồng trắng", en: "White Rose", color: [226, 232, 240] },
            { id: 9, raw: "Yellow Ixora", vn: "Hoa trang vàng", en: "Yellow Ixora", color: [249, 115, 22] }
        ]
    }
};

// Model mặc định: best_12_flowers
let currentModelKey = "best_12_flowers";
let activeModelConfig = MODEL_CONFIGS[currentModelKey];
let INPUT_SIZE = activeModelConfig.inputSize || DEFAULT_INPUT_SIZE;

let CLASS_NAMES = activeModelConfig.classes.map(c => c.en);
let CLASS_NAMES_VN = activeModelConfig.classes.map(c => c.vn);
let COLORS = activeModelConfig.classes.map(c => c.color);

// ============================================================
// DOM ELEMENTS
// ============================================================
const video = document.getElementById("video");
const overlay = document.getElementById("overlay");
const overlayCtx = overlay.getContext("2d");
const processingCanvas = document.getElementById("processingCanvas");
const processingCtx = processingCanvas.getContext("2d", { willReadFrequently: true });
const statusElement = document.getElementById("status");
const fpsElement = document.getElementById("fps");
const objectsElement = document.getElementById("objects");
const latencyElement = document.getElementById("latency");
const detectionsElement = document.getElementById("detections");
const loadButton = document.getElementById("loadButton");
const cameraButton = document.getElementById("cameraButton");
const switchCamButton = document.getElementById("switchCamButton");
const stopButton = document.getElementById("stopButton");
const imageUploadInput = document.getElementById("imageUpload");
const uploadButton = document.getElementById("uploadButton");
const confSlider = document.getElementById("confSlider");
const confValue = document.getElementById("confValue");
const modelSelect = document.getElementById("modelSelect");
const localModelFile = document.getElementById("localModelFile");
const localModelBtn = document.getElementById("localModelBtn");
const modelTag = document.getElementById("modelTag");
const headerBadge = document.getElementById("headerBadge");

// Canvas phụ nếu có segmentation
const maskCanvas = document.createElement("canvas");
const maskCtx = maskCanvas.getContext("2d", { willReadFrequently: true });

// ============================================================
// STATE
// ============================================================
let session = null;
let stream = null;
let running = false;
let starting = false;
let loadingModel = false;
let isInferencing = false;
let loopVersion = 0;
let lastInferenceTime = 0;
let lastTime = performance.now();
let frameCounter = 0;
let loggedOutputShapes = false;
let currentFacingMode = "environment"; // Mặc định camera sau trên điện thoại

// Bộ nhớ đệm tái sử dụng
let inputFloat32Array = null;
let maskImageData = null;
let maskLogits = null;

// ============================================================
// UTILITY FUNCTIONS
// ============================================================
function setStatus(text, type = "normal") {
    if (!statusElement) return;
    statusElement.textContent = text;
    statusElement.className = `status-text ${type}`;
}

window.addEventListener("resize", () => {
    if (video.videoWidth && video.videoHeight) {
        overlay.width = video.videoWidth;
        overlay.height = video.videoHeight;
    }
});

document.addEventListener("visibilitychange", () => {
    if (document.hidden && running) {
        stopCamera();
        setStatus("Tạm dừng - Tab đang ẩn", "warning");
    }
});

// Slider chỉnh ngưỡng tự tin realtime
if (confSlider && confValue) {
    confSlider.addEventListener("input", (e) => {
        CONF_THRESHOLD = parseFloat(e.target.value);
        confValue.textContent = `${Math.round(CONF_THRESHOLD * 100)}%`;
    });
}

// Cập nhật danh mục loài hoa trong accordion
function renderSpeciesGuide() {
    const guideGrid = document.querySelector(".species-grid");
    const summarySpan = document.querySelector(".species-guide summary span:first-child");
    if (!guideGrid) return;

    if (summarySpan) {
        summarySpan.textContent = `Danh mục ${activeModelConfig.classes.length} loài hoa được hỗ trợ nhận diện (${activeModelConfig.name})`;
    }

    guideGrid.replaceChildren();
    for (const c of activeModelConfig.classes) {
        const pill = document.createElement("div");
        pill.className = "species-pill";

        const badge = document.createElement("span");
        badge.className = "color-badge";
        badge.style.backgroundColor = `rgb(${c.color[0]}, ${c.color[1]}, ${c.color[2]})`;

        const text = document.createElement("span");
        text.innerHTML = `<strong>${c.id}: ${c.vn}</strong> <small style="color:var(--text-secondary)">(${c.en})</small>`;

        pill.appendChild(badge);
        pill.appendChild(text);
        guideGrid.appendChild(pill);
    }
}

// Chuyển đổi cấu hình model khi người dùng chọn model khác
function switchModelConfig(modelKey) {
    if (!MODEL_CONFIGS[modelKey] || loadingModel) return;
    currentModelKey = modelKey;
    activeModelConfig = MODEL_CONFIGS[modelKey];
    INPUT_SIZE = activeModelConfig.inputSize || DEFAULT_INPUT_SIZE;

    CLASS_NAMES = activeModelConfig.classes.map(c => c.en);
    CLASS_NAMES_VN = activeModelConfig.classes.map(c => c.vn);
    COLORS = activeModelConfig.classes.map(c => c.color);

    if (modelTag) {
        modelTag.textContent = activeModelConfig.badge;
    }
    if (headerBadge) {
        headerBadge.textContent = activeModelConfig.headerBadge;
    }

    renderSpeciesGuide();

    if (running) {
        stopCamera();
    }
    if (session) {
        try { session.release?.(); } catch (e) { console.warn(e); }
        session = null;
    }

    loadButton.classList.remove("btn-loaded");
    loadButton.textContent = "TẢI MODEL";
    loadButton.disabled = false;
    cameraButton.disabled = true;
    if (uploadButton) uploadButton.disabled = true;

    setStatus(`Đã chuyển sang ${activeModelConfig.name}. Nhấn 'TẢI MODEL' để bắt đầu.`, "normal");
}

if (modelSelect) {
    modelSelect.addEventListener("change", (e) => {
        switchModelConfig(e.target.value);
    });
}

// ============================================================
// LOAD ONNX MODEL (Hỗ trợ URL hoặc ArrayBuffer từ file máy tính)
// ============================================================
async function loadModel(customBuffer = null, customName = null) {
    if (loadingModel) return;
    loadingModel = true;
    if (running) stopCamera();
    loadButton.disabled = true;
    loadButton.classList.remove("btn-loaded");
    if (modelSelect) modelSelect.disabled = true;
    cameraButton.disabled = true;
    if (uploadButton) uploadButton.disabled = true;
    if (modelTag) modelTag.textContent = activeModelConfig.badge;
    try {
        const sourceDesc = customName || activeModelConfig.name;
        setStatus(`Đang khởi tạo model ${sourceDesc}...`, "loading");
        if (typeof ort === "undefined") {
            throw new Error("Không tìm thấy ONNX Runtime Web. Vui lòng kiểm tra kết nối mạng.");
        }

        if (session) {
            try { await session.release?.(); } catch (e) {}
            session = null;
        }

        const sourceData = customBuffer || activeModelConfig.path;
        let loadedProvider = "WASM CPU";

        // 1. Thử WebGPU trước (nếu trình duyệt hỗ trợ sẽ giảm độ trễ rất lớn)
        if (navigator.gpu) {
            try {
                session = await ort.InferenceSession.create(sourceData, {
                    executionProviders: ["webgpu"]
                });
                loadedProvider = "WebGPU (GPU)";
                console.log("[FlowerAI] Model loaded with WebGPU execution provider");
            } catch (gpuErr) {
                console.warn("[FlowerAI] WebGPU failed, fallback to WASM:", gpuErr);
                session = null;
            }
        }

        // 2. Fallback sang WebAssembly đa luồng nếu WebGPU không khả dụng
        if (!session) {
            const maxThreads = Math.min(4, navigator.hardwareConcurrency || 2);
            ort.env.wasm.numThreads = maxThreads;
            ort.env.wasm.proxy = false;

            session = await ort.InferenceSession.create(sourceData, {
                executionProviders: ["wasm"]
            });
            loadedProvider = `WASM (${ort.env.wasm.numThreads}T)`;
            console.log(`[FlowerAI] Model loaded with WASM (${ort.env.wasm.numThreads} threads)`);
        }

        processingCanvas.width = INPUT_SIZE;
        processingCanvas.height = INPUT_SIZE;
        loggedOutputShapes = false;

        if (customName && modelTag) {
            modelTag.textContent = `${customName} (${activeModelConfig.classes.length} classes)`;
        }
        setStatus(`Model sẵn sàng: ${sourceDesc} (${activeModelConfig.classes.length} loài hoa, ${INPUT_SIZE}x${INPUT_SIZE}, ${loadedProvider})`, "success");
        loadButton.classList.add("btn-loaded");
        loadButton.textContent = "✓ MODEL ĐÃ SẴN SÀNG";
        cameraButton.disabled = false;
        if (uploadButton) uploadButton.disabled = false;
        return true;
    } catch (error) {
        console.error("Load model error:", error);
        if (session) {
            try { await session.release?.(); } catch (e) { console.warn(e); }
            session = null;
        }
        loadButton.disabled = false;
        loadButton.classList.remove("btn-loaded");
        loadButton.textContent = "TẢI LẠI MODEL";
        
        const isFileProtocol = window.location.protocol === "file:";
        let errorHint = error.message;
        if (isFileProtocol && (error.message.includes("fetch") || error.message.includes("Failed"))) {
            errorHint = "Trình duyệt chặn tải model trực tiếp qua giao thức file:// (chính sách CORS).\n\n" +
                        "-> Cách 1: Bấm nút 'CHỌN FILE ONNX TỪ MÁY' để nạp file trực tiếp không cần web server.\n" +
                        "-> Cách 2: Chạy file 'start_server.bat' (hoặc python -m http.server 8000) và mở http://localhost:8000";
        } else if (/invalid model|DynamicQuantizeLinear|tensor\(float16\)/i.test(error.message)) {
            errorHint = `Model ${sourceDesc} không tương thích ONNX Runtime Web: node DynamicQuantizeLinear đang nhận tensor float16. Hãy xuất lại model với đầu vào FP32 hoặc sửa/xuất lại model quantization để dùng toán tử hỗ trợ float16.`;
        }
        setStatus("Lỗi tải model: " + errorHint, "error");
        alert("Không tải được model:\n\n" + errorHint);
        return false;
    } finally {
        loadingModel = false;
        if (modelSelect) modelSelect.disabled = false;
    }
}

// Nạp file ONNX trực tiếp từ ổ đĩa (Hữu ích khi mở bằng file://)
if (localModelBtn && localModelFile) {
    localModelBtn.addEventListener("click", () => localModelFile.click());
    localModelFile.addEventListener("change", async (e) => {
        const file = e.target.files && e.target.files[0];
        if (!file) return;
        try {
            setStatus(`Đang đọc file ${file.name} từ máy tính...`, "loading");
            const arrayBuffer = await file.arrayBuffer();
            const loaded = await loadModel(arrayBuffer, file.name);
            if (loaded) {
                setStatus(`Đã tải thành công model từ file: ${file.name}`, "success");
            }
        } catch (err) {
            console.error("Error reading local model:", err);
            setStatus("Lỗi nạp file: " + err.message, "error");
            alert("Lỗi nạp file ONNX từ máy:\n" + err.message);
        } finally {
            localModelFile.value = "";
        }
    });
}

// ============================================================
// CAMERA MANAGEMENT
// ============================================================
async function startCamera() {
    if (running || starting) return;
    if (!session) {
        alert("Vui lòng nhấn 'TẢI MODEL' trước khi mở Camera.");
        return;
    }

    starting = true;
    const token = ++loopVersion;
    let newStream = null;

    try {
        setStatus("Đang yêu cầu quyền truy cập Camera...", "loading");
        newStream = await navigator.mediaDevices.getUserMedia({
            video: {
                facingMode: { ideal: currentFacingMode },
                width: { ideal: 640 },
                height: { ideal: 480 }
            },
            audio: false
        });

        if (token !== loopVersion) {
            newStream.getTracks().forEach(track => track.stop());
            return;
        }

        stream = newStream;
        video.srcObject = stream;
        video.muted = true;
        video.playsInline = true;
        await video.play();

        if (token !== loopVersion) return;

        overlay.width = video.videoWidth || 640;
        overlay.height = video.videoHeight || 480;

        lastTime = performance.now();
        lastInferenceTime = 0;
        frameCounter = 0;
        running = true;

        if (fpsElement) fpsElement.textContent = "0.0";
        if (switchCamButton) switchCamButton.disabled = false;
        cameraButton.classList.add("active");
        setStatus("Camera đang chạy realtime", "success");

        requestAnimationFrame(time => inferenceLoop(time, token));
    } catch (error) {
        console.error("Camera error:", error);
        if (newStream) {
            newStream.getTracks().forEach(track => track.stop());
        }
        if (token === loopVersion) {
            stream = null;
            video.srcObject = null;
            setStatus("Không mở được camera: " + error.message, "error");
            alert("Không thể truy cập camera. Hãy chắc chắn đã cấp quyền camera trên trình duyệt.\n\n" + error.message);
        }
    } finally {
        starting = false;
    }
}

function stopCamera() {
    running = false;
    loopVersion++;
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        stream = null;
    }
    video.srcObject = null;
    overlayCtx.clearRect(0, 0, overlay.width, overlay.height);

    if (fpsElement) fpsElement.textContent = "0.0";
    if (objectsElement) objectsElement.textContent = "0";
    if (latencyElement) latencyElement.textContent = "0ms";
    if (switchCamButton) switchCamButton.disabled = true;
    cameraButton.classList.remove("active");

    detectionsElement.replaceChildren();
    const emptyMsg = document.createElement("div");
    emptyMsg.className = "detection-empty";
    emptyMsg.textContent = "Chưa phát hiện hoa nào";
    detectionsElement.appendChild(emptyMsg);

    setStatus(session ? "Model đã sẵn sàng - Camera đã dừng" : "Model chưa tải", "normal");
}

async function switchCamera() {
    if (!running) return;
    currentFacingMode = currentFacingMode === "environment" ? "user" : "environment";
    stopCamera();
    await startCamera();
}

// ============================================================
// IMAGE UPLOAD TEST (TIỆN LỢI TEST TRỰC TIẾP TỪ ẢNH)
// ============================================================
function handleImageUpload(event) {
    const file = event.target.files && event.target.files[0];
    if (!file) return;

    if (!session) {
        alert("Vui lòng nhấn 'TẢI MODEL' trước khi nhận diện ảnh.");
        return;
    }

    const reader = new FileReader();
    reader.onload = function(e) {
        const img = new Image();
        img.onload = async function() {
            stopCamera();

            // Cấu hình video overlay hiển thị kích thước ảnh
            overlay.width = img.width;
            overlay.height = img.height;

            setStatus("Đang nhận diện hoa từ ảnh...", "loading");
            const t0 = performance.now();

            const prep = preprocessSource(img, img.width, img.height);
            let outputs;
            try {
                outputs = await session.run({
                    [session.inputNames[0]]: prep.tensor
                });
            } finally {
                prep.tensor.dispose?.();
            }

            const tensors = Object.values(outputs);
            const predictions = tensors.find(t => t.dims.length === 3);
            const prototypes = tensors.find(t => t.dims.length === 4);

            const result = {
                outputs,
                predictions,
                prototypes,
                scale: prep.scale,
                offsetX: prep.offsetX,
                offsetY: prep.offsetY,
                newWidth: prep.newWidth,
                newHeight: prep.newHeight,
                sourceWidth: img.width,
                sourceHeight: img.height
            };

            const detections = postprocess(result);

            // Vẽ ảnh gốc lên canvas và đè bounding box
            const tempCanvas = document.createElement("canvas");
            tempCanvas.width = img.width;
            tempCanvas.height = img.height;
            const tempCtx = tempCanvas.getContext("2d");
            tempCtx.drawImage(img, 0, 0);

            // Đặt làm nền hiển thị
            video.poster = tempCanvas.toDataURL();
            drawBoundingBoxes(detections, img.width, img.height);
            drawLabels(detections);

            const elapsed = Math.round(performance.now() - t0);
            if (latencyElement) latencyElement.textContent = `${elapsed}ms`;
            setStatus(`Đã nhận diện: ${detections.length} hoa (${elapsed}ms)`, "success");

            disposeOutputs(outputs);
        };
        img.src = e.target.result;
    };
    reader.readAsDataURL(file);
}

// ============================================================
// PREPROCESSING: LETTERBOX + CHW FLOAT32
// ============================================================
function preprocessSource(source, srcWidth, srcHeight) {
    const scale = Math.min(
        INPUT_SIZE / srcWidth,
        INPUT_SIZE / srcHeight
    );
    const newWidth = srcWidth * scale;
    const newHeight = srcHeight * scale;
    const offsetX = (INPUT_SIZE - newWidth) / 2;
    const offsetY = (INPUT_SIZE - newHeight) / 2;

    processingCtx.fillStyle = "#727272"; // Padding màu xám chuẩn YOLO (114/255)
    processingCtx.fillRect(0, 0, INPUT_SIZE, INPUT_SIZE);
    processingCtx.drawImage(
        source,
        0, 0, srcWidth, srcHeight,
        offsetX, offsetY, newWidth, newHeight
    );

    const pixels = processingCtx.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE).data;
    const planeSize = INPUT_SIZE * INPUT_SIZE;

    if (!inputFloat32Array || inputFloat32Array.length !== 3 * planeSize) {
        inputFloat32Array = new Float32Array(3 * planeSize);
    }
    const input = inputFloat32Array;

    // Chuẩn hóa 0-1, định dạng Planar RGB (NCHW)
    for (let i = 0; i < planeSize; i++) {
        input[i] = pixels[i * 4] / 255.0;
        input[planeSize + i] = pixels[i * 4 + 1] / 255.0;
        input[2 * planeSize + i] = pixels[i * 4 + 2] / 255.0;
    }

    return {
        tensor: new ort.Tensor("float32", input, [1, 3, INPUT_SIZE, INPUT_SIZE]),
        scale,
        offsetX,
        offsetY,
        newWidth,
        newHeight
    };
}

function preprocess() {
    return preprocessSource(video, video.videoWidth, video.videoHeight);
}

// ============================================================
// INFERENCE
// ============================================================
function disposeOutputs(outputs) {
    if (outputs) {
        for (const tensor of Object.values(outputs)) {
            tensor.dispose?.();
        }
    }
}

async function runInference() {
    const prep = preprocess();
    let outputs;
    try {
        outputs = await session.run({
            [session.inputNames[0]]: prep.tensor
        });
    } finally {
        prep.tensor.dispose?.();
    }

    if (!loggedOutputShapes) {
        console.log(
            "ONNX Outputs Detected:",
            Object.fromEntries(
                Object.entries(outputs).map(([name, tensor]) => [name, tensor.dims])
            )
        );
        loggedOutputShapes = true;
    }

    const tensors = Object.values(outputs);
    const predictions = tensors.find(t => t.dims.length === 3);
    const prototypes = tensors.find(t => t.dims.length === 4);

    if (!predictions) {
        disposeOutputs(outputs);
        throw new Error(`Không tìm thấy output tensor dự đoán từ model ${activeModelConfig.name}`);
    }

    return {
        outputs,
        predictions,
        prototypes,
        scale: prep.scale,
        offsetX: prep.offsetX,
        offsetY: prep.offsetY,
        newWidth: prep.newWidth,
        newHeight: prep.newHeight,
        sourceWidth: video.videoWidth,
        sourceHeight: video.videoHeight
    };
}

// ============================================================
// POSTPROCESSING: DECODE BOXES & CLASS-AWARE NMS
// ============================================================
function postprocess(result) {
    const data = result.predictions.data;
    const dims = result.predictions.dims;

    // Hỗ trợ cả 2 định dạng: [1, 16, 8400] (Channels-First) hoặc [1, 8400, 16]
    const isChannelsFirst = dims[1] < dims[2];
    const numChannels = isChannelsFirst ? dims[1] : dims[2];
    const numBoxes = isChannelsFirst ? dims[2] : dims[1];
    const numClasses = Math.min(CLASS_NAMES.length, numChannels - 4);

    const hasPrototypes = Boolean(result.prototypes);
    const maskChannels = hasPrototypes ? result.prototypes.dims[1] : 0;

    const srcW = result.sourceWidth || video.videoWidth || INPUT_SIZE;
    const srcH = result.sourceHeight || video.videoHeight || INPUT_SIZE;

    const clampX = x => Math.max(0, Math.min(srcW, x));
    const clampY = y => Math.max(0, Math.min(srcH, y));

    const candidates = [];

    for (let i = 0; i < numBoxes; i++) {
        let bestClass = -1;
        let bestScore = 0;

        // Tìm class có score cao nhất trong số các loài hoa
        for (let c = 0; c < numClasses; c++) {
            const scoreIndex = isChannelsFirst ? (4 + c) * numBoxes + i : i * numChannels + 4 + c;
            const score = data[scoreIndex];
            if (score > bestScore) {
                bestScore = score;
                bestClass = c;
            }
        }

        if (bestScore < CONF_THRESHOLD) continue;

        const cx = isChannelsFirst ? data[0 * numBoxes + i] : data[i * numChannels + 0];
        const cy = isChannelsFirst ? data[1 * numBoxes + i] : data[i * numChannels + 1];
        const width = isChannelsFirst ? data[2 * numBoxes + i] : data[i * numChannels + 2];
        const height = isChannelsFirst ? data[3 * numBoxes + i] : data[i * numChannels + 3];

        if (![cx, cy, width, height].every(Number.isFinite) || width <= 0 || height <= 0) {
            continue;
        }

        const rawX1 = cx - width / 2;
        const rawY1 = cy - height / 2;
        const rawX2 = cx + width / 2;
        const rawY2 = cy + height / 2;

        const x1 = clampX((rawX1 - result.offsetX) / result.scale);
        const y1 = clampY((rawY1 - result.offsetY) / result.scale);
        const x2 = clampX((rawX2 - result.offsetX) / result.scale);
        const y2 = clampY((rawY2 - result.offsetY) / result.scale);

        if (x2 <= x1 || y2 <= y1) continue;

        let maskCoeffs = null;
        if (hasPrototypes && maskChannels > 0) {
            maskCoeffs = new Float32Array(maskChannels);
            for (let m = 0; m < maskChannels; m++) {
                const maskIndex = isChannelsFirst
                    ? (4 + numClasses + m) * numBoxes + i
                    : i * numChannels + 4 + numClasses + m;
                maskCoeffs[m] = data[maskIndex];
            }
        }

        candidates.push({
            x1, y1, x2, y2,
            rawX1, rawY1, rawX2, rawY2,
            score: bestScore,
            classId: bestClass,
            maskCoeffs
        });
    }

    return nms(candidates, IOU_THRESHOLD);
}

// ============================================================
// IOU & NON-MAX SUPPRESSION (NMS)
// ============================================================
function calculateIoU(a, b) {
    const xMin = Math.max(a.x1, b.x1);
    const yMin = Math.max(a.y1, b.y1);
    const xMax = Math.min(a.x2, b.x2);
    const yMax = Math.min(a.y2, b.y2);

    const intersection = Math.max(0, xMax - xMin) * Math.max(0, yMax - yMin);
    const areaA = (a.x2 - a.x1) * (a.y2 - a.y1);
    const areaB = (b.x2 - b.x1) * (b.y2 - b.y1);
    const union = areaA + areaB - intersection;

    return union > 0 ? intersection / union : 0;
}

function nms(boxes, threshold) {
    boxes.sort((a, b) => b.score - a.score);
    const selected = [];
    const suppressed = new Uint8Array(boxes.length);

    for (let i = 0; i < boxes.length; i++) {
        if (suppressed[i]) continue;
        const current = boxes[i];
        selected.push(current);

        for (let j = i + 1; j < boxes.length; j++) {
            if (suppressed[j]) continue;
            if (boxes[j].classId === current.classId) {
                if (calculateIoU(current, boxes[j]) >= threshold) {
                    suppressed[j] = 1;
                }
            }
        }
    }
    return selected;
}

// ============================================================
// VẼ BOUNDING BOX CHUYÊN NGHIỆP CHO FLOWER DETECTION
// ============================================================
function drawBoundingBoxes(detections, sourceW = video.videoWidth, sourceH = video.videoHeight) {
    const outW = overlay.width;
    const outH = overlay.height;
    overlayCtx.clearRect(0, 0, outW, outH);

    if (!detections.length || !outW || !outH) return;

    const scaleX = outW / (sourceW || outW);
    const scaleY = outH / (sourceH || outH);

    for (const det of detections) {
        const x1 = det.x1 * scaleX;
        const y1 = det.y1 * scaleY;
        const x2 = det.x2 * scaleX;
        const y2 = det.y2 * scaleY;
        const w = x2 - x1;
        const h = y2 - y1;

        if (w <= 0 || h <= 0) continue;

        const color = COLORS[det.classId % COLORS.length];
        const rgb = `${color[0]}, ${color[1]}, ${color[2]}`;
        const hex = `rgb(${rgb})`;

        // 1. Nền mờ bên trong bounding box
        overlayCtx.fillStyle = `rgba(${rgb}, 0.16)`;
        overlayCtx.fillRect(x1, y1, w, h);

        // 2. Viền bounding box
        overlayCtx.strokeStyle = hex;
        overlayCtx.lineWidth = 2.2;
        overlayCtx.strokeRect(x1, y1, w, h);

        // 3. Hiệu ứng 4 góc công nghệ cao (Corner Accents)
        const cornerLen = Math.min(18, Math.min(w, h) / 3.5);
        overlayCtx.lineWidth = 4;
        overlayCtx.beginPath();
        // Góc trên-trái
        overlayCtx.moveTo(x1, y1 + cornerLen);
        overlayCtx.lineTo(x1, y1);
        overlayCtx.lineTo(x1 + cornerLen, y1);
        // Góc trên-phải
        overlayCtx.moveTo(x2 - cornerLen, y1);
        overlayCtx.lineTo(x2, y1);
        overlayCtx.lineTo(x2, y1 + cornerLen);
        // Góc dưới-phải
        overlayCtx.moveTo(x2, y2 - cornerLen);
        overlayCtx.lineTo(x2, y2);
        overlayCtx.lineTo(x2 - cornerLen, y2);
        // Góc dưới-trái
        overlayCtx.moveTo(x1 + cornerLen, y2);
        overlayCtx.lineTo(x1, y2);
        overlayCtx.lineTo(x1, y2 - cornerLen);
        overlayCtx.stroke();

        // 4. Nhãn tên loài hoa + Độ chính xác (%)
        const vnName = CLASS_NAMES_VN[det.classId] || CLASS_NAMES[det.classId] || `Hoa ${det.classId}`;
        const confidence = (det.score * 100).toFixed(0);
        const labelText = `${vnName} • ${confidence}%`;

        overlayCtx.font = "bold 13px system-ui, -apple-system, BlinkMacSystemFont, sans-serif";
        const textMetrics = overlayCtx.measureText(labelText);
        const badgeW = textMetrics.width + 16;
        const badgeH = 24;

        let badgeX = x1;
        let badgeY = y1 - badgeH;
        if (badgeY < 0) badgeY = y1;
        if (badgeX + badgeW > outW) badgeX = Math.max(0, outW - badgeW);

        // Nền nhãn
        overlayCtx.fillStyle = hex;
        overlayCtx.beginPath();
        if (overlayCtx.roundRect) {
            overlayCtx.roundRect(badgeX, badgeY, badgeW, badgeH, [5, 5, 0, 0]);
        } else {
            overlayCtx.rect(badgeX, badgeY, badgeW, badgeH);
        }
        overlayCtx.fill();

        // Chữ nhãn
        overlayCtx.fillStyle = color[0] * 0.299 + color[1] * 0.587 + color[2] * 0.114 > 186 ? "#111827" : "#ffffff";
        overlayCtx.textBaseline = "middle";
        overlayCtx.fillText(labelText, badgeX + 8, badgeY + badgeH / 2);
    }
}

// ============================================================
// HỖ TRỢ VẼ MASK NẾU DÙNG MODEL SEGMENTATION
// ============================================================
function drawMasks(detections, result) {
    const outW = overlay.width;
    const outH = overlay.height;
    overlayCtx.clearRect(0, 0, outW, outH);
    if (!detections.length || !outW || !outH) return;

    const [, channels, mh, mw] = result.prototypes.dims;
    const proto = result.prototypes.data;
    const plane = mw * mh;

    if (!maskLogits || maskLogits.length !== plane) {
        maskLogits = new Float32Array(plane);
    }
    if (!maskImageData || maskCanvas.width !== outW || maskCanvas.height !== outH) {
        maskCanvas.width = outW;
        maskCanvas.height = outH;
        maskImageData = maskCtx.createImageData(outW, outH);
    }
    const rgba = maskImageData.data;
    rgba.fill(0);

    const stepX = result.newWidth * mw / (outW * INPUT_SIZE);
    const stepY = result.newHeight * mh / (outH * INPUT_SIZE);
    const originX = result.offsetX * mw / INPUT_SIZE;
    const originY = result.offsetY * mh / INPUT_SIZE;

    for (let i = detections.length - 1; i >= 0; i--) {
        const det = detections[i];
        if (!det.maskCoeffs) continue;
        const color = COLORS[det.classId % COLORS.length];

        const x1 = Math.max(0, Math.ceil((det.rawX1 - result.offsetX) * outW / result.newWidth - 0.5));
        const y1 = Math.max(0, Math.ceil((det.rawY1 - result.offsetY) * outH / result.newHeight - 0.5));
        const x2 = Math.min(outW, Math.ceil((det.rawX2 - result.offsetX) * outW / result.newWidth - 0.5));
        const y2 = Math.min(outH, Math.ceil((det.rawY2 - result.offsetY) * outH / result.newHeight - 0.5));
        if (x2 <= x1 || y2 <= y1) continue;

        const clampPX = x => Math.max(0, Math.min(mw - 1, x));
        const clampPY = y => Math.max(0, Math.min(mh - 1, y));
        const minPX = Math.floor(clampPX(originX + (x1 + 0.5) * stepX - 0.5));
        const minPY = Math.floor(clampPY(originY + (y1 + 0.5) * stepY - 0.5));
        const maxPX = Math.ceil(clampPX(originX + (x2 - 0.5) * stepX - 0.5));
        const maxPY = Math.ceil(clampPY(originY + (y2 - 0.5) * stepY - 0.5));

        for (let py = minPY; py <= maxPY; py++) {
            for (let px = minPX; px <= maxPX; px++) {
                const p = py * mw + px;
                let val = 0;
                for (let c = 0; c < channels; c++) {
                    val += det.maskCoeffs[c] * proto[c * plane + p];
                }
                maskLogits[p] = val;
            }
        }

        for (let y = y1; y < y2; y++) {
            const fy = clampPY(originY + (y + 0.5) * stepY - 0.5);
            const ya = Math.floor(fy);
            const yb = Math.min(ya + 1, mh - 1);
            const wy = fy - ya;

            for (let x = x1; x < x2; x++) {
                const fx = clampPX(originX + (x + 0.5) * stepX - 0.5);
                const xa = Math.floor(fx);
                const xb = Math.min(xa + 1, mw - 1);
                const wx = fx - xa;

                const top = maskLogits[ya * mw + xa] * (1 - wx) + maskLogits[ya * mw + xb] * wx;
                const bottom = maskLogits[yb * mw + xa] * (1 - wx) + maskLogits[yb * mw + xb] * wx;
                const logit = top * (1 - wy) + bottom * wy;

                if (logit > 0) {
                    const p = (y * outW + x) * 4;
                    rgba[p] = color[0];
                    rgba[p + 1] = color[1];
                    rgba[p + 2] = color[2];
                    rgba[p + 3] = 160;
                }
            }
        }
    }

    maskCtx.putImageData(maskImageData, 0, 0);
    overlayCtx.drawImage(maskCanvas, 0, 0);
}

// ============================================================
// HIỂN THỊ DANH SÁCH KẾT QUẢ DƯỚI BẢNG THÔNG TIN
// ============================================================
function drawLabels(detections) {
    if (objectsElement) objectsElement.textContent = String(detections.length);
    detectionsElement.replaceChildren();

    if (detections.length === 0) {
        const empty = document.createElement("div");
        empty.className = "detection-empty";
        empty.textContent = "Chưa phát hiện hoa nào";
        detectionsElement.appendChild(empty);
        return;
    }

    for (const det of detections) {
        const card = document.createElement("div");
        card.className = "detection-item";

        const color = COLORS[det.classId % COLORS.length];
        const dot = document.createElement("span");
        dot.className = "detection-dot";
        dot.style.backgroundColor = `rgb(${color[0]}, ${color[1]}, ${color[2]})`;
        dot.style.boxShadow = `0 0 8px rgba(${color[0]}, ${color[1]}, ${color[2]}, 0.8)`;

        const info = document.createElement("div");
        info.className = "detection-info";

        const title = document.createElement("div");
        title.className = "detection-title";

        const vn = document.createElement("strong");
        vn.textContent = CLASS_NAMES_VN[det.classId] || `Hoa ${det.classId}`;

        const en = document.createElement("span");
        en.className = "detection-vn";
        en.textContent = CLASS_NAMES[det.classId] ? ` (${CLASS_NAMES[det.classId]})` : "";

        title.appendChild(vn);
        title.appendChild(en);

        // Thanh tiến trình thể hiện độ tự tin
        const progressContainer = document.createElement("div");
        progressContainer.className = "detection-progress-bg";

        const progressBar = document.createElement("div");
        progressBar.className = "detection-progress-bar";
        const pct = Math.min(100, Math.max(0, det.score * 100));
        progressBar.style.width = `${pct}%`;
        progressBar.style.backgroundColor = `rgb(${color[0]}, ${color[1]}, ${color[2]})`;

        progressContainer.appendChild(progressBar);
        info.appendChild(title);
        info.appendChild(progressContainer);

        const scoreBadge = document.createElement("div");
        scoreBadge.className = "detection-score";
        scoreBadge.textContent = `${pct.toFixed(1)}%`;

        card.appendChild(dot);
        card.appendChild(info);
        card.appendChild(scoreBadge);

        detectionsElement.appendChild(card);
    }
}

// ============================================================
// FPS COUNTER
// ============================================================
function updateFPS() {
    frameCounter++;
    const now = performance.now();
    const elapsed = now - lastTime;
    if (elapsed >= 1000) {
        if (fpsElement) {
            fpsElement.textContent = (frameCounter * 1000 / elapsed).toFixed(1);
        }
        frameCounter = 0;
        lastTime = now;
    }
}

// ============================================================
// REALTIME INFERENCE LOOP
// ============================================================
async function inferenceLoop(timestamp, token) {
    if (!running || token !== loopVersion) return;

    if (
        (video.readyState >= 2 || video.currentTime > 0) &&
        timestamp - lastInferenceTime >= INFERENCE_INTERVAL &&
        !isInferencing
    ) {
        lastInferenceTime = timestamp;
        isInferencing = true;
        const t0 = performance.now();

        try {
            const result = await runInference();
            try {
                if (!running || token !== loopVersion) return;

                const detections = postprocess(result);

                if (result.prototypes) {
                    drawMasks(detections, result);
                } else {
                    drawBoundingBoxes(detections);
                }

                drawLabels(detections);

                const latency = Math.round(performance.now() - t0);
                if (latencyElement) latencyElement.textContent = `${latency}ms`;

                updateFPS();
            } finally {
                disposeOutputs(result.outputs);
            }
        } catch (error) {
            console.error("Inference loop error:", error);
            if (token === loopVersion) {
                stopCamera();
                setStatus("Lỗi nhận diện: " + error.message, "error");
            }
        } finally {
            isInferencing = false;
        }
    }

    if (running && token === loopVersion) {
        requestAnimationFrame(time => inferenceLoop(time, token));
    }
}

// ============================================================
// EVENT LISTENERS & INITIALIZATION
// ============================================================
loadButton.addEventListener("click", () => loadModel());
cameraButton.addEventListener("click", startCamera);
stopButton.addEventListener("click", stopCamera);

if (switchCamButton) {
    switchCamButton.addEventListener("click", switchCamera);
}

if (uploadButton && imageUploadInput) {
    uploadButton.addEventListener("click", () => imageUploadInput.click());
    imageUploadInput.addEventListener("change", handleImageUpload);
}

// Khởi tạo bảng danh mục và trạng thái ban đầu
renderSpeciesGuide();
setStatus("Nhấn 'TẢI MODEL' để bắt đầu nhận diện loài hoa (best.onnx)", "normal");