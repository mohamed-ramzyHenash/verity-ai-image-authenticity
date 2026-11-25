
"use strict";

/* =========================================
   Configuration
========================================= */

const API_BASE = (window.API_BASE ?? "").replace(/\/+$/, "");

const MAX_FILE_SIZE = 10 * 1024 * 1024;

const ALLOWED_TYPES = [
    "image/jpeg",
    "image/png"
];


/* =========================================
   DOM Elements
========================================= */

const $ = (id) => document.getElementById(id);

const elements = {
    dropzone: $("dropzone"),
    fileInput: $("fileInput"),
    browseButton: $("browseButton"),

    uploadPlaceholder: $("uploadPlaceholder"),
    imagePreview: $("imagePreview"),
    previewImage: $("previewImage"),
    scanLine: $("scanLine"),

    fileInformation: $("fileInformation"),
    fileName: $("fileName"),
    fileMeta: $("fileMeta"),
    removeButton: $("removeButton"),

    emptyState: $("emptyState"),
    loadingState: $("loadingState"),
    errorState: $("errorState"),
    errorMessage: $("errorMessage"),
    resultContent: $("resultContent"),

    systemStatus: $("systemStatus"),
    statusText: $("statusText"),

    resultBadge: $("resultBadge"),

    predictionTitle: $("predictionTitle"),
    predictionDescription: $("predictionDescription"),
    predictionSymbol: $("predictionSymbol"),

    confidenceValue: $("confidenceValue"),
    confidenceFill: $("confidenceFill"),
    confidenceProgress: $("confidenceProgress"),

    realValue: $("realValue"),
    aiValue: $("aiValue"),

    realFill: $("realFill"),
    aiFill: $("aiFill"),

    realProgress: $("realProgress"),
    aiProgress: $("aiProgress"),

    resultFileName: $("resultFileName"),

    retryButton: $("retryButton"),
    newAnalysisButton: $("newAnalysisButton")
};


/* =========================================
   Application State
========================================= */

let selectedFile = null;
let previewUrl = null;
let lastRequestFailed = false;
let requestController = null;


/* =========================================
   Utility Functions
========================================= */

function formatPercentage(value) {
    return `${(value * 100).toFixed(2)}%`;
}


function formatFileSize(bytes) {

    if (bytes < 1024 * 1024) {
        return `${(bytes / 1024).toFixed(1)} KB`;
    }

    return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}


function setStatus(status, message) {

    elements.systemStatus.classList.remove(
        "connected",
        "disconnected"
    );

    if (status === "connected") {
        elements.systemStatus.classList.add("connected");
    }

    if (status === "disconnected") {
        elements.systemStatus.classList.add("disconnected");
    }

    elements.statusText.textContent = message;
}


function setResultBadge(type, message) {

    elements.resultBadge.className = "result-badge";

    if (type === "success") {
        elements.resultBadge.classList.add("success");
    }

    if (type === "failed") {
        elements.resultBadge.classList.add("failed");
    }

    elements.resultBadge.textContent = message;
}


/* =========================================
   UI State Management
========================================= */

function showState(state) {

    const states = {
        empty: elements.emptyState,
        loading: elements.loadingState,
        error: elements.errorState,
        result: elements.resultContent
    };

    Object.entries(states).forEach(([name, element]) => {
        element.hidden = name !== state;
    });

    elements.scanLine.hidden = state !== "loading";

    if (state === "loading") {
        setResultBadge("", "PROCESSING");
    }

    if (state === "error") {
        setResultBadge("failed", "FAILED");
    }

    if (state === "empty") {
        setResultBadge("", "WAITING");
    }

    if (state === "result") {
        setResultBadge("success", "COMPLETED");
    }
}


function showError(message) {

    lastRequestFailed = selectedFile !== null;

    elements.errorMessage.textContent = message;

    showState("error");
}


/* =========================================
   Image Preview
========================================= */

function clearPreview() {

    if (previewUrl) {
        URL.revokeObjectURL(previewUrl);
        previewUrl = null;
    }

    elements.previewImage.onload = null;
    elements.previewImage.onerror = null;
    elements.previewImage.removeAttribute("src");

    elements.imagePreview.hidden = true;
    elements.uploadPlaceholder.hidden = false;
    elements.fileInformation.hidden = true;

    elements.dropzone.classList.remove(
        "has-image",
        "dragging"
    );
}


function displayPreview(file) {

    clearPreview();

    const currentPreviewUrl = URL.createObjectURL(file);

    previewUrl = currentPreviewUrl;

    elements.previewImage.src = currentPreviewUrl;

    elements.imagePreview.hidden = false;
    elements.uploadPlaceholder.hidden = true;
    elements.fileInformation.hidden = false;

    elements.dropzone.classList.add("has-image");

    elements.fileName.textContent = file.name;

    elements.fileMeta.textContent = formatFileSize(file.size);

    elements.previewImage.onload = () => {

        if (previewUrl !== currentPreviewUrl) {
            return;
        }

        const width = elements.previewImage.naturalWidth;
        const height = elements.previewImage.naturalHeight;

        elements.fileMeta.textContent =
            `${formatFileSize(file.size)} · ${width} × ${height} px`;

    };

}


/* =========================================
   File Validation
========================================= */

function validateFile(file) {

    if (!file) {
        return "Please select an image.";
    }

    if (!ALLOWED_TYPES.includes(file.type)) {
        return "Unsupported format. Please upload a JPEG or PNG image.";
    }

    if (file.size === 0) {
        return "The selected file is empty.";
    }

    if (file.size > MAX_FILE_SIZE) {
        return "File size exceeds the 10 MB limit.";
    }

    return null;
}


/* =========================================
   Prediction Validation
========================================= */

function validatePrediction(data) {

    if (!data || typeof data !== "object") {
        throw new Error("Invalid response received.");
    }

    if (typeof data.is_real !== "boolean") {
        throw new Error("The prediction result is invalid.");
    }

    const confidence = Number(data.confidence);
    const probabilityReal = Number(data.probability_real);
    const probabilityAI = Number(data.probability_ai);

    const values = [
        confidence,
        probabilityReal,
        probabilityAI
    ];

    const validProbabilities = values.every(
        value =>
            Number.isFinite(value) &&
            value >= 0 &&
            value <= 1
    );

    if (!validProbabilities) {
        throw new Error("The prediction values are invalid.");
    }

    const probabilitySum = probabilityReal + probabilityAI;

    if (Math.abs(probabilitySum - 1) > 0.01) {
        throw new Error("The prediction probabilities are inconsistent.");
    }

    const expectedConfidence = data.is_real
        ? probabilityReal
        : probabilityAI;

    if (Math.abs(confidence - expectedConfidence) > 0.01) {
        throw new Error("The confidence score is inconsistent.");
    }

    return {
        isReal: data.is_real,
        confidence,
        probabilityReal,
        probabilityAI
    };
}


/* =========================================
   Prediction Rendering
========================================= */

function renderResult(data) {

    const prediction = validatePrediction(data);

    const {
        isReal,
        confidence,
        probabilityReal,
        probabilityAI
    } = prediction;


    /* Prediction Header */

    elements.predictionTitle.textContent = isReal
        ? "REAL IMAGE"
        : "AI-GENERATED";

    elements.predictionTitle.className = isReal
        ? "real"
        : "ai";

    elements.predictionDescription.textContent = isReal
        ? "The image appears to be a real photograph."
        : "The image appears to be AI-generated.";

    elements.predictionSymbol.textContent = isReal
        ? "R"
        : "AI";

    elements.predictionSymbol.className =
        `prediction-symbol ${isReal ? "real" : "ai"}`;


    /* Confidence */

    elements.confidenceValue.textContent =
        formatPercentage(confidence);

    elements.confidenceValue.className = isReal
        ? "real"
        : "ai";

    elements.confidenceFill.className =
        `confidence-fill ${isReal ? "real" : "ai"}`;

    elements.confidenceFill.style.width =
        formatPercentage(confidence);

    elements.confidenceProgress.setAttribute(
        "aria-valuenow",
        (confidence * 100).toFixed(2)
    );


    /* Real Probability */

    elements.realValue.textContent =
        formatPercentage(probabilityReal);

    elements.realFill.style.width =
        formatPercentage(probabilityReal);

    elements.realProgress.setAttribute(
        "aria-valuenow",
        (probabilityReal * 100).toFixed(2)
    );


    /* AI Probability */

    elements.aiValue.textContent =
        formatPercentage(probabilityAI);

    elements.aiFill.style.width =
        formatPercentage(probabilityAI);

    elements.aiProgress.setAttribute(
        "aria-valuenow",
        (probabilityAI * 100).toFixed(2)
    );


    /* Analyzed File */

    elements.resultFileName.textContent =
        data.filename || selectedFile?.name || "Unknown image";


    showState("result");
}


/* =========================================
   API Error Handling
========================================= */

function getAPIError(response, data) {

    switch (response.status) {

        case 400:
            return "We couldn't read this image. It may be corrupted.";

        case 413:
            return "This image is too large. Please choose a smaller file.";

        case 415:
            return "Unsupported image format. Please upload a JPEG or PNG.";

        case 503:
            return "Image analysis is temporarily unavailable. Please try again later.";

        default:
            return "Something went wrong while analyzing your image.";
    }

}


/* =========================================
   Image Analysis
========================================= */

async function analyzeImage(file) {

    const validationError = validateFile(file);

    if (validationError) {

        selectedFile = null;

        lastRequestFailed = false;

        elements.fileInput.value = "";

        clearPreview();

        showError(validationError);

        return;
    }


    selectedFile = file;

    lastRequestFailed = false;

    displayPreview(file);

    showState("loading");


    const formData = new FormData();

    formData.append("file", file);


    const controller = new AbortController();

    requestController = controller;


    try {

        const response = await fetch(
            `${API_BASE}/api/predict`,
            {
                method: "POST",
                body: formData,
                signal: controller.signal
            }
        );


        let data;

        try {

            data = await response.json();

        } catch {

            throw new Error(
                "The analysis service returned an invalid response."
            );

        }


        if (!response.ok) {

            throw new Error(
                getAPIError(response, data)
            );

        }


        renderResult(data);


    } catch (error) {

        if (error.name === "AbortError") {
            return;
        }

        console.error("Image analysis error:", error);

        showError(
            error.message ||
            "Unable to analyze this image. Please try again."
        );


    } finally {

        if (requestController === controller) {
            requestController = null;
        }

    }

}


/* =========================================
   File Selection
========================================= */

function handleFileSelection(file) {

    if (!file) {
        return;
    }

    if (requestController) {
        requestController.abort();
        requestController = null;
    }

    analyzeImage(file);
}


/* Browse Button */

elements.browseButton.addEventListener("click", (event) => {

    event.stopPropagation();

    elements.fileInput.click();

});


/* Dropzone Click */

elements.dropzone.addEventListener("click", () => {

    if (!selectedFile) {
        elements.fileInput.click();
    }

});


/* Keyboard Accessibility */

elements.dropzone.addEventListener("keydown", (event) => {

    if (
        (event.key === "Enter" || event.key === " ") &&
        !selectedFile
    ) {

        event.preventDefault();

        elements.fileInput.click();

    }

});


/* File Input Change */

elements.fileInput.addEventListener("change", () => {

    const file = elements.fileInput.files[0];

    handleFileSelection(file);

});


/* =========================================
   Drag and Drop
========================================= */

["dragenter", "dragover"].forEach(eventName => {

    elements.dropzone.addEventListener(eventName, (event) => {

        event.preventDefault();

        elements.dropzone.classList.add("dragging");

    });

});


["dragleave", "drop"].forEach(eventName => {

    elements.dropzone.addEventListener(eventName, (event) => {

        event.preventDefault();

        elements.dropzone.classList.remove("dragging");

    });

});


elements.dropzone.addEventListener("drop", (event) => {

    const file = event.dataTransfer.files[0];

    handleFileSelection(file);

});


/* =========================================
   Reset Application
========================================= */

function resetApplication() {

    if (requestController) {

        requestController.abort();

        requestController = null;

    }

    selectedFile = null;

    lastRequestFailed = false;

    elements.fileInput.value = "";

    clearPreview();

    showState("empty");

}


/* Remove Selected Image */

elements.removeButton.addEventListener("click", () => {

    resetApplication();

});


/* Analyze Another Image */

elements.newAnalysisButton.addEventListener("click", () => {

    resetApplication();

    elements.fileInput.click();

});


/* Retry Failed Analysis */

elements.retryButton.addEventListener("click", () => {

    if (selectedFile && lastRequestFailed) {

        analyzeImage(selectedFile);

    } else {

        resetApplication();

    }

});


/* =========================================
   API Health Check
========================================= */

async function checkAPIHealth() {

    try {

        const response = await fetch(
            `${API_BASE}/api/health`,
            {
                method: "GET",
                cache: "no-store"
            }
        );


        if (!response.ok) {

            throw new Error("Service health check failed.");

        }


        const health = await response.json();


        if (health.model_loaded === true) {

            setStatus(
                "connected",
                "Service ready"
            );

        } else {

            setStatus(
                "disconnected",
                "Service unavailable"
            );

        }


    } catch (error) {

        console.error("Service connection error:", error);

        setStatus(
            "disconnected",
            "Service unavailable"
        );

    }

}


/* =========================================
   Application Initialization
========================================= */

function initializeApp() {

    showState("empty");

    checkAPIHealth();

}

initializeApp();
